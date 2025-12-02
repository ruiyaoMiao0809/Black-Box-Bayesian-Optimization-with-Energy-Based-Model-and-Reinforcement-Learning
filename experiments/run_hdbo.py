#!/usr/bin/env python
"""
Run REBMBO experiments on HDBO 200D benchmark.

This script reproduces the HDBO 200D results from the paper (Figure 2d, Table 1).
High-Dimensional Bayesian Optimization tests scalability to 200 dimensions.

Paper definition (Appendix B - HDBO-200D):
    f(x) = Σ_{i=1}^{200} e^(x_i)
    Domain: [-5, 5]^200 (scaled to [0, 1]^200)
    
    For MAXIMIZATION (negate the function):
    f(x) = -Σ_{i=1}^{200} e^(x_i)
    
    Optimal: x* = (-5, -5, ..., -5) in original domain
             x* = (0, 0, ..., 0) in [0, 1] scaled domain
    Optimal value: -200 * e^(-5) ≈ -1.348

Characteristics:
    - Very high dimensional (200D)
    - Additive structure but challenging
    - Tests algorithm scalability
    - Known ground-truth for evaluation
    - MUST use Sparse GP (classic GP is O(n³) = infeasible)

Usage:
    # Quick test (20 iterations, sparse GP)
    python run_hdbo.py --quick
    
    # Standard run (50 iterations, as in paper Table 1)
    python run_hdbo.py --n_iterations 50
    
    # Extended run (100 iterations)
    python run_hdbo.py --n_iterations 100
    
    # Multi-seed experiment
    python run_hdbo.py --seeds 1 2 3 4 5
    
    # Full paper reproduction (5 seeds, T=50 and T=100)
    python run_hdbo.py --full
    
    # Different GP variants (sparse recommended, deep also good)
    python run_hdbo.py --variant sparse  # Recommended
    python run_hdbo.py --variant deep    # Alternative

Author: REBMBO Project
Date: December 2024
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try different import paths
try:
    from src.rebmbo.algorithm import REBMBO, REBMBOConfig
except ImportError:
    try:
        from rebmbo.algorithm import REBMBO, REBMBOConfig
    except ImportError:
        from algorithm import REBMBO, REBMBOConfig

try:
    from src.utils.logger import ExperimentLogger
except ImportError:
    try:
        from utils.logger import ExperimentLogger
    except ImportError:
        ExperimentLogger = None

try:
    from src.utils.plotting import plot_convergence_curve, plot_lar_curve
except ImportError:
    try:
        from utils.plotting import plot_convergence_curve, plot_lar_curve
    except ImportError:
        plot_convergence_curve = None
        plot_lar_curve = None


# ============ HDBO 200D Benchmark Definition ============

class HDBO200DInfo:
    """Information about the HDBO 200D benchmark."""
    def __init__(self):
        self.name = "HDBO-200D"
        self.dim = 200
        self.bounds = (0.0, 1.0)  # Normalized bounds
        self.original_bounds = (-5.0, 5.0)
        
        # Optimal solution
        # In original domain: x* = (-5, -5, ..., -5)
        # In [0,1] domain: x* = (0, 0, ..., 0)
        # Optimal value: -200 * exp(-5) ≈ -1.348 (for maximization)
        self.optimal_x = np.zeros(200)
        self.optimal_value = -200 * np.exp(-5)  # ≈ -1.3476


def hdbo_200d_function(x: np.ndarray) -> float:
    """
    HDBO 200D benchmark function.
    
    Paper definition:
        f(x) = Σ_{i=1}^{200} e^(x_i)
    
    For maximization (negate):
        f(x) = -Σ_{i=1}^{200} e^(x_i)
    
    Args:
        x: Input vector in [0, 1]^200 (normalized domain)
        
    Returns:
        Negated sum of exponentials (for maximization)
    """
    # Ensure x is numpy array and flattened
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    x = np.asarray(x).flatten()
    
    # Scale from [0, 1] to [-5, 5]
    x_scaled = x * 10 - 5  # [0, 1] -> [-5, 5]
    
    # Compute sum of exponentials
    result = np.sum(np.exp(x_scaled))
    
    # Negate for maximization (we want to minimize the sum)
    return -result


def get_hdbo_benchmark():
    """Get the HDBO 200D benchmark function and info."""
    return hdbo_200d_function, HDBO200DInfo()


# ============ Configuration ============

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_hdbo_config(variant: str = "sparse", use_ppo: bool = True) -> REBMBOConfig:
    """
    Get optimized configuration for HDBO 200D benchmark.
    
    CRITICAL: For 200D, Sparse GP is strongly recommended.
    Classic GP has O(n³) complexity which is infeasible.
    
    Args:
        variant: GP variant - "sparse" (recommended), "deep", or "classic" (not recommended)
        use_ppo: Whether to use PPO module
        
    Returns:
        REBMBOConfig optimized for 200D
    """
    # Warn if using classic GP
    if variant == "classic":
        print("WARNING: Classic GP is not recommended for 200D due to O(n³) complexity!")
        print("         Consider using 'sparse' or 'deep' variant instead.")
    
    return REBMBOConfig(
        # Problem definition
        input_dim=200,
        bounds=(0.0, 1.0),
        
        # GP configuration (Module A)
        gp_variant=variant,
        gp_num_inducing=100,      # For sparse GP
        gp_hidden_dims=[128, 64], # For deep GP
        gp_latent_dim=64,         # For deep GP
        gp_train_epochs=200,
        gp_retrain_epochs=100,
        
        # EBM configuration (Module B) - Larger for 200D
        ebm_hidden_dims=[512, 512, 256],
        ebm_train_epochs=150,
        ebm_retrain_epochs=80,
        ebm_mcmc_steps=50,        # More steps for 200D
        ebm_mcmc_step_size=0.02,  # Smaller steps for stability
        ebm_num_negative_samples=512,
        
        # Acquisition function
        beta=3.0,    # Higher exploration for 200D
        gamma=0.2,   # Lower EBM weight
        
        # PPO configuration (Module C) - Larger for 200D action space
        ppo_hidden_dims=[1024, 512, 256],
        ppo_lr_actor=1e-4,        # Lower for stability
        ppo_lr_critic=5e-4,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_clip_epsilon=0.2,
        ppo_entropy_coef=0.1,     # Higher entropy for 200D exploration
        ppo_value_coef=0.5,
        ppo_epochs=15,
        ppo_mini_batch_size=64,
        lambda_energy=0.15,
        
        # State encoder - fewer grid points for 200D
        num_grid_points=20,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


# ============ Metrics ============

def compute_normalized_score(best_y: float, worst_y: float, optimal_y: float) -> float:
    """
    Compute normalized score as percentage of optimal.
    
    Score = (best_y - worst_y) / (optimal_y - worst_y) * 100
    """
    # Avoid division by zero
    if abs(optimal_y - worst_y) < 1e-10:
        return 100.0 if best_y >= optimal_y else 0.0
    
    score = (best_y - worst_y) / (optimal_y - worst_y) * 100
    return max(0, min(100, score))


def compute_lar(f_star: float, f_t: float, alpha: float = 0.1) -> float:
    """
    Compute Landscape-Aware Regret (simplified for known optimal).
    
    LAR = f(x*) - f(x_t)
    """
    return f_star - f_t


# ============ Single Experiment ============

def run_single_hdbo_experiment(
    variant: str = "sparse",
    n_init: int = 20,
    n_iterations: int = 50,
    seed: int = 42,
    use_ppo: bool = True,
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run a single REBMBO experiment on HDBO 200D benchmark.
    
    Args:
        variant: GP variant ("sparse" recommended, "deep", or "classic")
        n_init: Number of initial random samples
        n_iterations: Number of BO iterations
        seed: Random seed
        use_ppo: Whether to use PPO module
        save_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        Dictionary containing results and metrics
    """
    # Set seed
    set_seed(seed)
    
    # Get benchmark
    benchmark_fn, benchmark_info = get_hdbo_benchmark()
    
    # Print header
    if verbose:
        print("\n" + "="*70)
        print(f"REBMBO on {benchmark_info.name}")
        print(f"Variant: REBMBO-{variant.upper()}")
        print(f"Seed: {seed}")
        print(f"Iterations: {n_iterations}, Initial points: {n_init}")
        print(f"Using PPO: {use_ppo}")
        print(f"Optimal value: {benchmark_info.optimal_value:.4f}")
        print("="*70)
    
    # Create configuration
    config = get_hdbo_config(variant=variant, use_ppo=use_ppo)
    
    # Initialize REBMBO
    optimizer = REBMBO(config)
    
    # Generate initial samples
    if verbose:
        print(f"\nGenerating {n_init} initial samples in 200D space...")
    
    X_init = np.random.rand(n_init, benchmark_info.dim)
    y_init = np.array([benchmark_fn(x) for x in X_init])
    
    if verbose:
        print(f"Initial samples: best_y = {np.max(y_init):.4f}, "
              f"worst_y = {np.min(y_init):.4f}")
        print(f"Distance to optimal: {benchmark_info.optimal_value - np.max(y_init):.4f}")
    
    # Convert to torch tensors
    X_init_tensor = torch.tensor(X_init, dtype=torch.float32, device=config.device)
    y_init_tensor = torch.tensor(y_init, dtype=torch.float32, device=config.device)
    
    # Initialize optimizer with initial data
    optimizer.initialize(X_init_tensor, y_init_tensor)
    
    # Tracking
    history = {
        'iterations': [],
        'x': [],
        'y': [],
        'best_y': [],
        'lar': [],
        'time': []
    }
    
    best_y = np.max(y_init)
    start_time = time.time()
    
    # Record initial samples
    for i, (x_i, y_i) in enumerate(zip(X_init, y_init)):
        history['iterations'].append(0)
        # For 200D, only store first few components to save memory
        history['x'].append(x_i[:10].tolist())  # Store first 10 dims only
        history['y'].append(float(y_i))
        history['best_y'].append(float(np.max(y_init[:i+1])))
        history['lar'].append(float(benchmark_info.optimal_value - np.max(y_init[:i+1])))
        history['time'].append(0)
    
    # Wrapper function
    def objective_wrapper(x):
        """Wrapper to convert between numpy and torch."""
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = np.asarray(x)
        x_np = x_np.flatten()
        x_np = np.clip(x_np, 0, 1)
        return benchmark_fn(x_np)
    
    # Main optimization loop
    if verbose:
        print(f"\nStarting optimization...")
        print("-"*70)
    
    for t in range(1, n_iterations + 1):
        iter_start = time.time()
        
        # Run one step of REBMBO
        x_next, y_next = optimizer.step(
            objective_fn=objective_wrapper,
            use_ppo=use_ppo,
            verbose=False
        )
        
        # Convert to numpy
        x_next_np = x_next.cpu().numpy().flatten()
        
        # Update best
        if y_next > best_y:
            best_y = y_next
        
        # Compute metrics
        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start
        lar = benchmark_info.optimal_value - best_y
        
        # Record (only first 10 dims to save memory)
        history['iterations'].append(t)
        history['x'].append(x_next_np[:10].tolist())
        history['y'].append(float(y_next))
        history['best_y'].append(float(best_y))
        history['lar'].append(float(lar))
        history['time'].append(float(elapsed))
        
        # Progress output
        if verbose and (t % 5 == 0 or t == n_iterations or t == 1):
            score = compute_normalized_score(
                best_y, 
                min(history['y']), 
                benchmark_info.optimal_value
            )
            print(f"Iter {t:3d}: y={y_next:.4f}, best_y={best_y:.4f}, "
                  f"LAR={lar:.4f}, Score={score:.1f}%, time={iter_time:.2f}s")
    
    # Final statistics
    total_time = time.time() - start_time
    
    all_y = [float(y) for y in history['y']]
    worst_y = min(all_y)
    best_y_final = max(all_y)
    best_idx = all_y.index(best_y_final)
    
    normalized_score = compute_normalized_score(
        best_y_final, worst_y, benchmark_info.optimal_value
    )
    final_lar = benchmark_info.optimal_value - best_y_final
    
    results = {
        'benchmark': benchmark_info.name,
        'variant': variant,
        'seed': seed,
        'dim': benchmark_info.dim,
        'n_iterations': n_iterations,
        'n_init': n_init,
        'use_ppo': use_ppo,
        'optimal_value': float(benchmark_info.optimal_value),
        'best_x_partial': history['x'][best_idx],  # First 10 dims
        'best_y': float(best_y_final),
        'worst_y': float(worst_y),
        'final_lar': float(final_lar),
        'normalized_score': float(normalized_score),
        'total_time': float(total_time),
        'history': history
    }
    
    if verbose:
        print("-"*70)
        print(f"Optimization complete!")
        print(f"Best y: {best_y_final:.4f} (optimal: {benchmark_info.optimal_value:.4f})")
        print(f"Final LAR: {final_lar:.4f}")
        print(f"Normalized Score: {normalized_score:.2f}%")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print("="*70)
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        results_file = os.path.join(save_dir, f'results_seed{seed}.json')
        with open(results_file, 'w') as f:
            json_results = json.loads(
                json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
            )
            json.dump(json_results, f, indent=2)
        
        # Plot convergence
        if plot_convergence_curve is not None:
            fig_file = os.path.join(save_dir, f'convergence_seed{seed}.png')
            try:
                plot_convergence_curve(
                    history['iterations'],
                    history['best_y'],
                    title=f"{benchmark_info.name} - Seed {seed}",
                    save_path=fig_file
                )
            except Exception as e:
                print(f"Warning: Could not save plot: {e}")
    
    return results


# ============ Multi-Seed Experiment ============

def run_multi_seed_experiment(
    variant: str = "sparse",
    n_init: int = 20,
    n_iterations: int = 50,
    seeds: list = [1, 2, 3, 4, 5],
    use_ppo: bool = True,
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run multi-seed experiment for statistical analysis.
    """
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"results/hdbo_200d/multi_seed_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    print("\n" + "="*70)
    print(f"MULTI-SEED EXPERIMENT: HDBO 200D")
    print(f"Variant: REBMBO-{variant.upper()}")
    print(f"Seeds: {seeds}")
    print(f"Iterations: {n_iterations}, Initial points: {n_init}")
    print("="*70)
    
    for seed in seeds:
        print(f"\n{'>'*20} Running seed {seed} {'<'*20}")
        
        seed_save_dir = os.path.join(save_dir, f"seed_{seed}")
        
        results = run_single_hdbo_experiment(
            variant=variant,
            n_init=n_init,
            n_iterations=n_iterations,
            seed=seed,
            use_ppo=use_ppo,
            save_dir=seed_save_dir,
            verbose=verbose
        )
        
        all_results.append(results)
    
    # Aggregate statistics
    best_ys = [r['best_y'] for r in all_results]
    final_lars = [r['final_lar'] for r in all_results]
    scores = [r['normalized_score'] for r in all_results]
    times = [r['total_time'] for r in all_results]
    
    aggregated = {
        'benchmark': "HDBO-200D",
        'variant': variant,
        'seeds': seeds,
        'n_iterations': n_iterations,
        'n_init': n_init,
        'dim': 200,
        
        # Statistics
        'best_y_mean': float(np.mean(best_ys)),
        'best_y_std': float(np.std(best_ys)),
        'final_lar_mean': float(np.mean(final_lars)),
        'final_lar_std': float(np.std(final_lars)),
        'normalized_score_mean': float(np.mean(scores)),
        'normalized_score_std': float(np.std(scores)),
        'time_mean': float(np.mean(times)),
        'time_std': float(np.std(times)),
        
        # Individual results
        'individual_results': all_results
    }
    
    # Print summary
    print("\n" + "="*70)
    print("MULTI-SEED EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Benchmark: HDBO-200D")
    print(f"Variant: REBMBO-{variant.upper()}")
    print(f"Seeds: {seeds}")
    print("-"*70)
    print(f"Best y:           {aggregated['best_y_mean']:.4f} ± {aggregated['best_y_std']:.4f}")
    print(f"Final LAR:        {aggregated['final_lar_mean']:.4f} ± {aggregated['final_lar_std']:.4f}")
    print(f"Normalized Score: {aggregated['normalized_score_mean']:.2f}% ± {aggregated['normalized_score_std']:.2f}%")
    print(f"Time:             {aggregated['time_mean']:.2f}s ± {aggregated['time_std']:.2f}s")
    print("="*70)
    
    # Save aggregated results
    agg_file = os.path.join(save_dir, 'aggregated_results.json')
    with open(agg_file, 'w') as f:
        json.dump(aggregated, f, indent=2, 
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    
    print(f"\nResults saved to: {save_dir}")
    
    return aggregated


# ============ Main Entry Point ============

def main():
    parser = argparse.ArgumentParser(
        description="Run REBMBO on HDBO 200D benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (20 iterations)
  python run_hdbo.py --quick
  
  # Standard run (50 iterations, sparse GP)
  python run_hdbo.py
  
  # Extended run (100 iterations)
  python run_hdbo.py --n_iterations 100
  
  # Multi-seed experiment
  python run_hdbo.py --seeds 1 2 3 4 5
  
  # Full paper reproduction
  python run_hdbo.py --full

Notes:
  - Sparse GP (--variant sparse) is STRONGLY RECOMMENDED for 200D
  - Classic GP has O(n³) complexity and will be very slow
  - GPU is recommended for reasonable performance
        """
    )
    
    # Basic arguments
    parser.add_argument('--variant', type=str, default='sparse',
                        choices=['classic', 'sparse', 'deep'],
                        help='GP variant (default: sparse, RECOMMENDED for 200D)')
    parser.add_argument('--n_init', type=int, default=20,
                        help='Number of initial samples (default: 20)')
    parser.add_argument('--n_iterations', type=int, default=50,
                        help='Number of iterations (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for single run (default: 42)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Multiple seeds for statistical analysis')
    
    # Algorithm options
    parser.add_argument('--no_ppo', action='store_true',
                        help='Disable PPO module')
    
    # Output options
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    # Preset configurations
    parser.add_argument('--full', action='store_true',
                        help='Full paper reproduction (5 seeds, T=50)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (20 iterations)')
    parser.add_argument('--extended', action='store_true',
                        help='Extended run (100 iterations)')
    
    args = parser.parse_args()
    
    # Handle presets
    if args.full:
        args.seeds = [1, 2, 3, 4, 5]
        args.n_iterations = 50
        args.variant = 'sparse'  # Force sparse for 200D
    elif args.quick:
        args.n_iterations = 20
    elif args.extended:
        args.n_iterations = 100
    
    # Set default save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"results/hdbo_200d/{args.variant}_{timestamp}"
    
    # Run experiment(s)
    if args.seeds:
        results = run_multi_seed_experiment(
            variant=args.variant,
            n_init=args.n_init,
            n_iterations=args.n_iterations,
            seeds=args.seeds,
            use_ppo=not args.no_ppo,
            save_dir=args.save_dir,
            verbose=not args.quiet
        )
    else:
        results = run_single_hdbo_experiment(
            variant=args.variant,
            n_init=args.n_init,
            n_iterations=args.n_iterations,
            seed=args.seed,
            use_ppo=not args.no_ppo,
            save_dir=args.save_dir,
            verbose=not args.quiet
        )
    
    return results


if __name__ == "__main__":
    main()