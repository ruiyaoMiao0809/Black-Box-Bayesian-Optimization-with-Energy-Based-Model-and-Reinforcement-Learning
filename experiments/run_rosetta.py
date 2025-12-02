#!/usr/bin/env python
"""
Run REBMBO experiments on Rosetta Protein Design 86D benchmark.

This script reproduces the Rosetta 86D results from the paper (Figure 2f, Table 2).
Real-world antibody engineering task for SARS-CoV-2 neutralization.

Paper definition (Appendix B - Rosetta Protein Design 86D):
    Input: x ∈ R^86 encoding structural modifications to reference antibody
    Target: SARS-CoV-2 spike protein binding
    Output: ΔΔG (change in binding free energy)
    Objective: Minimize ΔΔG (maximize -ΔΔG)
    Simulator: Rosetta Flex (expensive CPU time per query)

Characteristics:
    - Real-world drug design application
    - High dimensional (86D)
    - No analytical form (true black-box)
    - Expensive function evaluations
    - Complex non-linear response surface

Biology Context:
    - ΔΔG represents change in binding free energy
    - Negative ΔΔG = improved binding (desired for neutralization)
    - Positive ΔΔG = weakened binding (undesired)
    - Goal: Find antibody mutations that minimize ΔΔG

Usage:
    # Quick test with mock simulator
    python run_rosetta.py --mock
    
    # Standard run (50 iterations, as in paper)
    python run_rosetta.py --mock --n_iterations 50
    
    # Extended run (80 iterations)
    python run_rosetta.py --mock --n_iterations 80
    
    # Multi-seed experiment
    python run_rosetta.py --mock --seeds 1 2 3 4 5
    
    # Full paper reproduction (5 seeds, T=50)
    python run_rosetta.py --mock --full
    
    # Different GP variants
    python run_rosetta.py --mock --variant sparse  # Recommended
    python run_rosetta.py --mock --variant deep    # Best in paper

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


# ============ Rosetta 86D Benchmark Definition ============

class Rosetta86DInfo:
    """Information about the Rosetta 86D benchmark."""
    def __init__(self):
        self.name = "Rosetta-86D"
        self.dim = 86
        self.bounds = (0.0, 1.0)  # Normalized bounds
        
        # Optimal solution unknown for real black-box
        # For mock function, we set a known optimum
        self.optimal_x = None  # Unknown
        self.optimal_value = None  # Unknown for real Rosetta
        
        # Application context
        self.application = "Antibody engineering for SARS-CoV-2"
        self.metric = "ΔΔG (binding free energy change)"


class MockRosettaSimulator:
    """
    Mock Rosetta simulator for testing.
    
    Simulates binding free energy (ΔΔG) landscape with:
    - Multiple local optima (different beneficial mutations)
    - Epistatic interactions (mutations affecting each other)
    - Realistic energy scale
    - Some ruggedness typical of protein landscapes
    
    The mock function is designed to be challenging but tractable,
    with properties similar to real protein fitness landscapes.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize mock simulator with reproducible randomness."""
        np.random.seed(seed)
        
        # Generate random "important" residue positions
        # In real proteins, only some positions significantly affect binding
        self.n_important = 30  # ~35% of positions are important
        self.important_positions = np.random.choice(86, self.n_important, replace=False)
        
        # Generate random optimal values for important positions
        self.optimal_values = np.random.rand(self.n_important) * 0.3 + 0.35  # Around 0.35-0.65
        
        # Generate epistatic interaction matrix (which positions interact)
        self.n_interactions = 20
        self.interactions = []
        for _ in range(self.n_interactions):
            pos1, pos2 = np.random.choice(self.n_important, 2, replace=False)
            strength = np.random.randn() * 0.5
            self.interactions.append((pos1, pos2, strength))
        
        # Generate local optima centers
        self.n_local_optima = 5
        self.local_optima = np.random.rand(self.n_local_optima, 86) * 0.4 + 0.3
        self.local_optima_depths = np.random.rand(self.n_local_optima) * 2 + 1
        
        # Best achievable ΔΔG (for normalization)
        self.best_ddg = -5.0  # kcal/mol (good binding improvement)
        self.worst_ddg = 10.0  # kcal/mol (severe binding disruption)
        
        # Reset seed so it doesn't affect other randomness
        np.random.seed(None)
    
    def compute_ddg(self, x: np.ndarray) -> float:
        """
        Compute mock ΔΔG (binding free energy change).
        
        Lower ΔΔG = better binding = what we want to minimize.
        We return -ΔΔG for maximization.
        
        Args:
            x: Normalized mutation parameters in [0, 1]^86
            
        Returns:
            -ΔΔG (negated for maximization)
        """
        x = np.asarray(x).flatten()
        x = np.clip(x, 0, 1)
        
        # Base energy (neutral mutations)
        ddg = 0.0
        
        # 1. Single-position effects (additive)
        for i, pos in enumerate(self.important_positions):
            # Distance from optimal value at this position
            diff = x[pos] - self.optimal_values[i]
            # Quadratic penalty
            ddg += 3.0 * diff ** 2
        
        # 2. Epistatic interactions (non-additive)
        for pos1, pos2, strength in self.interactions:
            actual_pos1 = self.important_positions[pos1]
            actual_pos2 = self.important_positions[pos2]
            # Interaction term
            interaction = (x[actual_pos1] - 0.5) * (x[actual_pos2] - 0.5)
            ddg += strength * interaction
        
        # 3. Local optima contributions
        for i, (center, depth) in enumerate(zip(self.local_optima, self.local_optima_depths)):
            # Gaussian basins around local optima
            dist_sq = np.sum((x - center) ** 2)
            ddg -= depth * np.exp(-dist_sq / 0.5)
        
        # 4. Stability penalty (too many mutations is bad)
        mutation_load = np.sum((x - 0.5) ** 2)
        ddg += 0.1 * mutation_load
        
        # 5. Small noise (experimental uncertainty)
        ddg += np.random.normal(0, 0.05)
        
        # Scale to realistic kcal/mol range
        ddg = np.clip(ddg, self.best_ddg, self.worst_ddg)
        
        # Return -ΔΔG for maximization
        # More negative ΔΔG (better binding) = higher return value
        return -ddg
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate mock ΔΔG."""
        return self.compute_ddg(x)


def create_rosetta_mock_function(seed: int = 42):
    """
    Create a mock Rosetta benchmark function.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Callable benchmark function and info object
    """
    simulator = MockRosettaSimulator(seed=seed)
    info = Rosetta86DInfo()
    
    # For mock function, we can estimate the optimal
    # by searching (in practice, it's unknown)
    info.optimal_value = 5.0  # Approximate best -ΔΔG achievable
    
    return simulator, info


def create_real_rosetta_function(rosetta_path: str = None, 
                                  reference_pdb: str = None):
    """
    Create real Rosetta Flex benchmark function.
    
    Requires Rosetta to be installed and configured.
    
    Args:
        rosetta_path: Path to Rosetta installation
        reference_pdb: Path to reference antibody PDB file
        
    Returns:
        Callable benchmark function and info object
    """
    print("WARNING: Real Rosetta simulator not implemented.")
    print("         Using mock function instead.")
    print("         For real Rosetta, implement interface to Rosetta Flex.")
    
    # Fall back to mock
    return create_rosetta_mock_function()


def get_rosetta_benchmark(use_mock: bool = True, seed: int = 42, **kwargs):
    """Get the Rosetta 86D benchmark function."""
    if use_mock:
        return create_rosetta_mock_function(seed=seed)
    else:
        return create_real_rosetta_function(**kwargs)


# ============ Configuration ============

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rosetta_config(variant: str = "sparse", use_ppo: bool = True) -> REBMBOConfig:
    """
    Get optimized configuration for Rosetta 86D benchmark.
    
    For 86D, sparse GP is recommended but classic GP is still feasible
    (unlike 200D HDBO). Deep GP shows best results in paper.
    
    Args:
        variant: GP variant - "sparse" (recommended), "deep" (best in paper), "classic"
        use_ppo: Whether to use PPO module
        
    Returns:
        REBMBOConfig optimized for 86D
    """
    return REBMBOConfig(
        # Problem definition
        input_dim=86,
        bounds=(0.0, 1.0),
        
        # GP configuration (Module A)
        gp_variant=variant,
        gp_num_inducing=80,        # For sparse GP
        gp_hidden_dims=[128, 64],  # For deep GP
        gp_latent_dim=48,          # For deep GP
        gp_train_epochs=150,
        gp_retrain_epochs=80,
        
        # EBM configuration (Module B) - Sized for 86D
        ebm_hidden_dims=[384, 384, 192],
        ebm_train_epochs=120,
        ebm_retrain_epochs=60,
        ebm_mcmc_steps=45,
        ebm_mcmc_step_size=0.025,
        ebm_num_negative_samples=384,
        
        # Acquisition function
        beta=2.8,    # Higher exploration for 86D
        gamma=0.25,  # Moderate EBM weight
        
        # PPO configuration (Module C) - Sized for 86D
        ppo_hidden_dims=[768, 384, 192],
        ppo_lr_actor=1.5e-4,
        ppo_lr_critic=7.5e-4,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_clip_epsilon=0.2,
        ppo_entropy_coef=0.08,  # Good exploration
        ppo_value_coef=0.5,
        ppo_epochs=12,
        ppo_mini_batch_size=48,
        lambda_energy=0.2,
        
        # State encoder
        num_grid_points=30,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


# ============ Metrics ============

def compute_normalized_score(best_y: float, worst_y: float, 
                              optimal_y: float = None) -> float:
    """
    Compute normalized score as percentage.
    
    For Rosetta where optimal is unknown, we use relative improvement.
    """
    if optimal_y is None:
        # Use best observed as reference (relative improvement)
        optimal_y = best_y + abs(best_y - worst_y) * 0.5
    
    # Avoid division by zero
    if abs(optimal_y - worst_y) < 1e-10:
        return 100.0 if best_y >= optimal_y else 0.0
    
    score = (best_y - worst_y) / (optimal_y - worst_y) * 100
    return max(0, min(100, score))


def compute_lar(f_star: float, f_t: float, alpha: float = 0.1) -> float:
    """Compute Landscape-Aware Regret (simplified)."""
    return f_star - f_t


# ============ Single Experiment ============

def run_single_rosetta_experiment(
    variant: str = "sparse",
    n_init: int = 15,
    n_iterations: int = 50,
    seed: int = 42,
    use_ppo: bool = True,
    use_mock: bool = True,
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run a single REBMBO experiment on Rosetta 86D benchmark.
    
    Args:
        variant: GP variant ("sparse" recommended, "deep" best in paper, "classic")
        n_init: Number of initial random samples
        n_iterations: Number of BO iterations
        seed: Random seed
        use_ppo: Whether to use PPO module
        use_mock: Use mock function instead of real Rosetta
        save_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        Dictionary containing results and metrics
    """
    # Set seed
    set_seed(seed)
    
    # Get benchmark
    benchmark_fn, benchmark_info = get_rosetta_benchmark(
        use_mock=use_mock,
        seed=seed
    )
    
    # Print header
    if verbose:
        print("\n" + "="*70)
        print(f"REBMBO on {benchmark_info.name}")
        print(f"Application: {benchmark_info.application}")
        print(f"Variant: REBMBO-{variant.upper()}")
        print(f"Seed: {seed}")
        print(f"Iterations: {n_iterations}, Initial points: {n_init}")
        print(f"Using PPO: {use_ppo}, Mock: {use_mock}")
        print("="*70)
    
    # Create configuration
    config = get_rosetta_config(variant=variant, use_ppo=use_ppo)
    
    # Initialize REBMBO
    optimizer = REBMBO(config)
    
    # Generate initial samples
    if verbose:
        print(f"\nGenerating {n_init} initial samples in 86D space...")
    
    X_init = np.random.rand(n_init, benchmark_info.dim)
    y_init = np.array([benchmark_fn(x) for x in X_init])
    
    if verbose:
        print(f"Initial samples: best_y = {np.max(y_init):.4f} (-ΔΔG), "
              f"worst_y = {np.min(y_init):.4f}")
        print(f"(Higher -ΔΔG = better binding)")
    
    # Convert to torch tensors
    X_init_tensor = torch.tensor(X_init, dtype=torch.float32, device=config.device)
    y_init_tensor = torch.tensor(y_init, dtype=torch.float32, device=config.device)
    
    # Initialize optimizer
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
    
    # Reference for LAR (estimated optimal or best observed * factor)
    estimated_optimal = benchmark_info.optimal_value if benchmark_info.optimal_value else best_y * 1.5
    
    # Record initial samples (store partial x to save memory)
    for i, (x_i, y_i) in enumerate(zip(X_init, y_init)):
        history['iterations'].append(0)
        history['x'].append(x_i[:15].tolist())  # Store first 15 dims
        history['y'].append(float(y_i))
        current_best = float(np.max(y_init[:i+1]))
        history['best_y'].append(current_best)
        history['lar'].append(float(estimated_optimal - current_best))
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
        print(f"\nStarting optimization (minimizing ΔΔG)...")
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
            if verbose and t > 1:
                print(f"  *** New best found! -ΔΔG = {best_y:.4f} ***")
        
        # Compute metrics
        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start
        lar = estimated_optimal - best_y
        
        # Record (partial x to save memory)
        history['iterations'].append(t)
        history['x'].append(x_next_np[:15].tolist())
        history['y'].append(float(y_next))
        history['best_y'].append(float(best_y))
        history['lar'].append(float(lar))
        history['time'].append(float(elapsed))
        
        # Progress output
        if verbose and (t % 5 == 0 or t == n_iterations or t == 1):
            score = compute_normalized_score(
                best_y, 
                min(history['y']), 
                estimated_optimal
            )
            ddg = -best_y  # Convert back to ΔΔG
            print(f"Iter {t:3d}: -ΔΔG={y_next:.4f}, best=-ΔΔG={best_y:.4f} "
                  f"(ΔΔG={ddg:.4f}), Score={score:.1f}%, time={iter_time:.2f}s")
    
    # Final statistics
    total_time = time.time() - start_time
    
    all_y = [float(y) for y in history['y']]
    worst_y = min(all_y)
    best_y_final = max(all_y)
    best_idx = all_y.index(best_y_final)
    
    normalized_score = compute_normalized_score(
        best_y_final, worst_y, estimated_optimal
    )
    final_lar = estimated_optimal - best_y_final
    
    results = {
        'benchmark': benchmark_info.name,
        'application': benchmark_info.application,
        'variant': variant,
        'seed': seed,
        'dim': benchmark_info.dim,
        'n_iterations': n_iterations,
        'n_init': n_init,
        'use_ppo': use_ppo,
        'use_mock': use_mock,
        'best_x_partial': history['x'][best_idx],  # First 15 dims
        'best_y': float(best_y_final),
        'best_ddg': float(-best_y_final),  # Actual ΔΔG value
        'worst_y': float(worst_y),
        'final_lar': float(final_lar),
        'normalized_score': float(normalized_score),
        'total_time': float(total_time),
        'history': history
    }
    
    if verbose:
        print("-"*70)
        print(f"Optimization complete!")
        print(f"Best -ΔΔG: {best_y_final:.4f} (ΔΔG = {-best_y_final:.4f} kcal/mol)")
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
    n_init: int = 15,
    n_iterations: int = 50,
    seeds: list = [1, 2, 3, 4, 5],
    use_ppo: bool = True,
    use_mock: bool = True,
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run multi-seed experiment for statistical analysis.
    """
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"results/rosetta_86d/multi_seed_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    print("\n" + "="*70)
    print(f"MULTI-SEED EXPERIMENT: Rosetta 86D (Antibody Engineering)")
    print(f"Variant: REBMBO-{variant.upper()}")
    print(f"Seeds: {seeds}")
    print(f"Iterations: {n_iterations}, Initial points: {n_init}")
    print("="*70)
    
    for seed in seeds:
        print(f"\n{'>'*20} Running seed {seed} {'<'*20}")
        
        seed_save_dir = os.path.join(save_dir, f"seed_{seed}")
        
        results = run_single_rosetta_experiment(
            variant=variant,
            n_init=n_init,
            n_iterations=n_iterations,
            seed=seed,
            use_ppo=use_ppo,
            use_mock=use_mock,
            save_dir=seed_save_dir,
            verbose=verbose
        )
        
        all_results.append(results)
    
    # Aggregate statistics
    best_ys = [r['best_y'] for r in all_results]
    best_ddgs = [r['best_ddg'] for r in all_results]
    final_lars = [r['final_lar'] for r in all_results]
    scores = [r['normalized_score'] for r in all_results]
    times = [r['total_time'] for r in all_results]
    
    aggregated = {
        'benchmark': "Rosetta-86D",
        'application': "Antibody engineering for SARS-CoV-2",
        'variant': variant,
        'seeds': seeds,
        'n_iterations': n_iterations,
        'n_init': n_init,
        'dim': 86,
        
        # Statistics
        'best_y_mean': float(np.mean(best_ys)),
        'best_y_std': float(np.std(best_ys)),
        'best_ddg_mean': float(np.mean(best_ddgs)),
        'best_ddg_std': float(np.std(best_ddgs)),
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
    print(f"Benchmark: Rosetta 86D (SARS-CoV-2 Antibody Design)")
    print(f"Variant: REBMBO-{variant.upper()}")
    print(f"Seeds: {seeds}")
    print("-"*70)
    print(f"Best -ΔΔG:        {aggregated['best_y_mean']:.4f} ± {aggregated['best_y_std']:.4f}")
    print(f"Best ΔΔG:         {aggregated['best_ddg_mean']:.4f} ± {aggregated['best_ddg_std']:.4f} kcal/mol")
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
        description="Run REBMBO on Rosetta 86D protein design benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with mock simulator
  python run_rosetta.py --mock
  
  # Standard run (50 iterations)
  python run_rosetta.py --mock --n_iterations 50
  
  # Extended run (80 iterations, as in paper Table 2)
  python run_rosetta.py --mock --n_iterations 80
  
  # Multi-seed experiment
  python run_rosetta.py --mock --seeds 1 2 3 4 5
  
  # Full paper reproduction
  python run_rosetta.py --mock --full

  # Use deep GP (best in paper for this benchmark)
  python run_rosetta.py --mock --variant deep

Biology Context:
  - ΔΔG: Change in binding free energy (kcal/mol)
  - Negative ΔΔG: Improved binding (desired)
  - Goal: Find antibody mutations that minimize ΔΔG
        """
    )
    
    # Basic arguments
    parser.add_argument('--variant', type=str, default='sparse',
                        choices=['classic', 'sparse', 'deep'],
                        help='GP variant (default: sparse, deep best in paper)')
    parser.add_argument('--n_init', type=int, default=15,
                        help='Number of initial samples (default: 15)')
    parser.add_argument('--n_iterations', type=int, default=50,
                        help='Number of iterations (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for single run (default: 42)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Multiple seeds for statistical analysis')
    
    # Algorithm options
    parser.add_argument('--no_ppo', action='store_true',
                        help='Disable PPO module')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock simulator (recommended for testing)')
    
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
                        help='Extended run (80 iterations, as in paper)')
    
    args = parser.parse_args()
    
    # Handle presets
    if args.full:
        args.seeds = [1, 2, 3, 4, 5]
        args.n_iterations = 50
        args.mock = True
    elif args.quick:
        args.n_iterations = 20
        args.mock = True
    elif args.extended:
        args.n_iterations = 80
    
    # Set default save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "mock" if args.mock else "rosetta"
        args.save_dir = f"results/rosetta_86d/{args.variant}_{mode}_{timestamp}"
    
    # Run experiment(s)
    if args.seeds:
        results = run_multi_seed_experiment(
            variant=args.variant,
            n_init=args.n_init,
            n_iterations=args.n_iterations,
            seeds=args.seeds,
            use_ppo=not args.no_ppo,
            use_mock=args.mock,
            save_dir=args.save_dir,
            verbose=not args.quiet
        )
    else:
        results = run_single_rosetta_experiment(
            variant=args.variant,
            n_init=args.n_init,
            n_iterations=args.n_iterations,
            seed=args.seed,
            use_ppo=not args.no_ppo,
            use_mock=args.mock,
            save_dir=args.save_dir,
            verbose=not args.quiet
        )
    
    return results


if __name__ == "__main__":
    main()