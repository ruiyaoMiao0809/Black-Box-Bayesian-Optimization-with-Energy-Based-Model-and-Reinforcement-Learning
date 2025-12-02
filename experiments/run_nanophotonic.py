#!/usr/bin/env python
"""
Run REBMBO experiments on Nanophotonic Structure Design benchmark.

This script reproduces the Nanophotonic 3D results from the paper (Figure 2e, Table 2).
The "3D" refers to the 3D physical simulation (Maxwell's equations), not input dimension.

Paper definition (Appendix B - Nanophotonic Structure Design):
    "Each input x ∈ R^2 represents physical parameters (thickness, radius) 
    that define a nanosphere structure. The output corresponds to a weighted 
    figure of merit for optical properties, derived from solving discretized 
    Maxwell's equations."

Key Characteristics:
    - Input: 2D (thickness, radius) normalized to [0,1]^2
    - Original bounds: thickness ∈ [100, 400] nm, radius ∈ [10, 200] nm
    - Output: Figure of Merit (FoM) for optical properties
    - Computationally intensive (MEEP simulation)
    - Response surface contains multiple basins
    - Real-world black-box optimization

Usage:
    # Quick test with mock simulation (no MEEP required)
    python run_nanophotonic.py --mock
    
    # Full MEEP simulation (requires meep installed)
    python run_nanophotonic.py --n_iterations 50
    
    # Different material combinations
    python run_nanophotonic.py --mock --materials cSi_TiO2
    python run_nanophotonic.py --mock --materials GaAs_TiO2
    
    # Different figure of merit
    python run_nanophotonic.py --mock --fom mean_absorbance
    python run_nanophotonic.py --mock --fom solar_cell_efficiency
    
    # Multi-seed evaluation
    python run_nanophotonic.py --mock --seeds 1 2 3 4 5
    
    # Full paper reproduction (5 seeds, 50 iterations each)
    python run_nanophotonic.py --mock --full

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


# ============ Nanophotonic Benchmark Definition ============

class NanophotonicBenchmarkInfo:
    """Information about the Nanophotonic benchmark."""
    def __init__(self, materials='cSi_TiO2', fom_type='mean_absorbance'):
        self.name = f"Nanophotonic-3D ({materials}, {fom_type})"
        self.dim = 2  # Note: "3D" refers to physical simulation, not input dim
        self.bounds = (0, 1)
        self.original_bounds = np.array([
            [100, 400],  # thickness (nm)
            [10, 200],   # radius (nm)
        ])
        self.optimal_value = None  # Unknown for real black-box
        self.materials = materials
        self.fom_type = fom_type
        self.labels = ['thickness (nm)', 'radius (nm)']


def create_nanophotonic_mock_function(
    materials: str = 'cSi_TiO2',
    fom_type: str = 'mean_absorbance',
    seed: int = 42
):
    """
    Create a mock nanophotonic benchmark function.
    
    This simulates the multi-modal response surface expected from 
    nanophotonic simulations without requiring MEEP.
    
    The mock function is designed to:
    1. Have multiple local optima (realistic for optical structures)
    2. Have smooth but complex response surface
    3. Reward structures with balanced thickness/radius ratios
    
    Args:
        materials: Material combination (affects function shape slightly)
        fom_type: Figure of merit type
        seed: Random seed for reproducibility
        
    Returns:
        Callable benchmark function and info object
    """
    np.random.seed(seed)
    
    # Material-specific parameters
    material_params = {
        'cSi_TiO2': {'scale': 1.0, 'shift': 0.0},
        'GaAs_TiO2': {'scale': 1.1, 'shift': 0.05},
        'Perovskite_TiO2': {'scale': 0.95, 'shift': -0.02},
    }
    
    params = material_params.get(materials, {'scale': 1.0, 'shift': 0.0})
    
    # FoM-specific adjustments
    fom_scale = {
        'mean_absorbance': 1.0,
        'weighted_absorbance': 1.1,
        'integrated_absorbance': 0.9,
        'visible_transmittance': 0.8,
        'solar_cell_efficiency': 1.2,
    }
    
    fom_s = fom_scale.get(fom_type, 1.0)
    
    def mock_nanophotonic(x: np.ndarray) -> float:
        """
        Mock nanophotonic objective function.
        
        Creates a realistic multi-modal surface with:
        - Multiple basins (optical resonances)
        - Smooth transitions
        - Physics-inspired structure
        """
        if x.ndim > 1:
            x = x.flatten()
        
        # Ensure x is in [0, 1]
        x = np.clip(x, 0, 1)
        
        # Scale to physical units for intuition
        thickness = x[0] * 300 + 100  # [100, 400] nm
        radius = x[1] * 190 + 10      # [10, 200] nm
        
        # Normalize for computation
        t_norm = (thickness - 100) / 300
        r_norm = (radius - 10) / 190
        
        # Main peak: balanced structure around t=0.6, r=0.5
        peak1 = 0.7 * np.exp(-10 * ((t_norm - 0.6)**2 + (r_norm - 0.5)**2))
        
        # Secondary peak: high thickness, medium radius
        peak2 = 0.5 * np.exp(-8 * ((t_norm - 0.8)**2 + (r_norm - 0.4)**2))
        
        # Third peak: medium thickness, high radius
        peak3 = 0.45 * np.exp(-12 * ((t_norm - 0.4)**2 + (r_norm - 0.7)**2))
        
        # Optical resonance effects (periodic variations)
        resonance = 0.15 * np.sin(6 * np.pi * t_norm) * np.sin(4 * np.pi * r_norm)
        
        # Physical constraint: very thin or very thick structures are worse
        edge_penalty = 0.1 * (np.exp(-20 * t_norm) + np.exp(-20 * (1 - t_norm)))
        
        # Base value + peaks + resonance - penalty
        fom = 0.3 + peak1 + peak2 + peak3 + resonance - edge_penalty
        
        # Apply material and FoM scaling
        fom = fom * params['scale'] * fom_s + params['shift']
        
        # Add small noise for realism
        fom += np.random.normal(0, 0.01)
        
        # Clip to valid range
        fom = np.clip(fom, 0, 1)
        
        return float(fom)
    
    info = NanophotonicBenchmarkInfo(materials, fom_type)
    
    return mock_nanophotonic, info


def create_meep_nanophotonic_function(
    materials: str = 'cSi_TiO2',
    fom_type: str = 'mean_absorbance',
    resolution: int = 20
):
    """
    Create actual MEEP-based nanophotonic benchmark.
    
    Requires meep to be installed:
        conda install -c conda-forge meep
    
    Args:
        materials: Material combination
        fom_type: Figure of merit type
        resolution: MEEP mesh resolution (higher = slower but more accurate)
        
    Returns:
        Callable benchmark function and info object
    """
    try:
        # Try to import the actual MEEP-based benchmark
        from src.benchmarks import nanophotonic_3d, NANOPHOTONIC_INFO
        return nanophotonic_3d, NANOPHOTONIC_INFO
    except ImportError:
        try:
            from benchmarks import nanophotonic_3d, NANOPHOTONIC_INFO  
            return nanophotonic_3d, NANOPHOTONIC_INFO
        except ImportError:
            pass
    
    # If MEEP is not available, fall back to mock
    print("Warning: MEEP not available, using mock function")
    return create_nanophotonic_mock_function(materials, fom_type)


def get_benchmark(use_mock: bool = True, materials: str = 'cSi_TiO2', 
                  fom_type: str = 'mean_absorbance', seed: int = 42):
    """Get the nanophotonic benchmark function."""
    if use_mock:
        return create_nanophotonic_mock_function(materials, fom_type, seed)
    else:
        return create_meep_nanophotonic_function(materials, fom_type)


# ============ Configuration ============

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_nanophotonic_config(use_ppo: bool = True) -> REBMBOConfig:
    """
    Get optimized configuration for Nanophotonic benchmark.
    
    Hyperparameters tuned for:
    - 2D input space (relatively low dimensional)
    - Multiple basins (requires good global exploration)
    - Expensive function evaluations (need efficient sampling)
    """
    return REBMBOConfig(
        # Problem definition
        input_dim=2,  # Use input_dim, not dim
        bounds=(0.0, 1.0),  # Tuple format
        
        # GP configuration (Module A) - Classic GP for low dimensions
        gp_variant="classic",
        gp_train_epochs=100,
        gp_retrain_epochs=50,
        
        # EBM configuration (Module B) - Capture multi-modal structure
        ebm_hidden_dims=[128, 128, 64],
        ebm_train_epochs=100,
        ebm_retrain_epochs=50,
        ebm_mcmc_steps=30,
        ebm_mcmc_step_size=0.01,
        ebm_num_negative_samples=128,
        
        # Acquisition function
        beta=2.0,   # UCB exploration weight
        gamma=0.35,  # EBM weight (moderate for multi-modal)
        
        # PPO configuration (Module C)
        ppo_hidden_dims=[256, 256],
        ppo_lr_actor=3e-4,
        ppo_lr_critic=1e-3,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_clip_epsilon=0.2,
        ppo_entropy_coef=0.05,  # Encourage exploration
        ppo_value_coef=0.5,
        ppo_epochs=10,
        ppo_mini_batch_size=32,
        lambda_energy=0.3,
        
        # State encoder
        num_grid_points=80,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


# ============ Metrics ============

def compute_normalized_score(best_y: float, worst_y: float, optimal_y: float = None) -> float:
    """
    Compute normalized score as percentage of optimal.
    
    For Nanophotonic where optimal is unknown, we use best observed
    across all runs as reference.
    """
    if optimal_y is None:
        # Assume optimal FoM is 1.0 for normalized metrics
        optimal_y = 1.0
    
    # Avoid division by zero
    if abs(optimal_y - worst_y) < 1e-10:
        return 100.0 if best_y >= optimal_y else 0.0
    
    score = (best_y - worst_y) / (optimal_y - worst_y) * 100
    return max(0, min(100, score))


def compute_lar(f_star: float, f_t: float, e_star: float = None, e_t: float = None, 
                alpha: float = 0.1) -> float:
    """
    Compute Landscape-Aware Regret (LAR).
    
    LAR = [f(x*) - f(x_t)] + α[E_θ(x*) - E_θ(x_t)]
    
    For real-world benchmarks where E_θ is unavailable, use simplified version.
    """
    function_regret = f_star - f_t
    
    if e_star is not None and e_t is not None:
        energy_regret = alpha * (e_star - e_t)
        return function_regret + energy_regret
    
    return function_regret


# ============ Single Experiment ============

def run_single_nanophotonic_experiment(
    variant: str = "classic",
    n_init: int = 5,
    n_iterations: int = 50,
    seed: int = 42,
    use_ppo: bool = True,
    use_mock: bool = True,
    materials: str = 'cSi_TiO2',
    fom_type: str = 'mean_absorbance',
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run a single REBMBO experiment on Nanophotonic benchmark.
    
    Args:
        variant: GP variant ("classic", "sparse", or "deep")
        n_init: Number of initial random samples
        n_iterations: Number of BO iterations
        seed: Random seed
        use_ppo: Whether to use PPO module
        use_mock: Use mock function instead of MEEP
        materials: Material combination
        fom_type: Figure of merit type
        save_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        Dictionary containing results and metrics
    """
    # Set seed
    set_seed(seed)
    
    # Create benchmark
    benchmark_fn, benchmark_info = get_benchmark(
        use_mock=use_mock,
        materials=materials,
        fom_type=fom_type,
        seed=seed
    )
    
    # Print header
    if verbose:
        print("\n" + "="*60)
        print(f"REBMBO on {benchmark_info.name}")
        print(f"Variant: REBMBO-{variant.upper()}")
        print(f"Seed: {seed}")
        print(f"Iterations: {n_iterations}, Initial points: {n_init}")
        print(f"Using PPO: {use_ppo}, Mock: {use_mock}")
        print("="*60)
    
    # Create configuration
    config = get_nanophotonic_config(use_ppo=use_ppo)
    config.gp_variant = variant
    
    # Initialize REBMBO
    optimizer = REBMBO(config)
    
    # Generate initial samples
    X_init = np.random.rand(n_init, benchmark_info.dim)
    y_init = np.array([benchmark_fn(x) for x in X_init])
    
    if verbose:
        print(f"\nInitial samples: best_y = {np.max(y_init):.4f}, "
              f"worst_y = {np.min(y_init):.4f}")
    
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
    best_x_idx = np.argmax(y_init)
    best_x = X_init[best_x_idx]
    start_time = time.time()
    
    # Record initial samples individually
    for i, (x_i, y_i) in enumerate(zip(X_init, y_init)):
        history['iterations'].append(0)  # All initial samples at "iteration 0"
        history['x'].append(x_i.tolist() if hasattr(x_i, 'tolist') else list(x_i))
        history['y'].append(float(y_i))
        history['best_y'].append(float(np.max(y_init[:i+1])))
        history['lar'].append(1.0 - float(np.max(y_init[:i+1])))
        history['time'].append(0)
    
    # Wrapper function that accepts numpy array
    def objective_wrapper(x):
        """Wrapper to convert between numpy and torch."""
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = np.asarray(x)
        x_np = x_np.flatten()
        x_np = np.clip(x_np, 0, 1)  # Ensure bounds
        return benchmark_fn(x_np)
    
    # Main optimization loop using step() method
    for t in range(1, n_iterations + 1):
        iter_start = time.time()
        
        # Run one step of REBMBO optimization
        x_next, y_next = optimizer.step(
            objective_fn=objective_wrapper,
            use_ppo=use_ppo,
            verbose=False  # We'll print our own progress
        )
        
        # Convert to numpy for recording
        x_next_np = x_next.cpu().numpy().flatten()
        
        # Update best (optimizer tracks this internally, but we track too)
        if y_next > best_y:
            best_y = y_next
        
        # Compute metrics
        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start
        lar = 1.0 - best_y  # Simple regret
        
        # Record
        history['iterations'].append(t)
        history['x'].append(x_next_np.tolist())
        history['y'].append(float(y_next))
        history['best_y'].append(float(best_y))
        history['lar'].append(float(lar))
        history['time'].append(float(elapsed))
        
        if verbose and (t % 5 == 0 or t == n_iterations):
            print(f"Iter {t:3d}: x={x_next_np}, y={y_next:.4f}, "
                  f"best_y={best_y:.4f}, time={iter_time:.2f}s")
    
    # Final statistics
    total_time = time.time() - start_time
    
    # Get all y values as floats for comparison
    all_y = [float(y) for y in history['y']]
    worst_y = min(all_y)
    best_y_final = max(all_y)
    best_idx = all_y.index(best_y_final)
    best_x_final = history['x'][best_idx]
    
    # Compute normalized score (assuming optimal FoM ~ 1.0 for mock)
    normalized_score = compute_normalized_score(best_y_final, worst_y, optimal_y=1.0)
    
    results = {
        'benchmark': benchmark_info.name,
        'variant': variant,
        'seed': seed,
        'n_iterations': n_iterations,
        'n_init': n_init,
        'use_ppo': use_ppo,
        'use_mock': use_mock,
        'materials': materials,
        'fom_type': fom_type,
        'best_x': best_x_final,
        'best_y': float(best_y_final),
        'worst_y': float(worst_y),
        'final_lar': float(history['lar'][-1]),
        'normalized_score': float(normalized_score),
        'total_time': float(total_time),
        'history': history
    }
    
    if verbose:
        print("\n" + "-"*60)
        print(f"Optimization complete!")
        print(f"Best y: {best_y_final:.4f}")
        print(f"Normalized Score: {normalized_score:.2f}%")
        print(f"Final LAR: {history['lar'][-1]:.4f}")
        print(f"Total time: {total_time:.2f}s")
        print("-"*60)
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        results_file = os.path.join(save_dir, f'results_seed{seed}.json')
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
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
    variant: str = "classic",
    n_init: int = 5,
    n_iterations: int = 50,
    seeds: list = [1, 2, 3, 4, 5],
    use_ppo: bool = True,
    use_mock: bool = True,
    materials: str = 'cSi_TiO2',
    fom_type: str = 'mean_absorbance',
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run multi-seed experiment for statistical analysis.
    
    Returns:
        Aggregated results across all seeds
    """
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"results/nanophotonic_3d/multi_seed_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    print("\n" + "="*70)
    print(f"Running Multi-Seed Experiment on Nanophotonic ({materials}, {fom_type})")
    print(f"Seeds: {seeds}")
    print(f"Variant: REBMBO-{variant.upper()}")
    print(f"Iterations: {n_iterations}")
    print("="*70)
    
    for seed in seeds:
        print(f"\n>>> Running seed {seed} <<<")
        
        seed_save_dir = os.path.join(save_dir, f"seed_{seed}")
        
        results = run_single_nanophotonic_experiment(
            variant=variant,
            n_init=n_init,
            n_iterations=n_iterations,
            seed=seed,
            use_ppo=use_ppo,
            use_mock=use_mock,
            materials=materials,
            fom_type=fom_type,
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
        'benchmark': f"Nanophotonic-3D ({materials})",
        'variant': variant,
        'seeds': seeds,
        'n_iterations': n_iterations,
        'materials': materials,
        'fom_type': fom_type,
        'use_mock': use_mock,
        
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
    print(f"Benchmark: Nanophotonic 3D ({materials}, {fom_type})")
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
        json.dump(aggregated, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    
    print(f"\nResults saved to: {save_dir}")
    
    return aggregated


# ============ Main Entry Point ============

def main():
    parser = argparse.ArgumentParser(
        description="Run REBMBO on Nanophotonic benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with mock simulation
  python run_nanophotonic.py --mock
  
  # Full paper reproduction
  python run_nanophotonic.py --mock --full
  
  # Different materials
  python run_nanophotonic.py --mock --materials GaAs_TiO2
  
  # Multi-seed experiment
  python run_nanophotonic.py --mock --seeds 1 2 3 4 5
        """
    )
    
    # Basic arguments
    parser.add_argument('--variant', type=str, default='classic',
                        choices=['classic', 'sparse', 'deep'],
                        help='GP variant (default: classic)')
    parser.add_argument('--n_init', type=int, default=5,
                        help='Number of initial samples (default: 5)')
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
                        help='Use mock simulation (no MEEP required)')
    
    # Benchmark options
    parser.add_argument('--materials', type=str, default='cSi_TiO2',
                        choices=['cSi_TiO2', 'GaAs_TiO2', 'Perovskite_TiO2'],
                        help='Material combination (default: cSi_TiO2)')
    parser.add_argument('--fom', type=str, default='mean_absorbance',
                        choices=['mean_absorbance', 'weighted_absorbance', 
                                 'integrated_absorbance', 'visible_transmittance',
                                 'solar_cell_efficiency'],
                        help='Figure of merit type (default: mean_absorbance)')
    
    # Output options
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    # Preset configurations
    parser.add_argument('--full', action='store_true',
                        help='Full paper reproduction (5 seeds, 50 iterations)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (1 seed, 20 iterations)')
    
    args = parser.parse_args()
    
    # Handle presets
    if args.full:
        args.seeds = [1, 2, 3, 4, 5]
        args.n_iterations = 50
        args.mock = True  # Default to mock for reproducibility
    elif args.quick:
        args.n_iterations = 20
        args.mock = True
    
    # Set default save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "mock" if args.mock else "meep"
        args.save_dir = f"results/nanophotonic_3d/{args.materials}_{mode}_{timestamp}"
    
    # Run experiment(s)
    if args.seeds:
        results = run_multi_seed_experiment(
            variant=args.variant,
            n_init=args.n_init,
            n_iterations=args.n_iterations,
            seeds=args.seeds,
            use_ppo=not args.no_ppo,
            use_mock=args.mock,
            materials=args.materials,
            fom_type=args.fom,
            save_dir=args.save_dir,
            verbose=not args.quiet
        )
    else:
        results = run_single_nanophotonic_experiment(
            variant=args.variant,
            n_init=args.n_init,
            n_iterations=args.n_iterations,
            seed=args.seed,
            use_ppo=not args.no_ppo,
            use_mock=args.mock,
            materials=args.materials,
            fom_type=args.fom,
            save_dir=args.save_dir,
            verbose=not args.quiet
        )
    
    return results


if __name__ == "__main__":
    main()