#!/usr/bin/env python
"""
Run BALLET-ICI baseline experiments on various benchmarks.

This script runs the BALLET-ICI baseline for comparison with REBMBO.
BALLET-ICI uses adaptive level-set estimation with iterative confidence
intervals for focused Bayesian optimization.

Usage:
    # Run on Ackley 5D
    python run_ballet_ici.py --benchmark ackley --dim 5
    
    # Run on Branin 2D
    python run_ballet_ici.py --benchmark branin
    
    # Run on Rosenbrock 8D
    python run_ballet_ici.py --benchmark rosenbrock --dim 8
    
    # Multi-seed experiment
    python run_ballet_ici.py --benchmark ackley --seeds 1 2 3 4 5
    
    # Compare with REBMBO results
    python run_ballet_ici.py --benchmark all --compare

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

# Import BALLET-ICI
try:
    from src.baselines.ballet_ici import BALLETICI, BALLETICIConfig, create_ballet_ici
except ImportError:
    try:
        from baselines.ballet_ici import BALLETICI, BALLETICIConfig, create_ballet_ici
    except ImportError:
        from ballet_ici import BALLETICI, BALLETICIConfig, create_ballet_ici


# ============ Benchmark Functions ============

def branin_2d(x: np.ndarray) -> float:
    """Branin function (2D), scaled to [0,1]^2."""
    x = np.asarray(x).flatten()
    x1 = x[0] * 15 - 5
    x2 = x[1] * 15
    
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    result = a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
    return -result


def ackley(x: np.ndarray, dim: int = 5) -> float:
    """Ackley function, scaled to [0,1]^d."""
    x = np.asarray(x).flatten()
    x_scaled = x * 10 - 5  # Scale to [-5, 5]
    
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    sum_sq = np.mean(x_scaled ** 2)
    sum_cos = np.mean(np.cos(c * x_scaled))
    
    result = -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.exp(1)
    return -result


def rosenbrock(x: np.ndarray, dim: int = 8) -> float:
    """Rosenbrock function, scaled to [0,1]^d."""
    x = np.asarray(x).flatten()
    x_scaled = x * 4 - 2  # Scale to [-2, 2]
    
    result = 0
    for i in range(len(x_scaled) - 1):
        result += 100 * (x_scaled[i+1] - x_scaled[i]**2)**2 + (1 - x_scaled[i])**2
    
    return -result


def hdbo_200d(x: np.ndarray) -> float:
    """HDBO 200D function."""
    x = np.asarray(x).flatten()
    x_scaled = x * 10 - 5
    return -np.sum(np.exp(x_scaled))


BENCHMARKS = {
    'branin': {
        'func': branin_2d,
        'dim': 2,
        'optimal': -0.398,
        'n_init': 5
    },
    'ackley': {
        'func': lambda x: ackley(x, 5),
        'dim': 5,
        'optimal': 0.0,
        'n_init': 10
    },
    'rosenbrock': {
        'func': lambda x: rosenbrock(x, 8),
        'dim': 8,
        'optimal': 0.0,
        'n_init': 10
    },
    'hdbo': {
        'func': hdbo_200d,
        'dim': 200,
        'optimal': -200 * np.exp(-5),
        'n_init': 20
    }
}


# ============ Configuration ============

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_ballet_config(benchmark: str, dim: int = None) -> BALLETICIConfig:
    """Get optimized configuration for each benchmark."""
    
    if benchmark == 'branin':
        return BALLETICIConfig(
            input_dim=2,
            bounds=(0.0, 1.0),
            beta=2.0,
            level_set_threshold=0.5,
            num_grid_points=50,
            gp_train_epochs=100,
            gp_retrain_epochs=30,
            use_local_gp=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    elif benchmark == 'ackley':
        d = dim or 5
        return BALLETICIConfig(
            input_dim=d,
            bounds=(0.0, 1.0),
            beta=2.5,
            level_set_threshold=0.4,
            num_grid_points=40,
            gp_train_epochs=150,
            gp_retrain_epochs=50,
            use_local_gp=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    elif benchmark == 'rosenbrock':
        d = dim or 8
        return BALLETICIConfig(
            input_dim=d,
            bounds=(0.0, 1.0),
            beta=2.0,
            level_set_threshold=0.5,
            num_grid_points=40,
            gp_train_epochs=150,
            gp_retrain_epochs=50,
            use_local_gp=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    elif benchmark == 'hdbo':
        return BALLETICIConfig(
            input_dim=200,
            bounds=(0.0, 1.0),
            beta=3.0,
            level_set_threshold=0.3,
            num_grid_points=30,
            gp_train_epochs=200,
            gp_retrain_epochs=80,
            use_local_gp=False,  # Too expensive for 200D
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    else:
        # Default configuration
        return BALLETICIConfig(
            input_dim=dim or 5,
            bounds=(0.0, 1.0),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )


# ============ Metrics ============

def compute_normalized_score(best_y: float, worst_y: float, optimal_y: float) -> float:
    """Compute normalized score as percentage."""
    if abs(optimal_y - worst_y) < 1e-10:
        return 100.0 if best_y >= optimal_y else 0.0
    score = (best_y - worst_y) / (optimal_y - worst_y) * 100
    return max(0, min(100, score))


# ============ Experiment Functions ============

def run_single_experiment(
    benchmark: str,
    dim: int = None,
    n_iterations: int = 50,
    seed: int = 42,
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """Run a single BALLET-ICI experiment."""
    
    set_seed(seed)
    
    # Get benchmark info
    bench_info = BENCHMARKS.get(benchmark)
    if bench_info is None:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    func = bench_info['func']
    d = dim or bench_info['dim']
    optimal = bench_info['optimal']
    n_init = bench_info['n_init']
    
    # Update function for custom dimension
    if benchmark == 'ackley' and dim:
        func = lambda x: ackley(x, dim)
    elif benchmark == 'rosenbrock' and dim:
        func = lambda x: rosenbrock(x, dim)
    
    if verbose:
        print("\n" + "="*60)
        print(f"BALLET-ICI on {benchmark.upper()}-{d}D")
        print(f"Seed: {seed}")
        print(f"Iterations: {n_iterations}, Initial points: {n_init}")
        print("="*60)
    
    # Create optimizer
    config = get_ballet_config(benchmark, d)
    config.input_dim = d
    optimizer = BALLETICI(config)
    
    # Generate initial samples
    X_init = torch.rand(n_init, d)
    y_init = torch.tensor([func(x.numpy()) for x in X_init], dtype=torch.float32)
    
    if verbose:
        print(f"Initial samples: best_y = {y_init.max():.4f}, worst_y = {y_init.min():.4f}")
    
    # Initialize
    optimizer.initialize(X_init, y_init)
    
    # Track history
    history = {
        'iterations': list(range(n_init)),
        'x': [x.tolist() for x in X_init],
        'y': y_init.tolist(),
        'best_y': [y_init[:i+1].max().item() for i in range(n_init)],
        'time': [0] * n_init
    }
    
    best_y = y_init.max().item()
    start_time = time.time()
    
    # Run optimization
    for t in range(n_iterations):
        x_next, y_next = optimizer.step(objective_fn=func, verbose=False)
        
        if y_next > best_y:
            best_y = y_next
        
        elapsed = time.time() - start_time
        
        history['iterations'].append(n_init + t)
        history['x'].append(x_next.cpu().numpy().tolist())
        history['y'].append(float(y_next))
        history['best_y'].append(float(best_y))
        history['time'].append(float(elapsed))
        
        if verbose and (t % 5 == 0 or t == n_iterations - 1):
            print(f"Iter {t+1:3d}: y={y_next:.4f}, best_y={best_y:.4f}")
    
    total_time = time.time() - start_time
    
    # Compute metrics
    worst_y = min(history['y'])
    normalized_score = compute_normalized_score(best_y, worst_y, optimal)
    
    results = {
        'benchmark': f"{benchmark}-{d}D",
        'method': 'BALLET-ICI',
        'seed': seed,
        'n_iterations': n_iterations,
        'n_init': n_init,
        'optimal_value': optimal,
        'best_y': float(best_y),
        'worst_y': float(worst_y),
        'normalized_score': float(normalized_score),
        'total_time': float(total_time),
        'history': history
    }
    
    if verbose:
        print("-"*60)
        print(f"Optimization complete!")
        print(f"Best y: {best_y:.4f} (optimal: {optimal:.4f})")
        print(f"Normalized Score: {normalized_score:.2f}%")
        print(f"Total time: {total_time:.2f}s")
        print("="*60)
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        results_file = os.path.join(save_dir, f'ballet_ici_{benchmark}_{d}d_seed{seed}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, 
                     default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    
    return results


def run_multi_seed_experiment(
    benchmark: str,
    dim: int = None,
    n_iterations: int = 50,
    seeds: list = [1, 2, 3, 4, 5],
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """Run multi-seed experiment for statistical analysis."""
    
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"results/baselines/ballet_ici/{benchmark}_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    print("\n" + "="*60)
    print(f"MULTI-SEED EXPERIMENT: BALLET-ICI on {benchmark.upper()}")
    print(f"Seeds: {seeds}")
    print("="*60)
    
    for seed in seeds:
        print(f"\n{'>'*20} Running seed {seed} {'<'*20}")
        
        results = run_single_experiment(
            benchmark=benchmark,
            dim=dim,
            n_iterations=n_iterations,
            seed=seed,
            save_dir=save_dir,
            verbose=verbose
        )
        all_results.append(results)
    
    # Aggregate statistics
    best_ys = [r['best_y'] for r in all_results]
    scores = [r['normalized_score'] for r in all_results]
    times = [r['total_time'] for r in all_results]
    
    aggregated = {
        'benchmark': all_results[0]['benchmark'],
        'method': 'BALLET-ICI',
        'seeds': seeds,
        'n_iterations': n_iterations,
        'best_y_mean': float(np.mean(best_ys)),
        'best_y_std': float(np.std(best_ys)),
        'normalized_score_mean': float(np.mean(scores)),
        'normalized_score_std': float(np.std(scores)),
        'time_mean': float(np.mean(times)),
        'time_std': float(np.std(times)),
        'individual_results': all_results
    }
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-SEED SUMMARY")
    print("="*60)
    print(f"Best y:           {aggregated['best_y_mean']:.4f} ± {aggregated['best_y_std']:.4f}")
    print(f"Normalized Score: {aggregated['normalized_score_mean']:.2f}% ± {aggregated['normalized_score_std']:.2f}%")
    print(f"Time:             {aggregated['time_mean']:.2f}s ± {aggregated['time_std']:.2f}s")
    print("="*60)
    
    # Save aggregated results
    agg_file = os.path.join(save_dir, 'aggregated_results.json')
    with open(agg_file, 'w') as f:
        json.dump(aggregated, f, indent=2,
                 default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    
    return aggregated


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(
        description="Run BALLET-ICI baseline experiments"
    )
    
    parser.add_argument('--benchmark', type=str, default='ackley',
                        choices=['branin', 'ackley', 'rosenbrock', 'hdbo', 'all'],
                        help='Benchmark to run')
    parser.add_argument('--dim', type=int, default=None,
                        help='Dimension (overrides default)')
    parser.add_argument('--n_iterations', type=int, default=50,
                        help='Number of iterations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for single run')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Multiple seeds for statistical analysis')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    if args.benchmark == 'all':
        benchmarks = ['branin', 'ackley', 'rosenbrock']
    else:
        benchmarks = [args.benchmark]
    
    for benchmark in benchmarks:
        if args.seeds:
            run_multi_seed_experiment(
                benchmark=benchmark,
                dim=args.dim,
                n_iterations=args.n_iterations,
                seeds=args.seeds,
                save_dir=args.save_dir,
                verbose=not args.quiet
            )
        else:
            run_single_experiment(
                benchmark=benchmark,
                dim=args.dim,
                n_iterations=args.n_iterations,
                seed=args.seed,
                save_dir=args.save_dir,
                verbose=not args.quiet
            )


if __name__ == "__main__":
    main()