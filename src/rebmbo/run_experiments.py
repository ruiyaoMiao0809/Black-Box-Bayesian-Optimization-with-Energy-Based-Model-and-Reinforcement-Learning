"""
Main Experiment Runner for REBMBO

Run experiments on various benchmarks and compare with baselines.
Reproduces results from the paper (Tables 1 and 2).
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rebmbo.algorithm import REBMBO, REBMBOConfig, create_rebmbo_c, create_rebmbo_s, create_rebmbo_d
from src.benchmarks import get_benchmark, list_benchmarks, BENCHMARKS


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    benchmark_name: str,
    variant: str = "classic",
    n_init: int = 5,
    n_iterations: int = 30,
    seed: int = 42,
    use_ppo: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run a single REBMBO experiment.
    
    Args:
        benchmark_name: Name of benchmark function
        variant: GP variant ('classic', 'sparse', 'deep')
        n_init: Number of initial random samples
        n_iterations: Number of optimization iterations
        seed: Random seed
        use_ppo: Whether to use PPO (vs EBM-UCB only)
        verbose: Print progress
    
    Returns:
        Results dictionary
    """
    set_seed(seed)
    
    # Get benchmark
    func, info = get_benchmark(benchmark_name)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {info.name}")
        print(f"Variant: REBMBO-{variant[0].upper()}")
        print(f"Seed: {seed}")
        print(f"Iterations: {n_iterations}")
        print(f"{'='*60}")
    
    # Create REBMBO instance
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = REBMBOConfig(
        input_dim=info.dim,
        bounds=info.bounds,
        gp_variant=variant,
        gp_train_epochs=100,
        gp_retrain_epochs=30,
        ebm_train_epochs=100,
        ebm_retrain_epochs=30,
        num_grid_points=min(50, max(20, info.dim * 2)),
        device=device
    )
    
    # Adjust for high dimensions
    if info.dim > 50:
        config.gp_variant = "sparse"
        config.gp_num_inducing = min(100, info.dim)
        config.num_grid_points = 30
        config.gp_train_epochs = 50
        config.gp_retrain_epochs = 20
        config.ebm_train_epochs = 50
        config.ebm_retrain_epochs = 20
    
    rebmbo = REBMBO(config)
    
    # Generate initial samples
    X_init = torch.rand(n_init, info.dim)
    y_init = torch.tensor([func(x.numpy()) for x in X_init])
    
    # Initialize and run
    start_time = time.time()
    rebmbo.initialize(X_init, y_init)
    best_x, best_y = rebmbo.optimize(
        objective_fn=func,
        n_iterations=n_iterations,
        use_ppo=use_ppo,
        verbose=verbose
    )
    total_time = time.time() - start_time
    
    # Get results
    results = rebmbo.get_results()
    results["benchmark"] = benchmark_name
    results["variant"] = variant
    results["seed"] = seed
    results["total_time"] = total_time
    results["optimal_value"] = info.optimal_value
    
    # Compute normalized score (higher is better, 0-100 scale)
    # Score = 100 * (1 - |best_y - optimal| / |initial_best - optimal|)
    initial_best = y_init.max().item()
    if info.optimal_value != initial_best:
        normalized_score = 100 * (1 - abs(best_y - info.optimal_value) / 
                                   abs(initial_best - info.optimal_value))
        normalized_score = max(0, min(100, normalized_score))
    else:
        normalized_score = 100
    
    results["normalized_score"] = normalized_score
    
    if verbose:
        print(f"\nResults:")
        print(f"  Best y: {best_y:.4f} (optimal: {info.optimal_value})")
        print(f"  Normalized score: {normalized_score:.2f}")
        print(f"  Total time: {total_time:.2f}s")
    
    return results


def run_multiple_seeds(
    benchmark_name: str,
    variant: str = "classic",
    n_init: int = 5,
    n_iterations: int = 30,
    seeds: List[int] = [1, 2, 3, 4, 5],
    use_ppo: bool = True,
    save_dir: str = None
) -> Dict:
    """
    Run experiments with multiple seeds and compute statistics.
    """
    print(f"\n{'#'*60}")
    print(f"# Running {benchmark_name} with {len(seeds)} seeds")
    print(f"# Variant: REBMBO-{variant[0].upper()}")
    print(f"{'#'*60}")
    
    all_results = []
    all_scores = []
    all_best_y = []
    
    for seed in seeds:
        results = run_single_experiment(
            benchmark_name=benchmark_name,
            variant=variant,
            n_init=n_init,
            n_iterations=n_iterations,
            seed=seed,
            use_ppo=use_ppo,
            verbose=True
        )
        all_results.append(results)
        all_scores.append(results["normalized_score"])
        all_best_y.append(results["best_y"])
    
    # Compute statistics
    summary = {
        "benchmark": benchmark_name,
        "variant": variant,
        "n_iterations": n_iterations,
        "mean_score": np.mean(all_scores),
        "std_score": np.std(all_scores),
        "mean_best_y": np.mean(all_best_y),
        "std_best_y": np.std(all_best_y),
        "scores": all_scores,
        "best_ys": all_best_y
    }
    
    print(f"\n{'='*60}")
    print(f"Summary: {benchmark_name} (REBMBO-{variant[0].upper()})")
    print(f"  Score: {summary['mean_score']:.2f} ± {summary['std_score']:.2f}")
    print(f"  Best y: {summary['mean_best_y']:.4f} ± {summary['std_best_y']:.4f}")
    print(f"{'='*60}")
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save summary
        summary_path = os.path.join(save_dir, f"{benchmark_name}_{variant}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results
        for i, results in enumerate(all_results):
            results_path = os.path.join(save_dir, f"{benchmark_name}_{variant}_seed{seeds[i]}.json")
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    serializable_results[k] = v.tolist()
                elif isinstance(v, torch.Tensor):
                    serializable_results[k] = v.cpu().numpy().tolist()
                else:
                    serializable_results[k] = v
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
    
    return summary


def plot_convergence(results_list: List[Dict], save_path: str = None):
    """
    Plot convergence curves for multiple experiments.
    """
    plt.figure(figsize=(10, 6))
    
    for results in results_list:
        label = f"{results['benchmark']} ({results['variant']})"
        best_y_curve = results['history']['best_y']
        plt.plot(best_y_curve, label=label, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Objective Value', fontsize=12)
    plt.title('REBMBO Convergence', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def run_synthetic_benchmarks(
    variants: List[str] = ["classic", "sparse", "deep"],
    n_iterations_list: Dict[str, List[int]] = None,
    seeds: List[int] = [1, 2, 3, 4, 5],
    save_dir: str = "results"
):
    """
    Run all synthetic benchmarks (Table 1 from paper).
    """
    if n_iterations_list is None:
        n_iterations_list = {
            "branin_2d": [30, 50],
            "ackley_5d": [30, 50],
            "rosenbrock_8d": [30, 50],
            "hdbo_200d": [50, 100]
        }
    
    all_summaries = []
    
    for benchmark_name, iterations_list in n_iterations_list.items():
        for n_iter in iterations_list:
            for variant in variants:
                # Skip deep variant for very high dimensions
                _, info = get_benchmark(benchmark_name)
                if variant == "deep" and info.dim > 100:
                    continue
                
                summary = run_multiple_seeds(
                    benchmark_name=benchmark_name,
                    variant=variant,
                    n_iterations=n_iter,
                    seeds=seeds,
                    save_dir=os.path.join(save_dir, benchmark_name)
                )
                summary["n_iterations"] = n_iter
                all_summaries.append(summary)
    
    # Create comparison table
    create_results_table(all_summaries, save_path=os.path.join(save_dir, "table1_synthetic.txt"))
    
    return all_summaries


def run_realworld_benchmarks(
    variants: List[str] = ["classic", "sparse", "deep"],
    n_iterations_list: Dict[str, List[int]] = None,
    seeds: List[int] = [1, 2, 3, 4, 5],
    save_dir: str = "results"
):
    """
    Run all real-world benchmarks (Table 2 from paper).
    """
    if n_iterations_list is None:
        n_iterations_list = {
            "nanophotonic_3d": [50, 80],
            "rosetta_86d": [50, 80],
            "natsbench_20d": [50, 80],
            "robot_trajectory_40d": [50, 80]
        }
    
    all_summaries = []
    
    for benchmark_name, iterations_list in n_iterations_list.items():
        for n_iter in iterations_list:
            for variant in variants:
                summary = run_multiple_seeds(
                    benchmark_name=benchmark_name,
                    variant=variant,
                    n_iterations=n_iter,
                    seeds=seeds,
                    save_dir=os.path.join(save_dir, benchmark_name)
                )
                summary["n_iterations"] = n_iter
                all_summaries.append(summary)
    
    # Create comparison table
    create_results_table(all_summaries, save_path=os.path.join(save_dir, "table2_realworld.txt"))
    
    return all_summaries


def create_results_table(summaries: List[Dict], save_path: str = None):
    """
    Create a formatted results table similar to paper Tables 1 & 2.
    """
    lines = []
    lines.append("="*80)
    lines.append("Results Summary")
    lines.append("="*80)
    lines.append(f"{'Benchmark':<20} {'Variant':<10} {'T':<6} {'Score (mean±std)':<20}")
    lines.append("-"*80)
    
    for s in summaries:
        line = f"{s['benchmark']:<20} {s['variant']:<10} {s['n_iterations']:<6} "
        line += f"{s['mean_score']:.2f}±{s['std_score']:.2f}"
        lines.append(line)
    
    lines.append("="*80)
    
    table_str = "\n".join(lines)
    print(table_str)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"\nSaved table to {save_path}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="REBMBO Experiments")
    
    parser.add_argument("--benchmark", type=str, default="branin_2d",
                        choices=list(BENCHMARKS.keys()),
                        help="Benchmark to run")
    parser.add_argument("--variant", type=str, default="classic",
                        choices=["classic", "sparse", "deep"],
                        help="GP variant")
    parser.add_argument("--n_init", type=int, default=5,
                        help="Number of initial samples")
    parser.add_argument("--n_iterations", type=int, default=30,
                        help="Number of optimization iterations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Random seeds to use")
    parser.add_argument("--no_ppo", action="store_true",
                        help="Disable PPO (use EBM-UCB only)")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--run_all_synthetic", action="store_true",
                        help="Run all synthetic benchmarks (Table 1)")
    parser.add_argument("--run_all_realworld", action="store_true",
                        help="Run all real-world benchmarks (Table 2)")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks")
    
    args = parser.parse_args()
    
    if args.list:
        list_benchmarks()
        return
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.run_all_synthetic:
        run_synthetic_benchmarks(save_dir=args.save_dir, seeds=args.seeds)
    elif args.run_all_realworld:
        run_realworld_benchmarks(save_dir=args.save_dir, seeds=args.seeds)
    else:
        # Run single benchmark
        run_multiple_seeds(
            benchmark_name=args.benchmark,
            variant=args.variant,
            n_init=args.n_init,
            n_iterations=args.n_iterations,
            seeds=args.seeds,
            use_ppo=not args.no_ppo,
            save_dir=os.path.join(args.save_dir, args.benchmark)
        )


if __name__ == "__main__":
    main()