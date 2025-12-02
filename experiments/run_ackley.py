#!/usr/bin/env python
"""
Run REBMBO experiments on Ackley 5D benchmark.

This script reproduces the Ackley 5D results from the paper (Figure 2b, Tables 1).
Ackley is a highly multimodal function that tests global exploration capability.

OPTIMIZED VERSION with improved hyperparameters:
- Increased iterations: 30 → 50
- Higher entropy coefficient: 0.01 → 0.05 (more exploration)
- Reduced EBM gamma: 0.5 → 0.3 (less EBM interference)
- Smaller mini_batch_size: 64 → 32 (more frequent updates)
- Improved EBM settings for value-weighted training

Usage:
    python run_ackley.py                     # Quick test
    python run_ackley.py --full              # Full paper reproduction
    python run_ackley.py --variant sparse    # Specific variant
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

from src.benchmarks import get_benchmark, ACKLEY_INFO
from src.rebmbo.algorithm import REBMBO, REBMBOConfig
from src.utils.config import load_config, get_config_path
from src.utils.logger import ExperimentLogger
from src.utils.metrics import (
    compute_lar, 
    compute_normalized_score,
    aggregate_results,
    MetricsTracker
)
from src.utils.plotting import (
    plot_convergence_curve,
    plot_lar_curve,
    create_figure_2_style_plot,
    plot_multiple_seeds,
    plot_comparison
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_ackley_experiment(
    variant: str = "classic",
    n_init: int = 5,
    n_iterations: int = 50,  # INCREASED: 30 → 50
    seed: int = 42,
    use_ppo: bool = True,
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run a single REBMBO experiment on Ackley 5D.
    
    Args:
        variant: GP variant ('classic', 'sparse', 'deep')
        n_init: Number of initial random samples
        n_iterations: Number of BO iterations
        seed: Random seed
        use_ppo: Whether to use PPO (vs EBM-UCB only)
        save_dir: Directory to save results
        verbose: Print progress
    
    Returns:
        Results dictionary
    """
    set_seed(seed)
    
    # Get benchmark
    func, info = get_benchmark("ackley_5d")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"REBMBO-{variant[0].upper()} on Ackley 5D")
        print(f"Seed: {seed}, Iterations: {n_iterations}")
        print(f"Using PPO: {use_ppo}")
        print(f"{'='*60}")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device: {device}")
    
    # ============================================================
    # OPTIMIZED CONFIG - Key hyperparameter changes
    # ============================================================
    config = REBMBOConfig(
        input_dim=info.dim,
        bounds=info.bounds,
        
        # GP Settings (Module A)
        gp_variant=variant,
        gp_train_epochs=100,
        gp_retrain_epochs=30,
        
        # EBM Settings (Module B) - OPTIMIZED
        ebm_hidden_dims=[128, 128, 64],
        ebm_train_epochs=100,
        ebm_retrain_epochs=30,
        ebm_mcmc_steps=30,           # INCREASED: 20 → 30 (better MCMC samples)
        ebm_mcmc_step_size=0.05,     # DECREASED: 0.1 → 0.05 (more stable)
        ebm_num_negative_samples=128,  # INCREASED: 100 → 128
        ebm_temperature=1.0,          # NEW: for value-weighted training
        
        # PPO Settings (Module C) - OPTIMIZED
        ppo_hidden_dims=[256, 256],
        ppo_lr_actor=3e-4,
        ppo_lr_critic=1e-3,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_clip_epsilon=0.2,
        ppo_epochs=10,
        ppo_mini_batch_size=32,      # DECREASED: 64 → 32 (more frequent updates)
        ppo_entropy_coef=0.05,       # INCREASED: 0.01 → 0.05 (more exploration)
        ppo_value_coef=0.5,
        ppo_max_grad_norm=0.5,
        
        # Acquisition & Exploration - OPTIMIZED
        num_grid_points=50,
        beta=2.0,                     # UCB exploration (keep as is)
        gamma=0.3,                    # DECREASED: 0.5 → 0.3 (reduce EBM weight)
        lambda_energy=0.2,            # DECREASED: 0.3 → 0.2 (less energy penalty)
        
        device=device
    )
    
    # Initialize logger
    logger = None
    if save_dir:
        logger = ExperimentLogger(
            log_dir=save_dir,
            experiment_name="ackley_5d",
            benchmark_name="ackley_5d",
            variant=variant,
            seed=seed
        )
    
    # Create REBMBO instance
    rebmbo = REBMBO(config)
    
    # Generate initial samples
    X_init = torch.rand(n_init, info.dim)
    y_init = torch.tensor([func(x.numpy()) for x in X_init], dtype=torch.float32)
    
    if verbose:
        print(f"Initial best y: {y_init.max().item():.4f}")
    
    # Initialize
    rebmbo.initialize(X_init, y_init)
    
    # Metrics tracker
    metrics_tracker = MetricsTracker(
        optimal_value=info.optimal_value,
        alpha=0.1
    )
    
    # Run optimization
    start_time = time.time()
    
    for t in range(n_iterations):
        x_t, y_t = rebmbo.step(
            objective_fn=func,
            use_ppo=use_ppo,
            verbose=verbose
        )
        
        # Get energy at current point
        x_tensor = x_t.unsqueeze(0) if x_t.dim() == 1 else x_t
        energy_t = rebmbo.ebm.get_energy(x_tensor).item()
        
        # Update metrics
        metrics_tracker.update(y_t, energy_t)
        
        # Log iteration
        if logger:
            # Get GP predictions at current point
            gp_mean, gp_std = rebmbo.gp.predict(x_tensor)
            
            # Compute LAR
            lar_values, lar_stats = compute_lar(
                metrics_tracker.y_values,
                metrics_tracker.energy_values,
                info.optimal_value,
                alpha=0.1
            )
            current_lar = lar_values[-1] if lar_values else 0
            
            logger.log_iteration(
                x=x_t.cpu().numpy(),
                y=y_t,
                energy=energy_t,
                gp_mean=gp_mean.mean().item(),
                gp_std=gp_std.mean().item(),
                lar=current_lar,
                ppo_stats=rebmbo.history.get("ppo_stats", [{}])[-1] if rebmbo.history.get("ppo_stats") else {}
            )
    
    total_time = time.time() - start_time
    
    # Get results
    results = rebmbo.get_results()
    
    # Compute final metrics
    lar_values, lar_stats = compute_lar(
        metrics_tracker.y_values,
        metrics_tracker.energy_values,
        info.optimal_value,
        alpha=0.1
    )
    
    normalized_score = compute_normalized_score(
        results["best_y"],
        y_init.max().item(),
        info.optimal_value
    )
    
    # Add metrics to results
    results["lar_values"] = lar_values
    results["lar_stats"] = lar_stats
    results["normalized_score"] = normalized_score
    results["total_time"] = total_time
    results["seed"] = seed
    results["variant"] = variant
    results["benchmark"] = "ackley_5d"
    results["optimal_value"] = info.optimal_value
    
    # Add LAR to history
    results["history"]["lar"] = lar_values
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results Summary:")
        print(f"  Best y: {results['best_y']:.4f} (optimal: {info.optimal_value})")
        print(f"  Normalized score: {normalized_score:.2f}")
        print(f"  Final LAR: {lar_stats['final_lar']:.4f}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"{'='*60}")
    
    # Save results
    if logger:
        final_results = logger.save_final_results(results)
    
    return results


def run_ackley_experiments(
    variants: list = ["classic", "sparse", "deep"],
    n_iterations_list: list = [50, 100],  # CHANGED: [30, 50] → [50, 100]
    seeds: list = [1, 2, 3, 4, 5],
    use_ppo: bool = True,
    save_dir: str = "results/ackley_5d",
    create_plots: bool = True
):
    """
    Run full Ackley 5D experiments with multiple seeds and variants.
    Reproduces results from the paper.
    
    Args:
        variants: List of GP variants to test
        n_iterations_list: List of iteration counts
        seeds: Random seeds for multiple runs
        use_ppo: Whether to use PPO
        save_dir: Base directory for results
        create_plots: Whether to create visualization plots
    """
    print("\n" + "#"*60)
    print("# REBMBO Ackley 5D Experiments (OPTIMIZED)")
    print("# Reproducing results from Table 1")
    print("#"*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}  # Store results by variant and n_iterations
    
    for n_iter in n_iterations_list:
        print(f"\n{'='*60}")
        print(f"T = {n_iter} iterations")
        print(f"{'='*60}")
        
        for variant in variants:
            variant_key = f"REBMBO-{variant[0].upper()}"
            
            print(f"\n--- {variant_key} ---")
            
            results_list = []
            
            for seed in seeds:
                result = run_single_ackley_experiment(
                    variant=variant,
                    n_init=5,
                    n_iterations=n_iter,
                    seed=seed,
                    use_ppo=use_ppo,
                    save_dir=os.path.join(save_dir, f"T{n_iter}", variant),
                    verbose=True
                )
                results_list.append(result)
            
            # Aggregate results
            aggregated = aggregate_results(results_list, ACKLEY_INFO.optimal_value)
            
            key = f"{variant_key}_T{n_iter}"
            all_results[key] = {
                "results_list": results_list,
                "aggregated": aggregated,
                "variant": variant,
                "n_iterations": n_iter
            }
            
            # Print summary
            print(f"\n{variant_key} (T={n_iter}) Summary:")
            print(f"  Score: {aggregated['score_mean']:.2f} ± {aggregated['score_std']:.2f}")
            print(f"  Best y: {aggregated['best_y_mean']:.4f} ± {aggregated['best_y_std']:.4f}")
    
    # Save aggregated results
    summary_file = os.path.join(save_dir, "summary.json")
    summary = {}
    for key, data in all_results.items():
        summary[key] = {
            "score_mean": data["aggregated"]["score_mean"],
            "score_std": data["aggregated"]["score_std"],
            "best_y_mean": data["aggregated"]["best_y_mean"],
            "best_y_std": data["aggregated"]["best_y_std"],
            "n_runs": data["aggregated"]["n_runs"]
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_file}")
    
    # Create plots
    if create_plots:
        create_ackley_plots(all_results, save_dir)
    
    return all_results


def create_ackley_plots(all_results: dict, save_dir: str):
    """
    Create visualization plots for Ackley 5D experiments.
    """
    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\nCreating visualization plots...")
    
    # 1. Convergence plot for T=50
    results_t50 = {}
    for key, data in all_results.items():
        if "_T50" in key:
            method_name = key.replace("_T50", "")
            results_t50[method_name] = data["results_list"]
    
    if results_t50:
        create_figure_2_style_plot(
            results_by_method=results_t50,
            benchmark_name="Ackley 5D (T=50)",
            metric='best_y',
            save_path=os.path.join(figures_dir, "ackley_5d_T50_convergence.png"),
            show=False
        )
        
        # LAR plot
        create_figure_2_style_plot(
            results_by_method=results_t50,
            benchmark_name="Ackley 5D (T=50)",
            metric='lar',
            save_path=os.path.join(figures_dir, "ackley_5d_T50_lar.png"),
            show=False
        )
    
    # 2. Convergence plot for T=100
    results_t100 = {}
    for key, data in all_results.items():
        if "_T100" in key:
            method_name = key.replace("_T100", "")
            results_t100[method_name] = data["results_list"]
    
    if results_t100:
        create_figure_2_style_plot(
            results_by_method=results_t100,
            benchmark_name="Ackley 5D (T=100)",
            metric='best_y',
            save_path=os.path.join(figures_dir, "ackley_5d_T100_convergence.png"),
            show=False
        )
        
        create_figure_2_style_plot(
            results_by_method=results_t100,
            benchmark_name="Ackley 5D (T=100)",
            metric='lar',
            save_path=os.path.join(figures_dir, "ackley_5d_T100_lar.png"),
            show=False
        )
    
    # 3. Individual variant plots with all seeds
    for key, data in all_results.items():
        variant_name = key.split("_T")[0]
        n_iter = data["n_iterations"]
        
        plot_multiple_seeds(
            all_results=data["results_list"],
            method_name=variant_name,
            metric='best_y',
            title=f"{variant_name} on Ackley 5D (T={n_iter})",
            save_path=os.path.join(figures_dir, f"ackley_5d_{variant_name}_T{n_iter}_seeds.png"),
            show=False
        )
    
    print(f"Saved plots to {figures_dir}")


def create_results_table():
    """Print a formatted results table similar to Table 1."""
    # This would typically load saved results
    print("\n" + "="*80)
    print("Ackley 5D Results (Table 1 Format)")
    print("="*80)
    print(f"{'Model':<15} {'T=50':<20} {'T=100':<20}")
    print("-"*80)
    # Results would be filled from saved data
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run REBMBO on Ackley 5D benchmark (OPTIMIZED)"
    )
    
    parser.add_argument(
        "--variant", type=str, default="classic",
        choices=["classic", "sparse", "deep"],
        help="GP variant to use"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=50,  # CHANGED: 30 → 50
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Random seeds (single for quick test, multiple for paper)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full paper reproduction (all variants, T=50/100, 5 seeds)"
    )
    parser.add_argument(
        "--no_ppo", action="store_true",
        help="Disable PPO (use EBM-UCB only)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results/ackley_5d",
        help="Directory to save results"
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Skip plot generation"
    )
    
    args = parser.parse_args()
    
    if args.full:
        # Full paper reproduction
        run_ackley_experiments(
            variants=["classic", "sparse", "deep"],
            n_iterations_list=[50, 100],  # CHANGED: [30, 50] → [50, 100]
            seeds=[1, 2, 3, 4, 5],
            use_ppo=not args.no_ppo,
            save_dir=args.save_dir,
            create_plots=not args.no_plots
        )
    else:
        # Single experiment or quick test
        if len(args.seeds) == 1:
            # Single run
            result = run_single_ackley_experiment(
                variant=args.variant,
                n_init=5,
                n_iterations=args.n_iterations,
                seed=args.seeds[0],
                use_ppo=not args.no_ppo,
                save_dir=args.save_dir,
                verbose=True
            )
            
            # Plot convergence
            if not args.no_plots:
                plot_convergence_curve(
                    result,
                    title=f"REBMBO-{args.variant[0].upper()} on Ackley 5D",
                    save_path=os.path.join(args.save_dir, "figures", "convergence.png"),
                    show=True
                )
        else:
            # Multiple seeds
            results_list = []
            for seed in args.seeds:
                result = run_single_ackley_experiment(
                    variant=args.variant,
                    n_init=5,
                    n_iterations=args.n_iterations,
                    seed=seed,
                    use_ppo=not args.no_ppo,
                    save_dir=args.save_dir,
                    verbose=True
                )
                results_list.append(result)
            
            # Aggregate and plot
            aggregated = aggregate_results(results_list, ACKLEY_INFO.optimal_value)
            print(f"\nAggregated Results:")
            print(f"  Score: {aggregated['score_mean']:.2f} ± {aggregated['score_std']:.2f}")
            print(f"  Best y: {aggregated['best_y_mean']:.4f} ± {aggregated['best_y_std']:.4f}")
            
            if not args.no_plots:
                os.makedirs(os.path.join(args.save_dir, "figures"), exist_ok=True)
                plot_multiple_seeds(
                    results_list,
                    method_name=f"REBMBO-{args.variant[0].upper()}",
                    metric='best_y',
                    title=f"REBMBO-{args.variant[0].upper()} on Ackley 5D",
                    save_path=os.path.join(args.save_dir, "figures", "convergence_multi.png"),
                    show=True
                )


if __name__ == "__main__":
    main()