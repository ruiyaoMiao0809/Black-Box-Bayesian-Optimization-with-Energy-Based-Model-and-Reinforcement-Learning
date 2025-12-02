"""
Plotting utilities for REBMBO experiments.
Creates publication-quality figures similar to the paper.

FIXED: 
1. Uses Agg backend to avoid GUI blocking issues
2. Automatically creates parent directories before saving figures
"""

import matplotlib
# Use Agg backend to avoid GUI blocking - MUST be before importing pyplot
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6
})

# Color scheme matching the paper
COLORS = {
    'REBMBO-C': '#1f77b4',  # Blue
    'REBMBO-S': '#2ca02c',  # Green  
    'REBMBO-D': '#9467bd',  # Purple
    'BALLET-ICI': '#ff7f0e',  # Orange
    'TuRBO': '#d62728',  # Red
    'EARL-BO': '#8c564b',  # Brown
    'Classic-BO': '#7f7f7f',  # Gray
    'Two-Step-EI': '#bcbd22',  # Olive
    'KG': '#17becf',  # Cyan
}

MARKERS = {
    'REBMBO-C': 'o',
    'REBMBO-S': 's',
    'REBMBO-D': '^',
    'BALLET-ICI': 'D',
    'TuRBO': 'v',
    'EARL-BO': '<',
    'Classic-BO': 'x',
    'Two-Step-EI': '+',
    'KG': '*',
}


def _save_figure(save_path: str, dpi: int = 300):
    """
    Save figure, automatically creating parent directories if needed.
    
    Args:
        save_path: Path to save the figure
        dpi: Resolution
    """
    # Create parent directories if they don't exist
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")


def plot_convergence_curve(
    results: Dict,
    title: str = "Convergence Curve",
    save_path: str = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot convergence curve for a single experiment.
    
    Args:
        results: Dictionary with 'history' containing 'best_y' list
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot (ignored with Agg backend)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    history = results.get('history', results)
    best_y = history.get('best_y', [])
    
    iterations = range(1, len(best_y) + 1)
    
    ax.plot(iterations, best_y, 'b-', linewidth=2, label='Best observed')
    
    # Mark the final best
    ax.scatter([len(best_y)], [best_y[-1]], c='red', s=100, zorder=5, 
               label=f'Final: {best_y[-1]:.4f}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Objective Value')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        _save_figure(save_path)
    
    # With Agg backend, show() does nothing, so we just close
    plt.close(fig)
    
    return fig, ax


def plot_lar_curve(
    lar_values: List[float],
    title: str = "Landscape-Aware Regret",
    save_path: str = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot LAR curve over iterations.
    
    Args:
        lar_values: List of LAR values
        title: Plot title
        save_path: Path to save figure
        show: Whether to display (ignored with Agg backend)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = range(1, len(lar_values) + 1)
    
    ax.plot(iterations, lar_values, 'b-', linewidth=2)
    ax.fill_between(iterations, 0, lar_values, alpha=0.3)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Landscape-Aware Regret (LAR)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        _save_figure(save_path)
    
    plt.close(fig)
    
    return fig, ax


def plot_comparison(
    results_dict: Dict[str, Dict],
    metric: str = 'best_y',
    title: str = "Method Comparison",
    save_path: str = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Compare multiple methods on the same plot.
    
    Args:
        results_dict: Dictionary mapping method names to results
        metric: Which metric to plot ('best_y', 'lar', etc.)
        title: Plot title
        save_path: Path to save
        show: Whether to display (ignored with Agg backend)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for method_name, results in results_dict.items():
        history = results.get('history', results)
        values = history.get(metric, [])
        
        if not values:
            continue
        
        iterations = range(1, len(values) + 1)
        
        color = COLORS.get(method_name, 'gray')
        marker = MARKERS.get(method_name, 'o')
        
        # Plot with markers every few points
        marker_every = max(1, len(values) // 10)
        
        ax.plot(iterations, values, color=color, linewidth=2, 
                label=method_name, marker=marker, markevery=marker_every)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        _save_figure(save_path)
    
    plt.close(fig)
    
    return fig, ax


def plot_with_confidence(
    mean_curve: List[float],
    std_curve: List[float],
    label: str = None,
    color: str = 'blue',
    ax: plt.Axes = None,
    alpha: float = 0.2
):
    """
    Plot curve with confidence band.
    
    Args:
        mean_curve: Mean values
        std_curve: Standard deviation values
        label: Line label
        color: Line color
        ax: Axes to plot on
        alpha: Confidence band transparency
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    iterations = range(1, len(mean_curve) + 1)
    mean = np.array(mean_curve)
    std = np.array(std_curve)
    
    ax.plot(iterations, mean, color=color, linewidth=2, label=label)
    ax.fill_between(iterations, mean - std, mean + std, 
                    color=color, alpha=alpha)
    
    return ax


def plot_multiple_seeds(
    all_results: List[Dict],
    method_name: str = "REBMBO",
    metric: str = 'best_y',
    title: str = None,
    save_path: str = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot results from multiple seeds with mean and std bands.
    
    Args:
        all_results: List of result dictionaries
        method_name: Name of method
        metric: Which metric to plot
        title: Plot title
        save_path: Path to save
        show: Whether to display (ignored with Agg backend)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract curves
    curves = []
    for result in all_results:
        history = result.get('history', result)
        values = history.get(metric, [])
        if values:
            curves.append(values)
    
    if not curves:
        print("No data to plot")
        return None, None
    
    # Pad to same length
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        padded.append(c + [c[-1]] * (max_len - len(c)))
    
    curves_arr = np.array(padded)
    mean_curve = np.mean(curves_arr, axis=0)
    std_curve = np.std(curves_arr, axis=0)
    
    # Plot
    color = COLORS.get(method_name, 'blue')
    iterations = range(1, max_len + 1)
    
    ax.plot(iterations, mean_curve, color=color, linewidth=2, 
            label=f'{method_name} (mean)')
    ax.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve,
                    color=color, alpha=0.2, label=f'{method_name} (±1 std)')
    
    # Plot individual runs (lighter)
    for curve in curves:
        ax.plot(range(1, len(curve) + 1), curve, color=color, 
                alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f"{method_name} - {metric}")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        _save_figure(save_path)
    
    plt.close(fig)
    
    return fig, ax


def create_figure_2_style_plot(
    results_by_method: Dict[str, List[Dict]],
    benchmark_name: str,
    metric: str = 'best_y',
    save_path: str = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create a publication-style plot similar to Figure 2 in the paper.
    Shows multiple methods with mean ± std bands.
    
    Args:
        results_by_method: Dict mapping method names to list of results
        benchmark_name: Name of benchmark
        metric: Which metric to plot (typically 'lar' for LAR)
        save_path: Path to save
        show: Whether to display (ignored with Agg backend)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for method_name, results_list in results_by_method.items():
        # Extract curves
        curves = []
        for result in results_list:
            history = result.get('history', result)
            values = history.get(metric, [])
            if values:
                curves.append(values)
        
        if not curves:
            continue
        
        # Pad to same length
        max_len = max(len(c) for c in curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
        curves_arr = np.array(padded)
        
        mean_curve = np.mean(curves_arr, axis=0)
        std_curve = np.std(curves_arr, axis=0)
        
        color = COLORS.get(method_name, 'gray')
        marker = MARKERS.get(method_name, 'o')
        iterations = range(1, max_len + 1)
        
        # Plot mean with markers
        marker_every = max(1, max_len // 8)
        ax.plot(iterations, mean_curve, color=color, linewidth=2,
                label=method_name, marker=marker, markevery=marker_every,
                markersize=6)
        
        # Confidence band
        ax.fill_between(iterations, mean_curve - std_curve, 
                        mean_curve + std_curve,
                        color=color, alpha=0.15)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Landscape-Aware Regret (LAR)' if metric == 'lar' 
                  else metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"{benchmark_name} (mean ± std, 5 runs)", fontsize=14)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0 if appropriate
    if metric == 'lar':
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(save_path)
    
    plt.close(fig)
    
    return fig, ax


def plot_energy_landscape(
    ebm_module,
    dim: int = 2,
    bounds: Tuple[float, float] = (0, 1),
    resolution: int = 50,
    title: str = "EBM Energy Landscape",
    save_path: str = None,
    show: bool = True
):
    """
    Visualize the EBM energy landscape (2D only).
    
    Args:
        ebm_module: Trained EBM module
        dim: Input dimension (must be 2)
        bounds: Domain bounds
        resolution: Grid resolution
        title: Plot title
        save_path: Path to save
        show: Whether to display (ignored with Agg backend)
    """
    if dim != 2:
        print("Energy landscape visualization only supports 2D")
        return None, None
    
    import torch
    
    # Create grid
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten and convert to tensor
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Get energies
    energies = ebm_module.get_energy(points_tensor).cpu().numpy()
    Z = energies.reshape(resolution, resolution)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    contour = ax.contourf(X, Y, Z, levels=50, cmap='RdYlBu_r')
    plt.colorbar(contour, ax=ax, label='Energy')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    
    if save_path:
        _save_figure(save_path)
    
    plt.close(fig)
    
    return fig, ax


def plot_optimization_trajectory(
    X: np.ndarray,
    y: np.ndarray,
    objective_fn=None,
    bounds: Tuple[float, float] = (0, 1),
    resolution: int = 100,
    title: str = "Optimization Trajectory",
    save_path: str = None,
    show: bool = True
):
    """
    Plot optimization trajectory on objective function contour (2D only).
    
    Args:
        X: Sampled points [N, 2]
        y: Function values [N]
        objective_fn: Objective function (for contour)
        bounds: Domain bounds
        resolution: Contour resolution
        title: Plot title
        save_path: Path to save
        show: Whether to display (ignored with Agg backend)
    """
    if X.shape[1] != 2:
        print("Trajectory visualization only supports 2D")
        return None, None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour if objective function provided
    if objective_fn is not None:
        x = np.linspace(bounds[0], bounds[1], resolution)
        y_grid = np.linspace(bounds[0], bounds[1], resolution)
        X_grid, Y_grid = np.meshgrid(x, y_grid)
        
        Z = np.zeros_like(X_grid)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = objective_fn(np.array([X_grid[i, j], Y_grid[i, j]]))
        
        contour = ax.contourf(X_grid, Y_grid, Z, levels=50, cmap='viridis', alpha=0.6)
        plt.colorbar(contour, ax=ax, label='Objective Value')
    
    # Plot trajectory
    ax.plot(X[:, 0], X[:, 1], 'w-', linewidth=1, alpha=0.5)
    
    # Color points by function value
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='hot', 
                         edgecolors='black', s=50, zorder=5)
    
    # Mark start and end
    ax.scatter(X[0, 0], X[0, 1], c='green', s=150, marker='o', 
               edgecolors='white', linewidths=2, label='Start', zorder=6)
    ax.scatter(X[-1, 0], X[-1, 1], c='red', s=150, marker='*', 
               edgecolors='white', linewidths=2, label='End', zorder=6)
    
    # Mark best point
    best_idx = np.argmax(y)
    ax.scatter(X[best_idx, 0], X[best_idx, 1], c='cyan', s=200, marker='*',
               edgecolors='black', linewidths=2, label='Best', zorder=7)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    if save_path:
        _save_figure(save_path)
    
    plt.close(fig)
    
    return fig, ax


def create_results_table_figure(
    results_table: Dict[str, Dict[str, str]],
    title: str = "Results Comparison",
    save_path: str = None,
    show: bool = True
):
    """
    Create a table figure for results (similar to Tables 1 & 2).
    
    Args:
        results_table: Nested dict with row names -> column names -> values
        title: Table title
        save_path: Path to save
        show: Whether to display (ignored with Agg backend)
    """
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(results_table).T
    
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 2))
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style row labels
    for i in range(1, len(df) + 1):
        table[(i, -1)].set_facecolor('#E8E8E8')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        _save_figure(save_path)
    
    plt.close(fig)
    
    return fig, ax


if __name__ == "__main__":
    # Test plotting functions
    import tempfile
    
    # Generate sample data
    np.random.seed(42)
    n_iter = 30
    
    # Simulate convergence curves
    base = -5 + np.random.randn(n_iter) * 0.5
    for i in range(1, n_iter):
        base[i] = max(base[i], base[i-1] + np.random.rand() * 0.3)
    
    results = {
        'history': {
            'best_y': base.tolist(),
            'lar': (5 - base).tolist()
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test convergence plot with nested directory
        nested_path = os.path.join(tmpdir, "nested", "subdir", "convergence.png")
        plot_convergence_curve(
            results, 
            title="Test Convergence",
            save_path=nested_path,
            show=False
        )
        print(f"✓ Saved to nested directory: {nested_path}")
        
        # Test LAR plot
        plot_lar_curve(
            results['history']['lar'],
            title="Test LAR",
            save_path=os.path.join(tmpdir, "lar.png"),
            show=False
        )
        
        print("✅ All plotting tests passed!")