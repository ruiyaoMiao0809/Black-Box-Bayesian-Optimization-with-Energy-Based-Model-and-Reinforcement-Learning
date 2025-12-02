"""
Metrics for REBMBO evaluation.
Implements Landscape-Aware Regret (LAR) and other performance metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import torch


def compute_simple_regret(
    y_values: List[float],
    optimal_value: float
) -> List[float]:
    """
    Compute simple regret at each iteration.
    
    Simple regret: R_t = f(x*) - max_{i<=t} f(x_i)
    
    Args:
        y_values: List of function values
        optimal_value: True optimal value f(x*)
    
    Returns:
        List of simple regrets
    """
    regrets = []
    best_so_far = float('-inf')
    
    for y in y_values:
        best_so_far = max(best_so_far, y)
        regret = optimal_value - best_so_far
        regrets.append(regret)
    
    return regrets


def compute_instantaneous_regret(
    y_values: List[float],
    optimal_value: float
) -> List[float]:
    """
    Compute instantaneous regret at each iteration.
    
    Instantaneous regret: r_t = f(x*) - f(x_t)
    
    Args:
        y_values: List of function values
        optimal_value: True optimal value f(x*)
    
    Returns:
        List of instantaneous regrets
    """
    return [optimal_value - y for y in y_values]


def compute_lar(
    y_values: List[float],
    energy_values: List[float],
    optimal_value: float,
    optimal_energy: float = None,
    alpha: float = 0.1
) -> Tuple[List[float], Dict]:
    """
    Compute Landscape-Aware Regret (LAR).
    
    LAR: R^{LAR}_t = [f(x*) - f(x_t)] + α[E_θ(x*) - E_θ(x_t)]
    
    - First term: Standard instantaneous regret
    - Second term: Energy regret (global exploration efficiency)
    
    Args:
        y_values: List of function values f(x_t)
        energy_values: List of EBM energy values E_θ(x_t)
        optimal_value: True optimal function value f(x*)
        optimal_energy: Energy at optimum (if None, use min observed)
        alpha: Weight for energy term
    
    Returns:
        Tuple of (LAR values, statistics dict)
    """
    if len(y_values) != len(energy_values):
        raise ValueError("y_values and energy_values must have same length")
    
    # Filter None values
    valid_energies = [e for e in energy_values if e is not None]
    
    # Estimate optimal energy if not provided
    if optimal_energy is None and valid_energies:
        optimal_energy = min(valid_energies)  # Low energy = good region
    elif optimal_energy is None:
        optimal_energy = 0
    
    lar_values = []
    function_regrets = []
    energy_regrets = []
    
    for y, e in zip(y_values, energy_values):
        # Function regret
        func_regret = optimal_value - y
        function_regrets.append(func_regret)
        
        # Energy regret (if energy available)
        if e is not None:
            energy_regret = optimal_energy - e  # Lower energy is better
        else:
            energy_regret = 0
        energy_regrets.append(energy_regret)
        
        # Combined LAR
        lar = func_regret + alpha * energy_regret
        lar_values.append(lar)
    
    # Compute statistics
    stats = {
        "final_lar": lar_values[-1] if lar_values else 0,
        "mean_lar": np.mean(lar_values),
        "std_lar": np.std(lar_values),
        "cumulative_lar": np.sum(lar_values),
        "final_function_regret": function_regrets[-1] if function_regrets else 0,
        "final_energy_regret": energy_regrets[-1] if energy_regrets else 0,
        "alpha": alpha
    }
    
    return lar_values, stats


def compute_normalized_score(
    best_y: float,
    initial_best_y: float,
    optimal_value: float
) -> float:
    """
    Compute normalized score (0-100 scale, higher is better).
    
    Score = 100 * (1 - |best_y - optimal| / |initial_best - optimal|)
    
    This measures improvement relative to initial random sampling.
    
    Args:
        best_y: Best found value
        initial_best_y: Best initial random sample value
        optimal_value: True optimal value
    
    Returns:
        Normalized score (0-100)
    """
    if initial_best_y == optimal_value:
        return 100.0
    
    gap_initial = abs(optimal_value - initial_best_y)
    gap_final = abs(optimal_value - best_y)
    
    if gap_initial == 0:
        return 100.0 if gap_final == 0 else 0.0
    
    score = 100 * (1 - gap_final / gap_initial)
    return max(0.0, min(100.0, score))


def compute_cumulative_regret(
    y_values: List[float],
    optimal_value: float
) -> List[float]:
    """
    Compute cumulative regret at each iteration.
    
    Cumulative regret: CR_T = Σ_{t=1}^T (f(x*) - f(x_t))
    
    Args:
        y_values: List of function values
        optimal_value: True optimal value
    
    Returns:
        List of cumulative regrets
    """
    cumulative = []
    total = 0
    
    for y in y_values:
        total += (optimal_value - y)
        cumulative.append(total)
    
    return cumulative


def compute_best_observed(y_values: List[float]) -> List[float]:
    """
    Compute best observed value at each iteration.
    
    Args:
        y_values: List of function values
    
    Returns:
        List of best values so far
    """
    best_values = []
    best = float('-inf')
    
    for y in y_values:
        best = max(best, y)
        best_values.append(best)
    
    return best_values


def aggregate_results(
    all_results: List[Dict],
    optimal_value: float = None
) -> Dict:
    """
    Aggregate results from multiple runs (different seeds).
    
    Args:
        all_results: List of result dictionaries from different runs
        optimal_value: True optimal value (optional)
    
    Returns:
        Aggregated statistics
    """
    if not all_results:
        return {}
    
    # Extract values
    best_ys = [r.get('best_y', r.get('y', [])[-1] if 'y' in r else 0) 
               for r in all_results]
    
    scores = [r.get('normalized_score', 0) for r in all_results]
    
    # Aggregate histories if available
    all_best_y_curves = []
    all_lar_curves = []
    
    for r in all_results:
        history = r.get('history', {})
        if 'best_y' in history:
            all_best_y_curves.append(history['best_y'])
        if 'lar' in history:
            all_lar_curves.append(history['lar'])
    
    # Compute mean curves
    mean_best_y_curve = None
    std_best_y_curve = None
    if all_best_y_curves:
        # Pad to same length
        max_len = max(len(c) for c in all_best_y_curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in all_best_y_curves]
        mean_best_y_curve = np.mean(padded, axis=0).tolist()
        std_best_y_curve = np.std(padded, axis=0).tolist()
    
    mean_lar_curve = None
    std_lar_curve = None
    if all_lar_curves:
        valid_curves = [c for c in all_lar_curves if all(x is not None for x in c)]
        if valid_curves:
            max_len = max(len(c) for c in valid_curves)
            padded = [c + [c[-1]] * (max_len - len(c)) for c in valid_curves]
            mean_lar_curve = np.mean(padded, axis=0).tolist()
            std_lar_curve = np.std(padded, axis=0).tolist()
    
    return {
        "n_runs": len(all_results),
        "best_y_mean": np.mean(best_ys),
        "best_y_std": np.std(best_ys),
        "best_y_values": best_ys,
        "score_mean": np.mean(scores),
        "score_std": np.std(scores),
        "scores": scores,
        "mean_best_y_curve": mean_best_y_curve,
        "std_best_y_curve": std_best_y_curve,
        "mean_lar_curve": mean_lar_curve,
        "std_lar_curve": std_lar_curve,
        "optimal_value": optimal_value
    }


def format_result_string(
    mean: float,
    std: float,
    precision: int = 2
) -> str:
    """
    Format mean ± std string for tables.
    
    Args:
        mean: Mean value
        std: Standard deviation
        precision: Decimal places
    
    Returns:
        Formatted string like "85.23±1.45"
    """
    return f"{mean:.{precision}f}±{std:.{precision}f}"


class MetricsTracker:
    """
    Tracks and computes metrics during optimization.
    """
    
    def __init__(self, optimal_value: float, alpha: float = 0.1):
        """
        Initialize tracker.
        
        Args:
            optimal_value: True optimal function value
            alpha: LAR weight parameter
        """
        self.optimal_value = optimal_value
        self.alpha = alpha
        
        self.y_values = []
        self.energy_values = []
        self.best_y = float('-inf')
        self.initial_best_y = None
    
    def update(self, y: float, energy: float = None):
        """Update with new observation."""
        self.y_values.append(y)
        self.energy_values.append(energy)
        
        if y > self.best_y:
            self.best_y = y
        
        if self.initial_best_y is None:
            self.initial_best_y = y
    
    def get_simple_regret(self) -> float:
        """Get current simple regret."""
        return self.optimal_value - self.best_y
    
    def get_normalized_score(self) -> float:
        """Get current normalized score."""
        if self.initial_best_y is None:
            return 0.0
        return compute_normalized_score(
            self.best_y,
            self.initial_best_y,
            self.optimal_value
        )
    
    def get_lar(self) -> Tuple[List[float], Dict]:
        """Get LAR values and statistics."""
        return compute_lar(
            self.y_values,
            self.energy_values,
            self.optimal_value,
            alpha=self.alpha
        )
    
    def get_all_metrics(self) -> Dict:
        """Get all current metrics."""
        lar_values, lar_stats = self.get_lar()
        
        return {
            "simple_regret": self.get_simple_regret(),
            "normalized_score": self.get_normalized_score(),
            "best_y": self.best_y,
            "n_iterations": len(self.y_values),
            "lar_stats": lar_stats,
            "simple_regrets": compute_simple_regret(self.y_values, self.optimal_value),
            "lar_values": lar_values,
            "best_observed": compute_best_observed(self.y_values)
        }


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Simulate optimization
    optimal = 0.0  # True optimum
    y_values = list(-np.abs(np.random.randn(30)))  # Simulated values
    energy_values = list(np.random.randn(30))
    
    # Compute LAR
    lar_values, stats = compute_lar(y_values, energy_values, optimal)
    print(f"LAR Stats: {stats}")
    
    # Compute normalized score
    initial_best = max(y_values[:5])
    final_best = max(y_values)
    score = compute_normalized_score(final_best, initial_best, optimal)
    print(f"Normalized Score: {score:.2f}")
    
    # Test tracker
    tracker = MetricsTracker(optimal_value=optimal)
    for y, e in zip(y_values, energy_values):
        tracker.update(y, e)
    
    metrics = tracker.get_all_metrics()
    print(f"Final simple regret: {metrics['simple_regret']:.4f}")
    print(f"Final score: {metrics['normalized_score']:.2f}")