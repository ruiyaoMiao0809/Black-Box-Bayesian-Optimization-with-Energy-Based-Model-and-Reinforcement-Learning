#!/usr/bin/env python
"""
BALLET-ICI: Bayesian Optimization with Adaptive Level-set Estimation
and Iterative Confidence Intervals

Reference:
    Zhang, F., Song, J., Bowden, J. C., Ladd, A., Yue, Y., Desautels, T., & Chen, Y. (2023).
    Learning regions of interest for Bayesian optimization with adaptive level-set estimation.
    In International Conference on Machine Learning (pp. 41579-41595). PMLR.

This baseline implementation provides:
    - Alternating global and local GP modeling
    - Iterative Confidence Interval (ICI) for region of interest identification
    - Level-set estimation for focused exploration
    
Interface compatible with REBMBO for fair comparison.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, Optional, Dict, List, Union
from dataclasses import dataclass, field
import time
import warnings

# Try to import gpytorch for GP modeling
try:
    import gpytorch
    from gpytorch.models import ExactGP
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False
    warnings.warn("gpytorch not found. Using simplified GP implementation.")


@dataclass
class BALLETICIConfig:
    """Configuration for BALLET-ICI algorithm."""
    
    # Problem definition
    input_dim: int = 2
    bounds: Tuple[float, float] = (0.0, 1.0)
    
    # GP configuration
    kernel_type: str = "rbf"  # "rbf", "matern", "mixed"
    noise_variance: float = 1e-4
    
    # Level-set estimation
    level_set_threshold: float = 0.5  # Percentile for level-set
    confidence_level: float = 0.95    # Confidence level for ICI
    
    # Region of interest
    roi_expansion_factor: float = 1.2  # How much to expand ROI
    min_roi_size: float = 0.1          # Minimum ROI size per dimension
    
    # Acquisition
    beta: float = 2.0  # UCB exploration parameter
    use_local_gp: bool = True  # Whether to use local GP refinement
    
    # Training
    gp_train_epochs: int = 100
    gp_retrain_epochs: int = 50
    learning_rate: float = 0.1
    
    # Grid for acquisition optimization
    num_grid_points: int = 50
    num_restarts: int = 5  # For multi-start optimization
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleGP:
    """
    Simple Gaussian Process implementation for when gpytorch is not available.
    Uses exact GP inference with RBF kernel.
    """
    
    def __init__(self, input_dim: int, noise_variance: float = 1e-4, 
                 length_scale: float = 0.2, device: str = "cpu"):
        self.input_dim = input_dim
        self.noise_variance = noise_variance
        self.length_scale = length_scale
        self.device = device
        
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
    
    def _rbf_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        # Compute squared distances
        X1 = X1.unsqueeze(1)  # (n1, 1, d)
        X2 = X2.unsqueeze(0)  # (1, n2, d)
        dist_sq = torch.sum((X1 - X2) ** 2, dim=-1)  # (n1, n2)
        return torch.exp(-0.5 * dist_sq / (self.length_scale ** 2))
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the GP to training data."""
        self.X_train = X.to(self.device)
        self.y_train = y.to(self.device)
        
        n = X.shape[0]
        K = self._rbf_kernel(self.X_train, self.X_train)
        K = K + self.noise_variance * torch.eye(n, device=self.device)
        
        # Compute inverse using Cholesky
        try:
            L = torch.linalg.cholesky(K)
            self.alpha = torch.cholesky_solve(self.y_train.unsqueeze(-1), L).squeeze(-1)
            self.K_inv = torch.cholesky_solve(torch.eye(n, device=self.device), L)
        except:
            # Fallback to direct inverse
            self.K_inv = torch.linalg.inv(K + 1e-6 * torch.eye(n, device=self.device))
            self.alpha = self.K_inv @ self.y_train
    
    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance at test points."""
        X = X.to(self.device)
        
        if self.X_train is None:
            # Prior prediction
            return torch.zeros(X.shape[0], device=self.device), \
                   torch.ones(X.shape[0], device=self.device)
        
        K_star = self._rbf_kernel(X, self.X_train)
        K_star_star = self._rbf_kernel(X, X)
        
        # Mean prediction
        mean = K_star @ self.alpha
        
        # Variance prediction
        v = K_star @ self.K_inv @ K_star.T
        var = torch.diag(K_star_star) - torch.diag(v)
        var = torch.clamp(var, min=1e-6)
        
        return mean, var


class GPyTorchGP(ExactGP if HAS_GPYTORCH else object):
    """GPyTorch-based Gaussian Process model."""
    
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: 'GaussianLikelihood', kernel_type: str = "rbf"):
        if not HAS_GPYTORCH:
            raise ImportError("gpytorch required for GPyTorchGP")
        
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        
        if kernel_type == "rbf":
            self.covar_module = ScaleKernel(RBFKernel())
        elif kernel_type == "matern":
            self.covar_module = ScaleKernel(MaternKernel(nu=2.5))
        else:  # mixed
            self.covar_module = ScaleKernel(
                RBFKernel() + MaternKernel(nu=2.5)
            )
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class RegionOfInterest:
    """
    Manages the Region of Interest (ROI) for focused exploration.
    
    The ROI is defined by level-set estimation based on the current
    best observations and GP predictions.
    """
    
    def __init__(self, input_dim: int, bounds: Tuple[float, float],
                 expansion_factor: float = 1.2, min_size: float = 0.1):
        self.input_dim = input_dim
        self.bounds = bounds
        self.expansion_factor = expansion_factor
        self.min_size = min_size
        
        # Initialize ROI to full domain
        self.roi_lower = torch.full((input_dim,), bounds[0])
        self.roi_upper = torch.full((input_dim,), bounds[1])
    
    def update(self, X: torch.Tensor, y: torch.Tensor, 
               threshold_percentile: float = 0.5) -> None:
        """
        Update ROI based on current observations.
        
        Args:
            X: Observed points (n x d)
            y: Observed values (n,)
            threshold_percentile: What fraction of points to include
        """
        if len(y) < 3:
            return  # Not enough points
        
        # Find threshold value
        threshold = torch.quantile(y, threshold_percentile)
        
        # Select points above threshold
        mask = y >= threshold
        if mask.sum() < 2:
            mask = y >= torch.quantile(y, 0.3)  # Relax threshold
        
        X_good = X[mask]
        
        if len(X_good) < 2:
            return
        
        # Compute bounding box of good points
        roi_lower = X_good.min(dim=0).values
        roi_upper = X_good.max(dim=0).values
        
        # Expand the ROI
        center = (roi_lower + roi_upper) / 2
        half_width = (roi_upper - roi_lower) / 2 * self.expansion_factor
        
        # Ensure minimum size
        half_width = torch.clamp(half_width, min=self.min_size / 2)
        
        # Update ROI with bounds checking
        self.roi_lower = torch.clamp(center - half_width, min=self.bounds[0])
        self.roi_upper = torch.clamp(center + half_width, max=self.bounds[1])
    
    def sample_within_roi(self, n_samples: int) -> torch.Tensor:
        """Sample uniformly within the ROI."""
        samples = torch.rand(n_samples, self.input_dim)
        samples = samples * (self.roi_upper - self.roi_lower) + self.roi_lower
        return samples
    
    def is_within_roi(self, X: torch.Tensor) -> torch.Tensor:
        """Check if points are within the ROI."""
        within = (X >= self.roi_lower) & (X <= self.roi_upper)
        return within.all(dim=-1)


class IterativeConfidenceInterval:
    """
    Implements Iterative Confidence Interval (ICI) for level-set estimation.
    
    ICI progressively refines the estimated level-set by updating
    confidence bounds based on new observations.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.z_score = self._get_z_score(confidence_level)
    
    def _get_z_score(self, confidence: float) -> float:
        """Get z-score for given confidence level."""
        # Approximate z-score for common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        return z_scores.get(confidence, 1.96)
    
    def compute_level_set(self, mean: torch.Tensor, std: torch.Tensor,
                          threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute level-set membership probabilities.
        
        Args:
            mean: GP posterior mean at test points
            std: GP posterior std at test points
            threshold: Level-set threshold value
            
        Returns:
            prob_above: Probability of being above threshold
            prob_below: Probability of being below threshold
        """
        # Compute z-scores
        z = (threshold - mean) / (std + 1e-6)
        
        # Use normal CDF approximation
        prob_below = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        prob_above = 1 - prob_below
        
        return prob_above, prob_below
    
    def get_uncertain_points(self, mean: torch.Tensor, std: torch.Tensor,
                             threshold: float) -> torch.Tensor:
        """
        Identify points with high uncertainty about level-set membership.
        
        Returns a mask of points that are uncertain (neither confidently
        above nor confidently below the threshold).
        """
        prob_above, prob_below = self.compute_level_set(mean, std, threshold)
        
        # Points are uncertain if neither probability exceeds confidence level
        uncertain = (prob_above < self.confidence_level) & \
                    (prob_below < self.confidence_level)
        
        return uncertain


class BALLETICI:
    """
    BALLET-ICI: Bayesian Optimization with Adaptive Level-set Estimation
    and Iterative Confidence Intervals.
    
    This method alternates between:
    1. Global GP for overall landscape modeling
    2. Local GP for refined modeling within the Region of Interest
    3. ICI for adaptive level-set estimation
    
    Interface compatible with REBMBO for fair comparison.
    """
    
    def __init__(self, config: BALLETICIConfig):
        """
        Initialize BALLET-ICI optimizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self.roi = RegionOfInterest(
            input_dim=config.input_dim,
            bounds=config.bounds,
            expansion_factor=config.roi_expansion_factor,
            min_size=config.min_roi_size
        )
        self.ici = IterativeConfidenceInterval(
            confidence_level=config.confidence_level
        )
        
        # GP models (initialized on first data)
        self.global_gp = None
        self.local_gp = None
        self.likelihood = None
        
        # Data storage
        self.X = None
        self.y = None
        
        # Tracking
        self.best_x = None
        self.best_y = float('-inf')
        self.iteration = 0
        
        # History
        self.history = {
            "x": [],
            "y": [],
            "best_y": [],
            "time": [],
            "roi_size": []
        }
    
    def _create_gp(self, X: torch.Tensor, y: torch.Tensor, 
                   use_gpytorch: bool = True) -> Union[SimpleGP, 'GPyTorchGP']:
        """Create a GP model."""
        if HAS_GPYTORCH and use_gpytorch:
            likelihood = GaussianLikelihood()
            likelihood.noise = self.config.noise_variance
            model = GPyTorchGP(X, y, likelihood, self.config.kernel_type)
            return model, likelihood
        else:
            model = SimpleGP(
                input_dim=self.config.input_dim,
                noise_variance=self.config.noise_variance,
                device=str(self.device)
            )
            model.fit(X, y)
            return model, None
    
    def _train_gp(self, model, likelihood, X: torch.Tensor, y: torch.Tensor,
                  epochs: int = 100):
        """Train a GPyTorch GP model."""
        if not HAS_GPYTORCH or likelihood is None:
            # SimpleGP is already fitted
            if hasattr(model, 'fit'):
                model.fit(X, y)
            return
        
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        likelihood.eval()
    
    def _predict_gp(self, model, likelihood, X: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions from GP model."""
        if not HAS_GPYTORCH or likelihood is None:
            # SimpleGP
            return model.predict(X)
        
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            pred = model(X)
            mean = pred.mean
            var = pred.variance
        
        return mean, var
    
    def initialize(self, X_init: torch.Tensor, y_init: torch.Tensor) -> None:
        """
        Initialize with initial observations.
        
        Args:
            X_init: Initial input points (n x d)
            y_init: Initial observations (n,)
        """
        self.X = X_init.to(self.device)
        self.y = y_init.to(self.device)
        
        # Update best
        best_idx = torch.argmax(self.y)
        self.best_x = self.X[best_idx].clone()
        self.best_y = self.y[best_idx].item()
        
        # Create and train global GP
        if HAS_GPYTORCH:
            self.likelihood = GaussianLikelihood().to(self.device)
            self.likelihood.noise = self.config.noise_variance
            self.global_gp = GPyTorchGP(
                self.X, self.y, self.likelihood, self.config.kernel_type
            ).to(self.device)
            self._train_gp(self.global_gp, self.likelihood, self.X, self.y,
                          epochs=self.config.gp_train_epochs)
        else:
            self.global_gp = SimpleGP(
                input_dim=self.config.input_dim,
                noise_variance=self.config.noise_variance,
                device=str(self.device)
            )
            self.global_gp.fit(self.X, self.y)
        
        # Update ROI
        self.roi.update(self.X.cpu(), self.y.cpu(), 
                       threshold_percentile=self.config.level_set_threshold)
        
        print(f"Initializing GP...")
        print(f"Initialization complete. Best initial y: {self.best_y:.4f}")
    
    def _compute_acquisition(self, X: torch.Tensor, 
                            use_local: bool = False) -> torch.Tensor:
        """
        Compute UCB acquisition values.
        
        Uses ICI-based level-set weighting to focus on promising regions.
        """
        # Get GP predictions
        if use_local and self.local_gp is not None:
            mean, var = self._predict_gp(self.local_gp, self.likelihood, X)
        else:
            mean, var = self._predict_gp(self.global_gp, self.likelihood, X)
        
        std = torch.sqrt(var + 1e-6)
        
        # Standard UCB
        ucb = mean + self.config.beta * std
        
        # Level-set weighting using ICI
        threshold = torch.quantile(self.y, self.config.level_set_threshold)
        prob_above, _ = self.ici.compute_level_set(mean, std, threshold.item())
        
        # Weight UCB by probability of being in level-set
        # Points likely in the level-set get higher weight
        weighted_ucb = ucb * (0.5 + 0.5 * prob_above)
        
        return weighted_ucb
    
    def _select_next_point(self) -> torch.Tensor:
        """
        Select the next point to evaluate.
        
        Uses a two-stage approach:
        1. Sample candidates from ROI
        2. Optimize acquisition function
        """
        n_grid = self.config.num_grid_points
        
        # Sample candidates from ROI and full domain
        roi_candidates = self.roi.sample_within_roi(n_grid * 2).to(self.device)
        
        # Also sample some from full domain for diversity
        full_candidates = torch.rand(n_grid, self.config.input_dim, device=self.device)
        full_candidates = full_candidates * (self.config.bounds[1] - self.config.bounds[0])
        full_candidates = full_candidates + self.config.bounds[0]
        
        candidates = torch.cat([roi_candidates, full_candidates], dim=0)
        
        # Compute acquisition values
        with torch.no_grad():
            acq_values = self._compute_acquisition(candidates, 
                                                   use_local=self.config.use_local_gp)
        
        # Select best candidate
        best_idx = torch.argmax(acq_values)
        x_next = candidates[best_idx]
        
        # Optional: refine with gradient-based optimization
        if self.config.num_restarts > 0:
            x_next = self._refine_candidate(x_next)
        
        return x_next
    
    def _refine_candidate(self, x_init: torch.Tensor, 
                          n_steps: int = 20) -> torch.Tensor:
        """Refine candidate using gradient ascent on acquisition."""
        x = x_init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=0.01)
        
        for _ in range(n_steps):
            optimizer.zero_grad()
            
            # Get GP predictions
            mean, var = self._predict_gp(self.global_gp, self.likelihood, 
                                         x.unsqueeze(0))
            std = torch.sqrt(var + 1e-6)
            
            # UCB as objective (maximize)
            acq = mean + self.config.beta * std
            loss = -acq.sum()
            
            loss.backward()
            optimizer.step()
            
            # Project back to bounds
            with torch.no_grad():
                x.clamp_(self.config.bounds[0], self.config.bounds[1])
        
        return x.detach()
    
    def _update_local_gp(self) -> None:
        """Update local GP with points in ROI."""
        if not self.config.use_local_gp:
            return
        
        # Get points within ROI
        within_roi = self.roi.is_within_roi(self.X.cpu())
        
        if within_roi.sum() < 3:
            self.local_gp = None
            return
        
        X_local = self.X[within_roi]
        y_local = self.y[within_roi]
        
        if HAS_GPYTORCH:
            self.local_gp = GPyTorchGP(
                X_local, y_local, self.likelihood, self.config.kernel_type
            ).to(self.device)
            self._train_gp(self.local_gp, self.likelihood, X_local, y_local,
                          epochs=self.config.gp_retrain_epochs)
        else:
            self.local_gp = SimpleGP(
                input_dim=self.config.input_dim,
                noise_variance=self.config.noise_variance,
                device=str(self.device)
            )
            self.local_gp.fit(X_local, y_local)
    
    def step(self, objective_fn: Callable[[torch.Tensor], float],
             verbose: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Perform one iteration of BALLET-ICI.
        
        Args:
            objective_fn: Black-box objective function
            verbose: Print progress
            
        Returns:
            x_next: Selected point
            y_next: Observed value
        """
        start_time = time.time()
        self.iteration += 1
        
        # Select next point
        x_next = self._select_next_point()
        
        # Evaluate objective
        if isinstance(x_next, torch.Tensor):
            x_np = x_next.cpu().numpy()
        else:
            x_np = x_next
        
        y_next = objective_fn(x_np)
        
        # Convert to tensors
        x_next_tensor = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        y_next_tensor = torch.tensor([y_next], dtype=torch.float32, device=self.device)
        
        # Update data
        self.X = torch.cat([self.X, x_next_tensor.unsqueeze(0)], dim=0)
        self.y = torch.cat([self.y, y_next_tensor], dim=0)
        
        # Update best
        if y_next > self.best_y:
            self.best_y = y_next
            self.best_x = x_next_tensor.clone()
        
        # Update global GP
        if HAS_GPYTORCH:
            self.global_gp = GPyTorchGP(
                self.X, self.y, self.likelihood, self.config.kernel_type
            ).to(self.device)
            self._train_gp(self.global_gp, self.likelihood, self.X, self.y,
                          epochs=self.config.gp_retrain_epochs)
        else:
            self.global_gp.fit(self.X, self.y)
        
        # Update ROI
        self.roi.update(self.X.cpu(), self.y.cpu(),
                       threshold_percentile=self.config.level_set_threshold)
        
        # Update local GP
        self._update_local_gp()
        
        # Record history
        elapsed = time.time() - start_time
        roi_size = (self.roi.roi_upper - self.roi.roi_lower).prod().item()
        
        self.history["x"].append(x_np.tolist() if hasattr(x_np, 'tolist') else list(x_np))
        self.history["y"].append(float(y_next))
        self.history["best_y"].append(float(self.best_y))
        self.history["time"].append(float(elapsed))
        self.history["roi_size"].append(float(roi_size))
        
        if verbose:
            print(f"Iter {self.iteration}: y={y_next:.4f}, best_y={self.best_y:.4f}, "
                  f"ROI_size={roi_size:.4f}, time={elapsed:.2f}s")
        
        return x_next_tensor, y_next
    
    def optimize(self, objective_fn: Callable[[torch.Tensor], float],
                 n_iterations: int,
                 verbose: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Run full optimization loop.
        
        Args:
            objective_fn: Black-box objective function
            n_iterations: Number of iterations
            verbose: Print progress
            
        Returns:
            best_x: Best found input
            best_y: Best found output
        """
        print(f"\nStarting BALLET-ICI optimization for {n_iterations} iterations...")
        print(f"Using local GP: {self.config.use_local_gp}")
        print("-" * 50)
        
        for t in range(n_iterations):
            self.step(objective_fn, verbose=verbose)
        
        print("-" * 50)
        print(f"Optimization complete!")
        print(f"Best x: {self.best_x.cpu().numpy()}")
        print(f"Best y: {self.best_y:.4f}")
        
        return self.best_x, self.best_y
    
    def get_results(self) -> Dict:
        """Get optimization results and statistics."""
        return {
            "best_x": self.best_x.cpu().numpy() if self.best_x is not None else None,
            "best_y": self.best_y,
            "X": self.X.cpu().numpy() if self.X is not None else None,
            "y": self.y.cpu().numpy() if self.y is not None else None,
            "history": self.history,
            "n_iterations": self.iteration,
            "final_roi": {
                "lower": self.roi.roi_lower.numpy().tolist(),
                "upper": self.roi.roi_upper.numpy().tolist()
            }
        }


def create_ballet_ici(input_dim: int, bounds: Tuple[float, float] = (0, 1),
                      **kwargs) -> BALLETICI:
    """
    Factory function to create BALLET-ICI optimizer.
    
    Args:
        input_dim: Dimension of input space
        bounds: Input bounds (min, max)
        **kwargs: Additional configuration options
        
    Returns:
        BALLETICI optimizer instance
    """
    config = BALLETICIConfig(
        input_dim=input_dim,
        bounds=bounds,
        **kwargs
    )
    return BALLETICI(config)


# ============ Test Code ============

if __name__ == "__main__":
    # Test BALLET-ICI on a simple 2D function
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define test function (negated Branin)
    def branin(x):
        """Branin function scaled to [0,1]^2."""
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
        return -result  # Negate for maximization
    
    print("Testing BALLET-ICI on Branin function...")
    print("="*60)
    
    # Create optimizer
    config = BALLETICIConfig(
        input_dim=2,
        bounds=(0, 1),
        beta=2.0,
        num_grid_points=50,
        gp_train_epochs=50,
        gp_retrain_epochs=20,
        device="cpu"
    )
    
    optimizer = BALLETICI(config)
    
    # Generate initial samples
    n_init = 5
    X_init = torch.rand(n_init, 2)
    y_init = torch.tensor([branin(x.numpy()) for x in X_init])
    
    print(f"Initial best y: {y_init.max().item():.4f}")
    
    # Initialize
    optimizer.initialize(X_init, y_init)
    
    # Run optimization
    best_x, best_y = optimizer.optimize(
        objective_fn=branin,
        n_iterations=10,
        verbose=True
    )
    
    # Print results
    results = optimizer.get_results()
    print(f"\nFinal Results:")
    print(f"Best x: {results['best_x']}")
    print(f"Best y: {results['best_y']:.4f}")
    print(f"True optimum: ~-0.398")