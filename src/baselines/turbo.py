"""
TuRBO: Scalable Global Optimization via Local Bayesian Optimization

Implementation based on:
    "Scalable Global Optimization via Local Bayesian Optimization"
    Eriksson et al., NeurIPS 2019

Key characteristics (from REBMBO paper Section 2.2):
    - Trust region local optimization with adaptive bounds
    - Trust region expands on success, contracts on failure
    - Lacks far-reaching jumps (global exploration)
    - Specializes in local trust-region expansions
    
TuRBO maintains a local trust region centered at the best observed point,
and uses Thompson Sampling or UCB within this region. The trust region
size adapts based on optimization progress.

This implementation provides TuRBO-1 (single trust region) as a baseline
for comparison with REBMBO in Table 1 and Table 2.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, Optional, Dict, List
from dataclasses import dataclass, field
import time
import math

# Try to import GPyTorch for GP modeling
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
    print("Warning: GPyTorch not available. Using basic GP implementation.")

# Try to import botorch for sobol sampling
try:
    from botorch.utils.sampling import draw_sobol_samples
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TuRBOConfig:
    """Configuration for TuRBO algorithm."""
    
    # Problem definition
    input_dim: int = 2
    bounds: Tuple[float, float] = (0.0, 1.0)
    
    # Trust region parameters (from original TuRBO paper)
    length_init: float = 0.8      # Initial trust region length (as fraction of domain)
    length_min: float = 0.5 ** 7  # Minimum trust region length before restart
    length_max: float = 1.6       # Maximum trust region length
    
    # Success/failure thresholds for trust region adaptation
    n_trust_updates: int = 1      # Evaluations before considering trust region update
    success_tolerance: int = 3    # Consecutive successes to expand
    failure_tolerance: int = 5    # Consecutive failures to shrink (dim-dependent default)
    
    # GP configuration
    gp_kernel: str = "matern"     # 'rbf', 'matern', or 'mixed'
    gp_noise_var: float = 1e-4
    gp_train_epochs: int = 100
    gp_retrain_epochs: int = 50
    gp_lr: float = 0.1
    
    # Acquisition configuration
    acquisition_type: str = "ts"  # 'ts' (Thompson Sampling) or 'ucb'
    beta: float = 2.0             # UCB exploration weight
    n_candidates: int = 5000      # Number of candidates for acquisition optimization
    
    # Batch size (TuRBO can do batch acquisition)
    batch_size: int = 1
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        # Default failure tolerance is dimension-dependent
        if self.failure_tolerance is None:
            self.failure_tolerance = max(5, self.input_dim)


# =============================================================================
# Simple GP Implementation (fallback)
# =============================================================================

class SimpleGPModel(nn.Module):
    """
    Simple GP implementation for environments without GPyTorch.
    Uses RBF kernel with learned lengthscale and variance.
    """
    
    def __init__(self, input_dim: int, noise_var: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.noise_var = noise_var
        
        # Learnable kernel parameters (ARD)
        self.log_lengthscale = nn.Parameter(torch.zeros(input_dim))
        self.log_variance = nn.Parameter(torch.tensor(0.0))
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        
        # Normalization
        self.y_mean = 0.0
        self.y_std = 1.0
        
    def rbf_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix with ARD."""
        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)
        
        # Scale inputs
        X1_scaled = X1 / lengthscale
        X2_scaled = X2 / lengthscale
        
        # Compute squared distances
        dist_sq = torch.cdist(X1_scaled, X2_scaled, p=2).pow(2)
        
        return variance * torch.exp(-0.5 * dist_sq)
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, num_epochs: int = 100, lr: float = 0.1):
        """Fit GP to data with hyperparameter optimization."""
        self.X_train = X.clone()
        self.y_train = y.clone()
        
        # Normalize targets
        self.y_mean = y.mean().item()
        self.y_std = y.std().item() + 1e-6
        y_normalized = (y - self.y_mean) / self.y_std
        
        # Optimize kernel hyperparameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for _ in range(num_epochs):
            optimizer.zero_grad()
            
            K = self.rbf_kernel(X, X)
            K_noise = K + self.noise_var * torch.eye(len(X), device=X.device)
            
            # Negative log marginal likelihood
            try:
                L = torch.linalg.cholesky(K_noise)
                alpha = torch.cholesky_solve(y_normalized.unsqueeze(1), L).squeeze()
                nll = 0.5 * y_normalized @ alpha + torch.log(L.diag()).sum()
                
                nll.backward()
                optimizer.step()
            except RuntimeError:
                continue
        
        # Precompute for prediction
        K = self.rbf_kernel(self.X_train, self.X_train)
        K_noise = K + self.noise_var * torch.eye(len(self.X_train), device=X.device)
        
        try:
            self.K_inv = torch.linalg.inv(K_noise)
            y_norm = (self.y_train - self.y_mean) / self.y_std
            self.alpha = self.K_inv @ y_norm
        except RuntimeError:
            self.K_inv = torch.eye(len(self.X_train), device=X.device) / self.noise_var
            y_norm = (self.y_train - self.y_mean) / self.y_std
            self.alpha = y_norm / self.noise_var
    
    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance at X."""
        if self.X_train is None:
            return torch.zeros(len(X)), torch.ones(len(X))
        
        K_star = self.rbf_kernel(X, self.X_train)
        K_star_star = self.rbf_kernel(X, X)
        
        # Mean
        mu = K_star @ self.alpha
        mu = mu * self.y_std + self.y_mean
        
        # Variance
        v = K_star @ self.K_inv @ K_star.T
        var = torch.diag(K_star_star - v).clamp(min=1e-6)
        var = var * (self.y_std ** 2)
        
        return mu, var
    
    def sample_posterior(self, X: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample from posterior distribution (for Thompson Sampling)."""
        mu, var = self.predict(X)
        std = torch.sqrt(var)
        
        # Sample from independent Gaussians (approximate)
        samples = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            n_samples, len(X), device=X.device
        )
        return samples


if HAS_GPYTORCH:
    class GPyTorchModel(ExactGP):
        """GP model using GPyTorch for better numerical stability."""
        
        def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, 
                     likelihood: GaussianLikelihood, kernel_type: str = "matern"):
            super().__init__(train_x, train_y, likelihood)
            
            self.mean_module = ConstantMean()
            
            # ARD kernel
            if kernel_type == "rbf":
                base_kernel = RBFKernel(ard_num_dims=train_x.shape[-1])
            elif kernel_type == "matern":
                base_kernel = MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
            elif kernel_type == "mixed":
                rbf = RBFKernel(ard_num_dims=train_x.shape[-1])
                matern = MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
                base_kernel = rbf + matern
            else:
                base_kernel = MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
            
            self.covar_module = ScaleKernel(base_kernel)
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)


# =============================================================================
# TuRBO GP Wrapper
# =============================================================================

class TuRBO_GP:
    """
    Gaussian Process module for TuRBO.
    Supports both GPyTorch and simple fallback implementations.
    """
    
    def __init__(self, config: TuRBOConfig):
        self.config = config
        self.device = config.device
        
        self.model = None
        self.likelihood = None
        self.X_train = None
        self.y_train = None
        
        # Normalization stats
        self.y_mean = 0.0
        self.y_std = 1.0
    
    def initialize(self, X: torch.Tensor, y: torch.Tensor):
        """Initialize GP with data."""
        self.X_train = X.to(self.device)
        self.y_train = y.to(self.device)
        
        # Normalize targets
        self.y_mean = y.mean().item()
        self.y_std = y.std().item() + 1e-6
        y_normalized = (self.y_train - self.y_mean) / self.y_std
        
        if HAS_GPYTORCH:
            self.likelihood = GaussianLikelihood().to(self.device)
            self.model = GPyTorchModel(
                self.X_train, y_normalized, self.likelihood, 
                kernel_type=self.config.gp_kernel
            ).to(self.device)
        else:
            self.model = SimpleGPModel(
                self.config.input_dim, 
                self.config.gp_noise_var
            ).to(self.device)
    
    def train(self, num_epochs: int = 100):
        """Train GP hyperparameters."""
        if self.model is None:
            return
        
        if HAS_GPYTORCH:
            self.model.train()
            self.likelihood.train()
            
            optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=self.config.gp_lr)
            
            mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
            
            y_normalized = (self.y_train - self.y_mean) / self.y_std
            
            for _ in range(num_epochs):
                optimizer.zero_grad()
                output = self.model(self.X_train)
                loss = -mll(output, y_normalized)
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            self.likelihood.eval()
        else:
            self.model.fit(self.X_train, self.y_train, num_epochs, self.config.gp_lr)
    
    def update(self, X_new: torch.Tensor, y_new: torch.Tensor, retrain_epochs: int = 50):
        """Update GP with new observations."""
        X_new = X_new.to(self.device)
        y_new = y_new.to(self.device)
        
        # Handle dimensions
        if X_new.dim() == 1:
            X_new = X_new.unsqueeze(0)
        if y_new.dim() == 0:
            y_new = y_new.unsqueeze(0)
        
        # Concatenate data
        if self.X_train is not None:
            self.X_train = torch.cat([self.X_train, X_new], dim=0)
            self.y_train = torch.cat([self.y_train, y_new], dim=0)
        else:
            self.X_train = X_new
            self.y_train = y_new
        
        # Update normalization
        self.y_mean = self.y_train.mean().item()
        self.y_std = self.y_train.std().item() + 1e-6
        
        # Reinitialize and retrain
        y_normalized = (self.y_train - self.y_mean) / self.y_std
        
        if HAS_GPYTORCH:
            self.likelihood = GaussianLikelihood().to(self.device)
            self.model = GPyTorchModel(
                self.X_train, y_normalized, self.likelihood,
                kernel_type=self.config.gp_kernel
            ).to(self.device)
            self.train(retrain_epochs)
        else:
            self.model = SimpleGPModel(
                self.config.input_dim,
                self.config.gp_noise_var
            ).to(self.device)
            self.model.fit(self.X_train, self.y_train, retrain_epochs, self.config.gp_lr)
    
    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and variance."""
        X = X.to(self.device)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        if self.model is None:
            return torch.zeros(len(X), device=self.device), torch.ones(len(X), device=self.device)
        
        if HAS_GPYTORCH:
            self.model.eval()
            self.likelihood.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(X))
                mu = observed_pred.mean * self.y_std + self.y_mean
                var = observed_pred.variance * (self.y_std ** 2)
            
            return mu, var
        else:
            with torch.no_grad():
                mu, var = self.model.predict(X)
            return mu, var
    
    def sample_posterior(self, X: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample from posterior (for Thompson Sampling)."""
        X = X.to(self.device)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        if HAS_GPYTORCH:
            self.model.eval()
            self.likelihood.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = self.model(X)
                samples = posterior.sample(torch.Size([n_samples]))
                # Unnormalize
                samples = samples * self.y_std + self.y_mean
            
            return samples
        else:
            with torch.no_grad():
                samples = self.model.sample_posterior(X, n_samples)
            return samples


# =============================================================================
# Trust Region Manager
# =============================================================================

class TrustRegion:
    """
    Manages the trust region for TuRBO.
    
    The trust region is a hyperrectangle centered at the best point,
    with side lengths determined by the current trust region length
    and the lengthscales of the GP kernel.
    """
    
    def __init__(self, config: TuRBOConfig):
        self.config = config
        self.dim = config.input_dim
        
        # Trust region state
        self.length = config.length_init
        self.length_min = config.length_min
        self.length_max = config.length_max
        
        # Success/failure tracking
        self.success_count = 0
        self.failure_count = 0
        self.success_tolerance = config.success_tolerance
        self.failure_tolerance = config.failure_tolerance
        
        # Center (best point so far)
        self.center = None
        self.best_value = float('-inf')
        
        # Restart counter
        self.n_restarts = 0
    
    def update_center(self, x: torch.Tensor, y: float):
        """Update trust region center if y is better."""
        if y > self.best_value:
            self.center = x.clone()
            self.best_value = y
            return True
        return False
    
    def update_state(self, improved: bool):
        """
        Update trust region based on success/failure.
        
        - Consecutive successes -> expand trust region
        - Consecutive failures -> shrink trust region
        """
        if improved:
            self.success_count += 1
            self.failure_count = 0
        else:
            self.failure_count += 1
            self.success_count = 0
        
        # Expand on success
        if self.success_count >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_count = 0
        
        # Shrink on failure
        if self.failure_count >= self.failure_tolerance:
            self.length = self.length / 2.0
            self.failure_count = 0
    
    def needs_restart(self) -> bool:
        """Check if trust region is too small and needs restart."""
        return self.length < self.length_min
    
    def restart(self, x_new: torch.Tensor, y_new: float):
        """Restart trust region at a new center."""
        self.length = self.config.length_init
        self.center = x_new.clone()
        self.best_value = y_new
        self.success_count = 0
        self.failure_count = 0
        self.n_restarts += 1
    
    def get_bounds(self, global_bounds: Tuple[float, float], 
                   lengthscales: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get trust region bounds.
        
        Returns:
            bounds: [2, dim] tensor with [lower_bounds, upper_bounds]
        """
        if self.center is None:
            # Return global bounds if no center
            lb = torch.full((self.dim,), global_bounds[0])
            ub = torch.full((self.dim,), global_bounds[1])
            return torch.stack([lb, ub])
        
        # Use lengthscales to scale trust region per dimension
        if lengthscales is not None:
            # Normalize lengthscales
            weights = lengthscales / lengthscales.mean()
            weights = weights.clamp(min=0.5, max=2.0)
        else:
            weights = torch.ones(self.dim, device=self.center.device)
        
        # Compute trust region bounds
        half_lengths = self.length * weights / 2.0
        
        lb = (self.center - half_lengths).clamp(min=global_bounds[0])
        ub = (self.center + half_lengths).clamp(max=global_bounds[1])
        
        return torch.stack([lb, ub])


# =============================================================================
# Acquisition Functions
# =============================================================================

class ThompsonSampling:
    """Thompson Sampling acquisition within trust region."""
    
    def __init__(self, config: TuRBOConfig):
        self.config = config
    
    def select(self, gp: TuRBO_GP, trust_region: TrustRegion,
               n_candidates: int = 5000) -> torch.Tensor:
        """
        Select next point using Thompson Sampling.
        
        1. Generate candidates within trust region
        2. Sample from GP posterior
        3. Return candidate with highest sample value
        """
        device = gp.X_train.device if gp.X_train is not None else self.config.device
        
        # Get trust region bounds
        tr_bounds = trust_region.get_bounds(self.config.bounds)
        tr_bounds = tr_bounds.to(device)
        
        # Generate candidates within trust region
        if HAS_BOTORCH:
            # Use Sobol sampling for better coverage
            candidates = draw_sobol_samples(
                bounds=tr_bounds,
                n=n_candidates,
                q=1
            ).squeeze(1)
        else:
            # Random sampling
            candidates = torch.rand(n_candidates, self.config.input_dim, device=device)
            candidates = candidates * (tr_bounds[1] - tr_bounds[0]) + tr_bounds[0]
        
        # Sample from posterior
        with torch.no_grad():
            samples = gp.sample_posterior(candidates, n_samples=1).squeeze(0)
        
        # Select best
        best_idx = samples.argmax()
        return candidates[best_idx]


class UCBAcquisition:
    """UCB acquisition within trust region."""
    
    def __init__(self, config: TuRBOConfig):
        self.config = config
        self.beta = config.beta
    
    def select(self, gp: TuRBO_GP, trust_region: TrustRegion,
               n_candidates: int = 5000) -> torch.Tensor:
        """
        Select next point using UCB.
        
        UCB(x) = mu(x) + beta * sigma(x)
        """
        device = gp.X_train.device if gp.X_train is not None else self.config.device
        
        # Get trust region bounds
        tr_bounds = trust_region.get_bounds(self.config.bounds)
        tr_bounds = tr_bounds.to(device)
        
        # Generate candidates within trust region
        candidates = torch.rand(n_candidates, self.config.input_dim, device=device)
        candidates = candidates * (tr_bounds[1] - tr_bounds[0]) + tr_bounds[0]
        
        # Compute UCB
        with torch.no_grad():
            mu, var = gp.predict(candidates)
            sigma = torch.sqrt(var + 1e-6)
            ucb = mu + self.beta * sigma
        
        # Select best
        best_idx = ucb.argmax()
        return candidates[best_idx]


# =============================================================================
# TuRBO Main Algorithm
# =============================================================================

class TuRBO:
    """
    TuRBO: Trust Region Bayesian Optimization
    
    Key characteristics:
    1. Maintains a trust region centered at the best point
    2. Trust region expands on success, contracts on failure
    3. Uses Thompson Sampling or UCB within trust region
    4. Restarts when trust region becomes too small
    
    Unlike REBMBO:
    - No global Energy-Based Model for exploration
    - No RL-based multi-step planning
    - Relies purely on local GP modeling within trust region
    """
    
    def __init__(self, config: TuRBOConfig):
        self.config = config
        self.device = config.device
        
        # GP surrogate
        self.gp = TuRBO_GP(config)
        
        # Trust region
        self.trust_region = TrustRegion(config)
        
        # Acquisition function
        if config.acquisition_type == "ts":
            self.acquisition = ThompsonSampling(config)
        else:
            self.acquisition = UCBAcquisition(config)
        
        # Data storage
        self.X = None
        self.y = None
        self.best_x = None
        self.best_y = float('-inf')
        
        # History
        self.history = {
            'x': [],
            'y': [],
            'best_y': [],
            'time': [],
            'tr_length': [],
            'n_restarts': []
        }
        
        self.iteration = 0
    
    def initialize(self, X_init: torch.Tensor, y_init: torch.Tensor):
        """
        Initialize TuRBO with initial data.
        
        Args:
            X_init: Initial input points [n_init, input_dim]
            y_init: Initial function values [n_init]
        """
        self.X = X_init.to(self.device)
        self.y = y_init.to(self.device)
        
        # Find best initial point
        best_idx = self.y.argmax()
        self.best_x = self.X[best_idx].clone()
        self.best_y = self.y[best_idx].item()
        
        # Initialize trust region at best point
        self.trust_region.center = self.best_x.clone()
        self.trust_region.best_value = self.best_y
        
        # Initialize and train GP
        print("Initializing GP...")
        self.gp.initialize(self.X, self.y)
        self.gp.train(num_epochs=self.config.gp_train_epochs)
        
        print(f"Initialization complete. Best initial y: {self.best_y:.4f}")
    
    def _select_next_point(self) -> torch.Tensor:
        """Select next query point using acquisition within trust region."""
        return self.acquisition.select(
            self.gp, 
            self.trust_region,
            n_candidates=self.config.n_candidates
        )
    
    def step(self, objective_fn: Callable, verbose: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Perform one iteration of TuRBO.
        
        Args:
            objective_fn: Black-box objective function
            verbose: Print progress
        
        Returns:
            x_t: Query point
            y_t: Function value
        """
        start_time = time.time()
        self.iteration += 1
        
        # Check if restart needed
        if self.trust_region.needs_restart():
            # Restart at a random point
            x_restart = torch.rand(self.config.input_dim, device=self.device)
            x_restart = x_restart * (self.config.bounds[1] - self.config.bounds[0]) + self.config.bounds[0]
            
            # Evaluate restart point
            x_restart_np = x_restart.cpu().numpy()
            y_restart = objective_fn(x_restart_np)
            
            self.trust_region.restart(x_restart, y_restart)
            
            # Update GP
            self.gp.update(
                x_restart.unsqueeze(0), 
                torch.tensor([y_restart], device=self.device),
                retrain_epochs=self.config.gp_retrain_epochs
            )
            
            if verbose:
                print(f"  -> Trust region restart #{self.trust_region.n_restarts}")
        
        # Select next point within trust region
        x_t = self._select_next_point()
        
        # Ensure bounds
        x_t = x_t.clamp(self.config.bounds[0], self.config.bounds[1])
        
        # Evaluate objective
        x_t_np = x_t.detach().cpu().numpy()
        y_t = objective_fn(x_t_np)
        
        # Check if improved
        improved = y_t > self.best_y
        
        # Update trust region
        self.trust_region.update_center(x_t, y_t)
        self.trust_region.update_state(improved)
        
        # Update best
        if improved:
            self.best_y = y_t
            self.best_x = x_t.clone()
        
        # Update GP
        x_t_tensor = x_t.unsqueeze(0) if x_t.dim() == 1 else x_t
        y_t_tensor = torch.tensor([y_t], device=self.device)
        self.gp.update(x_t_tensor, y_t_tensor, 
                      retrain_epochs=self.config.gp_retrain_epochs)
        
        # Record history
        elapsed_time = time.time() - start_time
        self.history['x'].append(x_t.detach().cpu().numpy())
        self.history['y'].append(y_t)
        self.history['best_y'].append(self.best_y)
        self.history['time'].append(elapsed_time)
        self.history['tr_length'].append(self.trust_region.length)
        self.history['n_restarts'].append(self.trust_region.n_restarts)
        
        if verbose:
            print(f"Iter {self.iteration}: x={x_t_np}, y={y_t:.4f}, "
                  f"best_y={self.best_y:.4f}, tr_len={self.trust_region.length:.4f}, "
                  f"time={elapsed_time:.2f}s")
        
        return x_t, y_t
    
    def optimize(self, objective_fn: Callable, n_iterations: int,
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
        print(f"\nStarting TuRBO optimization for {n_iterations} iterations...")
        print(f"Acquisition: {self.config.acquisition_type}")
        print(f"Initial trust region length: {self.trust_region.length:.4f}")
        print("-" * 60)
        
        for t in range(n_iterations):
            self.step(objective_fn, verbose=verbose)
        
        print("-" * 60)
        print(f"Optimization complete!")
        print(f"Best x: {self.best_x.cpu().numpy()}")
        print(f"Best y: {self.best_y:.4f}")
        print(f"Total restarts: {self.trust_region.n_restarts}")
        
        return self.best_x, self.best_y
    
    def get_results(self) -> Dict:
        """Get optimization results and statistics."""
        return {
            'best_x': self.best_x.cpu().numpy() if self.best_x is not None else None,
            'best_y': self.best_y,
            'X': self.X.cpu().numpy() if self.X is not None else None,
            'y': self.y.cpu().numpy() if self.y is not None else None,
            'history': self.history,
            'n_iterations': self.iteration,
            'n_restarts': self.trust_region.n_restarts,
            'final_tr_length': self.trust_region.length
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_turbo(input_dim: int, bounds: Tuple[float, float] = (0, 1),
                 **kwargs) -> TuRBO:
    """Create TuRBO instance with given configuration."""
    config = TuRBOConfig(
        input_dim=input_dim,
        bounds=bounds,
        **kwargs
    )
    return TuRBO(config)


def create_turbo_ts(input_dim: int, bounds: Tuple[float, float] = (0, 1),
                    **kwargs) -> TuRBO:
    """Create TuRBO with Thompson Sampling."""
    return create_turbo(input_dim, bounds, acquisition_type="ts", **kwargs)


def create_turbo_ucb(input_dim: int, bounds: Tuple[float, float] = (0, 1),
                     **kwargs) -> TuRBO:
    """Create TuRBO with UCB acquisition."""
    return create_turbo(input_dim, bounds, acquisition_type="ucb", **kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test TuRBO on a simple function
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define test function (Ackley 2D)
    def ackley_2d(x):
        """Ackley function (2D), scaled to [0,1]^2."""
        # Scale from [0,1] to [-5, 5]
        x_scaled = np.asarray(x) * 10 - 5
        
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x_scaled)
        
        sum1 = np.sum(x_scaled ** 2)
        sum2 = np.sum(np.cos(c * x_scaled))
        
        result = -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e
        return -result  # Negate for maximization
    
    print("Testing TuRBO on Ackley 2D function...")
    print("=" * 60)
    
    # Create TuRBO instance
    config = TuRBOConfig(
        input_dim=2,
        bounds=(0.0, 1.0),
        gp_train_epochs=50,
        gp_retrain_epochs=20,
        acquisition_type="ts",
        n_candidates=2000,
        device="cpu"
    )
    
    turbo = TuRBO(config)
    
    # Generate initial samples
    n_init = 5
    X_init = torch.rand(n_init, 2)
    y_init = torch.tensor([ackley_2d(x.numpy()) for x in X_init])
    
    print(f"Initial best y: {y_init.max():.4f}")
    
    # Initialize
    turbo.initialize(X_init, y_init)
    
    # Run optimization
    best_x, best_y = turbo.optimize(
        objective_fn=ackley_2d,
        n_iterations=15,
        verbose=True
    )
    
    # Print results
    results = turbo.get_results()
    print(f"\nFinal Results:")
    print(f"Best x: {results['best_x']}")
    print(f"Best y: {results['best_y']:.4f}")
    print(f"Total restarts: {results['n_restarts']}")
    print(f"Final TR length: {results['final_tr_length']:.4f}")
    print(f"True optimum: 0.0 at [0.5, 0.5] (scaled)")