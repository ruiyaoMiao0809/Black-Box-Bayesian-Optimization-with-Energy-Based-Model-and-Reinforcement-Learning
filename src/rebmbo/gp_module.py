"""
GP Module for REBMBO
Implements three GP variants: Classic, Sparse, and Deep GP
"""

import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from typing import Tuple, Optional, Literal
import numpy as np


class MixtureKernel(gpytorch.kernels.Kernel):
    """
    RBF + Matérn mixture kernel as described in the paper.
    k(x,x') = σ²[w_RBF * k_RBF(x,x') + w_Matern * k_Matern(x,x')]
    
    RBF captures smooth global trends, Matérn captures rough local variations.
    """
    def __init__(self, input_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.rbf = RBFKernel(ard_num_dims=input_dim)
        self.matern = MaternKernel(nu=2.5, ard_num_dims=input_dim)
        
        # Learnable mixture weights (initialized to 0.5 each)
        self.register_parameter(
            "raw_w_rbf", 
            nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        )
    
    @property
    def w_rbf(self):
        return torch.sigmoid(self.raw_w_rbf)
    
    @property
    def w_matern(self):
        return 1 - self.w_rbf
    
    def forward(self, x1, x2, diag=False, **params):
        rbf_term = self.rbf(x1, x2, diag=diag, **params)
        matern_term = self.matern(x1, x2, diag=diag, **params)
        
        if diag:
            return self.w_rbf * rbf_term + self.w_matern * matern_term
        else:
            return self.w_rbf * rbf_term + self.w_matern * matern_term


# ============ REBMBO-C: Classic GP ============
class ClassicGP(ExactGP):
    """
    Classic GP with exact O(n³) inference.
    Uses mixture kernel (RBF + Matérn) for local structure modeling.
    """
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, 
                 likelihood: GaussianLikelihood, input_dim: int):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MixtureKernel(input_dim))
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# ============ REBMBO-S: Sparse GP ============
class SparseGP(ApproximateGP):
    """
    Sparse GP with inducing points for O(nm²) complexity.
    Suitable for larger datasets and higher dimensions.
    """
    def __init__(self, inducing_points: torch.Tensor, input_dim: int):
        # Variational distribution q(u) for inducing points
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MixtureKernel(input_dim))
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# ============ REBMBO-D: Deep GP ============
class DeepFeatureExtractor(nn.Module):
    """
    Deep network Θ that maps inputs x to latent features φ_GP(x).
    """
    def __init__(self, input_dim: int, hidden_dims: list = [64, 64], 
                 latent_dim: int = 32):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepGP(ExactGP):
    """
    Deep Kernel GP for non-stationary, multi-scale problems.
    Uses a neural network to map inputs to latent space before GP.
    
    μ(x; D, Θ) = m^T φ_GP(x) + η(x)
    σ²(x; D, Θ) = φ_GP(x)^T K^{-1} φ_GP(x) + 1/β
    """
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: GaussianLikelihood, input_dim: int,
                 hidden_dims: list = [64, 64], latent_dim: int = 32):
        super().__init__(train_x, train_y, likelihood)
        
        self.feature_extractor = DeepFeatureExtractor(
            input_dim, hidden_dims, latent_dim
        )
        self.mean_module = ConstantMean()
        # GP kernel operates in latent space
        self.covar_module = ScaleKernel(MixtureKernel(latent_dim))
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        # Extract deep features
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return MultivariateNormal(mean_x, covar_x)


# ============ Unified GP Wrapper ============
class GPModule:
    """
    Unified interface for all GP variants in REBMBO.
    Provides μ(x) and σ(x) for the RL state.
    """
    def __init__(self, 
                 variant: Literal["classic", "sparse", "deep"] = "classic",
                 input_dim: int = 2,
                 num_inducing: int = 50,
                 hidden_dims: list = [64, 64],
                 latent_dim: int = 32,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.variant = variant
        self.input_dim = input_dim
        self.num_inducing = num_inducing
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.device = device
        
        self.model = None
        self.likelihood = None
        self.mll = None
        self.is_trained = False
    
    def initialize(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """Initialize GP model with initial data."""
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        
        self.likelihood = GaussianLikelihood().to(self.device)
        
        if self.variant == "classic":
            self.model = ClassicGP(
                train_x, train_y, self.likelihood, self.input_dim
            ).to(self.device)
            self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
            
        elif self.variant == "sparse":
            # Select inducing points via k-means or uniformly
            indices = torch.randperm(len(train_x))[:min(self.num_inducing, len(train_x))]
            inducing_points = train_x[indices].clone()
            
            self.model = SparseGP(inducing_points, self.input_dim).to(self.device)
            self.mll = VariationalELBO(self.likelihood, self.model, 
                                        num_data=len(train_y))
            
        elif self.variant == "deep":
            self.model = DeepGP(
                train_x, train_y, self.likelihood, self.input_dim,
                self.hidden_dims, self.latent_dim
            ).to(self.device)
            self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        self.train_x = train_x
        self.train_y = train_y
    
    def train(self, num_epochs: int = 100, lr: float = 0.1):
        """Train GP hyperparameters via marginal likelihood optimization."""
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            optimizer.step()
        
        self.is_trained = True
    
    def update(self, new_x: torch.Tensor, new_y: torch.Tensor, 
               retrain_epochs: int = 50):
        """Update GP with new observation and retrain."""
        new_x = new_x.to(self.device)
        new_y = new_y.to(self.device)
        
        # Concatenate new data
        self.train_x = torch.cat([self.train_x, new_x.unsqueeze(0) 
                                   if new_x.dim() == 1 else new_x], dim=0)
        self.train_y = torch.cat([self.train_y, new_y.unsqueeze(0) 
                                   if new_y.dim() == 0 else new_y], dim=0)
        
        # Re-initialize model with updated data
        if self.variant == "classic" or self.variant == "deep":
            self.model.set_train_data(self.train_x, self.train_y, strict=False)
        else:
            # For sparse GP, re-initialize
            self.initialize(self.train_x, self.train_y)
        
        # Retrain
        self.train(num_epochs=retrain_epochs)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get posterior mean μ(x) and std σ(x).
        These are used as part of the RL state.
        """
        self.model.eval()
        self.likelihood.eval()
        
        x = x.to(self.device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(x)
            mean = posterior.mean
            std = posterior.stddev
        
        return mean, std
    
    def get_posterior_at_points(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get μ(x) and σ(x) for constructing RL state."""
        return self.predict(x)


if __name__ == "__main__":
    # Test the GP module
    torch.manual_seed(42)
    
    # Generate synthetic data
    train_x = torch.rand(20, 2)
    train_y = torch.sin(train_x[:, 0] * 3) + torch.cos(train_x[:, 1] * 3) + 0.1 * torch.randn(20)
    
    for variant in ["classic", "sparse", "deep"]:
        print(f"\n{'='*50}")
        print(f"Testing {variant.upper()} GP")
        print('='*50)
        
        gp = GPModule(variant=variant, input_dim=2)
        gp.initialize(train_x, train_y)
        gp.train(num_epochs=50)
        
        # Test prediction
        test_x = torch.rand(5, 2)
        mean, std = gp.predict(test_x)
        
        print(f"Test points shape: {test_x.shape}")
        print(f"Mean predictions: {mean}")
        print(f"Std predictions: {std}")
        
        # Test update
        new_x = torch.rand(1, 2)
        new_y = torch.tensor([0.5])
        gp.update(new_x, new_y)
        
        mean_after, std_after = gp.predict(test_x)
        print(f"Mean after update: {mean_after}")