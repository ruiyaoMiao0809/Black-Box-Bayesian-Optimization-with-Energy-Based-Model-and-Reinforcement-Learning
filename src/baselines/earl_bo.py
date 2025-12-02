"""
EARL-BO: Efficient Adaptive Reinforcement Learning for Bayesian Optimization

Implementation based on:
    "EARL-BO: Reinforcement Learning for Multi-Step Lookahead, 
     High-Dimensional Bayesian Optimization"
    Cheon et al., 2024 (arXiv:2411.00171)

Key characteristics (from REBMBO paper Section 2.2):
    - RL-based multi-step BO method
    - Heavily dependent on local GP precision
    - No EBM global guidance signals
    - Uses a learned policy for multi-step acquisition

EARL-BO formulates BO as an MDP and uses RL to learn an acquisition policy,
but unlike REBMBO, it does not incorporate global energy-based exploration signals.

This implementation provides a baseline for comparison with REBMBO in Table 1 and Table 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple, Optional, Dict, List
from dataclasses import dataclass
import time

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


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EARLBOConfig:
    """Configuration for EARL-BO algorithm."""
    
    # Problem definition
    input_dim: int = 2
    bounds: Tuple[float, float] = (0.0, 1.0)
    
    # GP configuration
    gp_kernel: str = "matern"  # 'rbf', 'matern', or 'mixed'
    gp_noise_var: float = 1e-4
    gp_train_epochs: int = 100
    gp_retrain_epochs: int = 50
    gp_lr: float = 0.1
    
    # RL Policy configuration  
    policy_hidden_dims: List[int] = None  # Default: [256, 256]
    policy_lr: float = 3e-4
    policy_gamma: float = 0.99  # Discount factor
    policy_gae_lambda: float = 0.95  # GAE lambda
    policy_clip_epsilon: float = 0.2  # PPO clip
    policy_entropy_coef: float = 0.01  # Entropy bonus
    policy_value_coef: float = 0.5
    policy_epochs: int = 10
    policy_mini_batch_size: int = 32
    policy_buffer_size: int = 64
    
    # UCB acquisition parameters
    beta: float = 2.0  # UCB exploration weight
    
    # State representation
    num_grid_points: int = 50  # Grid points for state encoding
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.policy_hidden_dims is None:
            self.policy_hidden_dims = [256, 256]


# =============================================================================
# Gaussian Process Module (Local Surrogate)
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
        
        # Learnable kernel parameters
        self.log_lengthscale = nn.Parameter(torch.zeros(input_dim))
        self.log_variance = nn.Parameter(torch.tensor(0.0))
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        
    def rbf_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)
        
        # Scale inputs
        X1_scaled = X1 / lengthscale
        X2_scaled = X2 / lengthscale
        
        # Compute squared distances
        dist_sq = torch.cdist(X1_scaled, X2_scaled, p=2).pow(2)
        
        return variance * torch.exp(-0.5 * dist_sq)
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, num_epochs: int = 100, lr: float = 0.1):
        """Fit GP to data."""
        self.X_train = X.clone()
        self.y_train = y.clone()
        
        # Normalize targets
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-6
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
                # Numerical issues, skip this step
                continue
        
        # Precompute for prediction
        K = self.rbf_kernel(self.X_train, self.X_train)
        K_noise = K + self.noise_var * torch.eye(len(self.X_train), device=X.device)
        
        try:
            self.K_inv = torch.linalg.inv(K_noise)
            self.alpha = self.K_inv @ ((self.y_train - self.y_mean) / self.y_std)
        except RuntimeError:
            self.K_inv = torch.eye(len(self.X_train), device=X.device) / self.noise_var
            self.alpha = ((self.y_train - self.y_mean) / self.y_std) / self.noise_var
    
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


if HAS_GPYTORCH:
    class GPyTorchModel(ExactGP):
        """GP model using GPyTorch for better numerical stability."""
        
        def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, 
                     likelihood: GaussianLikelihood, kernel_type: str = "matern"):
            super().__init__(train_x, train_y, likelihood)
            
            self.mean_module = ConstantMean()
            
            if kernel_type == "rbf":
                base_kernel = RBFKernel(ard_num_dims=train_x.shape[-1])
            elif kernel_type == "matern":
                base_kernel = MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
            elif kernel_type == "mixed":
                # Mixture of RBF and Matérn (as in REBMBO paper)
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


class EARL_GP:
    """
    Gaussian Process module for EARL-BO.
    Provides local surrogate modeling with posterior mean and variance.
    """
    
    def __init__(self, config: EARLBOConfig):
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
            y_normalized = (self.y_train - self.y_mean) / self.y_std
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


# =============================================================================
# RL Policy Network (Actor-Critic)
# =============================================================================

class PolicyNetwork(nn.Module):
    """
    Actor-Critic policy network for EARL-BO.
    
    The actor outputs continuous actions in the bounded domain.
    The critic estimates state values for advantage computation.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int], 
                 action_low: float = 0.0, action_high: float = 1.0):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0
        self.action_center = (action_high + action_low) / 2.0
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head (outputs mean of action distribution)
        self.actor_mean = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
        )
        
        # Learnable log std
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (outputs state value)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_mean: Mean of action distribution
            action_log_std: Log std of action distribution
            value: State value estimate
        """
        features = self.shared(state)
        
        # Actor output (bounded via tanh)
        raw_mean = self.actor_mean(features)
        action_mean = torch.tanh(raw_mean) * self.action_scale + self.action_center
        
        # Value estimate
        value = self.critic(features).squeeze(-1)
        
        return action_mean, self.actor_log_std.expand_as(action_mean), value
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_mean, action_log_std, value = self.forward(state)
        
        if deterministic:
            return action_mean, torch.zeros(1, device=state.device), value
        
        # Sample from Gaussian
        std = torch.exp(action_log_std).clamp(min=1e-6, max=1.0)
        dist = torch.distributions.Normal(action_mean, std)
        
        # Sample and compute log prob
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        
        # Bound action
        action = raw_action.clamp(self.action_low, self.action_high)
        
        return action, log_prob, value


class StateEncoder:
    """
    Encodes GP posterior into a fixed-size state vector.
    
    Unlike REBMBO, EARL-BO only uses GP information (no EBM energy).
    """
    
    def __init__(self, config: EARLBOConfig):
        self.config = config
        self.device = config.device
        self.num_grid_points = config.num_grid_points
        
        # Create grid for state encoding
        self._create_grid()
    
    def _create_grid(self):
        """Create evaluation grid for state encoding."""
        # 1D grid points
        grid_1d = torch.linspace(
            self.config.bounds[0], 
            self.config.bounds[1], 
            self.num_grid_points
        )
        
        # For multi-dimensional, use random points to avoid explosion
        if self.config.input_dim > 1:
            self.grid = torch.rand(
                self.num_grid_points, 
                self.config.input_dim,
                device=self.device
            ) * (self.config.bounds[1] - self.config.bounds[0]) + self.config.bounds[0]
        else:
            self.grid = grid_1d.unsqueeze(1).to(self.device)
    
    def get_state_dim(self) -> int:
        """Get dimension of encoded state."""
        # mu + sigma at grid points
        return self.num_grid_points * 2
    
    def encode(self, gp: EARL_GP) -> torch.Tensor:
        """
        Encode GP posterior into state vector.
        
        State = [mu(grid), sigma(grid)]
        """
        with torch.no_grad():
            mu, var = gp.predict(self.grid)
            sigma = torch.sqrt(var + 1e-6)
            
            # Normalize
            mu_normalized = (mu - mu.mean()) / (mu.std() + 1e-6)
            sigma_normalized = sigma / (sigma.max() + 1e-6)
            
            state = torch.cat([mu_normalized, sigma_normalized])
        
        return state


# =============================================================================
# UCB Acquisition Function
# =============================================================================

class UCBAcquisition:
    """
    Standard UCB acquisition function (without EBM term).
    
    alpha_UCB(x) = mu(x) + beta * sigma(x)
    """
    
    def __init__(self, beta: float = 2.0):
        self.beta = beta
    
    def __call__(self, x: torch.Tensor, gp: EARL_GP) -> torch.Tensor:
        """Compute UCB value at x."""
        mu, var = gp.predict(x)
        sigma = torch.sqrt(var + 1e-6)
        return mu + self.beta * sigma
    
    def optimize(self, gp: EARL_GP, bounds: torch.Tensor, 
                 n_candidates: int = 1000) -> torch.Tensor:
        """
        Find the argmax of UCB using random sampling.
        
        Returns:
            x_best: Best point found
        """
        device = bounds.device
        
        # Generate random candidates
        candidates = torch.rand(n_candidates, bounds.shape[1], device=device)
        candidates = candidates * (bounds[1] - bounds[0]) + bounds[0]
        
        # Evaluate UCB
        ucb_values = self(candidates, gp)
        
        # Return best
        best_idx = ucb_values.argmax()
        return candidates[best_idx]


# =============================================================================
# Experience Buffer for RL
# =============================================================================

class ExperienceBuffer:
    """Simple experience buffer for PPO-style updates."""
    
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.clear()
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float,
            value: torch.Tensor, log_prob: torch.Tensor, done: bool = False):
        """Add experience to buffer."""
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.log_probs.append(log_prob.detach())
        self.dones.append(done)
    
    def __len__(self):
        return len(self.states)
    
    def get_batch(self) -> Tuple:
        """Get all experiences as tensors."""
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack(self.values) if self.values[0].dim() > 0 else torch.tensor(self.values)
        log_probs = torch.stack(self.log_probs) if self.log_probs[0].dim() > 0 else torch.tensor(self.log_probs)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        return states, actions, rewards, values, log_probs, dones


# =============================================================================
# EARL-BO Main Algorithm
# =============================================================================

class EARL_BO:
    """
    EARL-BO: Efficient Adaptive Reinforcement Learning for Bayesian Optimization
    
    Key differences from REBMBO:
    1. No Energy-Based Model (EBM) for global exploration
    2. Uses only GP posterior for state representation
    3. Standard UCB without energy term
    
    This serves as a baseline to demonstrate the value of REBMBO's 
    EBM-enhanced global exploration.
    """
    
    def __init__(self, config: EARLBOConfig):
        self.config = config
        self.device = config.device
        
        # GP surrogate (Module A equivalent)
        self.gp = EARL_GP(config)
        
        # State encoder (GP-only)
        self.state_encoder = StateEncoder(config)
        
        # Policy network
        state_dim = self.state_encoder.get_state_dim()
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            action_dim=config.input_dim,
            hidden_dims=config.policy_hidden_dims,
            action_low=config.bounds[0],
            action_high=config.bounds[1]
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.policy_lr
        )
        
        # UCB acquisition (fallback)
        self.ucb = UCBAcquisition(beta=config.beta)
        
        # Experience buffer
        self.buffer = ExperienceBuffer(config.policy_buffer_size)
        
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
            'time': []
        }
        
        self.iteration = 0
    
    def initialize(self, X_init: torch.Tensor, y_init: torch.Tensor):
        """
        Initialize EARL-BO with initial data.
        
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
        
        # Initialize and train GP
        print("Initializing GP...")
        self.gp.initialize(self.X, self.y)
        self.gp.train(num_epochs=self.config.gp_train_epochs)
        
        print(f"Initialization complete. Best initial y: {self.best_y:.4f}")
    
    def _get_state(self) -> torch.Tensor:
        """Get current RL state from GP."""
        return self.state_encoder.encode(self.gp)
    
    def _select_next_point(self, use_rl: bool = True) -> torch.Tensor:
        """
        Select next query point.
        
        Args:
            use_rl: If True, use RL policy; otherwise use UCB optimization
        """
        if use_rl:
            state = self._get_state()
            action, log_prob, value = self.policy.get_action(state.unsqueeze(0))
            
            # Store for policy update
            self._current_state = state
            self._current_log_prob = log_prob.squeeze()
            self._current_value = value.squeeze()
            
            return action.squeeze(0)
        else:
            # Fallback: optimize UCB directly
            bounds = torch.tensor([
                [self.config.bounds[0]] * self.config.input_dim,
                [self.config.bounds[1]] * self.config.input_dim
            ], device=self.device)
            
            return self.ucb.optimize(self.gp, bounds)
    
    def _compute_reward(self, y: float) -> float:
        """
        Compute reward for RL.
        
        Unlike REBMBO, EARL-BO uses only function value (no energy term).
        Reward = normalized function value
        """
        # Normalize reward
        if len(self.history['y']) > 0:
            y_min = min(self.history['y'])
            y_max = max(self.history['y'])
            if y_max - y_min > 1e-6:
                reward = (y - y_min) / (y_max - y_min)
            else:
                reward = 0.5
        else:
            reward = 0.5
        
        # Bonus for improvement
        if y > self.best_y:
            reward += 0.5
        
        return float(reward)
    
    def _update_policy(self):
        """Update policy using PPO."""
        if len(self.buffer) < self.config.policy_mini_batch_size:
            return {}
        
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get_batch()
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        # Compute returns and advantages using GAE
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.policy_gamma * next_value - values[t]
            gae = delta + self.config.policy_gamma * self.config.policy_gae_lambda * gae
            
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        
        returns = torch.tensor(returns, device=self.device)
        advantages = torch.tensor(advantages, device=self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        total_loss = 0
        for _ in range(self.config.policy_epochs):
            # Get current policy outputs
            action_mean, action_log_std, new_values = self.policy(states)
            
            # Compute new log probs
            std = torch.exp(action_log_std).clamp(min=1e-6, max=1.0)
            dist = torch.distributions.Normal(action_mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.policy_clip_epsilon, 
                               1 + self.config.policy_clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = (policy_loss + 
                   self.config.policy_value_coef * value_loss - 
                   self.config.policy_entropy_coef * entropy)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear buffer after update
        self.buffer.clear()
        
        return {
            'policy_loss': total_loss / self.config.policy_epochs,
            'entropy': entropy.item()
        }
    
    def step(self, objective_fn: Callable, use_rl: bool = True, 
             verbose: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Perform one iteration of EARL-BO.
        
        Args:
            objective_fn: Black-box objective function
            use_rl: Whether to use RL policy
            verbose: Print progress
        
        Returns:
            x_t: Query point
            y_t: Function value
        """
        start_time = time.time()
        self.iteration += 1
        
        # Select next point
        x_t = self._select_next_point(use_rl=use_rl)
        
        # Ensure bounds
        x_t = x_t.clamp(self.config.bounds[0], self.config.bounds[1])
        
        # Evaluate objective
        x_t_np = x_t.detach().cpu().numpy()
        y_t = objective_fn(x_t_np)
        
        # Convert for storage
        x_t_tensor = x_t.unsqueeze(0) if x_t.dim() == 1 else x_t
        y_t_tensor = torch.tensor([y_t], device=self.device)
        
        # Update GP
        self.gp.update(x_t_tensor, y_t_tensor, 
                      retrain_epochs=self.config.gp_retrain_epochs)
        
        # Update best
        if y_t > self.best_y:
            self.best_y = y_t
            self.best_x = x_t.clone()
        
        # Compute reward and store experience
        if use_rl and hasattr(self, '_current_state'):
            reward = self._compute_reward(y_t)
            self.buffer.add(
                self._current_state,
                x_t,
                reward,
                self._current_value,
                self._current_log_prob
            )
            
            # Update policy if buffer is full
            if len(self.buffer) >= self.config.policy_buffer_size:
                self._update_policy()
        
        # Record history
        elapsed_time = time.time() - start_time
        self.history['x'].append(x_t.detach().cpu().numpy())
        self.history['y'].append(y_t)
        self.history['best_y'].append(self.best_y)
        self.history['time'].append(elapsed_time)
        
        if verbose:
            print(f"Iter {self.iteration}: x={x_t_np}, y={y_t:.4f}, "
                  f"best_y={self.best_y:.4f}, time={elapsed_time:.2f}s")
        
        return x_t, y_t
    
    def optimize(self, objective_fn: Callable, n_iterations: int,
                 use_rl: bool = True, verbose: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Run full optimization loop.
        
        Args:
            objective_fn: Black-box objective function
            n_iterations: Number of iterations
            use_rl: Whether to use RL policy
            verbose: Print progress
        
        Returns:
            best_x: Best found input
            best_y: Best found output
        """
        print(f"\nStarting EARL-BO optimization for {n_iterations} iterations...")
        print(f"Using RL: {use_rl}")
        print("-" * 50)
        
        for t in range(n_iterations):
            self.step(objective_fn, use_rl=use_rl, verbose=verbose)
        
        print("-" * 50)
        print(f"Optimization complete!")
        print(f"Best x: {self.best_x.cpu().numpy()}")
        print(f"Best y: {self.best_y:.4f}")
        
        return self.best_x, self.best_y
    
    def get_results(self) -> Dict:
        """Get optimization results and statistics."""
        return {
            'best_x': self.best_x.cpu().numpy() if self.best_x is not None else None,
            'best_y': self.best_y,
            'X': self.X.cpu().numpy() if self.X is not None else None,
            'y': self.y.cpu().numpy() if self.y is not None else None,
            'history': self.history,
            'n_iterations': self.iteration
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_earl_bo(input_dim: int, bounds: Tuple[float, float] = (0, 1),
                   **kwargs) -> EARL_BO:
    """Create EARL-BO instance with given configuration."""
    config = EARLBOConfig(
        input_dim=input_dim,
        bounds=bounds,
        **kwargs
    )
    return EARL_BO(config)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test EARL-BO on a simple function
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define test function (Ackley 2D)
    def ackley_2d(x):
        """Ackley function (2D), scaled to [0,1]^2."""
        # Scale from [0,1] to [-5, 5]
        x_scaled = x * 10 - 5
        
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x_scaled)
        
        sum1 = np.sum(x_scaled ** 2)
        sum2 = np.sum(np.cos(c * x_scaled))
        
        result = -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e
        return -result  # Negate for maximization
    
    print("Testing EARL-BO on Ackley 2D function...")
    print("=" * 60)
    
    # Create EARL-BO instance
    config = EARLBOConfig(
        input_dim=2,
        bounds=(0.0, 1.0),
        gp_train_epochs=50,
        gp_retrain_epochs=20,
        num_grid_points=30,
        policy_buffer_size=16,
        device="cpu"
    )
    
    earl_bo = EARL_BO(config)
    
    # Generate initial samples
    n_init = 5
    X_init = torch.rand(n_init, 2)
    y_init = torch.tensor([ackley_2d(x.numpy()) for x in X_init])
    
    print(f"Initial best y: {y_init.max():.4f}")
    
    # Initialize
    earl_bo.initialize(X_init, y_init)
    
    # Run optimization
    best_x, best_y = earl_bo.optimize(
        objective_fn=ackley_2d,
        n_iterations=10,
        use_rl=True,
        verbose=True
    )
    
    # Print results
    results = earl_bo.get_results()
    print(f"\nFinal Results:")
    print(f"Best x: {results['best_x']}")
    print(f"Best y: {results['best_y']:.4f}")
    print(f"True optimum: 0.0 at [0.5, 0.5] (scaled)")