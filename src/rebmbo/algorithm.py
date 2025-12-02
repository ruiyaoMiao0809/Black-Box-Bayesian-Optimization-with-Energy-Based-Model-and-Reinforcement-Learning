"""
REBMBO: Reinforced Energy-Based Model for Bayesian Optimization

Main algorithm that integrates:
- Module A: Gaussian Process for local modeling
- Module B: Energy-Based Model for global exploration  
- Module C: PPO for multi-step planning

Algorithm 1 from the paper.

FIXED:
1. PPO now initialized with action_low/action_high for bounded actions
2. Removed redundant clamp in _select_next_point (PPO handles bounds via tanh)
3. Uses state_encoder.get_state_dim() for correct state dimension
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, Optional, Literal, Dict, List
from dataclasses import dataclass
import time

from .gp_module import GPModule
from .ebm_module import EBMModule, EBMUCBAcquisition
from .ppo_module import PPOModule, StateEncoder


@dataclass
class REBMBOConfig:
    """Configuration for REBMBO algorithm."""
    # Problem settings
    input_dim: int = 2
    bounds: Tuple[float, float] = (0.0, 1.0)
    
    # GP settings
    gp_variant: Literal["classic", "sparse", "deep"] = "classic"
    gp_num_inducing: int = 50
    gp_hidden_dims: list = None  # For deep GP
    gp_latent_dim: int = 32
    gp_train_epochs: int = 100
    gp_retrain_epochs: int = 50
    
    # EBM settings (IMPROVED defaults)
    ebm_hidden_dims: list = None
    ebm_mcmc_steps: int = 30        # Increased from 20 for better MCMC samples
    ebm_mcmc_step_size: float = 0.05  # Decreased for stability
    ebm_num_negative_samples: int = 128  # Increased from 100
    ebm_train_epochs: int = 100
    ebm_retrain_epochs: int = 50
    ebm_temperature: float = 1.0    # Temperature for value-weighted training
    
    # PPO settings (OPTIMIZED defaults)
    ppo_hidden_dims: list = None
    ppo_lr_actor: float = 3e-4
    ppo_lr_critic: float = 1e-3
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    ppo_mini_batch_size: int = 32   # DECREASED: 64 → 32 for more frequent updates
    ppo_entropy_coef: float = 0.05  # INCREASED: 0.01 → 0.05 for more exploration
    ppo_value_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    
    # Acquisition function settings (OPTIMIZED defaults)
    beta: float = 2.0   # UCB exploration parameter
    gamma: float = 0.3  # DECREASED: 0.5 → 0.3 (reduce EBM weight)
    lambda_energy: float = 0.2  # DECREASED: 0.3 → 0.2 (less energy penalty)
    
    # State encoding
    num_grid_points: int = 50
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.gp_hidden_dims is None:
            self.gp_hidden_dims = [64, 64]
        if self.ebm_hidden_dims is None:
            self.ebm_hidden_dims = [128, 128, 64]
        if self.ppo_hidden_dims is None:
            self.ppo_hidden_dims = [256, 256]


class LandscapeAwareRegret:
    """
    Landscape-Aware Regret (LAR) metric:
    
    R^{LAR}_t = [f(x*) - f(x_t)] + α[E_θ(x*) - E_θ(x_t)]
    
    - First term: Standard regret (local suboptimality)
    - Second term: Energy regret (global exploration efficiency)
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.regrets = []
        self.function_regrets = []
        self.energy_regrets = []
    
    def compute(self, f_star: float, f_t: float,
                e_star: float, e_t: float) -> float:
        """Compute LAR at iteration t."""
        func_regret = f_star - f_t
        energy_regret = e_star - e_t
        
        lar = func_regret + self.alpha * energy_regret
        
        self.regrets.append(lar)
        self.function_regrets.append(func_regret)
        self.energy_regrets.append(energy_regret)
        
        return lar
    
    def get_cumulative(self) -> float:
        """Get cumulative LAR."""
        return sum(self.regrets)
    
    def get_statistics(self) -> Dict:
        """Get LAR statistics."""
        return {
            "final_lar": self.regrets[-1] if self.regrets else 0,
            "cumulative_lar": sum(self.regrets),
            "mean_lar": np.mean(self.regrets) if self.regrets else 0,
            "std_lar": np.std(self.regrets) if self.regrets else 0,
            "function_regrets": self.function_regrets,
            "energy_regrets": self.energy_regrets
        }


class REBMBO:
    """
    Reinforced Energy-Based Model for Bayesian Optimization.
    
    Combines:
    1. GP surrogate for local uncertainty estimation
    2. EBM for global structure exploration
    3. PPO for multi-step lookahead planning
    
    The algorithm iteratively:
    1. Updates GP with new observations
    2. Retrains EBM to capture global energy landscape
    3. Uses PPO to select next query point based on state (GP + EBM)
    4. Evaluates expensive black-box function
    5. Updates all modules with new data
    """
    
    def __init__(self, config: REBMBOConfig):
        self.config = config
        self.device = config.device
        
        # Initialize modules
        self._init_modules()
        
        # Data storage
        self.X = None  # Observed inputs
        self.y = None  # Observed outputs
        
        # Best observed values
        self.best_x = None
        self.best_y = float('-inf')
        
        # Tracking
        self.iteration = 0
        self.history = {
            "x": [],
            "y": [],
            "best_y": [],
            "time": [],
            "ppo_stats": []
        }
        
        # Regret metric
        self.lar_metric = LandscapeAwareRegret(alpha=0.1)
    
    def _init_modules(self):
        """Initialize GP, EBM, and PPO modules."""
        config = self.config
        
        # Module A: Gaussian Process
        self.gp = GPModule(
            variant=config.gp_variant,
            input_dim=config.input_dim,
            num_inducing=config.gp_num_inducing,
            hidden_dims=config.gp_hidden_dims,
            latent_dim=config.gp_latent_dim,
            device=config.device
        )
        
        # Module B: Energy-Based Model
        # IMPROVED: Uses value-weighted training (higher y → lower energy)
        self.ebm = EBMModule(
            input_dim=config.input_dim,
            hidden_dims=config.ebm_hidden_dims,
            mcmc_steps=config.ebm_mcmc_steps,
            mcmc_step_size=config.ebm_mcmc_step_size,
            num_negative_samples=config.ebm_num_negative_samples,
            bounds=config.bounds,
            temperature=config.ebm_temperature,  # Controls softmax sharpness for weighting
            device=config.device
        )
        
        # EBM-UCB acquisition function
        self.acquisition = EBMUCBAcquisition(
            beta=config.beta,
            gamma=config.gamma
        )
        
        # State encoder for PPO
        self.state_encoder = StateEncoder(
            input_dim=config.input_dim,
            num_grid_points=config.num_grid_points,
            bounds=config.bounds,
            device=config.device
        )
        
        # Module C: PPO
        # FIXED: Use state_encoder.get_state_dim() for correct dimension
        state_dim = self.state_encoder.get_state_dim()
        
        # FIXED: Pass action bounds to PPO for tanh squashing
        # OPTIMIZED: Added entropy_coef for more exploration
        self.ppo = PPOModule(
            state_dim=state_dim,
            action_dim=config.input_dim,
            hidden_dims=config.ppo_hidden_dims,
            lr_actor=config.ppo_lr_actor,
            lr_critic=config.ppo_lr_critic,
            gamma=config.ppo_gamma,
            gae_lambda=config.ppo_gae_lambda,
            clip_epsilon=config.ppo_clip_epsilon,
            entropy_coef=config.ppo_entropy_coef,      # ADDED: for exploration
            value_coef=config.ppo_value_coef,          # ADDED
            max_grad_norm=config.ppo_max_grad_norm,    # ADDED
            ppo_epochs=config.ppo_epochs,
            mini_batch_size=config.ppo_mini_batch_size,
            lambda_energy=config.lambda_energy,
            action_low=config.bounds[0],
            action_high=config.bounds[1],
            device=config.device
        )
    
    def initialize(self, X_init: torch.Tensor, y_init: torch.Tensor):
        """
        Initialize REBMBO with initial dataset D_0.
        
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
        
        # Step 1: Train GP on D_0
        print("Initializing GP...")
        self.gp.initialize(self.X, self.y)
        self.gp.train(num_epochs=self.config.gp_train_epochs)
        
        # Step 2: Train EBM on D_0
        # CRITICAL: Pass y values for value-weighted training
        # This makes EBM assign lower energy to regions with higher y values
        print("Initializing EBM...")
        self.ebm.train(
            self.X,
            y=self.y,  # IMPORTANT: Pass y for value-weighted training
            num_epochs=self.config.ebm_train_epochs,
            verbose=False
        )
        
        print(f"Initialization complete. Best initial y: {self.best_y:.4f}")
    
    def _get_state(self) -> torch.Tensor:
        """Get current RL state from GP and EBM."""
        return self.state_encoder.encode(self.gp, self.ebm)
    
    def _select_next_point(self, use_ppo: bool = True) -> torch.Tensor:
        """
        Select next query point.
        
        If use_ppo=True, uses PPO policy.
        Otherwise, falls back to EBM-UCB optimization.
        
        FIXED: No longer needs to clamp action - PPO now outputs bounded
        actions via tanh squashing in the ActorNetwork.
        """
        if use_ppo:
            state = self._get_state()
            action, log_prob, value = self.ppo.select_action(state)
            
            # FIXED: PPO now outputs bounded actions via tanh squashing
            # No need to clamp anymore - actions are already in [bounds[0], bounds[1]]
            
            # Store for PPO update
            self._current_state = state
            self._current_log_prob = log_prob
            self._current_value = value
            
            return action
        else:
            # Fallback: optimize EBM-UCB directly
            bounds = torch.tensor([
                [self.config.bounds[0]] * self.config.input_dim,
                [self.config.bounds[1]] * self.config.input_dim
            ], device=self.device)
            
            return self.acquisition.optimize(
                self.gp, self.ebm, bounds
            )
    
    def step(self, objective_fn: Callable[[torch.Tensor], float],
             use_ppo: bool = True, verbose: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Perform one iteration of REBMBO.
        
        Args:
            objective_fn: Black-box objective function f(x)
            use_ppo: Whether to use PPO for action selection
            verbose: Print iteration info
        
        Returns:
            x_t: Query point
            y_t: Function value
        """
        start_time = time.time()
        self.iteration += 1
        
        # Step 1: Update GP with current data (Module A)
        # Already done in previous iteration or initialization
        
        # Step 2: Retrain EBM (Module B)
        # CRITICAL: Pass y values for value-weighted training
        # This makes EBM learn which regions have high function values
        if self.iteration > 1:
            self.ebm.train(
                self.X,
                y=self.y,  # IMPORTANT: Pass y for value-weighted training
                num_epochs=self.config.ebm_retrain_epochs,
                verbose=False
            )
        
        # Step 3: Get state and select action (Module C)
        x_t = self._select_next_point(use_ppo=use_ppo)
        
        # Step 4: Evaluate black-box function
        with torch.no_grad():
            if x_t.dim() == 1:
                x_t_np = x_t.cpu().numpy()
            else:
                x_t_np = x_t.squeeze(0).cpu().numpy()
            y_t = objective_fn(x_t_np)
        
        y_t_tensor = torch.tensor([y_t], dtype=torch.float32, device=self.device)
        x_t_tensor = x_t.unsqueeze(0) if x_t.dim() == 1 else x_t
        
        # Step 5: Compute reward and update PPO
        if use_ppo:
            energy_t = self.ebm.get_energy(x_t_tensor).item()
            reward = self.ppo.compute_reward(y_t, energy_t)
            
            done = False  # In BO, we don't have a terminal state
            
            self.ppo.store_transition(
                self._current_state,
                x_t,
                reward,
                self._current_log_prob,
                self._current_value,
                done
            )
            
            # Update PPO periodically (every few steps for efficiency)
            ppo_stats = {}
            if len(self.ppo.buffer) >= self.config.ppo_mini_batch_size:
                ppo_stats = self.ppo.update()
        else:
            ppo_stats = {}
        
        # Step 6: Update dataset
        self.X = torch.cat([self.X, x_t_tensor], dim=0)
        self.y = torch.cat([self.y, y_t_tensor], dim=0)
        
        # Update GP with new data
        self.gp.update(x_t_tensor, y_t_tensor,
                       retrain_epochs=self.config.gp_retrain_epochs)
        
        # Update best observed
        if y_t > self.best_y:
            self.best_y = y_t
            self.best_x = x_t.clone()
        
        # Record history
        elapsed_time = time.time() - start_time
        self.history["x"].append(x_t.cpu().numpy())
        self.history["y"].append(y_t)
        self.history["best_y"].append(self.best_y)
        self.history["time"].append(elapsed_time)
        self.history["ppo_stats"].append(ppo_stats)
        
        if verbose:
            print(f"Iter {self.iteration}: x={x_t.cpu().numpy()}, y={y_t:.4f}, "
                  f"best_y={self.best_y:.4f}, time={elapsed_time:.2f}s")
        
        return x_t, y_t
    
    def optimize(self, objective_fn: Callable[[torch.Tensor], float],
                 n_iterations: int,
                 use_ppo: bool = True,
                 verbose: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Run full optimization loop.
        
        Args:
            objective_fn: Black-box objective function
            n_iterations: Number of iterations (budget T)
            use_ppo: Whether to use PPO
            verbose: Print progress
        
        Returns:
            best_x: Best found input
            best_y: Best found output
        """
        print(f"\nStarting REBMBO optimization for {n_iterations} iterations...")
        print(f"GP variant: {self.config.gp_variant}")
        print(f"Using PPO: {use_ppo}")
        print("-" * 50)
        
        for t in range(n_iterations):
            self.step(objective_fn, use_ppo=use_ppo, verbose=verbose)
        
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
            "lar_stats": self.lar_metric.get_statistics(),
            "n_iterations": self.iteration
        }
    
    def save(self, path: str):
        """Save model state."""
        torch.save({
            "config": self.config,
            "X": self.X,
            "y": self.y,
            "best_x": self.best_x,
            "best_y": self.best_y,
            "iteration": self.iteration,
            "history": self.history
        }, path)
        
        # Save PPO separately
        self.ppo.save(path.replace(".pt", "_ppo.pt"))
    
    @classmethod
    def load(cls, path: str) -> 'REBMBO':
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        
        rebmbo = cls(checkpoint["config"])
        rebmbo.X = checkpoint["X"]
        rebmbo.y = checkpoint["y"]
        rebmbo.best_x = checkpoint["best_x"]
        rebmbo.best_y = checkpoint["best_y"]
        rebmbo.iteration = checkpoint["iteration"]
        rebmbo.history = checkpoint["history"]
        
        # Reinitialize modules with loaded data
        if rebmbo.X is not None:
            rebmbo.gp.initialize(rebmbo.X, rebmbo.y)
            rebmbo.gp.train()
            rebmbo.ebm.train(rebmbo.X)
        
        rebmbo.ppo.load(path.replace(".pt", "_ppo.pt"))
        
        return rebmbo


# ============ Convenience functions for different variants ============

def create_rebmbo_c(input_dim: int, bounds: Tuple[float, float] = (0, 1),
                    **kwargs) -> REBMBO:
    """Create REBMBO with Classic GP."""
    config = REBMBOConfig(
        input_dim=input_dim,
        bounds=bounds,
        gp_variant="classic",
        **kwargs
    )
    return REBMBO(config)


def create_rebmbo_s(input_dim: int, bounds: Tuple[float, float] = (0, 1),
                    num_inducing: int = 50, **kwargs) -> REBMBO:
    """Create REBMBO with Sparse GP."""
    config = REBMBOConfig(
        input_dim=input_dim,
        bounds=bounds,
        gp_variant="sparse",
        gp_num_inducing=num_inducing,
        **kwargs
    )
    return REBMBO(config)


def create_rebmbo_d(input_dim: int, bounds: Tuple[float, float] = (0, 1),
                    hidden_dims: list = [64, 64], latent_dim: int = 32,
                    **kwargs) -> REBMBO:
    """Create REBMBO with Deep GP."""
    config = REBMBOConfig(
        input_dim=input_dim,
        bounds=bounds,
        gp_variant="deep",
        gp_hidden_dims=hidden_dims,
        gp_latent_dim=latent_dim,
        **kwargs
    )
    return REBMBO(config)


if __name__ == "__main__":
    # Test REBMBO on a simple synthetic function
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define test function (Branin-Hoo)
    def branin(x):
        """Branin-Hoo function (2D), scaled to [0,1]^2."""
        # Scale from [0,1] to original domain
        x1 = x[0] * 15 - 5  # [0,1] -> [-5, 10]
        x2 = x[1] * 15      # [0,1] -> [0, 15]
        
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        
        result = a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
        return -result  # Negate for maximization
    
    print("Testing REBMBO on Branin function...")
    print("="*60)
    
    # Create REBMBO instance
    config = REBMBOConfig(
        input_dim=2,
        bounds=(0, 1),
        gp_variant="classic",
        gp_train_epochs=50,
        gp_retrain_epochs=20,
        ebm_train_epochs=50,
        ebm_retrain_epochs=20,
        num_grid_points=25,
        device="cpu"  # Use CPU for testing
    )
    
    rebmbo = REBMBO(config)
    
    # Generate initial random samples
    n_init = 5
    X_init = torch.rand(n_init, 2)
    y_init = torch.tensor([branin(x.numpy()) for x in X_init])
    
    print(f"Initial best y: {y_init.max():.4f}")
    
    # Initialize
    rebmbo.initialize(X_init, y_init)
    
    # Run optimization
    best_x, best_y = rebmbo.optimize(
        objective_fn=branin,
        n_iterations=10,
        use_ppo=True,
        verbose=True
    )
    
    # Print results
    results = rebmbo.get_results()
    print(f"\nFinal Results:")
    print(f"Best x: {results['best_x']}")
    print(f"Best y: {results['best_y']:.4f}")
    print(f"True optimum: ~-0.398 at multiple locations")