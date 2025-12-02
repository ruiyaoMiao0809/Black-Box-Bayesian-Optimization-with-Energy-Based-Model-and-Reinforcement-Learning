"""
PPO Module for REBMBO
Implements Proximal Policy Optimization for multi-step lookahead planning.
Formulates Bayesian Optimization as an MDP solved via RL.

FIXED:
1. StateEncoder now correctly generates exactly num_grid_points points
2. ActorNetwork now uses tanh squashing to constrain actions to [0, 1]
3. Better initialization for bounded action spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, List, Optional
import numpy as np
from collections import deque


class ActorNetwork(nn.Module):
    """
    Policy network π_ϕ(a|s) that outputs action distribution.
    
    Input (state): [μ(x), σ(x), E_θ(x)] - GP posterior + EBM energy
    Output: Mean and std of Gaussian policy over action space
    
    FIXED: Uses tanh squashing to constrain actions to [action_low, action_high]
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [256, 256],
                 log_std_min: float = -5, log_std_max: float = 2,
                 action_low: float = 0.0, action_high: float = 1.0):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.LayerNorm(h_dim)
            ])
            prev_dim = h_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        # Initialize with appropriate weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for bounded action space."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Initialize mean head to output near 0 (which maps to action_bias after tanh)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0)
        
        # Initialize log_std head to output reasonable initial std
        nn.init.constant_(self.log_std_head.weight, 0)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # exp(-1) ≈ 0.37 std
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action distribution parameters.
        
        Returns:
            mean: Action mean in unbounded space [batch, action_dim]
            log_std: Log standard deviation [batch, action_dim]
        """
        # Replace NaN in input state
        state = torch.nan_to_num(state, nan=0.0)
        
        features = self.shared(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Replace NaN and clamp for numerical stability
        mean = torch.nan_to_num(mean, nan=0.0)
        mean = torch.clamp(mean, -10.0, 10.0)  # Prevent extreme values
        
        log_std = torch.nan_to_num(log_std, nan=-1.0)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def _squash_action(self, action: torch.Tensor) -> torch.Tensor:
        """Apply tanh squashing and scale to [action_low, action_high]."""
        return torch.tanh(action) * self.action_scale + self.action_bias
    
    def _unsquash_action(self, squashed_action: torch.Tensor) -> torch.Tensor:
        """Inverse of squash: map from [action_low, action_high] to unbounded."""
        # Clamp to avoid numerical issues at boundaries
        normalized = (squashed_action - self.action_bias) / self.action_scale
        normalized = torch.clamp(normalized, -0.999, 0.999)
        return torch.atanh(normalized)
    
    def _log_prob_correction(self, unbounded_action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability correction for tanh squashing.
        
        log π(a|s) = log μ(u|s) - sum(log(1 - tanh²(u)))
        where a = tanh(u) * scale + bias
        """
        # log(1 - tanh²(x)) = log(sech²(x)) = 2 * log(sech(x))
        # = 2 * (log(2) - x - softplus(-2x))
        correction = 2 * (np.log(2) - unbounded_action - F.softplus(-2 * unbounded_action))
        # Account for scaling
        correction = correction - np.log(self.action_scale)
        return correction.sum(dim=-1)
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: Current state
            deterministic: If True, return mean action
        
        Returns:
            action: Sampled (or mean) action, squashed to [action_low, action_high]
            log_prob: Log probability of action (with squashing correction)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            unbounded_action = mean
            squashed_action = self._squash_action(unbounded_action)
            log_prob = torch.zeros(mean.size(0), device=mean.device)
        else:
            dist = Normal(mean, std)
            unbounded_action = dist.rsample()  # Reparameterized sample
            squashed_action = self._squash_action(unbounded_action)
            
            # Log probability with squashing correction
            log_prob = dist.log_prob(unbounded_action).sum(dim=-1)
            log_prob = log_prob - self._log_prob_correction(unbounded_action)
        
        return squashed_action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, 
                         action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.
        
        Args:
            state: States [batch, state_dim]
            action: Actions in [action_low, action_high] space [batch, action_dim]
        
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of the policy distribution (approximate)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Convert squashed action back to unbounded space
        unbounded_action = self._unsquash_action(action)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(unbounded_action).sum(dim=-1)
        log_prob = log_prob - self._log_prob_correction(unbounded_action)
        
        # Entropy (approximate, ignoring squashing)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Value network V(s) for advantage estimation.
    """
    def __init__(self, state_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.LayerNorm(h_dim)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value V(s)."""
        return self.network(state).squeeze(-1)


class RolloutBuffer:
    """
    Buffer to store rollout data for PPO training.
    """
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def add(self, state: torch.Tensor, action: torch.Tensor,
            reward: float, log_prob: torch.Tensor, 
            value: torch.Tensor, done: bool):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def get_tensors(self, device: str) -> Tuple[torch.Tensor, ...]:
        """Convert buffer to tensors."""
        states = torch.stack(self.states).to(device)
        actions = torch.stack(self.actions).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        log_probs = torch.stack(self.log_probs).to(device)
        values = torch.stack(self.values).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        
        return states, actions, rewards, log_probs, values, dones
    
    def __len__(self):
        return len(self.states)


class PPOModule:
    """
    Proximal Policy Optimization for multi-step planning in REBMBO.
    
    MDP Formulation:
    - State s_t = [μ(x), σ(x), E_θ(x)] - GP posterior + EBM energy
    - Action a_t = proposed query point x ∈ X
    - Reward r_t = f(a_t) - λ·E_θ(a_t)
    
    The PPO agent learns to select sampling points that balance:
    1. Immediate function improvement (f(a_t))
    2. Global exploration (low energy regions)
    
    FIXED: Actor now outputs actions bounded to [0, 1] via tanh squashing
    """
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [256, 256],
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 lambda_energy: float = 0.3,
                 action_low: float = 0.0,
                 action_high: float = 1.0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.lambda_energy = lambda_energy
        self.action_low = action_low
        self.action_high = action_high
        self.device = device
        
        # Networks with bounded action space
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dims,
            action_low=action_low, action_high=action_high
        ).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dims).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Training statistics
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": []
        }
    
    def construct_state(self, gp_mean: torch.Tensor, gp_std: torch.Tensor,
                        ebm_energy: torch.Tensor) -> torch.Tensor:
        """
        Construct RL state from GP posterior and EBM energy.
        
        State s_t = [μ(x), σ(x), E_θ(x)]
        """
        # Normalize each component for stable training
        gp_mean_norm = (gp_mean - gp_mean.mean()) / (gp_mean.std() + 1e-8)
        gp_std_norm = gp_std / (gp_std.max() + 1e-8)
        energy_norm = (ebm_energy - ebm_energy.mean()) / (ebm_energy.std() + 1e-8)
        
        state = torch.cat([gp_mean_norm, gp_std_norm, energy_norm], dim=-1)
        return state
    
    def select_action(self, state: torch.Tensor, 
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action (next query point) given current state.
        
        Args:
            state: Current state [state_dim] or [1, state_dim]
            deterministic: If True, return mean action
        
        Returns:
            action: Selected action [action_dim], already bounded to [action_low, action_high]
            log_prob: Log probability of action
            value: State value estimate
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic)
            value = self.critic(state)
        
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)
    
    def compute_reward(self, function_value: float, energy_value: float) -> float:
        """
        Compute reward for RL agent.
        
        r_t = f(a_t) - λ * E_θ(a_t)
        
        Higher function value and lower energy are rewarded.
        """
        return function_value - self.lambda_energy * energy_value
    
    def store_transition(self, state: torch.Tensor, action: torch.Tensor,
                        reward: float, log_prob: torch.Tensor,
                        value: torch.Tensor, done: bool):
        """Store transition in buffer."""
        self.buffer.add(state, action, reward, log_prob, value, done)
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, next_value: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        if next_value is None:
            next_value = torch.tensor(0.0, device=self.device)
        
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        returns = torch.zeros(T, device=self.device)
        
        gae = 0
        next_val = next_value
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self) -> dict:
        """
        Perform PPO update using collected rollout data.
        """
        if len(self.buffer) < self.mini_batch_size:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "approx_kl": 0}
        
        # Get data from buffer
        states, actions, rewards, old_log_probs, values, dones = \
            self.buffer.get_tensors(self.device)
        
        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0
        
        batch_size = len(states)
        
        for epoch in range(self.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions with current policy
                log_probs, entropy = self.actor.evaluate_actions(
                    batch_states, batch_actions
                )
                values_pred = self.critic(batch_states)
                
                # Policy ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                    1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                        self.value_coef * value_loss + 
                        self.entropy_coef * entropy_loss)
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (ratio.log())).mean()
                    total_kl += approx_kl.item()
                
                num_updates += 1
        
        # Clear buffer after update
        self.buffer.clear()
        
        # Average statistics
        if num_updates > 0:
            stats = {
                "policy_loss": total_policy_loss / num_updates,
                "value_loss": total_value_loss / num_updates,
                "entropy": total_entropy / num_updates,
                "approx_kl": total_kl / num_updates
            }
        else:
            stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "approx_kl": 0}
        
        for k, v in stats.items():
            self.training_stats[k].append(v)
        
        return stats
    
    def save(self, path: str):
        """Save model checkpoints."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])


class StateEncoder:
    """
    Encodes GP and EBM information into a fixed-size state vector.
    
    Strategy: Evaluate GP and EBM at a fixed grid of representative points.
    
    FIXED: Now correctly generates exactly num_grid_points points using
    Sobol sequence for better coverage.
    """
    def __init__(self, input_dim: int, num_grid_points: int = 50,
                 bounds: Tuple[float, float] = (0, 1),
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.input_dim = input_dim
        self.num_grid_points = num_grid_points
        self.bounds = bounds
        self.device = device
        
        # Create fixed grid for state encoding
        self.grid_points = self._create_grid()
        
        # Store actual state dimension for PPO initialization
        self.state_dim = self.num_grid_points * 3  # μ, σ, E at each point
    
    def _create_grid(self) -> torch.Tensor:
        """
        Create a fixed grid of exactly num_grid_points points for evaluation.
        Uses Sobol sequence for quasi-random coverage.
        """
        try:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=self.input_dim, scramble=True)
            grid = sobol.draw(self.num_grid_points)
            # Scale to bounds
            grid = grid * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        except Exception:
            # Fallback to random sampling
            grid = torch.rand(self.num_grid_points, self.input_dim)
            grid = grid * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        
        return grid.to(self.device)
    
    def get_state_dim(self) -> int:
        """Return the actual state dimension."""
        return self.state_dim
    
    def encode(self, gp_module, ebm_module) -> torch.Tensor:
        """
        Encode current GP and EBM state into a vector.
        
        State = [μ(grid), σ(grid), E(grid)] - flattened
        
        Includes NaN protection for numerical stability.
        """
        # Evaluate GP at grid points
        gp_mean, gp_std = gp_module.predict(self.grid_points)
        
        # Evaluate EBM at grid points
        ebm_energy = ebm_module.get_energy(self.grid_points)
        
        # Normalize with NaN protection
        # GP mean normalization
        gp_mean_std = gp_mean.std()
        if gp_mean_std < 1e-8 or torch.isnan(gp_mean_std):
            gp_mean_norm = torch.zeros_like(gp_mean)
        else:
            gp_mean_norm = (gp_mean - gp_mean.mean()) / (gp_mean_std + 1e-8)
        
        # GP std normalization
        gp_std_max = gp_std.max()
        if gp_std_max < 1e-8 or torch.isnan(gp_std_max):
            gp_std_norm = torch.ones_like(gp_std) * 0.5
        else:
            gp_std_norm = gp_std / (gp_std_max + 1e-8)
        
        # Energy normalization
        energy_std = ebm_energy.std()
        if energy_std < 1e-8 or torch.isnan(energy_std):
            energy_norm = torch.zeros_like(ebm_energy)
        else:
            energy_norm = (ebm_energy - ebm_energy.mean()) / (energy_std + 1e-8)
        
        # Replace any remaining NaN with 0
        gp_mean_norm = torch.nan_to_num(gp_mean_norm, nan=0.0)
        gp_std_norm = torch.nan_to_num(gp_std_norm, nan=0.5)
        energy_norm = torch.nan_to_num(energy_norm, nan=0.0)
        
        state = torch.cat([gp_mean_norm, gp_std_norm, energy_norm])
        
        return state


if __name__ == "__main__":
    # Test PPO module with bounded actions
    torch.manual_seed(42)
    
    state_dim = 150  # 50 grid points * 3 (mean, std, energy)
    action_dim = 5   # 5D Ackley
    
    print("Testing PPO Module with Bounded Actions...")
    print("="*60)
    
    ppo = PPOModule(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128],
        action_low=0.0,
        action_high=1.0,
        device="cpu"
    )
    
    # Test action bounds
    print("\n1. Testing action bounds...")
    test_state = torch.randn(state_dim)
    
    all_actions = []
    for _ in range(100):
        action, _, _ = ppo.select_action(test_state, deterministic=False)
        all_actions.append(action)
    
    all_actions = torch.stack(all_actions)
    print(f"   Action shape: {all_actions.shape}")
    print(f"   Action min: {all_actions.min().item():.4f} (should be >= 0)")
    print(f"   Action max: {all_actions.max().item():.4f} (should be <= 1)")
    print(f"   Action mean: {all_actions.mean().item():.4f} (should be ~0.5)")
    
    assert all_actions.min() >= 0, "Actions should be >= 0!"
    assert all_actions.max() <= 1, "Actions should be <= 1!"
    print("   ✓ All actions within bounds!")
    
    # Test deterministic action
    print("\n2. Testing deterministic action...")
    det_action, _, _ = ppo.select_action(test_state, deterministic=True)
    print(f"   Deterministic action: {det_action.numpy()}")
    print(f"   Action in [0, 1]: {(det_action >= 0).all() and (det_action <= 1).all()}")
    
    # Simulate training
    print("\n3. Testing training loop...")
    for episode in range(3):
        state = torch.randn(state_dim)
        total_reward = 0
        
        for step in range(10):
            action, log_prob, value = ppo.select_action(state)
            
            # Simulate reward (Ackley-like: better near center)
            distance_from_center = ((action - 0.5) ** 2).sum().item()
            reward = -distance_from_center  # Reward being close to center
            
            done = (step == 9)
            ppo.store_transition(state, action, reward, log_prob, value, done)
            
            state = torch.randn(state_dim)
            total_reward += reward
        
        print(f"   Episode {episode + 1}: Total reward = {total_reward:.2f}")
    
    # Update policy
    print("\n4. Updating policy...")
    stats = ppo.update()
    print(f"   Update stats: {stats}")
    
    # Test StateEncoder
    print("\n" + "="*60)
    print("Testing StateEncoder...")
    
    encoder = StateEncoder(
        input_dim=5,
        num_grid_points=50,
        bounds=(0, 1),
        device="cpu"
    )
    
    print(f"Grid points shape: {encoder.grid_points.shape}")
    print(f"State dimension: {encoder.get_state_dim()}")
    assert encoder.grid_points.shape == (50, 5), "Grid shape mismatch!"
    assert encoder.get_state_dim() == 150, "State dim mismatch!"
    print("✓ StateEncoder test passed!")
    
    print("\n" + "="*60)
    print("✅ All PPO tests passed!")