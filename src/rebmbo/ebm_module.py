"""
EBM Module for REBMBO (Optimized Version)
Implements Energy-Based Model with short-run MCMC training.
The EBM captures global structure and guides exploration.

OPTIMIZATIONS:
1. Value-weighted training: Good points (high y) get lower energy
2. Improved negative sampling: Better exploration of search space
3. Enhanced energy network with residual connections
4. Adaptive MCMC step size
5. Temperature-based softmax weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, dim: int, use_spectral_norm: bool = True):
        super().__init__()
        linear1 = nn.Linear(dim, dim)
        linear2 = nn.Linear(dim, dim)
        
        if use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
            linear2 = nn.utils.spectral_norm(linear2)
        
        self.block = nn.Sequential(
            linear1,
            nn.LeakyReLU(0.2),
            linear2
        )
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class EnergyNetwork(nn.Module):
    """
    Neural network parameterizing the energy function E_θ(x).
    Lower energy = more promising region for optimization.
    
    IMPROVED: Added residual connections and layer normalization.
    """
    def __init__(self, input_dim: int, hidden_dims: list = [128, 128, 64],
                 use_spectral_norm: bool = True, use_residual: bool = True):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        if use_spectral_norm:
            self.input_proj = nn.utils.spectral_norm(self.input_proj)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                self.hidden_layers.append(
                    ResidualBlock(hidden_dims[i], use_spectral_norm)
                )
            else:
                linear = nn.Linear(hidden_dims[i], hidden_dims[i+1])
                if use_spectral_norm:
                    linear = nn.utils.spectral_norm(linear)
                self.hidden_layers.append(
                    nn.Sequential(linear, nn.LeakyReLU(0.2))
                )
            self.norms.append(nn.LayerNorm(hidden_dims[i+1]))
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)
        if use_spectral_norm:
            self.output = nn.utils.spectral_norm(self.output)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return energy E_θ(x) for each input."""
        h = F.leaky_relu(self.input_proj(x), 0.2)
        
        for layer, norm in zip(self.hidden_layers, self.norms):
            h = layer(h)
            h = norm(h)
        
        return self.output(h).squeeze(-1)


class AdaptiveLangevinSampler:
    """
    Improved Langevin Dynamics MCMC sampler with adaptive step size.
    
    Features:
    - Adaptive step size based on gradient magnitude
    - Momentum for faster mixing
    - Noise annealing for better convergence
    """
    def __init__(self, step_size: float = 0.1, num_steps: int = 20,
                 noise_scale: float = 0.01, momentum: float = 0.9,
                 adapt_step_size: bool = True):
        self.step_size = step_size
        self.num_steps = num_steps
        self.noise_scale = noise_scale
        self.momentum = momentum
        self.adapt_step_size = adapt_step_size
    
    def sample(self, energy_fn: nn.Module, init_samples: torch.Tensor,
               bounds: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        """
        Run short-run MCMC starting from init_samples.
        """
        samples = init_samples.clone().requires_grad_(True)
        velocity = torch.zeros_like(samples)
        
        for step in range(self.num_steps):
            # Compute energy and gradient
            energy = energy_fn(samples)
            grad = torch.autograd.grad(energy.sum(), samples, 
                                       create_graph=False)[0]
            
            # Adaptive step size based on gradient norm
            if self.adapt_step_size:
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                adaptive_step = self.step_size / (1 + grad_norm)
            else:
                adaptive_step = self.step_size
            
            # Noise annealing: reduce noise over steps
            noise_factor = 1.0 - 0.5 * (step / self.num_steps)
            noise = torch.randn_like(samples) * self.noise_scale * noise_factor
            
            # Momentum-based update
            velocity = self.momentum * velocity - adaptive_step * grad
            samples = samples + velocity + np.sqrt(self.step_size) * noise
            
            # Clamp to valid domain
            samples = samples.clamp(bounds[0], bounds[1])
            samples = samples.detach().requires_grad_(True)
        
        return samples.detach()


class EBMModule:
    """
    Energy-Based Model for global exploration in REBMBO.
    
    OPTIMIZATIONS:
    1. Value-weighted training: Points with higher function values get 
       trained to have lower energy
    2. Improved negative sampling: Mix of MCMC, random, and anti-optimal samples
    3. Temperature-controlled softmax weighting
    4. Better regularization
    
    The trained EBM provides:
    - E_θ(x): Energy at point x (lower = more promising)
    - -E_θ(x): Used in EBM-UCB acquisition function
    """
    def __init__(self, input_dim: int, hidden_dims: list = [128, 128, 64],
                 mcmc_steps: int = 20, mcmc_step_size: float = 0.1,
                 num_negative_samples: int = 100,
                 bounds: Tuple[float, float] = (0, 1),
                 temperature: float = 1.0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.input_dim = input_dim
        self.bounds = bounds
        self.device = device
        self.num_negative_samples = num_negative_samples
        self.temperature = temperature
        
        # Improved energy network with residual connections
        self.energy_net = EnergyNetwork(
            input_dim, hidden_dims, 
            use_spectral_norm=True,
            use_residual=True
        ).to(device)
        
        # Adaptive Langevin sampler
        self.sampler = AdaptiveLangevinSampler(
            step_size=mcmc_step_size, 
            num_steps=mcmc_steps,
            adapt_step_size=True
        )
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.energy_net.parameters(), 
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        
        # Buffer for persistent chains
        self.sample_buffer = None
        
        # Store function values for weighted training
        self.y_values = None
    
    def _initialize_buffer(self, batch_size: int):
        """Initialize sample buffer with random samples."""
        self.sample_buffer = torch.rand(
            batch_size, self.input_dim, device=self.device
        ) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
    
    def _compute_weights(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weights based on function values.
        Higher y → higher weight → lower energy after training.
        
        Uses softmax with temperature for smooth weighting.
        """
        # Handle edge cases
        if len(y) <= 1:
            return torch.ones_like(y)
        
        # Normalize y values (handle constant y case)
        y_std = y.std()
        if y_std < 1e-8:
            # All y values are the same, use uniform weights
            return torch.ones_like(y)
        
        y_normalized = (y - y.mean()) / (y_std + 1e-8)
        
        # Softmax weighting: higher y gets higher weight
        weights = F.softmax(y_normalized / self.temperature, dim=0)
        
        # Scale weights to have mean 1
        weights = weights * len(weights)
        
        return weights
    
    def _generate_negative_samples(self, data: torch.Tensor, 
                                    y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate diverse negative samples:
        1. MCMC samples from current energy model (50%)
        2. Random samples from domain (30%)
        3. Anti-optimal samples: far from best points (20%)
        """
        n_mcmc = int(self.num_negative_samples * 0.5)
        n_random = int(self.num_negative_samples * 0.3)
        n_anti = self.num_negative_samples - n_mcmc - n_random
        
        neg_samples = []
        
        # 1. MCMC samples
        if self.sample_buffer is None:
            self._initialize_buffer(n_mcmc)
        
        # Ensure buffer has correct size
        if self.sample_buffer.size(0) < n_mcmc:
            self._initialize_buffer(n_mcmc)
        
        mcmc_init = self.sample_buffer[:n_mcmc].clone()
        mcmc_samples = self.sampler.sample(
            self.energy_net, mcmc_init, self.bounds
        )
        neg_samples.append(mcmc_samples)
        
        # Update buffer
        if self.sample_buffer.size(0) >= n_mcmc:
            self.sample_buffer[:n_mcmc] = mcmc_samples
        
        # 2. Random samples (uniform exploration)
        random_samples = torch.rand(
            n_random, self.input_dim, device=self.device
        ) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        neg_samples.append(random_samples)
        
        # 3. Anti-optimal samples (encourage exploration away from best)
        if y is not None and len(y) > 0:
            # Find best points
            top_k = min(5, len(y))
            _, best_indices = torch.topk(y, top_k)
            best_points = data[best_indices]
            
            # Generate points far from best (reflect across center)
            center = torch.tensor([0.5] * self.input_dim, device=self.device)
            anti_samples = 2 * center - best_points.mean(dim=0, keepdim=True)
            anti_samples = anti_samples.expand(n_anti, -1)
            
            # Add noise for diversity
            anti_samples = anti_samples + torch.randn(n_anti, self.input_dim, device=self.device) * 0.2
            anti_samples = anti_samples.clamp(self.bounds[0], self.bounds[1])
            neg_samples.append(anti_samples)
        else:
            # Fallback to random
            extra_random = torch.rand(
                n_anti, self.input_dim, device=self.device
            ) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            neg_samples.append(extra_random)
        
        return torch.cat(neg_samples, dim=0)
    
    def train_step(self, data: torch.Tensor, 
                   y: Optional[torch.Tensor] = None) -> dict:
        """
        Single training step using VALUE-WEIGHTED contrastive divergence.
        
        Key difference from standard CD:
        - Points with higher y values get higher weight
        - This makes the EBM learn to assign low energy to promising regions
        
        Args:
            data: Observed data points [batch_size, dim]
            y: Function values for each point [batch_size] (IMPORTANT!)
        
        Returns:
            Dictionary with loss components
        """
        data = data.to(self.device)
        batch_size = data.size(0)
        
        # Skip training if batch is too small
        if batch_size < 2:
            return {
                "loss": 0.0,
                "cd_loss": 0.0,
                "energy_pos": 0.0,
                "energy_neg": 0.0,
                "reg_loss": 0.0
            }
        
        if y is not None:
            y = y.to(self.device)
        
        self.energy_net.train()
        self.optimizer.zero_grad()
        
        # ===== Positive Phase (with value weighting) =====
        energy_pos = self.energy_net(data)
        
        # Check for NaN in energy
        if torch.isnan(energy_pos).any():
            print("Warning: NaN detected in positive energy, skipping update")
            return {
                "loss": 0.0,
                "cd_loss": 0.0,
                "energy_pos": 0.0,
                "energy_neg": 0.0,
                "reg_loss": 0.0
            }
        
        if y is not None:
            # CRITICAL: Weight energy by function value
            # Higher y → higher weight → stronger push for low energy
            weights = self._compute_weights(y)
            loss_pos = (weights * energy_pos).mean()
        else:
            loss_pos = energy_pos.mean()
        
        # ===== Negative Phase =====
        neg_samples = self._generate_negative_samples(data, y)
        energy_neg = self.energy_net(neg_samples)
        
        # Check for NaN in negative energy
        if torch.isnan(energy_neg).any():
            print("Warning: NaN detected in negative energy, skipping update")
            return {
                "loss": 0.0,
                "cd_loss": 0.0,
                "energy_pos": energy_pos.mean().item(),
                "energy_neg": 0.0,
                "reg_loss": 0.0
            }
        
        loss_neg = energy_neg.mean()
        
        # ===== Contrastive Loss =====
        # Positive: minimize weighted energy of good points
        # Negative: maximize energy of negative samples
        cd_loss = loss_pos - loss_neg
        
        # ===== Regularization =====
        # L2 on energies to prevent unbounded growth
        reg_energy = 0.005 * (energy_pos ** 2).mean() + 0.005 * (energy_neg ** 2).mean()
        
        # Gradient penalty for smoothness
        data_grad = data.clone().detach().requires_grad_(True)
        energy_for_grad = self.energy_net(data_grad)
        grad_data = torch.autograd.grad(
            energy_for_grad.sum(), data_grad, create_graph=True
        )[0]
        grad_penalty = 0.01 * (grad_data ** 2).sum(dim=-1).mean()
        
        # Total loss
        loss = cd_loss + reg_energy + grad_penalty
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print("Warning: NaN detected in loss, skipping update")
            return {
                "loss": 0.0,
                "cd_loss": cd_loss.item() if not torch.isnan(cd_loss) else 0.0,
                "energy_pos": energy_pos.mean().item(),
                "energy_neg": energy_neg.mean().item(),
                "reg_loss": 0.0
            }
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.energy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "cd_loss": cd_loss.item(),
            "energy_pos": energy_pos.mean().item(),
            "energy_neg": energy_neg.mean().item(),
            "reg_loss": (reg_energy + grad_penalty).item()
        }
    
    def train(self, data: torch.Tensor, 
              y: Optional[torch.Tensor] = None,
              num_epochs: int = 100, 
              batch_size: int = 32, 
              verbose: bool = False):
        """
        Train EBM on observed data.
        
        IMPORTANT: Pass y values for value-weighted training!
        
        Args:
            data: All observed data points [N, dim]
            y: Function values for each point [N] (CRITICAL for good performance)
            num_epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Whether to print progress
        """
        data = data.to(self.device)
        if y is not None:
            y = y.to(self.device)
            self.y_values = y
        
        # Create dataset with y values
        if y is not None:
            dataset = torch.utils.data.TensorDataset(data, y)
        else:
            dataset = torch.utils.data.TensorDataset(data)
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=min(batch_size, len(data)), shuffle=True
        )
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_cd_loss = 0
            
            for batch in dataloader:
                if y is not None:
                    batch_data, batch_y = batch
                    metrics = self.train_step(batch_data, batch_y)
                else:
                    batch_data = batch[0]
                    metrics = self.train_step(batch_data, None)
                
                epoch_loss += metrics["loss"]
                epoch_cd_loss += metrics["cd_loss"]
            
            # Update learning rate
            self.scheduler.step()
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
                      f"CD Loss: {epoch_cd_loss:.4f}")
    
    def get_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute E_θ(x) for given points.
        
        Args:
            x: Input points [batch_size, dim]
        
        Returns:
            Energy values [batch_size]
        """
        self.energy_net.eval()
        x = x.to(self.device)
        
        with torch.no_grad():
            energy = self.energy_net(x)
            
            # Replace NaN with 0 to prevent propagation
            if torch.isnan(energy).any():
                energy = torch.nan_to_num(energy, nan=0.0)
        
        return energy
    
    def get_negative_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute -E_θ(x) for use in EBM-UCB acquisition.
        Higher values indicate more promising regions.
        """
        return -self.get_energy(x)
    
    def sample_from_model(self, num_samples: int, 
                          num_chains: int = 1) -> torch.Tensor:
        """
        Generate samples from the learned energy model.
        """
        self.energy_net.eval()
        
        init_samples = torch.rand(
            num_samples * num_chains, self.input_dim, device=self.device
        ) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        
        # Run longer MCMC
        long_sampler = AdaptiveLangevinSampler(
            step_size=self.sampler.step_size,
            num_steps=100
        )
        
        samples = long_sampler.sample(
            self.energy_net, init_samples, self.bounds
        )
        
        return samples


class EBMUCBAcquisition:
    """
    EBM-UCB Acquisition Function:
    
    α_EBM-UCB(x) = μ(x) + β·σ(x) - γ·E_θ(x)
    
    - μ(x), σ(x): GP posterior mean and std
    - β: Exploration-exploitation tradeoff for GP
    - γ: Weight of EBM global guidance
    - E_θ(x): Energy from EBM (lower = more promising)
    
    IMPROVED: Better optimization with gradient-based search
    """
    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        self.beta = beta
        self.gamma = gamma
    
    def __call__(self, x: torch.Tensor, 
                 gp_mean: torch.Tensor, 
                 gp_std: torch.Tensor,
                 energy: torch.Tensor) -> torch.Tensor:
        """
        Compute EBM-UCB acquisition value.
        """
        ucb = gp_mean + self.beta * gp_std
        ebm_term = -self.gamma * energy
        
        return ucb + ebm_term
    
    def optimize(self, gp_module, ebm_module, 
                 bounds: torch.Tensor,
                 num_restarts: int = 10,
                 num_samples: int = 1000) -> torch.Tensor:
        """
        Find the point that maximizes EBM-UCB acquisition.
        
        IMPROVED: Multi-stage optimization
        1. Initial screening with random + Sobol samples
        2. Local optimization from top candidates
        """
        device = ebm_module.device
        dim = bounds.size(1)
        
        # Stage 1: Generate diverse candidates
        try:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=dim, scramble=True)
            sobol_samples = sobol.draw(num_samples // 2).to(device)
            sobol_samples = sobol_samples * (bounds[1] - bounds[0]) + bounds[0]
        except:
            sobol_samples = torch.rand(num_samples // 2, dim, device=device)
            sobol_samples = sobol_samples * (bounds[1] - bounds[0]) + bounds[0]
        
        random_samples = torch.rand(num_samples // 2, dim, device=device)
        random_samples = random_samples * (bounds[1] - bounds[0]) + bounds[0]
        
        candidates = torch.cat([sobol_samples, random_samples], dim=0)
        
        # Evaluate acquisition
        gp_mean, gp_std = gp_module.predict(candidates)
        energy = ebm_module.get_energy(candidates)
        acq_values = self(candidates, gp_mean, gp_std, energy)
        
        # Stage 2: Local optimization from top candidates
        top_k = min(num_restarts, num_samples)
        _, top_indices = torch.topk(acq_values, top_k)
        top_candidates = candidates[top_indices]
        
        best_value = float('-inf')
        best_point = None
        
        for i in range(top_k):
            x = top_candidates[i:i+1].clone().requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=0.05)
            
            for _ in range(30):
                optimizer.zero_grad()
                gp_mean, gp_std = gp_module.predict(x)
                energy = ebm_module.get_energy(x)
                neg_acq = -self(x, gp_mean, gp_std, energy)
                neg_acq.sum().backward()
                optimizer.step()
                
                with torch.no_grad():
                    x.data.clamp_(bounds[0].item(), bounds[1].item())
            
            with torch.no_grad():
                gp_mean, gp_std = gp_module.predict(x)
                energy = ebm_module.get_energy(x)
                value = self(x, gp_mean, gp_std, energy).item()
            
            if value > best_value:
                best_value = value
                best_point = x.detach().squeeze(0)
        
        return best_point


if __name__ == "__main__":
    # Test optimized EBM module
    torch.manual_seed(42)
    
    input_dim = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create optimized EBM
    ebm = EBMModule(
        input_dim=input_dim,
        hidden_dims=[128, 128, 64],
        mcmc_steps=20,
        temperature=1.0,
        device=device
    )
    
    # Generate synthetic data with known structure
    n_samples = 50
    
    # Good points near (0.5, 0.5, ...)
    good_points = torch.randn(n_samples // 2, input_dim) * 0.1 + 0.5
    good_points = good_points.clamp(0, 1)
    good_y = torch.randn(n_samples // 2) * 0.5 - 2
    
    # Bad points near boundaries
    bad_points = torch.rand(n_samples // 2, input_dim)
    bad_points = (bad_points > 0.5).float()
    bad_points = bad_points + torch.randn_like(bad_points) * 0.1
    bad_points = bad_points.clamp(0, 1)
    bad_y = torch.randn(n_samples // 2) * 0.5 - 8
    
    data = torch.cat([good_points, bad_points], dim=0)
    y = torch.cat([good_y, bad_y], dim=0)
    
    print(f"Data shape: {data.shape}")
    print(f"Good points y mean: {good_y.mean():.2f}")
    print(f"Bad points y mean: {bad_y.mean():.2f}")
    
    print("\nTraining EBM with value weighting...")
    ebm.train(data, y=y, num_epochs=100, verbose=True)
    
    # Test: good points should have lower energy than bad points
    energy_good = ebm.get_energy(good_points).mean()
    energy_bad = ebm.get_energy(bad_points).mean()
    
    print(f"\nEnergy of good points (should be LOWER): {energy_good:.4f}")
    print(f"Energy of bad points (should be HIGHER): {energy_bad:.4f}")
    
    if energy_good < energy_bad:
        print("✓ SUCCESS: EBM correctly learned to assign lower energy to better points!")
    else:
        print("✗ ISSUE: EBM did not learn the correct energy landscape")
    
    # Test center vs boundary
    center = torch.ones(10, input_dim, device=device) * 0.5
    boundary = torch.zeros(10, input_dim, device=device)
    boundary[:, :2] = 1
    
    energy_center = ebm.get_energy(center).mean()
    energy_boundary = ebm.get_energy(boundary).mean()
    
    print(f"\nEnergy at center (0.5, ...): {energy_center:.4f}")
    print(f"Energy at boundary (0/1, ...): {energy_boundary:.4f}")