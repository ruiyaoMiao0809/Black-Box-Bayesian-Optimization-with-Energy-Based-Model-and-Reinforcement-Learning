"""
Configuration loader for REBMBO experiments.
Loads YAML config files and provides a structured interface.
"""

import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import os


@dataclass
class GPConfig:
    """GP Module configuration."""
    variant: str = "classic"
    kernel: str = "rbf_matern_mix"
    num_inducing: int = 50
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    latent_dim: int = 32
    train_epochs: int = 100
    retrain_epochs: int = 50
    learning_rate: float = 0.1


@dataclass
class EBMConfig:
    """EBM Module configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])
    mcmc_steps: int = 20
    mcmc_step_size: float = 0.1
    num_negative_samples: int = 100
    train_epochs: int = 100
    retrain_epochs: int = 50
    learning_rate: float = 0.001


@dataclass
class PPOConfig:
    """PPO Module configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class AcquisitionConfig:
    """Acquisition function configuration."""
    beta: float = 2.0
    gamma: float = 0.5
    lambda_energy: float = 0.3


@dataclass
class OptimizationConfig:
    """Optimization settings."""
    n_init: int = 5
    n_iterations_short: int = 30
    n_iterations_long: int = 50
    use_ppo: bool = True


@dataclass
class ExperimentConfig:
    """Experiment settings."""
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    save_results: bool = True
    verbose: bool = True


@dataclass
class TaskConfig:
    """Task/benchmark configuration."""
    name: str = "unknown"
    dim: int = 2
    bounds: Tuple[float, float] = (0.0, 1.0)
    original_bounds: Tuple[float, float] = (-5.0, 5.0)
    optimal_value: float = 0.0
    description: str = ""


@dataclass
class Config:
    """Complete REBMBO configuration."""
    task: TaskConfig = field(default_factory=TaskConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    ebm: EBMConfig = field(default_factory=EBMConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    num_grid_points: int = 50
    device: str = "auto"
    
    def get_device(self) -> str:
        """Get actual device string."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    def to_rebmbo_config(self):
        """Convert to REBMBOConfig for algorithm."""
        from src.rebmbo.algorithm import REBMBOConfig
        
        return REBMBOConfig(
            input_dim=self.task.dim,
            bounds=self.task.bounds,
            gp_variant=self.gp.variant,
            gp_num_inducing=self.gp.num_inducing,
            gp_hidden_dims=self.gp.hidden_dims,
            gp_latent_dim=self.gp.latent_dim,
            gp_train_epochs=self.gp.train_epochs,
            gp_retrain_epochs=self.gp.retrain_epochs,
            ebm_hidden_dims=self.ebm.hidden_dims,
            ebm_mcmc_steps=self.ebm.mcmc_steps,
            ebm_mcmc_step_size=self.ebm.mcmc_step_size,
            ebm_num_negative_samples=self.ebm.num_negative_samples,
            ebm_train_epochs=self.ebm.train_epochs,
            ebm_retrain_epochs=self.ebm.retrain_epochs,
            ppo_hidden_dims=self.ppo.hidden_dims,
            ppo_lr_actor=self.ppo.lr_actor,
            ppo_lr_critic=self.ppo.lr_critic,
            ppo_gamma=self.ppo.gamma,
            ppo_gae_lambda=self.ppo.gae_lambda,
            ppo_clip_epsilon=self.ppo.clip_epsilon,
            ppo_epochs=self.ppo.ppo_epochs,
            ppo_mini_batch_size=self.ppo.mini_batch_size,
            beta=self.acquisition.beta,
            gamma=self.acquisition.gamma,
            lambda_energy=self.acquisition.lambda_energy,
            num_grid_points=self.num_grid_points,
            device=self.get_device()
        )


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Config object with all settings
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Parse task config
    task_dict = yaml_config.get('task', {})
    bounds = task_dict.get('bounds', [0.0, 1.0])
    if isinstance(bounds, list):
        bounds = tuple(bounds)
    task_config = TaskConfig(
        name=task_dict.get('name', 'unknown'),
        dim=task_dict.get('dim', 2),
        bounds=bounds,
        original_bounds=tuple(task_dict.get('original_bounds', [-5.0, 5.0])),
        optimal_value=task_dict.get('optimal_value', 0.0),
        description=task_dict.get('description', '')
    )
    
    # Parse GP config
    gp_dict = yaml_config.get('gp', {})
    gp_config = GPConfig(
        variant=gp_dict.get('variant', 'classic'),
        kernel=gp_dict.get('kernel', 'rbf_matern_mix'),
        num_inducing=gp_dict.get('num_inducing', 50),
        hidden_dims=gp_dict.get('hidden_dims', [64, 64]),
        latent_dim=gp_dict.get('latent_dim', 32),
        train_epochs=gp_dict.get('train_epochs', 100),
        retrain_epochs=gp_dict.get('retrain_epochs', 50),
        learning_rate=gp_dict.get('learning_rate', 0.1)
    )
    
    # Parse EBM config
    ebm_dict = yaml_config.get('ebm', {})
    ebm_config = EBMConfig(
        hidden_dims=ebm_dict.get('hidden_dims', [128, 128, 64]),
        mcmc_steps=ebm_dict.get('mcmc_steps', 20),
        mcmc_step_size=ebm_dict.get('mcmc_step_size', 0.1),
        num_negative_samples=ebm_dict.get('num_negative_samples', 100),
        train_epochs=ebm_dict.get('train_epochs', 100),
        retrain_epochs=ebm_dict.get('retrain_epochs', 50),
        learning_rate=ebm_dict.get('learning_rate', 0.001)
    )
    
    # Parse PPO config
    ppo_dict = yaml_config.get('ppo', {})
    ppo_config = PPOConfig(
        hidden_dims=ppo_dict.get('hidden_dims', [256, 256]),
        lr_actor=ppo_dict.get('lr_actor', 3e-4),
        lr_critic=ppo_dict.get('lr_critic', 1e-3),
        gamma=ppo_dict.get('gamma', 0.99),
        gae_lambda=ppo_dict.get('gae_lambda', 0.95),
        clip_epsilon=ppo_dict.get('clip_epsilon', 0.2),
        ppo_epochs=ppo_dict.get('ppo_epochs', 10),
        mini_batch_size=ppo_dict.get('mini_batch_size', 64),
        entropy_coef=ppo_dict.get('entropy_coef', 0.01),
        value_coef=ppo_dict.get('value_coef', 0.5),
        max_grad_norm=ppo_dict.get('max_grad_norm', 0.5)
    )
    
    # Parse acquisition config
    acq_dict = yaml_config.get('acquisition', {})
    acq_config = AcquisitionConfig(
        beta=acq_dict.get('beta', 2.0),
        gamma=acq_dict.get('gamma', 0.5),
        lambda_energy=acq_dict.get('lambda_energy', 0.3)
    )
    
    # Parse optimization config
    opt_dict = yaml_config.get('optimization', {})
    opt_config = OptimizationConfig(
        n_init=opt_dict.get('n_init', 5),
        n_iterations_short=opt_dict.get('n_iterations_short', 30),
        n_iterations_long=opt_dict.get('n_iterations_long', 50),
        use_ppo=opt_dict.get('use_ppo', True)
    )
    
    # Parse experiment config
    exp_dict = yaml_config.get('experiment', {})
    exp_config = ExperimentConfig(
        seeds=exp_dict.get('seeds', [1, 2, 3, 4, 5]),
        save_results=exp_dict.get('save_results', True),
        verbose=exp_dict.get('verbose', True)
    )
    
    # Parse state config
    state_dict = yaml_config.get('state', {})
    num_grid_points = state_dict.get('num_grid_points', 50)
    
    # Device
    device = yaml_config.get('device', 'auto')
    
    return Config(
        task=task_config,
        gp=gp_config,
        ebm=ebm_config,
        ppo=ppo_config,
        acquisition=acq_config,
        optimization=opt_config,
        experiment=exp_config,
        num_grid_points=num_grid_points,
        device=device
    )


def get_config_path(benchmark_name: str) -> Path:
    """Get path to config file for a benchmark."""
    # Try multiple locations
    possible_paths = [
        Path(__file__).parent.parent.parent / "configs" / f"{benchmark_name}.yaml",
        Path("configs") / f"{benchmark_name}.yaml",
        Path(f"configs/{benchmark_name}.yaml")
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(f"Config file for {benchmark_name} not found")


if __name__ == "__main__":
    # Test config loading
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    config_path = get_config_path("ackley_5d")
    print(f"Loading config from: {config_path}")
    
    config = load_config(config_path)
    
    print(f"\nTask: {config.task.name}")
    print(f"Dimension: {config.task.dim}")
    print(f"GP variant: {config.gp.variant}")
    print(f"Device: {config.get_device()}")
    print(f"Seeds: {config.experiment.seeds}")