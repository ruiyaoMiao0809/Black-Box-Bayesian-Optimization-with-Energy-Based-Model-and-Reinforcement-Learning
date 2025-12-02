"""
REBMBO: Reinforced Energy-Based Model for Bayesian Optimization

A novel approach combining:
- Gaussian Processes for local uncertainty estimation
- Energy-Based Models for global structure exploration
- PPO for multi-step lookahead planning

Paper: "Optimizing the Unknown: Black Box Bayesian Optimization with 
        Energy-Based Model and Reinforcement Learning"
"""

from .algorithm import (
    REBMBO,
    REBMBOConfig,
    LandscapeAwareRegret,
    create_rebmbo_c,
    create_rebmbo_s,
    create_rebmbo_d
)
from .gp_module import GPModule, ClassicGP, SparseGP, DeepGP
from .ebm_module import EBMModule, EBMUCBAcquisition, EnergyNetwork
from .ppo_module import PPOModule, ActorNetwork, CriticNetwork, StateEncoder

__version__ = "1.0.0"
__author__ = "REBMBO Authors"

__all__ = [
    # Main algorithm
    "REBMBO",
    "REBMBOConfig",
    "LandscapeAwareRegret",
    
    # Factory functions
    "create_rebmbo_c",
    "create_rebmbo_s", 
    "create_rebmbo_d",
    
    # GP Module
    "GPModule",
    "ClassicGP",
    "SparseGP",
    "DeepGP",
    
    # EBM Module
    "EBMModule",
    "EBMUCBAcquisition",
    "EnergyNetwork",
    
    # PPO Module
    "PPOModule",
    "ActorNetwork",
    "CriticNetwork",
    "StateEncoder"
]