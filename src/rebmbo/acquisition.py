# src/rebmbo/acquisition.py
from typing import Tuple
import torch

def ebm_ucb_acquisition(
    gp_mean: torch.Tensor,
    gp_std: torch.Tensor,
    ebm_energy: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """
    Calculates the EBM-UCB acquisition function value.
    α_EBM-UCB(x) = μ_f,t(x) + β * σ_f,t(x) - γ * E_θ(x)
    [cite: 184]

    Args:
        gp_mean (torch.Tensor): Posterior mean μ_f,t(x) from Module A.
        gp_std (torch.Tensor): Posterior std dev σ_f,t(x) from Module A.
        ebm_energy (torch.Tensor): Energy E_θ(x) from Module B.
        beta (float): Weight for GP uncertainty (local exploration).
        gamma (float): Weight for EBM energy (global exploration).

    Returns:
        torch.Tensor: The EBM-UCB acquisition value.
    """
    # This function *only* needs the computed values, not the original 'x'
    return gp_mean + beta * gp_std - gamma * ebm_energy