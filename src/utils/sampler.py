# src/utils/sampler.py

import torch
import numpy as np
from typing import Optional

def sample_uniform(
    n: int,
    dim: int,
    bounds: torch.Tensor,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    在给定边界内生成 n 个均匀随机采样的点。

    Args:
        n (int): 要生成的样本数量。
        dim (int): 样本的维度 (输入维度)。
        bounds (torch.Tensor): 形状为 (2, dim) 或 (2, 1) 的边界张量。
                                bounds[0] 是下界，bounds[1] 是上界。
        device (str, optional): 生成张量的设备。

    Returns:
        torch.Tensor: 形状为 (n, dim) 的均匀采样点。
    """
    if bounds.dim() == 1:
        # 如果 bounds 只有 [min, max]
        bounds = bounds.unsqueeze(1).repeat(1, dim)
    
    # 获取下界和上界
    lower_bounds = bounds[0].to(torch.float32)
    upper_bounds = bounds[1].to(torch.float32)

    # 计算范围 (range = max - min)
    span = upper_bounds - lower_bounds

    # 生成 [0, 1] 上的均匀随机数
    uniform_samples = torch.rand(n, dim, device=device)

    # 缩放到指定的边界内: x = min + rand * span
    X = lower_bounds + uniform_samples * span
    
    return X.to(device)