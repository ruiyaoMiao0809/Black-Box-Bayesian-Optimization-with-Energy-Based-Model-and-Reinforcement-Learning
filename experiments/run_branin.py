# experiments/run_branin.py
import os
import torch
import numpy as np
import time

# Import from our source files
from src.utils.config import load_config
from src.utils.logger import JSONLogger
from src.rebmbo.algorithm import REBMBO
from src.utils.plotting import plot_convergence  # Import the updated plotter

def branin(x: torch.Tensor) -> torch.Tensor:
    """
    Standard 2D Branin function.
    The function is *negated* because REBMBO aims to *maximize* the objective.
    The true minimum is ~0.397887, so the true maximum is ~-0.397887.
    
    Args:
        x (torch.Tensor): A (N, 2) tensor of input points.

    Returns:
        torch.Tensor: A (N,) tensor of objective values.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    # Standard Branin definition
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * torch.cos(x1)
    
    # Negate the function for maximization
    return -(term1 + term2 + s)

def main():
    print("Loading config and setting up...")
    
    # 1. Load configuration
    cfg = load_config("configs/branin_2d.yaml")
    bounds = torch.tensor(cfg['bounds'], dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 2. Set up logger
    log_dir = os.path.join("results", "logs", cfg['task_name'])
    logger = JSONLogger(log_dir)

    # Define the logging callback function
    def log_fn(record):
        logger.log(record)
        print(f"Iter: {record['iter']}, y_t: {record['y_t']:.3f}, Best y: {record['best_y']:.3f}, Reward: {record['reward']:.3f}")

    # 3. Initialize REBMBO Algorithm
    algo = REBMBO(
        f=branin,
        input_dim=cfg['input_dim'],
        beta=cfg['beta'],
        gamma=cfg['gamma'],
        lam=cfg['lambda'],
        device=device,
        gp_cfg=cfg.get("gp", {}),
        ebm_cfg=cfg.get("ebm", {}),
        ppo_cfg=cfg.get("ppo", {}),
    )

    # 4. Run Optimization
    start_time = time.time()
    
    # Run initial sampling
    algo.initialize(n0=cfg['n0'], bounds=bounds)
    
    # Run the main BO loop
    algo.run(T=cfg['T'], bounds=bounds, log_fn=log_fn)
    
    end_time = time.time()
    print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")
    print(f"Best value found: {algo.y.max().item()}")

    # 5. Generate and save plots
    print("Optimization finished. Generating convergence plot...")
    
    # Define log file and output path
    log_file_path = os.path.join(log_dir, "log.jsonl")
    figure_path = os.path.join(
        "results", "figures", f"{cfg['task_name']}_convergence.png"
    )
    
    # Call the updated plotting function
    # This now passes the task name and the known global optimum
    plot_convergence(
        log_file_path, 
        figure_path, 
        task_name=cfg['task_name'],
        global_optimum=-0.397887  # Pass the known optimum for Branin
    )


if __name__ == "__main__":
    main()