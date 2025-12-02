# REBMBO: Reinforced Energy-Based Model for Bayesian Optimization

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Optimizing the Unknown: Black Box Bayesian Optimization with Energy-Based Model and Reinforcement Learning"** (NeurIPS 2025).

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Configuration](#configuration)
- [Citation](#citation)

---

## Overview

**REBMBO** (Reinforced Energy-Based Model for Bayesian Optimization) addresses the fundamental **one-step myopia** problem in traditional Bayesian Optimization by integrating three synergistic modules:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          REBMBO Framework                           │
├─────────────────┬─────────────────────┬─────────────────────────────┤
│    Module A     │      Module B       │         Module C            │
│   GP Surrogate  │   EBM Global Guide  │      PPO Multi-Step         │
│                 │                     │                             │
│   μ(x), σ(x)    │    E_θ(x) Energy    │    π_ϕ(a|s) Policy          │
│  Local modeling │  Global exploration │   Adaptive planning         │
└─────────────────┴─────────────────────┴─────────────────────────────┘
```

### Core Algorithm

The algorithm formulates each BO iteration as a **Markov Decision Process (MDP)**:

- **State**: `s_t = (μ_t(x), σ_t(x), E_θ(x))` — GP posterior + EBM energy
- **Action**: `a_t ∈ X` — Next query point
- **Reward**: `r_t = f(a_t) - λ·E_θ(a_t)` — Function value + global exploration bonus

### EBM-UCB Acquisition Function

```
α_EBM-UCB(x) = μ(x) + β·σ(x) - γ·E_θ(x)
               ├────────────┤   └──────────┘
               Standard UCB    Global energy guidance
```

---

## Key Contributions

1. **EBM-Enhanced Acquisition**: Integrates Energy-Based Model signals into GP-UCB to capture global structural information beyond local uncertainty.

2. **MDP Formulation with PPO**: Models Bayesian Optimization as an MDP and uses Proximal Policy Optimization for adaptive multi-step lookahead.

3. **Landscape-Aware Regret (LAR)**: A theoretically justified metric that extends standard regret with an energy-informed global term:
   ```
   R^LAR_t = [f(x*) - f(x_t)] + α·[E_θ(x*) - E_θ(x_t)]
   ```

4. **Three GP Variants**:
   | Variant | Complexity | Best For |
   |---------|------------|----------|
   | REBMBO-C (Classic) | O(n³) | Low-dimensional, small datasets |
   | REBMBO-S (Sparse) | O(nm²) | High-dimensional, large datasets |
   | REBMBO-D (Deep) | Neural kernel | Non-stationary, multi-scale functions |

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/REBMBO.git
cd REBMBO

# Create conda environment
conda create -n rebmbo python=3.10
conda activate rebmbo

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.rebmbo import REBMBO; print('Installation successful!')"
```

### Dependencies (requirements.txt)

```
torch>=2.0.0
gpytorch>=1.11
botorch>=0.9.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

---

## Project Structure

```
REBMBO_Optimizing-the-Unknown/
│
├── configs/                          # YAML configuration files
│   ├── branin_2d.yaml               # Branin function (2D)
│   ├── ackley_5d.yaml               # Ackley function (5D)
│   ├── rosenbrock_8d.yaml           # Rosenbrock function (8D)
│   ├── hdbo_200d.yaml               # High-dimensional BO (200D)
│   ├── nanophotonic_3d.yaml         # Nanophotonic design (real-world)
│   └── rosetta_86d.yaml             # Protein design (real-world)
│
├── experiments/                      # Experiment runner scripts
│   ├── run_branin.py
│   ├── run_ackley.py
│   ├── run_rosenbrock.py
│   ├── run_hdbo.py
│   ├── run_nanophotonic.py
│   └── run_rosetta.py
│
├── src/                              # Source code
│   ├── rebmbo/                      # Core REBMBO implementation
│   │   ├── algorithm.py             # Main REBMBO class
│   │   ├── gp_module.py             # Module A: Gaussian Process
│   │   ├── ebm_module.py            # Module B: Energy-Based Model
│   │   ├── ppo_module.py            # Module C: PPO Agent
│   │   ├── acquisition.py           # EBM-UCB acquisition function
│   │   └── run_experiments.py       # Experiment utilities
│   │
│   ├── baselines/                   # Baseline methods
│   │   ├── turbo.py                 # TuRBO [Eriksson et al.]
│   │   ├── ballet_ici.py            # BALLET-ICI [Zhang et al.]
│   │   └── earl_bo.py               # EARL-BO [Cheon et al.]
│   │
│   ├── utils/                       # Utility functions
│   │   ├── config.py                # Configuration loader
│   │   ├── logger.py                # Experiment logging
│   │   ├── metrics.py               # LAR and other metrics
│   │   ├── plotting.py              # Visualization
│   │   └── sampler.py               # Benchmark functions
│   │
│   └── benchmarks.py                # Benchmark definitions
│
├── results/                          # Experimental results
│   ├── experimental_figures/        # Paper figures (PDFs and PNGs)
│   │   ├── rebmbo_framework.png     # Framework diagram
│   │   ├── UML.pdf                  # Architecture UML
│   │   ├── Ackley_regret_comparison.pdf
│   │   ├── Branin_regret_comparison.pdf
│   │   ├── HDBO200D_V4.pdf
│   │   ├── rosenbrock.pdf
│   │   ├── nanophotonic_benchmark.pdf
│   │   ├── rosetta_protein_benchmark.pdf
│   │   └── ...                      # Additional analysis figures
│   │
│   ├── figures/                     # Generated convergence plots
│   │   ├── ackley_5d_convergence.png
│   │   ├── branin_2d_convergence.png
│   │   └── rosenbrock_8d_convergence.png
│   │
│   ├── logs/                        # Detailed experiment logs
│   │   ├── ackley_5d/              # Ackley experiment logs
│   │   ├── branin_2d/              # Branin experiment logs
│   │   ├── rosenbrock_8d/          # Rosenbrock experiment logs
│   │   ├── hdbo_200d/              # HDBO experiment logs
│   │   └── nanophotonic_3d/        # Nanophotonic experiment logs
│   │
│   ├── rosetta_86d/                 # Rosetta experiment results
│   │
│   └── tables/                      # Summary tables (CSV/JSON)
│
├── README.md
└── requirements.txt
```

---

## Quick Start

### Basic Usage

```python
import torch
import numpy as np
from src.rebmbo import REBMBO, REBMBOConfig

# Define objective function (negated for maximization)
def branin(x):
    x1 = x[0] * 15 - 5  # Scale to [-5, 10]
    x2 = x[1] * 15      # Scale to [0, 15]
    a, b, c = 1, 5.1/(4*np.pi**2), 5/np.pi
    r, s, t = 6, 10, 1/(8*np.pi)
    return -(a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)

# Create configuration
config = REBMBOConfig(
    input_dim=2,
    bounds=(0, 1),
    gp_variant="classic",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Initialize REBMBO
optimizer = REBMBO(config)

# Generate initial samples
X_init = torch.rand(5, 2)
y_init = torch.tensor([branin(x.numpy()) for x in X_init])

# Initialize and optimize
optimizer.initialize(X_init, y_init)
best_x, best_y = optimizer.optimize(
    objective_fn=branin,
    n_iterations=50,
    use_ppo=True,
    verbose=True
)

print(f"Best found: x={best_x}, y={best_y:.4f}")
```

---

## Reproducing Paper Results

### Synthetic Benchmarks

#### Branin 2D

```powershell
# Quick test (30 iterations)
python experiments\run_branin.py

# Standard run (T=50, as in paper Table 1)
python experiments\run_branin.py --n_iterations 50

# Multi-seed experiment for statistical significance
python experiments\run_branin.py --seeds 1 2 3 4 5 --n_iterations 50

# Full paper reproduction
python experiments\run_branin.py --full
```

#### Ackley 5D

```powershell
# Quick test
python experiments\run_ackley.py

# Standard run (T=50)
python experiments\run_ackley.py --n_iterations 50

# Multi-seed experiment
python experiments\run_ackley.py --seeds 1 2 3 4 5 --n_iterations 50

# Full paper reproduction
python experiments\run_ackley.py --full
```

#### Rosenbrock 8D

```powershell
# Quick test
python experiments\run_rosenbrock.py

# Standard run (T=50)
python experiments\run_rosenbrock.py --n_iterations 50

# Extended run (T=100)
python experiments\run_rosenbrock.py --n_iterations 100

# Multi-seed experiment
python experiments\run_rosenbrock.py --seeds 1 2 3 4 5 --n_iterations 50

# Full paper reproduction
python experiments\run_rosenbrock.py --full
```

#### HDBO 200D (High-Dimensional)

```powershell
# Quick test with sparse GP (recommended for 200D)
python experiments\run_hdbo.py --variant sparse

# Standard run (T=50, as in paper Table 1)
python experiments\run_hdbo.py --n_iterations 50

# Extended run (T=100)
python experiments\run_hdbo.py --n_iterations 100

# Multi-seed statistical experiment
python experiments\run_hdbo.py --seeds 1 2 3 4 5 --n_iterations 50

# Full paper reproduction
python experiments\run_hdbo.py --full
```

### Real-World Benchmarks

#### Nanophotonic Structure Design (3D)

```powershell
# Quick test with mock simulator
python experiments\run_nanophotonic.py --mock

# Standard run (T=50, as in paper Table 2)
python experiments\run_nanophotonic.py --mock --n_iterations 50

# Extended run (T=80)
python experiments\run_nanophotonic.py --mock --n_iterations 80

# Multi-seed experiment
python experiments\run_nanophotonic.py --mock --seeds 1 2 3 4 5

# Full paper reproduction
python experiments\run_nanophotonic.py --mock --full
```

#### Rosetta Protein Design (86D)

```powershell
# Quick test with mock simulator
python experiments\run_rosetta.py --mock

# Try Deep GP (best variant for Rosetta in paper)
python experiments\run_rosetta.py --mock --variant deep

# Extended run with Deep GP (T=80)
python experiments\run_rosetta.py --mock --n_iterations 80 --variant deep

# Multi-seed experiment (check stability)
python experiments\run_rosetta.py --mock --seeds 1 2 3 --variant sparse

# Full paper reproduction with Deep GP
python experiments\run_rosetta.py --mock --full --variant deep
```

### Complete Paper Reproduction

To reproduce all results from the paper:

```powershell
# Run all synthetic benchmarks
python experiments\run_branin.py --full
python experiments\run_ackley.py --full
python experiments\run_rosenbrock.py --full
python experiments\run_hdbo.py --full

# Run all real-world benchmarks (with mock simulators)
python experiments\run_nanophotonic.py --mock --full
python experiments\run_rosetta.py --mock --full --variant deep
```

### Expected Runtime

| Benchmark | T=50 | T=100 | Hardware |
|-----------|------|-------|----------|
| Branin 2D | ~2 min | ~4 min | CPU/GPU |
| Ackley 5D | ~4 min | ~8 min | CPU/GPU |
| Rosenbrock 8D | ~7 min | ~14 min | CPU/GPU |
| HDBO 200D | ~20 min | ~40 min | GPU recommended |
| Nanophotonic 3D | ~10 min | ~18 min | CPU/GPU |
| Rosetta 86D | ~25 min | ~50 min | GPU recommended |


### Result Files

After running experiments, results are saved in:

```
results/
├── logs/
│   └── <benchmark_name>/
│       └── <benchmark>_<variant>_seed<N>_<timestamp>/
│           ├── config.json           # Experiment configuration
│           ├── experiment.log        # Detailed log
│           ├── final_results.json    # Summary results
│           └── iterations_*.jsonl    # Per-iteration data
├── figures/
│   └── <benchmark>_convergence.png   # Convergence plots
└── experimental_figures/
    └── *.pdf, *.png                  # Paper figures
```

---

## Configuration

### Example Configuration (ackley_5d.yaml)

```yaml
# Benchmark settings
benchmark:
  name: ackley_5d
  dim: 5
  bounds: [-5, 5]
  optimal_value: 0.0
  optimal_x: [0, 0, 0, 0, 0]

# Module A: Gaussian Process
gp:
  variant: classic          # classic, sparse, deep
  kernel: rbf_matern_mix    # RBF + Matern mixture
  num_inducing: 50          # For sparse GP
  train_epochs: 100
  retrain_epochs: 30

# Module B: Energy-Based Model
ebm:
  hidden_dims: [128, 128, 64]
  mcmc_steps: 30
  mcmc_step_size: 0.01
  num_negative_samples: 128
  train_epochs: 100
  retrain_epochs: 30

# Module C: PPO Agent
ppo:
  hidden_dims: [256, 256]
  lr_actor: 0.0003
  lr_critic: 0.001
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.05
  ppo_epochs: 10

# Acquisition function
acquisition:
  beta: 2.0     # UCB exploration weight
  gamma: 0.3    # EBM weight
  lambda_energy: 0.3  # Energy penalty in reward

# Optimization
optimization:
  n_init: 5
  n_iterations: 50
  use_ppo: true

# Device
device: auto  # auto, cuda, cpu
```

### Key Hyperparameters by Dimension

| Parameter | Low-D (2-5) | Medium-D (8-20) | High-D (50+) |
|-----------|-------------|-----------------|--------------|
| `n_init` | 5 | 10-15 | 20+ |
| `gp_variant` | classic | classic/sparse | sparse |
| `num_inducing` | - | 50-80 | 100+ |
| `ebm_hidden` | [128,128,64] | [256,256,128] | [512,512,256] |
| `ppo_hidden` | [256,256] | [512,256] | [1024,512] |
| `beta` (UCB) | 2.0 | 2.5 | 3.0 |
| `entropy_coef` | 0.05 | 0.08 | 0.1 |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{miao2025rebmbo,
  title={Optimizing the Unknown: Black Box Bayesian Optimization with 
         Energy-Based Model and Reinforcement Learning},
  author={Miao, Ruiyao and Xiao, Junren and Tsang, Shiya and 
          Xiong, Hui and Wu, Yingnian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```