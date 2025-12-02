"""
Benchmark Functions for REBMBO Experiments

Includes:
- Synthetic: Branin 2D, Ackley 5D, Rosenbrock 8D, HDBO 200D
- Real-world proxies: Nanophotonic 3D, Rosetta 86D (simulated)

Updated: Rosenbrock 8D now follows paper definition exactly:
- Domain: [-2, 2]^8 (scaled to [0, 1]^8)
- f(x) = Σ_{i=1}^{7} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
- Global minimum: f(x*) = 0 at x* = (1, 1, ..., 1)
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkInfo:
    """Information about a benchmark function."""
    name: str
    dim: int
    bounds: Tuple[float, float]
    optimal_value: float
    optimal_x: np.ndarray = None
    original_bounds: Tuple[float, float] = None  # Original domain before scaling


# ============ Synthetic Benchmarks ============

def branin_2d(x: np.ndarray) -> float:
    """
    Branin-Hoo function (2D).
    
    Domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    Scaled to [0, 1]^2 for standardization.
    
    Global minimum: f(x*) ≈ 0.398 at multiple locations
    """
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


BRANIN_INFO = BenchmarkInfo(
    name="Branin-2D",
    dim=2,
    bounds=(0, 1),
    optimal_value=-0.398
)


def ackley_5d(x: np.ndarray) -> float:
    """
    Ackley function (5D) as defined in the paper (Appendix B.2).
    
    f(x) = -a·exp(-b·√(1/d·Σx_i²)) - exp(1/d·Σcos(c·x_i)) + a + e
    
    Parameters (from paper):
        a = 20
        b = 0.2
        c = 2π
    
    Original domain: [-32.768, 32.768]^5
    Scaled domain: [0, 1]^5 for standardization
    
    Global minimum: f(x*) = 0 at x* = (0, ..., 0)
    
    This function has many local minima and one global minimum,
    making it excellent for testing global exploration capability.
    """
    # Scale from [0,1] to [-5, 5]
    x_scaled = x * 10 - 5  # [0,1] -> [-5, 5]
    
    d = len(x_scaled)
    
    # Paper parameters
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi
    
    # Ackley formula
    sum_sq = np.sum(x_scaled ** 2)
    sum_cos = np.sum(np.cos(c * x_scaled))
    
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    
    result = term1 + term2 + a + np.e
    
    return -result  # Negate for maximization (paper minimizes, we maximize)


ACKLEY_INFO = BenchmarkInfo(
    name="Ackley-5D",
    dim=5,
    bounds=(0, 1),
    optimal_value=0.0,  # Maximum after negation
    optimal_x=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),  # Maps to origin in scaled space
    original_bounds=(-5, 5)
)


def rosenbrock_8d(x: np.ndarray) -> float:
    """
    Rosenbrock function (8D) as defined in the paper (Appendix B.3).
    
    Also known as the "Banana function" due to its curved valley.
    
    Formula (from paper):
        f(x) = Σ_{i=1}^{7} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    
    Original domain: [-2, 2]^8 (as specified in paper)
    Scaled domain: [0, 1]^8 for standardization
    
    Global minimum: f(x*) = 0 at x* = (1, 1, ..., 1)
    
    Characteristics:
    - Uni-modal (single global minimum)
    - Narrow, curved valley leading to minimum
    - Difficult for local methods due to curvature
    - Tests surrogate model's ability to capture curvature
    - Tests exploration strategy to avoid premature convergence
    
    In scaled [0, 1] domain:
    - Optimal point is at x* = (0.75, 0.75, ..., 0.75)
      because 0.75 * 4 - 2 = 1.0 (maps to 1 in original domain)
    """
    # Scale from [0, 1] to [-2, 2] as specified in paper
    x_scaled = x * 4 - 2  # [0,1] -> [-2, 2]
    
    # Rosenbrock formula: Σ_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    result = 0.0
    for i in range(len(x_scaled) - 1):
        result += 100.0 * (x_scaled[i+1] - x_scaled[i]**2)**2 + (1.0 - x_scaled[i])**2
    
    return -result  # Negate for maximization (paper minimizes, we maximize)


# Compute optimal_x: x* = 1 in original domain -> (1 - (-2)) / 4 = 0.75 in [0,1]
ROSENBROCK_INFO = BenchmarkInfo(
    name="Rosenbrock-8D",
    dim=8,
    bounds=(0, 1),
    optimal_value=0.0,  # Maximum after negation (minimum is 0)
    optimal_x=np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
    original_bounds=(-2, 2)
)


def hdbo_200d(x: np.ndarray) -> float:
    """
    High-Dimensional BO benchmark (200D).
    
    Sum of squared distances from random target points.
    Designed to test scalability in high dimensions.
    """
    # Create deterministic "optimal" point
    np.random.seed(42)
    optimal = np.random.rand(200)
    
    # Distance-based objective
    diff = x - optimal
    result = -np.sum(diff**2)  # Negative because we maximize
    
    return result


HDBO_INFO = BenchmarkInfo(
    name="HDBO-200D",
    dim=200,
    bounds=(0, 1),
    optimal_value=0.0
)


# ============ Real-World Inspired Benchmarks ============

def nanophotonic_3d(x: np.ndarray) -> float:
    """
    Nanophotonic structure optimization (3D).
    
    Simulates optimizing optical properties of a nanostructure.
    Based on simplified physics model.
    
    Variables represent layer thicknesses/compositions.
    """
    # Simplified physics-inspired model
    wavelength = 550  # nm, target wavelength
    
    # Layer parameters from x
    t1, t2, t3 = x * 100  # Scale to nm
    
    # Simplified Fresnel coefficients simulation
    n1 = 1.5 + 0.5 * np.sin(t1 / 50)
    n2 = 2.0 + 0.3 * np.cos(t2 / 30)
    n3 = 1.8 + 0.4 * np.sin(t3 / 40)
    
    # Transmission efficiency (simplified)
    phase1 = 2 * np.pi * n1 * t1 / wavelength
    phase2 = 2 * np.pi * n2 * t2 / wavelength
    phase3 = 2 * np.pi * n3 * t3 / wavelength
    
    # Interference pattern
    transmission = np.abs(
        np.exp(1j * phase1) + 0.5 * np.exp(1j * phase2) + 0.3 * np.exp(1j * phase3)
    )**2
    
    # Add bandwidth consideration
    bandwidth_penalty = -0.1 * ((t1 - 50)**2 + (t2 - 60)**2 + (t3 - 55)**2) / 1000
    
    return transmission + bandwidth_penalty


NANOPHOTONIC_INFO = BenchmarkInfo(
    name="Nanophotonic-3D",
    dim=3,
    bounds=(0, 1),
    optimal_value=3.0  # Approximate
)


def rosetta_86d(x: np.ndarray) -> float:
    """
    Rosetta protein structure prediction (86D).
    
    Simplified version of protein energy landscape optimization.
    Variables represent dihedral angles and sidechain conformations.
    """
    # Scale to angle domain [-pi, pi]
    angles = (x - 0.5) * 2 * np.pi
    
    # Simplified energy function components
    
    # 1. Backbone torsion energy
    backbone_energy = 0
    for i in range(0, len(angles)-1, 2):
        phi, psi = angles[i], angles[i+1]
        # Ramachandran-like energy landscape
        backbone_energy += -np.cos(phi) - 0.5 * np.cos(psi) - 0.3 * np.cos(phi + psi)
    
    # 2. Van der Waals-like repulsion
    positions = np.cumsum(np.sin(angles[:20]))  # Simplified 1D chain
    vdw_energy = 0
    for i in range(len(positions)):
        for j in range(i+2, len(positions)):
            dist = abs(positions[j] - positions[i])
            if dist < 0.5:
                vdw_energy -= 10 * (0.5 - dist)**2
    
    # 3. Hydrogen bonding (simplified)
    hbond_energy = 0
    for i in range(0, len(angles)-6, 3):
        # Check for H-bond pattern
        if np.cos(angles[i] - angles[i+6]) > 0.7:
            hbond_energy += 1
    
    # 4. Electrostatic (simplified)
    charges = np.sin(angles[:10])
    elec_energy = -0.1 * np.sum(charges)**2
    
    total_energy = backbone_energy + vdw_energy + hbond_energy + elec_energy
    
    return total_energy


ROSETTA_INFO = BenchmarkInfo(
    name="Rosetta-86D",
    dim=86,
    bounds=(0, 1),
    optimal_value=50.0  # Approximate
)


def natsbench_20d(x: np.ndarray) -> float:
    """
    NATS-Bench NAS benchmark (20D).
    
    Simulates neural architecture search.
    Variables encode architecture choices.
    """
    # Decode architecture parameters
    num_layers = int(x[0] * 4) + 1  # 1-5 layers
    width_mult = 0.5 + x[1] * 1.5    # Width multiplier 0.5-2.0
    
    # Layer-specific parameters
    layer_params = x[2:12].reshape(5, 2)  # [type, connections]
    
    # Regularization parameters
    dropout = x[12] * 0.5
    weight_decay = 10**(x[13] * 4 - 6)  # 1e-6 to 1e-2
    
    # Training hyperparameters
    lr = 10**(x[14] * 3 - 4)  # 1e-4 to 1e-1
    batch_size = int(32 + x[15] * 96)  # 32-128
    
    # Simulated accuracy based on architecture choices
    base_acc = 0.7
    
    # Depth bonus (with diminishing returns)
    depth_bonus = 0.05 * np.log1p(num_layers)
    
    # Width bonus
    width_bonus = 0.03 * width_mult if width_mult < 1.5 else 0.03 * 1.5 - 0.02 * (width_mult - 1.5)
    
    # Connection pattern bonus
    conn_bonus = 0.02 * np.mean(layer_params[:, 1])
    
    # Regularization effect
    reg_effect = -0.1 * (dropout - 0.2)**2 - 0.05 * np.log10(weight_decay + 1e-10)**2 / 100
    
    # Learning rate effect
    lr_effect = -0.1 * (np.log10(lr) + 2)**2 / 10
    
    # Noise to simulate training variance
    noise = 0.01 * np.sin(x[16] * 100) * np.cos(x[17] * 100)
    
    accuracy = base_acc + depth_bonus + width_bonus + conn_bonus + reg_effect + lr_effect + noise
    
    return np.clip(accuracy, 0, 1)


NATSBENCH_INFO = BenchmarkInfo(
    name="NATS-Bench-20D",
    dim=20,
    bounds=(0, 1),
    optimal_value=0.95
)


def robot_trajectory_40d(x: np.ndarray) -> float:
    """
    Robot trajectory optimization (40D).
    
    Optimizes a 10-waypoint trajectory in 4D space (x, y, z, gripper).
    Objective: minimize path length + reach target + avoid obstacles.
    """
    # Reshape to waypoints
    waypoints = x.reshape(10, 4)  # 10 waypoints, 4D each
    
    # Scale to workspace [-1, 1]^3 + gripper [0, 1]
    waypoints[:, :3] = waypoints[:, :3] * 2 - 1
    
    # Target position
    target = np.array([0.8, 0.5, 0.3])
    
    # 1. Path smoothness (minimize acceleration)
    velocities = np.diff(waypoints[:, :3], axis=0)
    accelerations = np.diff(velocities, axis=0)
    smoothness = -np.sum(accelerations**2)
    
    # 2. Target reaching (final waypoint close to target)
    target_dist = np.linalg.norm(waypoints[-1, :3] - target)
    reaching = -10 * target_dist**2
    
    # 3. Obstacle avoidance (spherical obstacles)
    obstacles = [
        (np.array([0.3, 0.3, 0.3]), 0.2),
        (np.array([0.5, 0.7, 0.2]), 0.15),
        (np.array([0.2, 0.5, 0.6]), 0.18)
    ]
    
    collision_penalty = 0
    for wp in waypoints:
        for obs_center, obs_radius in obstacles:
            dist = np.linalg.norm(wp[:3] - obs_center)
            if dist < obs_radius:
                collision_penalty -= 5 * (obs_radius - dist)
    
    # 4. Path length (energy efficiency)
    path_length = np.sum(np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1))
    efficiency = -0.5 * path_length
    
    # 5. Gripper timing (should close near target)
    gripper_timing = waypoints[:, 3]
    gripper_score = gripper_timing[-1] - 0.3 * np.sum(gripper_timing[:-3])
    
    total = smoothness + reaching + collision_penalty + efficiency + gripper_score
    
    return total


ROBOT_TRAJECTORY_INFO = BenchmarkInfo(
    name="Robot-Trajectory-40D",
    dim=40,
    bounds=(0, 1),
    optimal_value=-1.0  # Approximate
)


# ============ Benchmark Registry ============

BENCHMARKS = {
    "branin_2d": (branin_2d, BRANIN_INFO),
    "ackley_5d": (ackley_5d, ACKLEY_INFO),
    "rosenbrock_8d": (rosenbrock_8d, ROSENBROCK_INFO),
    "hdbo_200d": (hdbo_200d, HDBO_INFO),
    "nanophotonic_3d": (nanophotonic_3d, NANOPHOTONIC_INFO),
    "rosetta_86d": (rosetta_86d, ROSETTA_INFO),
    "natsbench_20d": (natsbench_20d, NATSBENCH_INFO),
    "robot_trajectory_40d": (robot_trajectory_40d, ROBOT_TRAJECTORY_INFO)
}


def get_benchmark(name: str) -> Tuple[Callable, BenchmarkInfo]:
    """Get benchmark function and info by name."""
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[name]


def list_benchmarks():
    """List all available benchmarks."""
    print("\nAvailable Benchmarks:")
    print("-" * 60)
    for name, (_, info) in BENCHMARKS.items():
        opt_x_str = ""
        if info.optimal_x is not None:
            opt_x_str = f", optimal_x={info.optimal_x[:3]}..." if len(info.optimal_x) > 3 else f", optimal_x={info.optimal_x}"
        print(f"  {name}: {info.name} (dim={info.dim}, optimal={info.optimal_value}{opt_x_str})")
    print("-" * 60)


def test_rosenbrock():
    """Test Rosenbrock function at key points."""
    print("\n" + "="*60)
    print("Testing Rosenbrock 8D Function")
    print("="*60)
    
    func, info = get_benchmark("rosenbrock_8d")
    
    # Test at optimal point (0.75, 0.75, ..., 0.75) in [0,1] domain
    # which maps to (1, 1, ..., 1) in [-2, 2] domain
    x_optimal = np.array([0.75] * 8)
    y_optimal = func(x_optimal)
    print(f"\nOptimal point in [0,1] domain: x = {x_optimal}")
    print(f"Maps to in [-2,2] domain: x = {x_optimal * 4 - 2}")
    print(f"f(x*) = {y_optimal:.6f} (should be ~0.0)")
    
    # Test at center (0.5, 0.5, ..., 0.5) -> (0, 0, ..., 0) in original
    x_center = np.array([0.5] * 8)
    y_center = func(x_center)
    print(f"\nCenter point in [0,1] domain: x = {x_center}")
    print(f"Maps to in [-2,2] domain: x = {x_center * 4 - 2}")
    print(f"f(center) = {y_center:.6f}")
    
    # Test at random point
    np.random.seed(42)
    x_random = np.random.rand(8)
    y_random = func(x_random)
    print(f"\nRandom point: x = {x_random[:4]}...")
    print(f"f(random) = {y_random:.6f}")
    
    # Test at corners
    x_corner_low = np.array([0.0] * 8)  # Maps to (-2, -2, ..., -2)
    y_corner_low = func(x_corner_low)
    print(f"\nLow corner (0,0,...,0) -> (-2,-2,...,-2): f = {y_corner_low:.6f}")
    
    x_corner_high = np.array([1.0] * 8)  # Maps to (2, 2, ..., 2)
    y_corner_high = func(x_corner_high)
    print(f"High corner (1,1,...,1) -> (2,2,...,2): f = {y_corner_high:.6f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Test all benchmarks
    print("Testing all benchmark functions...")
    print("="*60)
    
    for name, (func, info) in BENCHMARKS.items():
        # Random test point
        x = np.random.rand(info.dim)
        y = func(x)
        print(f"{info.name}: f(random) = {y:.4f}")
    
    print("\n" + "="*60)
    list_benchmarks()
    
    # Detailed Rosenbrock test
    test_rosenbrock()