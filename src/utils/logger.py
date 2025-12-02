"""
Logging utilities for REBMBO experiments.
Records experiment progress, metrics, and results.

FIXED: Properly handles numpy scalar types (float32, float64, int32, int64) for JSON serialization
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch


class Logger:
    """
    Simple JSON-lines logger for experiment tracking.
    Each line is a JSON object representing one iteration.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
        
        self.entries = []
        self.start_time = time.time()
    
    def log(self, data: Dict[str, Any]):
        """
        Log a single entry (e.g., one iteration).
        
        Args:
            data: Dictionary of values to log
        """
        # Add timestamp
        entry = {
            "timestamp": time.time() - self.start_time,
            "datetime": datetime.now().isoformat(),
            **self._serialize(data)
        }
        
        self.entries.append(entry)
        
        # Write to file immediately (append mode)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def _serialize(self, data: Dict) -> Dict:
        """Convert numpy/torch arrays and scalars to JSON-serializable types."""
        result = {}
        for k, v in data.items():
            result[k] = self._serialize_value(v)
        return result
    
    def _serialize_value(self, v):
        """Serialize a single value to JSON-compatible type."""
        if v is None:
            return None
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, torch.Tensor):
            return v.cpu().numpy().tolist()
        elif isinstance(v, dict):
            return self._serialize(v)
        elif isinstance(v, (list, tuple)):
            return [self._serialize_value(x) for x in v]
        # FIXED: Handle numpy scalar types
        elif isinstance(v, (np.float32, np.float64, np.floating)):
            return float(v)
        elif isinstance(v, (np.int32, np.int64, np.integer)):
            return int(v)
        elif isinstance(v, np.bool_):
            return bool(v)
        # Handle Python native types
        elif isinstance(v, (int, float, str, bool)):
            return v
        else:
            # Try to convert to string as last resort
            try:
                return str(v)
            except:
                return None
    
    def get_entries(self) -> List[Dict]:
        """Get all logged entries."""
        return self.entries
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save final summary to a separate file."""
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self._serialize(summary), f, indent=2)
    
    def load(self, log_file: str) -> List[Dict]:
        """Load entries from a log file."""
        entries = []
        with open(log_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line.strip()))
        return entries


class ExperimentLogger:
    """
    Comprehensive experiment logger with multiple tracking capabilities.
    Tracks iterations, metrics, model states, and more.
    """
    
    def __init__(self, 
                 log_dir: str,
                 experiment_name: str,
                 benchmark_name: str,
                 variant: str = "classic",
                 seed: int = 42,
                 config: Dict = None):
        """
        Initialize experiment logger.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of experiment
            benchmark_name: Name of benchmark function
            variant: REBMBO variant (classic, sparse, deep)
            seed: Random seed
            config: Configuration dictionary
        """
        self.benchmark_name = benchmark_name
        self.variant = variant
        self.seed = seed
        self.config = config or {}
        
        # Create directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{benchmark_name}_{variant}_seed{seed}_{timestamp}"
        self.log_dir = Path(log_dir) / benchmark_name / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.iteration_logger = Logger(self.log_dir, "iterations")
        self.metrics_logger = Logger(self.log_dir, "metrics")
        
        # Setup Python logger
        self.setup_python_logger()
        
        # Tracking variables
        self.iteration = 0
        self.best_y = float('-inf')
        self.start_time = time.time()
        
        # Store history
        self.history = {
            "x": [],
            "y": [],
            "best_y": [],
            "lar": [],
            "time": [],
            "energy": [],
            "gp_mean": [],
            "gp_std": [],
            "ppo_stats": []
        }
        
        # Log initial config
        self._log_config()
    
    def setup_python_logger(self):
        """Setup Python logging."""
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicate logs
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(self.log_dir / "experiment.log")
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _log_config(self):
        """Log experiment configuration."""
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self._serialize({
                "benchmark": self.benchmark_name,
                "variant": self.variant,
                "seed": self.seed,
                "config": self.config,
                "start_time": datetime.now().isoformat()
            }), f, indent=2)
    
    def _serialize(self, data) -> Any:
        """Serialize data for JSON, handling numpy and torch types."""
        if data is None:
            return None
        elif isinstance(data, dict):
            return {k: self._serialize(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize(x) for x in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        # Handle numpy scalar types
        elif isinstance(data, (np.float32, np.float64, np.floating)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64, np.integer)):
            return int(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, (int, float, str, bool)):
            return data
        else:
            try:
                return str(data)
            except:
                return None
    
    def log_iteration(self, 
                      x: np.ndarray,
                      y: float,
                      energy: float = None,
                      gp_mean: float = None,
                      gp_std: float = None,
                      lar: float = None,
                      ppo_stats: Dict = None,
                      extra: Dict = None):
        """
        Log a single optimization iteration.
        
        Args:
            x: Query point
            y: Function value
            energy: EBM energy at x
            gp_mean: GP mean at x
            gp_std: GP std at x
            lar: Landscape-Aware Regret
            ppo_stats: PPO training statistics
            extra: Additional data to log
        """
        self.iteration += 1
        elapsed = time.time() - self.start_time
        
        # Convert to Python native types
        y_float = float(y) if y is not None else None
        energy_float = float(energy) if energy is not None else None
        gp_mean_float = float(gp_mean) if gp_mean is not None else None
        gp_std_float = float(gp_std) if gp_std is not None else None
        lar_float = float(lar) if lar is not None else None
        
        # Update best
        if y_float is not None and y_float > self.best_y:
            self.best_y = y_float
        
        # Store in history
        self.history["x"].append(x.tolist() if isinstance(x, np.ndarray) else x)
        self.history["y"].append(y_float)
        self.history["best_y"].append(self.best_y)
        self.history["time"].append(elapsed)
        self.history["lar"].append(lar_float)
        self.history["energy"].append(energy_float)
        self.history["gp_mean"].append(gp_mean_float)
        self.history["gp_std"].append(gp_std_float)
        self.history["ppo_stats"].append(ppo_stats or {})
        
        # Create log entry
        entry = {
            "iteration": self.iteration,
            "x": x.tolist() if isinstance(x, np.ndarray) else x,
            "y": y_float,
            "best_y": self.best_y,
            "time": elapsed,
            "energy": energy_float,
            "gp_mean": gp_mean_float,
            "gp_std": gp_std_float,
            "lar": lar_float
        }
        
        if ppo_stats:
            entry["ppo"] = self._serialize(ppo_stats)
        if extra:
            entry.update(self._serialize(extra))
        
        self.iteration_logger.log(entry)
        
        # Log to Python logger
        self.logger.info(
            f"Iter {self.iteration}: y={y_float:.4f}, best_y={self.best_y:.4f}, "
            f"time={elapsed:.2f}s"
        )
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log evaluation metrics."""
        self.metrics_logger.log(self._serialize(metrics))
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def get_history(self) -> Dict:
        """Get complete history."""
        return self.history
    
    def save_final_results(self, results: Dict[str, Any]):
        """Save final experiment results."""
        results_file = self.log_dir / "final_results.json"
        
        final_results = {
            "benchmark": self.benchmark_name,
            "variant": self.variant,
            "seed": self.seed,
            "n_iterations": self.iteration,
            "best_y": self.best_y,
            "total_time": time.time() - self.start_time,
            "history": self.history,
            **results
        }
        
        with open(results_file, 'w') as f:
            json.dump(self._serialize(final_results), f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        
        return final_results


def load_experiment_logs(log_dir: str) -> Dict:
    """
    Load all logs from an experiment directory.
    
    Args:
        log_dir: Path to experiment log directory
    
    Returns:
        Dictionary with all logged data
    """
    log_dir = Path(log_dir)
    
    result = {}
    
    # Load config
    config_file = log_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            result["config"] = json.load(f)
    
    # Load iterations
    iterations_files = list(log_dir.glob("iterations_*.jsonl"))
    if iterations_files:
        iterations = []
        with open(iterations_files[0]) as f:
            for line in f:
                iterations.append(json.loads(line.strip()))
        result["iterations"] = iterations
    
    # Load final results
    results_file = log_dir / "final_results.json"
    if results_file.exists():
        with open(results_file) as f:
            result["final_results"] = json.load(f)
    
    return result


if __name__ == "__main__":
    # Test the logger
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test simple logger with numpy types
        logger = Logger(tmpdir, "test")
        
        # Test with numpy float32 (the problematic type)
        logger.log({
            "iteration": 1, 
            "value": np.float32(0.5),
            "int_val": np.int64(42),
            "array": np.array([1.0, 2.0, 3.0])
        })
        logger.log({
            "iteration": 2, 
            "value": np.float64(0.7),
            "nested": {"a": np.float32(1.0), "b": np.int32(2)}
        })
        print(f"Logged {len(logger.entries)} entries successfully")
        
        # Test experiment logger
        exp_logger = ExperimentLogger(
            log_dir=tmpdir,
            experiment_name="test",
            benchmark_name="ackley_5d",
            variant="classic",
            seed=42
        )
        
        for i in range(5):
            exp_logger.log_iteration(
                x=np.random.rand(5).astype(np.float32),
                y=np.float32(np.random.randn()),  # numpy float32
                energy=np.float64(np.random.randn()),  # numpy float64
                lar=np.float32(np.random.rand())
            )
        
        history = exp_logger.get_history()
        print(f"History length: {len(history['y'])}")
        
        results = exp_logger.save_final_results({"test": "value"})
        print(f"Best y: {results['best_y']:.4f}")
        
        print("\nAll tests passed! Logger correctly handles numpy types.")