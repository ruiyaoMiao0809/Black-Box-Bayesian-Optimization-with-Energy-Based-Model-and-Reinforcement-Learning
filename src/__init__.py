"""
REBMBO: Reinforced Energy-Based Model for Bayesian Optimization

Main package providing:
- rebmbo: Core algorithm modules
- utils: Utility functions
- benchmarks: Benchmark functions
- baselines: Baseline methods for comparison
"""

from .benchmarks import get_benchmark, list_benchmarks, BENCHMARKS

__version__ = "1.0.0"
__author__ = "REBMBO Authors"

__all__ = [
    'get_benchmark',
    'list_benchmarks', 
    'BENCHMARKS',
]