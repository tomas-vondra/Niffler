"""
Niffler Optimization Package

This package provides parameter optimization capabilities for trading strategies.
It includes grid search, random search, and other optimization algorithms.

Main Components:
- BaseOptimizer: Abstract base class for all optimizers
- GridSearchOptimizer: Exhaustive grid search optimization
- RandomSearchOptimizer: Random sampling optimization with smart estimation
- ParameterSpace: Parameter space definition and validation
- OptimizationResult: Results container

Usage:
    from niffler.optimization import GridSearchOptimizer, RandomSearchOptimizer, ParameterSpace
    
    # Define parameter space
    param_space = ParameterSpace({
        'param1': {'type': 'int', 'min': 1, 'max': 10},
        'param2': {'type': 'float', 'min': 0.1, 'max': 1.0, 'step': 0.1}
    })
    
    # Create optimizer
    optimizer = GridSearchOptimizer(
        strategy_class=YourStrategy,
        parameter_space=param_space,
        data=your_data
    )
    
    # Run optimization
    results = optimizer.optimize()
"""

from .base_optimizer import BaseOptimizer
from .grid_search_optimizer import GridSearchOptimizer
from .random_search_optimizer import RandomSearchOptimizer
from .parameter_space import ParameterSpace
from .optimization_result import OptimizationResult
from .optimizer_factory import create_optimizer

__all__ = [
    'BaseOptimizer',
    'GridSearchOptimizer', 
    'RandomSearchOptimizer',
    'ParameterSpace',
    'OptimizationResult',
    'create_optimizer'
]

__version__ = '1.0.0'