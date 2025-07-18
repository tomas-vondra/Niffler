from typing import Dict, Type, List, Optional
import pandas as pd
from niffler.strategies.base_strategy import BaseStrategy
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
from .parameter_space import ParameterSpace, SIMPLE_MA_PARAMETER_SPACE
from .base_optimizer import BaseOptimizer
from .grid_search_optimizer import GridSearchOptimizer
from .random_search_optimizer import RandomSearchOptimizer
from .optimization_result import OptimizationResult


# Optimizer registry
OPTIMIZER_CLASSES = {
    'grid': GridSearchOptimizer,
    'random': RandomSearchOptimizer
}


def create_optimizer(
    method: str,
    strategy_class: Type[BaseStrategy],
    parameter_space: ParameterSpace,
    data: pd.DataFrame,
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    sort_by: str = 'total_return',
    n_jobs: Optional[int] = None
) -> BaseOptimizer:
    """
    Create an optimizer instance based on the method name.
    
    Args:
        method: Optimization method name ('grid', 'random')
        strategy_class: Strategy class to optimize
        parameter_space: Parameter search space
        data: Historical price data for backtesting
        initial_capital: Starting capital for backtests (default: 10000.0)
        commission: Commission rate for trades (default: 0.001)
        sort_by: Metric to sort results by (default: 'total_return')
        n_jobs: Number of parallel jobs (default: auto-detect)
        
    Returns:
        Optimizer instance
        
    Raises:
        ValueError: If method is not supported
    """
    if method not in OPTIMIZER_CLASSES:
        available = ', '.join(OPTIMIZER_CLASSES.keys())
        raise ValueError(f"Unknown optimization method '{method}'. Available: {available}")
    
    return OPTIMIZER_CLASSES[method](
        strategy_class=strategy_class,
        parameter_space=parameter_space,
        data=data,
        initial_capital=initial_capital,
        commission=commission,
        sort_by=sort_by,
        n_jobs=n_jobs
    )


def get_available_optimizers() -> List[str]:
    """Get list of available optimizer methods."""
    return list(OPTIMIZER_CLASSES.keys())


# Strategy class mapping for CLI
STRATEGY_CLASSES = {
    'simple_ma': SimpleMAStrategy
}

# Parameter space mapping for CLI  
PARAMETER_SPACES = {
    'simple_ma': SIMPLE_MA_PARAMETER_SPACE
}


def get_strategy_class(name: str) -> Type[BaseStrategy]:
    """Get strategy class by name."""
    if name not in STRATEGY_CLASSES:
        available = ', '.join(STRATEGY_CLASSES.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_CLASSES[name]


def get_parameter_space(name: str) -> ParameterSpace:
    """Get parameter space by strategy name."""
    if name not in PARAMETER_SPACES:
        available = ', '.join(PARAMETER_SPACES.keys())
        raise ValueError(f"No parameter space defined for strategy '{name}'. Available: {available}")
    return PARAMETER_SPACES[name]