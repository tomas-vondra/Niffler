from typing import Dict, Type, List, Optional
import pandas as pd
from niffler.backtesting.run_config import RunConfig
from niffler.strategies.base_strategy import BaseStrategy
from niffler.strategies.registry import get_parameter_spec
from .parameter_space import ParameterSpace
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
    sort_by: str = 'total_return',
    n_jobs: Optional[int] = None,
    run_config: Optional[RunConfig] = None,
    max_results_in_memory: Optional[int] = None
) -> BaseOptimizer:
    """
    Create an optimizer instance based on the method name.

    Args:
        method: Optimization method name ('grid', 'random')
        strategy_class: Strategy class to optimize
        parameter_space: Parameter search space
        data: Historical price data for backtesting
        sort_by: Metric to sort results by (default: 'total_return')
        n_jobs: Number of parallel jobs (default: auto-detect)
        run_config: Engine settings every candidate backtest runs under
            (default: None, i.e. :class:`RunConfig`'s defaults, which are the
            engine's own). It carries capital, commission, cost model,
            benchmark, annualisation and the significance gate together, so a
            caller cannot configure some of them and lose the rest
        max_results_in_memory: Results retained before the worst-scoring half is
            discarded (default: None, i.e. the optimizer's own cap). Raise it to
            keep every combination, which whole-grid statistics such as
            :mod:`niffler.optimization.plateau` require

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
        sort_by=sort_by,
        n_jobs=n_jobs,
        run_config=run_config,
        max_results_in_memory=max_results_in_memory
    )


def get_available_optimizers() -> List[str]:
    """Get list of available optimizer methods."""
    return list(OPTIMIZER_CLASSES.keys())


def get_parameter_space(name: str) -> ParameterSpace:
    """Build the optimisation search space for a registered strategy.

    The space itself is declared by the strategy as a plain dict
    (``PARAMETER_SPEC``) and lives in :mod:`niffler.strategies.registry`; this
    function only wraps it in the optimization layer's ``ParameterSpace``. That
    split is why :mod:`niffler.strategies` never has to import
    :mod:`niffler.optimization`.

    The strategy name to class lookup itself is *not* here - import
    ``get_strategy_class`` from :mod:`niffler.strategies.registry`, which is the
    single registry every CLI derives its ``--strategy`` choices from.

    Args:
        name: Registered strategy name, e.g. ``'simple_ma'``.

    Returns:
        A validated ParameterSpace for that strategy.

    Raises:
        ValueError: If the strategy is not registered or declares no
            PARAMETER_SPEC.
    """
    return ParameterSpace(get_parameter_spec(name))