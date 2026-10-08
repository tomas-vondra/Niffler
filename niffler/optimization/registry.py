"""The single registry of optimizers addressable by name.

The counterpart of :mod:`niffler.strategies.registry`, :mod:`niffler.risk.registry`
and :mod:`niffler.exporters.registry`, and the last of the four to take this
shape. The name-to-class map used to live in ``optimizer_factory.py`` beside an
unrelated helper, so "where do I add an optimizer" had a different answer than
"where do I add a strategy". Every CLI derives its ``--optimization-method``
choices from :func:`get_available_optimizers`.

Adding an optimizer is now **one entry in** :data:`OPTIMIZER_CLASSES`.
"""

from typing import Dict, List, Optional, Type

import pandas as pd

from niffler.backtesting.run_config import RunConfig
from niffler.strategies.base_strategy import BaseStrategy

from .base_optimizer import BaseOptimizer
from .grid_search_optimizer import GridSearchOptimizer
from .parameter_space import ParameterSpace
from .random_search_optimizer import RandomSearchOptimizer


# The registry. Adding an optimizer means adding a line here - nothing else.
OPTIMIZER_CLASSES: Dict[str, Type[BaseOptimizer]] = {
    'grid': GridSearchOptimizer,
    'random': RandomSearchOptimizer,
}


def get_available_optimizers() -> List[str]:
    """Return the registered optimizer names, for CLI ``choices``.

    Returns:
        The names in registration order.
    """
    return list(OPTIMIZER_CLASSES.keys())


def get_optimizer_class(name: str) -> Type[BaseOptimizer]:
    """Look up an optimizer class by its registered name.

    Args:
        name: Registered optimizer name, e.g. ``'grid'``.

    Returns:
        The optimizer class, not an instance.

    Raises:
        ValueError: If the name is not registered. The message lists what is.
    """
    if name not in OPTIMIZER_CLASSES:
        available = ', '.join(OPTIMIZER_CLASSES.keys())
        raise ValueError(f"Unknown optimization method '{name}'. Available: {available}")
    return OPTIMIZER_CLASSES[name]


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
        method: Registered optimizer name ('grid', 'random')
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
        ValueError: If method is not registered
    """
    return get_optimizer_class(method)(
        strategy_class=strategy_class,
        parameter_space=parameter_space,
        data=data,
        sort_by=sort_by,
        n_jobs=n_jobs,
        run_config=run_config,
        max_results_in_memory=max_results_in_memory
    )
