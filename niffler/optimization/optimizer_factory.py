"""Backwards-compatible home of the optimizer entry points.

The name-to-class map and the functions that read it live in
:mod:`niffler.optimization.registry`; they are re-exported here so existing
imports keep working. :func:`get_parameter_space` stays: it wraps a strategy's
declared search space and has nothing to do with which optimizer runs.
"""

from niffler.strategies.registry import get_parameter_spec
from .parameter_space import ParameterSpace
from .registry import (
    OPTIMIZER_CLASSES,
    create_optimizer,
    get_available_optimizers,
    get_optimizer_class,
)

__all__ = [
    'OPTIMIZER_CLASSES',
    'create_optimizer',
    'get_available_optimizers',
    'get_optimizer_class',
    'get_parameter_space',
]


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
