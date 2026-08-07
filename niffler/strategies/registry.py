"""The single registry of trading strategies addressable by name.

Adding a strategy to Niffler is **one entry in :data:`STRATEGY_CLASSES`**. Every
command line entry point that takes ``--strategy`` derives its choices from this
module, so a newly registered strategy is immediately available to
``backtest.py``, ``optimize.py`` and ``analyze.py`` without touching any of them.

Two rules keep that promise true, and both are enforced by
``tests/test_strategies/test_registry.py``:

1. **The whole library constructs strategies as** ``strategy_class(**parameters)``
   (see :meth:`niffler.optimization.base_optimizer.BaseOptimizer._evaluate_parameters`,
   ``WalkForwardAnalyzer`` and ``MonteCarloAnalyzer``). Every parameter a strategy
   exposes must therefore be an ``__init__`` keyword argument with a default, and
   every strategy must accept ``position_size`` and ``risk_manager``.
2. **A strategy owns its own search space** through a ``PARAMETER_SPEC`` class
   attribute, whose keys must be a subset of those ``__init__`` keyword arguments.

This module deliberately does **not** import from :mod:`niffler.optimization`.
A ``PARAMETER_SPEC`` is a plain dict; the optimization layer wraps it in a
``ParameterSpace`` (see
:func:`niffler.optimization.optimizer_factory.get_parameter_space`). Importing
``ParameterSpace`` here would make :mod:`niffler.strategies` depend on
:mod:`niffler.optimization`, whose ``__init__`` imports ``optimizer_factory``,
which imports this module - a circular import. The dependency runs one way only:
optimization knows about strategies, never the reverse.
"""

import inspect
from typing import Any, Dict, List, Optional, Set, Type

from .base_strategy import BaseStrategy
from .breakout_strategy import BreakoutStrategy
from .rsi_strategy import RSIStrategy
from .simple_ma_strategy import SimpleMAStrategy


# The registry. Adding a strategy means adding a line here - nothing else.
STRATEGY_CLASSES: Dict[str, Type[BaseStrategy]] = {
    'simple_ma': SimpleMAStrategy,
    'rsi': RSIStrategy,
    'breakout': BreakoutStrategy,
}


def get_available_strategies() -> List[str]:
    """Return the registered strategy names, for CLI ``choices``.

    Returns:
        Strategy names in registration order.
    """
    return list(STRATEGY_CLASSES.keys())


def get_strategy_class(name: str) -> Type[BaseStrategy]:
    """Look up a strategy class by its registered name.

    Args:
        name: Registered strategy name, e.g. ``'simple_ma'``.

    Returns:
        The strategy class, not an instance.

    Raises:
        ValueError: If the name is not registered. The message lists what is.
    """
    if name not in STRATEGY_CLASSES:
        available = ', '.join(STRATEGY_CLASSES.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_CLASSES[name]


def get_parameter_spec(name: str) -> Dict[str, Dict[str, Any]]:
    """Return a strategy's declared optimisation search space as plain data.

    The spec is copied on the way out so a caller - or a ``ParameterSpace``
    built from it - cannot mutate the class attribute shared by every other
    caller in the process.

    Args:
        name: Registered strategy name.

    Returns:
        A mapping of parameter name to its ``{'type', 'min', 'max', 'step'}``
        configuration, in the form
        :class:`niffler.optimization.parameter_space.ParameterSpace` expects.

    Raises:
        ValueError: If the name is not registered, or the registered class
            declares no ``PARAMETER_SPEC``.
    """
    strategy_class = get_strategy_class(name)
    spec = getattr(strategy_class, 'PARAMETER_SPEC', None)
    if not spec:
        raise ValueError(
            f"Strategy '{name}' ({strategy_class.__name__}) declares no PARAMETER_SPEC, "
            f"so it cannot be optimised. Add one as a class attribute."
        )
    return {key: dict(config) for key, config in spec.items()}


def get_strategy_parameter_names(name: str) -> Set[str]:
    """Return the parameter names a registered strategy's constructor accepts.

    ``risk_manager`` is excluded: it is plumbing supplied by the caller, not a
    strategy parameter a user tunes or optimises. Callers use this to reject an
    unknown parameter with a message naming the strategy, rather than letting it
    surface as a bare ``TypeError``.

    Args:
        name: Registered strategy name.

    Returns:
        The accepted keyword argument names.

    Raises:
        ValueError: If the name is not registered.
    """
    strategy_class = get_strategy_class(name)
    signature = inspect.signature(strategy_class.__init__)
    return {
        parameter_name
        for parameter_name in signature.parameters
        if parameter_name not in ('self', 'risk_manager')
    }


def create_strategy(
    name: str,
    parameters: Optional[Dict[str, Any]] = None,
    risk_manager: Any = None,
) -> BaseStrategy:
    """Construct a registered strategy from a parameter dict.

    This is the generic construction path used by ``scripts/backtest.py``, so a
    new strategy needs no ``if`` branch there. Unknown parameter names surface as
    a ``ValueError`` naming the offending keys rather than an opaque ``TypeError``,
    because the parameters usually come from user-supplied JSON.

    Args:
        name: Registered strategy name.
        parameters: Keyword arguments for the strategy's ``__init__``.
        risk_manager: Optional risk manager forwarded to the strategy.

    Returns:
        The constructed strategy instance.

    Raises:
        ValueError: If the name is not registered, or the parameters are not
            accepted by the strategy's constructor.
    """
    strategy_class = get_strategy_class(name)
    kwargs = dict(parameters or {})

    try:
        return strategy_class(risk_manager=risk_manager, **kwargs)
    except TypeError as e:
        raise ValueError(
            f"Invalid parameters for strategy '{name}' ({strategy_class.__name__}): {e}"
        ) from e
