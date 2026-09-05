"""The single registry of risk managers addressable by name.

The counterpart of :mod:`niffler.strategies.registry`, and it exists for the same
reason. ``backtest.py`` used to hardcode ``choices=['none', 'fixed']`` and then
construct the manager from an ``if`` chain, two lines above a comment explaining
that *strategy* construction is generic precisely so that no script needs such a
chain. Two extension points sat next to each other in one function; one had been
fixed and the other had not. ``KellyRiskManager`` existed as a class the whole
time and the string ``kelly`` appeared in no script - finishing it would not have
made it selectable.

Adding a risk manager is now **one entry in** :data:`RISK_MANAGER_CLASSES`.

``'none'`` is not an entry. The absence of a risk manager is ``None``, not a
do-nothing manager: a null object would make "no risk management" and "a risk
manager that permits everything" indistinguishable in the engine, and the engine
branches on ``risk_manager is None`` to skip stop processing entirely.
:func:`create_risk_manager` therefore accepts ``'none'`` and returns ``None`` -
the one name that maps to no class.
"""

import inspect
from typing import Any, Dict, List, Optional, Set, Type

from .base_risk_manager import BaseRiskManager
from .fixed_risk_manager import FixedRiskManager


#: The name reserved for "run without risk management".
NO_RISK_MANAGER = 'none'

# The registry. Adding a risk manager means adding a line here - nothing else.
RISK_MANAGER_CLASSES: Dict[str, Type[BaseRiskManager]] = {
    'fixed': FixedRiskManager,
}


def get_available_risk_managers() -> List[str]:
    """Return the selectable names, for CLI ``choices``.

    ``'none'`` leads, because it is the default and it is a real choice rather
    than the absence of one.

    Returns:
        ``'none'`` followed by the registered names in registration order.
    """
    return [NO_RISK_MANAGER] + list(RISK_MANAGER_CLASSES.keys())


def get_risk_manager_class(name: str) -> Type[BaseRiskManager]:
    """Look up a risk manager class by its registered name.

    Args:
        name: Registered risk manager name, e.g. ``'fixed'``.

    Returns:
        The risk manager class, not an instance.

    Raises:
        ValueError: If the name is not registered. The message lists what is.
            ``'none'`` is rejected here too - it maps to no class, so a caller
            asking for its class is asking the wrong question.
    """
    if name not in RISK_MANAGER_CLASSES:
        available = ', '.join(RISK_MANAGER_CLASSES.keys())
        raise ValueError(f"Unknown risk manager '{name}'. Available: {available}")
    return RISK_MANAGER_CLASSES[name]


def get_risk_manager_parameter_names(name: str) -> Set[str]:
    """Return the parameter names a registered manager's constructor accepts.

    Args:
        name: Registered risk manager name.

    Returns:
        The accepted keyword argument names.

    Raises:
        ValueError: If the name is not registered.
    """
    manager_class = get_risk_manager_class(name)
    signature = inspect.signature(manager_class.__init__)
    return {
        parameter_name
        for parameter_name in signature.parameters
        if parameter_name != 'self'
    }


def create_risk_manager(
    name: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Optional[BaseRiskManager]:
    """Construct a registered risk manager from a parameter dict.

    The generic construction path, so a newly registered manager needs no ``if``
    branch in any script. An argument the chosen manager does not accept is an
    error naming the ones it does, rather than a bare ``TypeError`` or - worse -
    a silently ignored setting that leaves the user believing the run was
    configured.

    Args:
        name: Registered risk manager name, or ``'none'``.
        parameters: Keyword arguments for the manager's ``__init__``.

    Returns:
        The constructed manager, or ``None`` for ``'none'``.

    Raises:
        ValueError: If the name is not registered, or a parameter is not
            accepted by the manager's constructor.
    """
    if name == NO_RISK_MANAGER:
        return None

    manager_class = get_risk_manager_class(name)
    kwargs = dict(parameters or {})

    accepted = get_risk_manager_parameter_names(name)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"Risk manager '{name}' ({manager_class.__name__}) does not accept "
            f"{', '.join(unknown)}. Accepted: {', '.join(sorted(accepted))}"
        )

    return manager_class(**kwargs)
