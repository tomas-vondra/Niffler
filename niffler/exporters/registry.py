"""The single registry of result exporters addressable by name.

Adding an exporter to Niffler is **one entry in :data:`EXPORTER_CLASSES`**. The
options it accepts are read off its ``__init__`` signature with
:func:`inspect.signature`, exactly as
:func:`niffler.strategies.registry.get_strategy_parameter_names` does, so there is
no hand-maintained per-exporter kwargs list to forget. Forgetting one used to be
silent: the exporter was constructed with defaults, the export "succeeded", and a
CSV file landed in the working directory instead of the ``--csv-output-dir`` the
user passed.

Two rules keep the promise true, and both are enforced by
``tests/test_exporters/test_registry.py``:

1. **Every constructor parameter has a default**, so ``create_exporter(name)``
   with no options works for every registered exporter.
2. **No exporter takes ``**kwargs``**, which would make the derived option set
   "everything" and put the silent-default bug straight back.

Metadata is deliberately *not* built here or in :class:`BaseExporter`: it is built
once by :meth:`niffler.exporters.exporter_manager.ExporterManager.create_metadata`
and handed to every exporter, so there is one document shape rather than one per
exporter.
"""

import inspect
from typing import Any, Dict, List, Optional, Set, Type

from .base_exporter import BaseExporter
from .console_exporter import ConsoleExporter
from .csv_exporter import CSVExporter
from .elasticsearch_exporter import ElasticsearchExporter


# The registry. Adding an exporter means adding a line here - nothing else.
EXPORTER_CLASSES: Dict[str, Type[BaseExporter]] = {
    'console': ConsoleExporter,
    'csv': CSVExporter,
    'elasticsearch': ElasticsearchExporter,
}


def get_available_exporters() -> List[str]:
    """Return the registered exporter names, for CLI ``choices`` and help text.

    Returns:
        Exporter names in registration order.
    """
    return list(EXPORTER_CLASSES.keys())


def get_exporter_class(name: str) -> Type[BaseExporter]:
    """Look up an exporter class by its registered name.

    Args:
        name: Registered exporter name; surrounding whitespace and case are
            ignored, because the names arrive from a comma-separated CLI list.

    Returns:
        The exporter class, not an instance.

    Raises:
        ValueError: If the name is not registered. The message lists what is.
    """
    normalized = name.strip().lower()
    if normalized not in EXPORTER_CLASSES:
        available = ', '.join(EXPORTER_CLASSES.keys())
        raise ValueError(f"Unknown exporter type: {normalized}. Available types: {available}")
    return EXPORTER_CLASSES[normalized]


def get_exporter_option_names(name: str) -> Set[str]:
    """Return the constructor option names a registered exporter accepts.

    Derived from the signature rather than listed by hand, so a new exporter's
    options are picked up by every caller the moment it is registered.

    Args:
        name: Registered exporter name.

    Returns:
        The accepted keyword argument names.

    Raises:
        ValueError: If the name is not registered.
    """
    exporter_class = get_exporter_class(name)
    signature = inspect.signature(exporter_class.__init__)
    return {
        parameter_name
        for parameter_name, parameter in signature.parameters.items()
        if parameter_name != 'self'
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }


def create_exporter(name: str, options: Optional[Dict[str, Any]] = None) -> BaseExporter:
    """Construct a registered exporter from an option dict.

    This is the generic construction path, so a new exporter needs no ``if``
    branch anywhere. An option the exporter does not accept is an error naming
    what it does accept - the same rule ``--params`` follows for strategies -
    rather than being dropped, which would run the exporter on defaults while the
    user believes it was configured.

    Args:
        name: Registered exporter name.
        options: Keyword arguments for the exporter's ``__init__``.

    Returns:
        The constructed exporter instance.

    Raises:
        ValueError: If the name is not registered, or an option is not accepted
            by the exporter's constructor.
    """
    exporter_class = get_exporter_class(name)
    kwargs = dict(options or {})

    accepted = get_exporter_option_names(name)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"Exporter '{name.strip().lower()}' ({exporter_class.__name__}) does not "
            f"accept: {', '.join(unknown)}. It accepts: {', '.join(sorted(accepted))}"
        )

    return exporter_class(**kwargs)
