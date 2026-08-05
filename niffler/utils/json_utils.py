"""
JSON Utilities

Shared helpers for producing standards-compliant JSON from backtest data.

``json.dump`` accepts ``Infinity``/``-Infinity``/``NaN`` literals by default, which are
*not* valid JSON (RFC 8259) and are rejected by Elasticsearch, most JavaScript
consumers and many strict parsers. Metrics such as ``profit_factor`` or
``sharpe_ratio`` legitimately become ``inf``/``nan`` for degenerate backtests, so every
writer of JSON in the project should sanitise its payload first.

Usage::

    from niffler.utils.json_utils import safe_json_dump, sanitize_numeric_values

    with open(path, "w") as f:
        safe_json_dump(payload, f, indent=2, default=str)
"""

import json
import math
from typing import Any, IO

import numpy as np

__all__ = [
    'sanitize_numeric_values',
    'safe_json_dump',
    'safe_json_dumps',
]


def _is_non_finite(value: Any) -> bool:
    """
    Check whether a scalar is a non-finite number (``inf``, ``-inf`` or ``NaN``).

    Args:
        value: Scalar value to inspect

    Returns:
        True if the value is a float-like value that is not finite
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))
    return False


def sanitize_numeric_values(data: Any) -> Any:
    """
    Recursively replace non-finite numbers with ``None`` and normalise numpy scalars.

    ``inf``/``-inf``/``NaN`` are replaced by ``None`` (JSON ``null``) and numpy
    integer/floating scalars are converted to their Python equivalents so they can be
    serialised without a custom encoder.

    Args:
        data: Arbitrary JSON-like structure (dict, list, tuple or scalar)

    Returns:
        A structure of the same shape with all non-finite numbers replaced by None
    """
    if isinstance(data, dict):
        return {key: sanitize_numeric_values(value) for key, value in data.items()}

    if isinstance(data, (list, tuple)):
        return [sanitize_numeric_values(item) for item in data]

    if _is_non_finite(data):
        return None

    if isinstance(data, np.floating):
        return float(data)

    if isinstance(data, np.integer):
        return int(data)

    if isinstance(data, np.bool_):
        return bool(data)

    return data


def safe_json_dumps(data: Any, **kwargs: Any) -> str:
    """
    Serialise data to a JSON string, guaranteeing standards-compliant output.

    Non-finite numbers are converted to ``null`` before serialisation and
    ``allow_nan=False`` is enforced so that any remaining non-finite value raises
    instead of silently emitting an invalid ``Infinity``/``NaN`` literal.

    Args:
        data: Data to serialise
        **kwargs: Additional keyword arguments forwarded to ``json.dumps``
            (``allow_nan`` is always forced to False)

    Returns:
        Valid JSON text

    Raises:
        ValueError: If a non-finite value survives sanitisation (e.g. inside a custom
            object serialised by a ``default`` hook)
    """
    kwargs['allow_nan'] = False
    return json.dumps(sanitize_numeric_values(data), **kwargs)


def safe_json_dump(data: Any, fp: IO[str], **kwargs: Any) -> None:
    """
    Write data to a file object as standards-compliant JSON.

    See :func:`safe_json_dumps` for the sanitisation rules.

    Args:
        data: Data to serialise
        fp: Writable text file object
        **kwargs: Additional keyword arguments forwarded to ``json.dump``
            (``allow_nan`` is always forced to False)

    Raises:
        ValueError: If a non-finite value survives sanitisation
    """
    kwargs['allow_nan'] = False
    json.dump(sanitize_numeric_values(data), fp, **kwargs)
