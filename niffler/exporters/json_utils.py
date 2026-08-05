"""
JSON Utilities (compatibility re-export)

The helpers moved to :mod:`niffler.utils.json_utils` because they are generic
stdlib+numpy code with nothing export-specific in them, and importing them from
here forced every consumer - the optimizer among them - to execute
``niffler/exporters/__init__.py`` and with it the optional Elasticsearch client.

This module stays as a thin re-export so existing imports keep working. New code
should import from :mod:`niffler.utils.json_utils`.
"""

from ..utils.json_utils import (
    safe_json_dump,
    safe_json_dumps,
    sanitize_numeric_values,
)

__all__ = [
    'sanitize_numeric_values',
    'safe_json_dump',
    'safe_json_dumps',
]
