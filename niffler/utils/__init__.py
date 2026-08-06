"""
Niffler Utilities Package

Layer-neutral helpers shared by the backtesting, optimization and export layers.
Nothing in here may import from those layers, so importing a helper never drags an
optional third-party dependency (such as the Elasticsearch client) along with it.
"""

from .json_utils import safe_json_dump, safe_json_dumps, sanitize_numeric_values
from .provenance import collect_provenance, format_provenance_summary

__all__ = [
    'collect_provenance',
    'format_provenance_summary',
    'safe_json_dump',
    'safe_json_dumps',
    'sanitize_numeric_values',
]
