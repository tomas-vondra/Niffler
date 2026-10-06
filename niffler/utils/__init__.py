"""
Niffler Utilities Package

Layer-neutral helpers shared by the backtesting, optimization and export layers.
Nothing in here may import from those layers, so importing a helper never drags an
optional third-party dependency (such as the Elasticsearch client) along with it.
"""

from .json_utils import safe_json_dump, safe_json_dumps, sanitize_numeric_values
from .provenance import collect_provenance, format_provenance_summary
from .run_identity import (
    RUN_KINDS,
    ExperimentMismatchError,
    RunIdentity,
    format_run_identity,
    mint_run_id,
    new_run_identity,
    resolve_experiment,
)

__all__ = [
    'RUN_KINDS',
    'ExperimentMismatchError',
    'RunIdentity',
    'collect_provenance',
    'format_provenance_summary',
    'format_run_identity',
    'mint_run_id',
    'new_run_identity',
    'resolve_experiment',
    'safe_json_dump',
    'safe_json_dumps',
    'sanitize_numeric_values',
]
