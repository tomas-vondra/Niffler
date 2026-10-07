"""What an exporter is handed for a run that is not a single backtest.

A backtest export carries a ``BacktestResult`` plus a metadata document. An
optimization, a walk-forward, a Monte Carlo run, a cross-asset comparison and a
screen have no such object in common - but each already produces one JSON
document. :class:`RunRecord` is that document together with the identity of the
run that produced it, so every exporter receives the same thing whichever
script is running.

The same record also carries the run in the shape a document store needs: one
``summary`` and its ``details`` rows, each stamped with one shared ``header``.
Elasticsearch has no joins, so the header is copied onto every document; that
copy is what lets a dashboard filter trials, folds and summaries by experiment
with a single clause.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.run_identity import RunIdentity

#: The kinds of detail row a run can carry. One tuple, so an exporter and a
#: mapping file never spell them differently.
DETAIL_TRIAL = 'trial'
DETAIL_FOLD = 'fold'
DETAIL_SIMULATION = 'simulation'
DETAIL_COMPARISON = 'comparison'

DETAIL_TYPES: Tuple[str, ...] = (
    DETAIL_TRIAL,
    DETAIL_FOLD,
    DETAIL_SIMULATION,
    DETAIL_COMPARISON,
)


@dataclass(frozen=True)
class RunRecord:
    """One run's exportable result.

    Built by :meth:`niffler.exporters.exporter_manager.ExporterManager.create_run_record`,
    which is the only place the ``run`` and ``provenance`` blocks and the shared
    header are assembled.

    Attributes:
        identity: The run's identity.
        strategy_key: Registry name of the strategy, or None for a run that
            spans several.
        document: The complete JSON-safe document, ``run`` and ``provenance``
            blocks included. This is what lands on disk.
        header: Fields every exported document of this run carries: identity,
            strategy key, symbol, engine settings, code and data fingerprint.
        summary: The run-level result, one document per run.
        details: Detail rows by type (see :data:`DETAIL_TYPES`): one trial,
            fold, simulation or comparison each.
        provenance: The full provenance record, or None. A run over several
            datasets has one record per file and carries them in ``document``
            only - each of its detail rows names its own data fingerprint.
    """

    identity: RunIdentity
    strategy_key: Optional[str]
    document: Dict[str, Any]
    header: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    provenance: Optional[Dict[str, Any]] = None
