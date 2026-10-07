"""What an exporter is handed for a run that is not a single backtest.

A backtest export carries a ``BacktestResult`` plus a metadata document. An
optimization, a walk-forward, a Monte Carlo run, a cross-asset comparison and a
screen have no such object in common - but each already produces one JSON
document. :class:`RunRecord` is that document together with the identity of the
run that produced it, so every exporter receives the same thing whichever
script is running.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..utils.run_identity import RunIdentity


@dataclass(frozen=True)
class RunRecord:
    """One run's exportable result.

    Built by :meth:`niffler.exporters.exporter_manager.ExporterManager.create_run_record`,
    which is the only place the ``run`` and ``provenance`` blocks are attached.

    Attributes:
        identity: The run's identity.
        strategy_key: Registry name of the strategy, or None for a run that
            spans several.
        document: The complete JSON-safe document, ``run`` and ``provenance``
            blocks included. This is what lands on disk.
    """

    identity: RunIdentity
    strategy_key: Optional[str]
    document: Dict[str, Any]
