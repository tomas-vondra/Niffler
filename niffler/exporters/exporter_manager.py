"""
Exporter Manager

Coordinates multiple exporters for backtesting results with unique identification.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Set, Tuple
from .base_exporter import BaseExporter
from .registry import (
    create_exporter,
    get_available_exporters,
    get_exporter_class,
    get_exporter_option_names,
    get_exporters_supporting,
)
from .run_record import DETAIL_TYPES, RunRecord
from ..backtesting.backtest_result import BacktestResult
from ..utils.run_identity import RUN_KIND_BACKTEST, RunIdentity, new_run_identity

logger = logging.getLogger(__name__)


def _is_single_provenance(provenance: Optional[Dict[str, Any]]) -> bool:
    """True for one provenance record, False for one-per-dataset keyed by path."""
    return isinstance(provenance, dict) and (
        'code' in provenance or 'data' in provenance or 'environment' in provenance)


@dataclass
class ExportSummary:
    """
    Outcome of a multi-exporter export run.

    Attributes:
        successes: Class names of the exporters that completed successfully
        failures: (exporter class name, error message) pairs for exporters that failed
        run_id: The run ID that was used for this export run
    """

    successes: List[str]
    failures: List[Tuple[str, str]]
    run_id: str

    @property
    def ok(self) -> bool:
        """True when every configured exporter succeeded."""
        return not self.failures


class ExporterManager:
    """Manages multiple exporters for backtesting results.

    The name→class map lives in :mod:`niffler.exporters.registry`, not here: a
    second map in the manager is exactly the shadowing that made ``analyze.py``
    reject strategies ``optimize.py`` accepted.
    """

    def __init__(self):
        """Initialize the exporter manager."""
        self.exporters: List[BaseExporter] = []
        # (requested name, error) for exporters that could not even be constructed.
        # Carried into every ExportSummary so a misconfigured exporter cannot be
        # mistaken for one that ran.
        self.creation_failures: List[Tuple[str, str]] = []
    
    @classmethod
    def get_available_exporter_names(cls) -> List[str]:
        """Get the names of all available exporter types."""
        return get_available_exporters()

    def add_exporter(self, exporter: BaseExporter) -> None:
        """
        Add an exporter to the manager.
        
        Args:
            exporter: Exporter instance to add
        """
        self.exporters.append(exporter)
    
    def create_exporter_by_name(self, name: str, **kwargs) -> BaseExporter:
        """Create and add a single exporter by name.

        Options are matched against the exporter's constructor signature by
        :func:`niffler.exporters.registry.create_exporter`, so an option this
        exporter does not accept raises instead of being dropped.

        Args:
            name: Registered exporter name
            **kwargs: Constructor options for that exporter

        Returns:
            The constructed exporter, already added to this manager

        Raises:
            ValueError: If the name is not registered, or an option is not
                accepted by the exporter
        """
        exporter = create_exporter(name, kwargs)
        self.add_exporter(exporter)
        return exporter

    def create_exporters_from_list(self, exporter_names: List[str],
                                   kind: Optional[str] = None,
                                   **kwargs) -> List[Tuple[str, str]]:
        """
        Create multiple exporters from a list of names, broadcasting the options.

        The options are a single pool shared by every requested exporter, and each
        exporter is handed the subset its constructor declares. The subset is read
        off the signature, so registering an exporter is the only edit its options
        need. An option **no** requested exporter accepts is a caller error and
        raises: silently dropping it is how ``--csv-output-dir`` used to be ignored
        while the run still reported success.

        An exporter whose constructor rejects its configuration (an unknown name, an
        invalid Elasticsearch scheme, an unparsable port) is recorded rather than
        quietly skipped: the failures are returned, kept on ``creation_failures`` and
        folded into every subsequent :class:`ExportSummary`, so a run that was asked
        for an exporter it never built cannot report success.

        Args:
            exporter_names: Names of the exporter types to create
            kind: The kind of run about to be exported (see
                :data:`niffler.utils.run_identity.RUN_KINDS`). When given, an
                exporter that does not support it is a caller error and raises
                here - which callers do before the computation, so an unusable
                exporter is not discovered after the work is done
            **kwargs: Configuration pool broadcast to the exporter constructors

        Returns:
            List of (requested name, error message) pairs for the exporters that
            could not be created; empty when every requested exporter was built

        Raises:
            ValueError: If an option is accepted by none of the requested exporters,
                or a requested exporter does not support ``kind``
        """
        accepted_by: Dict[str, Set[str]] = {}
        unsupported: List[str] = []
        for name in exporter_names:
            try:
                exporter_class = get_exporter_class(name)
            except ValueError:
                continue  # Unknown names are reported per-exporter below.
            accepted_by[name] = get_exporter_option_names(name)
            if kind is not None and kind not in exporter_class.SUPPORTED_KINDS:
                unsupported.append(name)

        if unsupported:
            supporting = ', '.join(get_exporters_supporting(kind)) or 'none'
            raise ValueError(
                f"Exporter(s) {', '.join(unsupported)} cannot export a {kind} run. "
                f"Exporters that can: {supporting}"
            )

        if kwargs and accepted_by:
            self._reject_orphan_options(set(kwargs), accepted_by)

        failures: List[Tuple[str, str]] = []
        for name in exporter_names:
            accepted = accepted_by.get(name)
            options = kwargs if accepted is None else {
                key: value for key, value in kwargs.items() if key in accepted
            }
            try:
                self.create_exporter_by_name(name, **options)
            except Exception as e:
                logger.error(f"Could not create exporter '{name}': {e}")
                failures.append((name, str(e)))

        self.creation_failures.extend(failures)
        return failures

    @staticmethod
    def _reject_orphan_options(supplied: Set[str], accepted_by: Dict[str, Set[str]]) -> None:
        """Raise when an option is accepted by none of the requested exporters.

        Args:
            supplied: Option names the caller passed
            accepted_by: Requested exporter name to the options it accepts

        Raises:
            ValueError: Naming the orphaned options and what each exporter accepts
        """
        union = set().union(*accepted_by.values())
        orphans = sorted(supplied - union)
        if not orphans:
            return

        rendered = '; '.join(
            f"{name} accepts {', '.join(sorted(options))}"
            for name, options in accepted_by.items()
        )
        raise ValueError(
            f"No requested exporter accepts: {', '.join(orphans)}. {rendered}"
        )


    def export_backtest_result(self, result: BacktestResult, strategy_params: Dict[str, Any],
                              symbol: str, initial_capital: float, commission: float,
                              provenance: Optional[Dict[str, Any]] = None,
                              cost_model: str = None,
                              risk_manager: Optional[Dict[str, Any]] = None,
                              identity: Optional[RunIdentity] = None,
                              strategy_key: Optional[str] = None) -> ExportSummary:
        """
        Export backtest results using all configured exporters.

        Each exporter is isolated: a failing exporter never prevents the others from
        running, but every failure is recorded in the returned summary so the caller
        can report it (or exit non-zero) instead of silently losing data. Exporters
        that could not be constructed at all (see :meth:`create_exporters_from_list`)
        are reported as failures too.

        Args:
            result: BacktestResult object containing all backtest data
            strategy_params: Strategy parameters used in the backtest
            symbol: Trading symbol
            initial_capital: Initial capital amount
            commission: Commission rate
            provenance: Optional run provenance record (see
                :func:`niffler.utils.provenance.collect_provenance`). It is collected
                **once** by the caller that owns the run and shared by every exporter:
                collecting it here per exporter would re-hash the input data file for
                each destination
            cost_model: Description of the transaction cost model in force, so the
                export says which market assumption produced these numbers
            risk_manager: The run's risk configuration as
                :func:`niffler.risk.registry.describe_risk_manager` renders it, so
                the export says which position sizing and stops produced these
                numbers rather than leaving "no risk management" indistinguishable
                from "risk management not recorded"
            identity: The run's identity, minted once by the caller that owns the
                run (see :mod:`niffler.utils.run_identity`). A library caller that
                passes none gets a fresh, unnamed backtest identity
            strategy_key: Registry name of the strategy (``'rsi'``), recorded
                beside the display name so grouping does not key on prose

        Returns:
            ExportSummary describing which exporters succeeded, which failed and the
            run ID that was used
        """
        if identity is None:
            identity = new_run_identity(RUN_KIND_BACKTEST)
        run_id = identity.run_id

        # Create metadata
        metadata = self.create_metadata(
            result, strategy_params, symbol, initial_capital, commission, provenance,
            cost_model, risk_manager, identity=identity, strategy_key=strategy_key
        )

        return self._export_with_all(
            run_id,
            lambda exporter: exporter.export_backtest_result(result, run_id, metadata)
        )

    def export_run(self, record: RunRecord) -> ExportSummary:
        """
        Export a run that is not a single backtest using all configured exporters.

        The same isolation and failure accounting as
        :meth:`export_backtest_result`: one exporter failing never stops the
        others, and every failure is in the returned summary.

        Args:
            record: The run to export, built by :meth:`create_run_record`

        Returns:
            ExportSummary describing which exporters succeeded and which failed
        """
        return self._export_with_all(
            record.identity.run_id, lambda exporter: exporter.export_run(record)
        )

    def _export_with_all(self, run_id: str,
                         export: Callable[[BaseExporter], None]) -> ExportSummary:
        """Run one export call against every exporter, recording each outcome."""
        successes: List[str] = []
        # Exporters that never got constructed failed just as surely as ones that
        # raised while exporting - both mean the data did not reach their sink.
        failures: List[Tuple[str, str]] = list(self.creation_failures)

        for exporter in self.exporters:
            exporter_name = exporter.__class__.__name__
            try:
                export(exporter)
                successes.append(exporter_name)
            except Exception as e:
                # Continue with other exporters even if one fails, but record the failure
                exporter_logger = getattr(exporter, 'logger', None) or logger
                exporter_logger.error(f"Export failed for {exporter_name}: {e}")
                failures.append((exporter_name, str(e)))

        summary = ExportSummary(
            successes=successes, failures=failures, run_id=run_id
        )

        if not summary.ok:
            requested = len(self.exporters) + len(self.creation_failures)
            logger.error(
                f"{len(failures)} of {requested} exporter(s) failed for "
                f"run {run_id}: "
                f"{', '.join(name for name, _ in failures)}"
            )

        return summary

    @staticmethod
    def create_run_record(identity: RunIdentity, strategy_key: Optional[str],
                          body: Dict[str, Any],
                          provenance: Optional[Dict[str, Any]] = None,
                          settings: Optional[Dict[str, Any]] = None,
                          symbol: Optional[str] = None,
                          summary: Optional[Dict[str, Any]] = None,
                          details: Optional[Dict[str, List[Dict[str, Any]]]] = None
                          ) -> RunRecord:
        """
        Build the record of a run that is not a single backtest.

        The counterpart of :meth:`create_metadata`, and likewise the only builder:
        the ``run`` and ``provenance`` blocks are attached here and nowhere else,
        so every saved document names its run the same way. The registry key is
        recorded beside the identity because ``strategy_name`` is the display
        string - a grouping keyed on prose splits a strategy's history in two the
        day its display name is edited.

        Args:
            identity: The run's identity, minted once by the caller that owns the run
            strategy_key: Registry name of the strategy, or None for a run that
                spans several
            body: The script's own result document
            provenance: Optional run provenance record, collected once by the
                caller. A run over several datasets passes one record per file,
                keyed by path
            settings: The engine settings the run used
                (``RunConfig.to_metadata()``), copied onto every exported document
            symbol: The instrument, or None for a run over several
            summary: The run-level result, for exporters that store one document
                per run
            details: Detail rows by type (see
                :data:`niffler.exporters.run_record.DETAIL_TYPES`)

        Returns:
            The record handed to every exporter

        Raises:
            ValueError: If ``details`` names a type that does not exist
        """
        details = dict(details or {})
        unknown = sorted(set(details) - set(DETAIL_TYPES))
        if unknown:
            raise ValueError(
                f"Unknown detail type(s): {', '.join(unknown)}. "
                f"Types: {', '.join(DETAIL_TYPES)}"
            )

        run_block = {**identity.to_metadata(), 'strategy_key': strategy_key}

        document = dict(body)
        if provenance is not None:
            document['provenance'] = provenance
        document['run'] = run_block

        # A single record has 'code' and 'data' blocks. One record per dataset is
        # keyed by path instead: the code is the same for all of them, and no
        # single data fingerprint describes the run - each detail row names its own.
        single = provenance if _is_single_provenance(provenance) else None
        any_record = single or next(iter((provenance or {}).values()), None)
        if not isinstance(any_record, dict):
            any_record = {}
        code = any_record.get('code') or {}
        data = (single or {}).get('data') or {}

        header = {
            **run_block,
            'symbol': symbol,
            **(settings or {}),
            'git_sha': code.get('git_sha'),
            # None when it could not be determined: False would assert a
            # cleanliness nobody checked.
            'git_dirty': code.get('dirty'),
            'data_sha256': data.get('sha256'),
        }

        return RunRecord(
            identity=identity,
            strategy_key=strategy_key,
            document=document,
            header=header,
            summary=dict(summary or {}),
            details=details,
            provenance=single,
        )

    def create_metadata(self, result: BacktestResult, strategy_params: Dict[str, Any],
                        symbol: str, initial_capital: float, commission: float,
                        provenance: Optional[Dict[str, Any]] = None,
                        cost_model: str = None,
                        risk_manager: Optional[Dict[str, Any]] = None,
                        identity: Optional[RunIdentity] = None,
                        strategy_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Create standardized metadata for a backtest.

        Args:
            result: BacktestResult object
            strategy_params: Strategy parameters used in the backtest
            symbol: Trading symbol
            initial_capital: Initial capital amount
            commission: Commission rate
            provenance: Optional run provenance record, included under a
                ``provenance`` key when supplied
            cost_model: Description of the transaction cost model in force
            risk_manager: The run's risk configuration, as
                :func:`niffler.risk.registry.describe_risk_manager` renders it
            identity: The run's identity; its fields are included when supplied
            strategy_key: Registry name of the strategy

        Returns:
            Dictionary containing standardized metadata
        """
        metadata = {
            # Identity first: run_id, kind, experiment, parent_run_id, profile.
            **(identity.to_metadata() if identity is not None else {}),
            'strategy_key': strategy_key,
            'cost_model': cost_model,
            'risk_manager': risk_manager,
            'total_commission': getattr(result, 'total_commission', 0.0),
            'total_slippage': getattr(result, 'total_slippage', 0.0),
            'strategy_name': result.strategy_name,
            'strategy_params': strategy_params,
            'symbol': symbol,
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'initial_capital': initial_capital,
            'final_capital': result.final_capital,
            'commission': commission,
            'total_return': result.total_return,
            'total_return_pct': result.total_return_pct,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'average_win': result.average_win,
            'average_loss': result.average_loss,
            'largest_win': result.largest_win,
            'largest_loss': result.largest_loss,
            'num_winning_trades': result.num_winning_trades,
            'num_losing_trades': result.num_losing_trades,
            **self._benchmark_metadata(result),
            **self._significance_metadata(result),
        }

        # Only present when a record was collected, so a caller that opts out does not
        # get a null field indexed into Elasticsearch for every run.
        if provenance is not None:
            metadata['provenance'] = provenance

        return metadata

    @staticmethod
    def _benchmark_metadata(result: BacktestResult) -> Dict[str, Any]:
        """
        Benchmark comparison fields for the exported document.

        Read defensively so a BacktestResult built by an older caller still
        exports. The fields stay None when no benchmark ran, which is a
        different statement from a zero excess return and must survive as one
        all the way into Elasticsearch.

        Args:
            result: BacktestResult to read the comparison from

        Returns:
            Dictionary of benchmark fields
        """
        return {
            'benchmark_name': getattr(result, 'benchmark_name', None),
            'benchmark_return_pct': getattr(result, 'benchmark_return_pct', None),
            'benchmark_sharpe_ratio': getattr(result, 'benchmark_sharpe_ratio', None),
            'benchmark_max_drawdown': getattr(result, 'benchmark_max_drawdown', None),
            'benchmark_total_cost': getattr(result, 'benchmark_total_cost', None),
            'benchmark_error': getattr(result, 'benchmark_error', None),
            'excess_return_pct': getattr(result, 'excess_return_pct', None),
            'information_ratio': getattr(result, 'information_ratio', None),
        }

    @staticmethod
    def _significance_metadata(result: BacktestResult) -> Dict[str, Any]:
        """
        Statistical significance fields for the exported document.

        ``is_significant`` is exported as a tri-state: True, False, or **None**
        when the sample was below the gate. Flattening the third case to False
        would turn "we cannot tell" into "we tested and found nothing", which is
        a claim the data does not support.

        Args:
            result: BacktestResult to read the assessment from

        Returns:
            Dictionary of significance fields
        """
        is_significant = getattr(result, 'is_significant', None)
        return {
            'round_trip_count': getattr(result, 'round_trip_count', 0),
            'mean_trade_return_pct': getattr(result, 'mean_trade_return_pct', None),
            't_statistic': getattr(result, 't_statistic', None),
            'p_value': getattr(result, 'p_value', None),
            'sharpe_ci_low': getattr(result, 'sharpe_ci_low', None),
            'sharpe_ci_high': getattr(result, 'sharpe_ci_high', None),
            'sharpe_ci_confidence': getattr(result, 'sharpe_ci_confidence', None),
            'significance_min_trades': getattr(result, 'significance_min_trades', 0),
            'is_sample_sufficient': getattr(result, 'is_sample_sufficient', False),
            'is_significant': is_significant,
            'significance_verdict': getattr(result, 'significance_verdict', ''),
        }

    def get_exporter_count(self) -> int:
        """Get the number of configured exporters."""
        return len(self.exporters)
    
    def clear_exporters(self) -> None:
        """Remove all exporters and forget any recorded construction failures."""
        self.exporters.clear()
        self.creation_failures.clear()
    
    def get_exporter_names(self) -> List[str]:
        """Get the names of all configured exporters."""
        return [exporter.__class__.__name__ for exporter in self.exporters]