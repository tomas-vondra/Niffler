"""
Exporter Manager

Coordinates multiple exporters for backtesting results with unique identification.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from .base_exporter import BaseExporter
from .console_exporter import ConsoleExporter
from .csv_exporter import CSVExporter
from .elasticsearch_exporter import ElasticsearchExporter
from ..backtesting.backtest_result import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ExportSummary:
    """
    Outcome of a multi-exporter export run.

    Attributes:
        successes: Class names of the exporters that completed successfully
        failures: (exporter class name, error message) pairs for exporters that failed
        backtest_id: The backtest ID that was used for this export run
    """

    successes: List[str]
    failures: List[Tuple[str, str]]
    backtest_id: str

    @property
    def ok(self) -> bool:
        """True when every configured exporter succeeded."""
        return not self.failures


class ExporterManager:
    """Manages multiple exporters for backtesting results."""
    
    # Simple dictionary mapping exporter names to their classes
    EXPORTER_TYPES = {
        'console': ConsoleExporter,
        'csv': CSVExporter,
        'elasticsearch': ElasticsearchExporter
    }
    
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
        return list(cls.EXPORTER_TYPES.keys())
    
    def add_exporter(self, exporter: BaseExporter) -> None:
        """
        Add an exporter to the manager.
        
        Args:
            exporter: Exporter instance to add
        """
        self.exporters.append(exporter)
    
    def create_exporter_by_name(self, name: str, **kwargs) -> BaseExporter:
        """Create and add an exporter by name."""
        name = name.strip().lower()
        
        if name not in self.EXPORTER_TYPES:
            available = ', '.join(self.get_available_exporter_names())
            raise ValueError(f"Unknown exporter type: {name}. Available types: {available}")
        
        exporter_class = self.EXPORTER_TYPES[name]
        
        # Filter kwargs based on exporter type to avoid passing invalid parameters
        filtered_kwargs = {}
        if name == 'csv':
            if 'output_dir' in kwargs:
                filtered_kwargs['output_dir'] = kwargs['output_dir']
        elif name == 'elasticsearch':
            for key in ['host', 'port', 'index_prefix', 'scheme', 'api_key',
                        'username', 'password', 'timeout', 'verify_certs']:
                if key in kwargs:
                    filtered_kwargs[key] = kwargs[key]
        # console exporter doesn't take specific parameters beyond config
        
        # Always allow config parameter for all exporters
        if 'config' in kwargs:
            filtered_kwargs['config'] = kwargs['config']
        
        exporter = exporter_class(**filtered_kwargs)
        self.add_exporter(exporter)
        return exporter
    
    def create_exporters_from_list(self, exporter_names: List[str],
                                   **kwargs) -> List[Tuple[str, str]]:
        """
        Create multiple exporters from a list of names.

        An exporter whose constructor rejects its configuration (an unknown name, an
        invalid Elasticsearch scheme, an unparsable port) is recorded rather than
        quietly skipped: the failures are returned, kept on ``creation_failures`` and
        folded into every subsequent :class:`ExportSummary`, so a run that was asked
        for an exporter it never built cannot report success.

        Args:
            exporter_names: Names of the exporter types to create
            **kwargs: Configuration forwarded to each exporter constructor

        Returns:
            List of (requested name, error message) pairs for the exporters that
            could not be created; empty when every requested exporter was built
        """
        failures: List[Tuple[str, str]] = []
        for name in exporter_names:
            try:
                self.create_exporter_by_name(name, **kwargs)
            except Exception as e:
                logger.error(f"Could not create exporter '{name}': {e}")
                failures.append((name, str(e)))

        self.creation_failures.extend(failures)
        return failures


    def export_backtest_result(self, result: BacktestResult, strategy_params: Dict[str, Any],
                              symbol: str, initial_capital: float, commission: float,
                              backtest_id: str = None,
                              provenance: Optional[Dict[str, Any]] = None) -> ExportSummary:
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
            backtest_id: Optional custom backtest ID (generates one if not provided)
            provenance: Optional run provenance record (see
                :func:`niffler.utils.provenance.collect_provenance`). It is collected
                **once** by the caller that owns the run and shared by every exporter:
                collecting it here per exporter would re-hash the input data file for
                each destination

        Returns:
            ExportSummary describing which exporters succeeded, which failed and the
            backtest ID that was used
        """
        # Generate backtest ID if not provided
        if backtest_id is None:
            backtest_id = self._generate_backtest_id()

        # Create metadata
        metadata = self._create_metadata(
            result, strategy_params, symbol, initial_capital, commission, provenance
        )

        successes: List[str] = []
        # Exporters that never got constructed failed just as surely as ones that
        # raised while exporting - both mean the data did not reach their sink.
        failures: List[Tuple[str, str]] = list(self.creation_failures)

        # Export using all exporters
        for exporter in self.exporters:
            exporter_name = exporter.__class__.__name__
            try:
                exporter.export_backtest_result(result, backtest_id, metadata)
                successes.append(exporter_name)
            except Exception as e:
                # Continue with other exporters even if one fails, but record the failure
                exporter_logger = getattr(exporter, 'logger', None) or logger
                exporter_logger.error(f"Export failed for {exporter_name}: {e}")
                failures.append((exporter_name, str(e)))

        summary = ExportSummary(
            successes=successes, failures=failures, backtest_id=backtest_id
        )

        if not summary.ok:
            requested = len(self.exporters) + len(self.creation_failures)
            logger.error(
                f"{len(failures)} of {requested} exporter(s) failed for "
                f"backtest {backtest_id}: "
                f"{', '.join(name for name, _ in failures)}"
            )

        return summary
    
    def _generate_backtest_id(self) -> str:
        """Generate a unique backtest ID."""
        return str(uuid.uuid4())
    
    def _create_metadata(self, result: BacktestResult, strategy_params: Dict[str, Any],
                        symbol: str, initial_capital: float, commission: float,
                        provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

        Returns:
            Dictionary containing standardized metadata
        """
        metadata = {
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
            'num_losing_trades': result.num_losing_trades
        }

        # Only present when a record was collected, so a caller that opts out does not
        # get a null field indexed into Elasticsearch for every run.
        if provenance is not None:
            metadata['provenance'] = provenance

        return metadata

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