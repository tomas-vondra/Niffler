"""
Base Exporter Abstract Class

Defines the interface that all exporters must implement for backtesting results.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Optional, Tuple
import logging

from ..backtesting.backtest_result import BacktestResult
from ..utils.run_identity import RUN_KIND_BACKTEST

if TYPE_CHECKING:
    from .run_record import RunRecord


class ExportError(RuntimeError):
    """
    Raised when an exporter cannot complete an export.

    Exporters must never return normally without having exported the data: a silent
    ``return`` is indistinguishable from success to ``ExporterManager``, which would then
    report the run as fully exported and let the CLI exit 0 while nothing was written.
    """


class BaseExporter(ABC):
    """Abstract base class for result exporters."""

    #: The kinds of run this exporter can export (see
    #: :data:`niffler.utils.run_identity.RUN_KINDS`). Checked when the exporter
    #: is created, before any computation: an exporter that cannot handle the
    #: run must not be discovered after a 1632-trial grid has finished.
    SUPPORTED_KINDS: Tuple[str, ...] = (RUN_KIND_BACKTEST,)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the exporter with optional configuration.
        
        Args:
            config: Dictionary containing exporter-specific configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def export_backtest_result(self, result: BacktestResult, run_id: str,
                              metadata: Dict[str, Any]) -> None:
        """
        Export a complete backtest result.

        Implementations must raise on any failure (``ExportError`` for their own
        preconditions) rather than returning early, so that ExporterManager records the
        failure instead of reporting a successful export.

        Args:
            result: BacktestResult object containing all backtest data
            run_id: Unique identifier for this backtest run
            metadata: Additional metadata about the backtest (strategy params, config, etc.)

        Raises:
            Exception: If the export could not be completed
        """
        pass

    def export_run(self, record: 'RunRecord') -> None:
        """
        Export a run that is not a single backtest (an optimization, an analysis,
        a comparison, a screen).

        The default refuses rather than doing nothing: an exporter that returned
        normally here would be reported as having exported the run.

        Args:
            record: The run to export

        Raises:
            ExportError: If this exporter does not support the run's kind
        """
        raise ExportError(
            f"{self.__class__.__name__} cannot export a {record.identity.kind} run. "
            f"It supports: {', '.join(self.SUPPORTED_KINDS)}"
        )

    def require_valid_result(self, result: BacktestResult, destination: str) -> None:
        """
        Assert that a result is exportable, raising instead of skipping silently.

        Args:
            result: BacktestResult to validate
            destination: Human-readable name of the export destination, used in the
                error message (e.g. "CSV", "Elasticsearch")

        Raises:
            ExportError: If the result does not contain the data required to export
        """
        if not self.validate_result(result):
            message = f"Invalid backtest result, cannot export to {destination}"
            self.logger.error(message)
            raise ExportError(message)


    # No run-id generator here: niffler.utils.run_identity.mint_run_id is the
    # only mint site, and the id is minted by the caller that owns the run.

    # No create_metadata here on purpose. The document is built once, by
    # ExporterManager.create_metadata, and handed to every exporter: a second
    # builder on the base class silently produced a document missing the trade
    # statistics, benchmark and significance fields.

    def validate_result(self, result: BacktestResult) -> bool:
        """
        Validate that the backtest result contains required data.
        
        Args:
            result: BacktestResult to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(result, BacktestResult):
            self.logger.error("Result is not a BacktestResult instance")
            return False
        
        if result.portfolio_values is None or result.portfolio_values.empty:
            self.logger.error("Portfolio values are missing or empty")
            return False
        
        if not hasattr(result, 'trades') or result.trades is None:
            self.logger.warning("No trades data available")
        
        return True