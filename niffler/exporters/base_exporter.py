"""
Base Exporter Abstract Class

Defines the interface that all exporters must implement for backtesting results.
"""

from abc import ABC, abstractmethod
import uuid
from typing import Dict, Any, Optional
import logging

from ..backtesting.backtest_result import BacktestResult


class ExportError(RuntimeError):
    """
    Raised when an exporter cannot complete an export.

    Exporters must never return normally without having exported the data: a silent
    ``return`` is indistinguishable from success to ``ExporterManager``, which would then
    report the run as fully exported and let the CLI exit 0 while nothing was written.
    """


class BaseExporter(ABC):
    """Abstract base class for backtesting result exporters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the exporter with optional configuration.
        
        Args:
            config: Dictionary containing exporter-specific configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def export_backtest_result(self, result: BacktestResult, backtest_id: str,
                              metadata: Dict[str, Any]) -> None:
        """
        Export a complete backtest result.

        Implementations must raise on any failure (``ExportError`` for their own
        preconditions) rather than returning early, so that ExporterManager records the
        failure instead of reporting a successful export.

        Args:
            result: BacktestResult object containing all backtest data
            backtest_id: Unique identifier for this backtest run
            metadata: Additional metadata about the backtest (strategy params, config, etc.)

        Raises:
            Exception: If the export could not be completed
        """
        pass

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


    def generate_backtest_id(self) -> str:
        """Generate a unique backtest ID."""
        return str(uuid.uuid4())
    
    def create_metadata(self, result: BacktestResult, strategy_params: Dict[str, Any],
                       symbol: str, initial_capital: float, commission: float,
                       cost_model: str = None) -> Dict[str, Any]:
        """
        Create standardized metadata for a backtest.
        
        Args:
            result: BacktestResult object
            strategy_params: Strategy parameters used in the backtest
            symbol: Trading symbol
            initial_capital: Initial capital amount
            commission: Commission rate
            cost_model: Description of the transaction cost model in force, when
                the caller knows it

        Returns:
            Dictionary containing standardized metadata
        """
        return {
            'cost_model': cost_model,
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
            'total_trades': result.total_trades
        }
    
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