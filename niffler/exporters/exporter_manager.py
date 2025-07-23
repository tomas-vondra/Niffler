"""
Exporter Manager

Coordinates multiple exporters for backtesting results with unique identification.
"""

import uuid
import os
from typing import Dict, Any, List, Type, Callable
from .base_exporter import BaseExporter
from .console_exporter import ConsoleExporter
from .csv_exporter import CSVExporter
from .elasticsearch_exporter import ElasticsearchExporter
from ..backtesting.backtest_result import BacktestResult


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
            for key in ['host', 'port', 'index_prefix']:
                if key in kwargs:
                    filtered_kwargs[key] = kwargs[key]
        # console exporter doesn't take specific parameters beyond config
        
        # Always allow config parameter for all exporters
        if 'config' in kwargs:
            filtered_kwargs['config'] = kwargs['config']
        
        exporter = exporter_class(**filtered_kwargs)
        self.add_exporter(exporter)
        return exporter
    
    def create_exporters_from_list(self, exporter_names: List[str], **kwargs) -> None:
        """Create multiple exporters from a list of names."""
        for name in exporter_names:
            try:
                self.create_exporter_by_name(name, **kwargs)
            except ValueError as e:
                print(f"Warning: {e}, skipping")
    
    def export_backtest_result(self, result: BacktestResult, strategy_params: Dict[str, Any],
                              symbol: str, initial_capital: float, commission: float,
                              backtest_id: str = None) -> str:
        """
        Export backtest results using all configured exporters.
        
        Args:
            result: BacktestResult object containing all backtest data
            strategy_params: Strategy parameters used in the backtest
            symbol: Trading symbol
            initial_capital: Initial capital amount
            commission: Commission rate
            backtest_id: Optional custom backtest ID (generates one if not provided)
            
        Returns:
            The backtest ID that was used
        """
        # Generate backtest ID if not provided
        if backtest_id is None:
            backtest_id = self._generate_backtest_id()
        
        # Create metadata
        metadata = self._create_metadata(
            result, strategy_params, symbol, initial_capital, commission
        )
        
        # Export using all exporters
        for exporter in self.exporters:
            try:
                exporter.export_backtest_result(result, backtest_id, metadata)
            except Exception as e:
                exporter.logger.error(f"Export failed for {exporter.__class__.__name__}: {e}")
                # Continue with other exporters even if one fails
        
        return backtest_id
    
    def _generate_backtest_id(self) -> str:
        """Generate a unique backtest ID."""
        return str(uuid.uuid4())
    
    def _create_metadata(self, result: BacktestResult, strategy_params: Dict[str, Any],
                        symbol: str, initial_capital: float, commission: float) -> Dict[str, Any]:
        """
        Create standardized metadata for a backtest.
        
        Args:
            result: BacktestResult object
            strategy_params: Strategy parameters used in the backtest
            symbol: Trading symbol
            initial_capital: Initial capital amount
            commission: Commission rate
            
        Returns:
            Dictionary containing standardized metadata
        """
        return {
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
    
    def get_exporter_count(self) -> int:
        """Get the number of configured exporters."""
        return len(self.exporters)
    
    def clear_exporters(self) -> None:
        """Remove all exporters."""
        self.exporters.clear()
    
    def get_exporter_names(self) -> List[str]:
        """Get the names of all configured exporters."""
        return [exporter.__class__.__name__ for exporter in self.exporters]