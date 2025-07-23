"""
CSV Exporter

Exports backtest results to CSV files for analysis and external tools.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .base_exporter import BaseExporter
from ..backtesting.backtest_result import BacktestResult


class CSVExporter(BaseExporter):
    """Exporter that saves backtest results to CSV files."""
    
    def __init__(self, output_dir: str = ".", config: Dict[str, Any] = None):
        """
        Initialize CSV exporter.
        
        Args:
            output_dir: Directory where CSV files will be saved
            config: Additional configuration options
        """
        super().__init__(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_backtest_result(self, result: BacktestResult, backtest_id: str, 
                              metadata: Dict[str, Any]) -> None:
        """
        Export backtest results to CSV files.
        
        Args:
            result: BacktestResult object containing all backtest data
            backtest_id: Unique identifier for this backtest run
            metadata: Additional metadata about the backtest
        """
        if not self.validate_result(result):
            self.logger.error("Invalid backtest result, skipping CSV export")
            return
        
        base_filename = self._generate_filename(result, backtest_id)
        
        try:
            # Export metadata
            self._export_metadata(metadata, backtest_id, base_filename)
            
            # Export portfolio values
            portfolio_file = self._export_portfolio_values(result, backtest_id, base_filename)
            
            # Export trades
            trades_file = self._export_trades(result, backtest_id, base_filename)
            
            self.logger.info(f"CSV export completed:")
            self.logger.info(f"  Portfolio values: {portfolio_file}")
            if trades_file:
                self.logger.info(f"  Trades: {trades_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to export CSV files: {e}")
            raise
    
    def _generate_filename(self, result: BacktestResult, backtest_id: str) -> str:
        """Generate base filename for CSV files."""
        # Create a readable filename with key information
        start_date = result.start_date.strftime('%Y%m%d')
        end_date = result.end_date.strftime('%Y%m%d')
        return f"{result.symbol}_{result.strategy_name}_{start_date}_{end_date}_{backtest_id[:8]}"
    
    def _export_metadata(self, metadata: Dict[str, Any], backtest_id: str, base_filename: str) -> str:
        """Export backtest metadata to JSON file."""
        metadata_file = self.output_dir / f"{base_filename}_metadata.json"
        
        # Add backtest_id to metadata
        metadata_with_id = {**metadata, 'backtest_id': backtest_id}
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata_with_id, f, indent=2, default=str)
        
        return str(metadata_file)
    
    def _export_portfolio_values(self, result: BacktestResult, backtest_id: str, base_filename: str) -> str:
        """Export portfolio values to CSV."""
        portfolio_file = self.output_dir / f"{base_filename}_portfolio.csv"
        
        # Create DataFrame with portfolio values
        portfolio_df = pd.DataFrame({
            'timestamp': result.portfolio_values.index,
            'portfolio_value': result.portfolio_values.values,
            'backtest_id': backtest_id
        })
        
        portfolio_df.to_csv(portfolio_file, index=False)
        return str(portfolio_file)
    
    def _export_trades(self, result: BacktestResult, backtest_id: str, base_filename: str) -> str:
        """Export trades to CSV."""
        if not result.trades:
            self.logger.info("No trades to export")
            return ""
        
        trades_file = self.output_dir / f"{base_filename}_trades.csv"
        
        # Create DataFrame with trade details
        trades_data = []
        for trade in result.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'price': trade.price,
                'quantity': trade.quantity,
                'value': trade.value,
                'backtest_id': backtest_id
            })
        
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(trades_file, index=False)
        return str(trades_file)
    
    def set_output_directory(self, output_dir: str) -> None:
        """Set the output directory for CSV files."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)