"""
Console Exporter

Exports backtest results to console with human-readable formatting.
"""

from typing import Dict, Any
from .base_exporter import BaseExporter
from ..backtesting.backtest_result import BacktestResult


class ConsoleExporter(BaseExporter):
    """Exporter that prints formatted backtest results to console."""
    
    def export_backtest_result(self, result: BacktestResult, backtest_id: str, 
                              metadata: Dict[str, Any]) -> None:
        """
        Export backtest results to console with formatted output.
        
        Args:
            result: BacktestResult object containing all backtest data
            backtest_id: Unique identifier for this backtest run
            metadata: Additional metadata about the backtest
        """
        if not self.validate_result(result):
            self.logger.error("Invalid backtest result, skipping console export")
            return
        
        self._print_backtest_results(result, backtest_id)
    
    def _print_backtest_results(self, result: BacktestResult, backtest_id: str) -> None:
        """Print formatted backtest results to console."""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Backtest ID: {backtest_id}")
        print(f"Strategy: {result.strategy_name}")
        print(f"Symbol: {result.symbol}")
        print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Initial Capital: ${result.initial_capital:,.2f}")
        print(f"  Final Capital: ${result.final_capital:,.2f}")
        print(f"  Total Return: ${result.total_return:,.2f}")
        print(f"  Total Return %: {result.total_return_pct:.2f}%")
        print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"  Win Rate: {result.win_rate:.1f}%")
        print(f"  Total Trades: {result.total_trades}")
        
        if result.trades:
            print(f"\nFIRST 5 TRADES:")
            for i, trade in enumerate(result.trades[:5]):
                print(f"  {i+1}. {trade.timestamp.strftime('%Y-%m-%d')} - "
                      f"{trade.side.value.upper()} {trade.quantity:.4f} @ ${trade.price:.2f}")
            
            if len(result.trades) > 5:
                print(f"  ... and {len(result.trades) - 5} more trades")
        
        print(f"{'='*60}\n")