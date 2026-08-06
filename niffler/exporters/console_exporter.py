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

        Raises:
            ExportError: If the result does not contain exportable data
        """
        self.require_valid_result(result, "console")

        self._print_backtest_results(result, backtest_id, metadata)
    
    def _print_transaction_costs(self, result: BacktestResult,
                                 metadata: Dict[str, Any]) -> None:
        """
        Print what the run paid to trade, and under which cost model.

        The block is printed even when everything in it is zero: a run with no
        slippage is a frictionless run, and saying so out loud is the difference
        between a stated assumption and a silent one.

        Args:
            result: BacktestResult holding the cost totals
            metadata: Backtest metadata; its 'cost_model' entry is reported when
                the caller supplied one
        """
        print("\nTRANSACTION COSTS:")
        cost_model = (metadata or {}).get('cost_model')
        if cost_model:
            print(f"  Cost Model: {cost_model}")
        print(f"  Total Commission: ${getattr(result, 'total_commission', 0.0):,.2f}")
        print(f"  Total Slippage: ${getattr(result, 'total_slippage', 0.0):,.2f}")

    def _print_backtest_results(self, result: BacktestResult, backtest_id: str,
                                metadata: Dict[str, Any] = None) -> None:
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

        self._print_transaction_costs(result, metadata)
        
        if result.trades:
            print(f"\nFIRST 5 TRADES:")
            for i, trade in enumerate(result.trades[:5]):
                print(f"  {i+1}. {trade.timestamp.strftime('%Y-%m-%d')} - "
                      f"{trade.side.value.upper()} {trade.quantity:.4f} @ ${trade.price:.2f}")
            
            if len(result.trades) > 5:
                print(f"  ... and {len(result.trades) - 5} more trades")
        
        print(f"{'='*60}\n")