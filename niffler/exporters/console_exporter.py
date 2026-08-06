"""
Console Exporter

Exports backtest results to console with human-readable formatting.
"""

from typing import Dict, Any
from .base_exporter import BaseExporter
from ..backtesting.backtest_result import BacktestResult
from ..utils.provenance import format_provenance_summary


class ConsoleExporter(BaseExporter):
    """Exporter that prints formatted backtest results to console."""

    #: Rule used to fence off the blocks a reader must not skim past.
    _BANNER = '!' * 66


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

    def _print_benchmark_comparison(self, result: BacktestResult) -> None:
        """
        Print the strategy beside the passive alternative it has to beat.

        A return with nothing to compare it against is not a result. This block
        puts the two side by side, states the excess in percentage points, and
        says out loud that the benchmark paid the same costs - a cost-free
        benchmark would make every strategy look good.

        Args:
            result: BacktestResult carrying the benchmark fields
        """
        error = getattr(result, 'benchmark_error', None)
        if error:
            print("\nBENCHMARK COMPARISON:")
            print(f"  {self._BANNER}")
            print("  NO BENCHMARK: the passive comparison could not be established,")
            print("  so there is nothing to judge this return against.")
            print(f"  Reason: {error}")
            print(f"  {self._BANNER}")
            return

        name = getattr(result, 'benchmark_name', None)
        if not name:
            print("\nBENCHMARK COMPARISON: none requested "
                  "(--benchmark none), so this return stands unmeasured.")
            return

        print(f"\nBENCHMARK COMPARISON ({name}):")
        print(f"  {'':<20}{'Strategy':>14}{'Benchmark':>14}")
        print(f"  {'Total Return %':<20}{result.total_return_pct:>14.2f}"
              f"{result.benchmark_return_pct:>14.2f}")
        print(f"  {'Sharpe Ratio':<20}{result.sharpe_ratio:>14.3f}"
              f"{result.benchmark_sharpe_ratio:>14.3f}")
        print(f"  {'Max Drawdown %':<20}{result.max_drawdown:>14.2f}"
              f"{result.benchmark_max_drawdown:>14.2f}")
        print(f"  Excess Return: {result.excess_return_pct:+.2f} percentage points "
              f"({'ahead of' if result.excess_return_pct > 0 else 'behind'} buy-and-hold)")
        if result.information_ratio is not None:
            print(f"  Information Ratio: {result.information_ratio:.3f} "
                  f"(annualised active return / tracking error)")
        if result.benchmark_total_cost is not None:
            print(f"  The benchmark paid ${result.benchmark_total_cost:,.2f} in commission "
                  f"and slippage")
            print(f"  under the same cost model, so it is not being flattered by free fills.")

    def _print_significance(self, result: BacktestResult) -> None:
        """
        Print whether the edge is distinguishable from noise - or refuse to.

        Below the minimum-trades gate the numbers are still shown, but every one
        of them is labelled as not meaningful and no verdict is rendered. A
        confident-looking p-value beside a six-trade sample is worse than no
        p-value at all.

        Args:
            result: BacktestResult carrying the significance fields
        """
        verdict = getattr(result, 'significance_verdict', '')
        if not verdict:
            return

        count = getattr(result, 'round_trip_count', 0)
        sufficient = getattr(result, 'is_sample_sufficient', False)
        caveat = "" if sufficient else f" (NOT meaningful at n={count})"

        print("\nSTATISTICAL SIGNIFICANCE:")
        if not sufficient:
            print(f"  {self._BANNER}")
            print(f"  {verdict}")
            print(f"  {self._BANNER}")
        else:
            print(f"  {verdict}")

        if result.mean_trade_return_pct is not None:
            print(f"  Mean Round-Trip Return: {result.mean_trade_return_pct:+.3f}%{caveat}")
        if result.t_statistic is not None and result.p_value is not None:
            print(f"  t-statistic: {result.t_statistic:.3f}   "
                  f"p-value: {result.p_value:.4f} (two-sided){caveat}")
        if result.sharpe_ci_low is not None and result.sharpe_ci_high is not None:
            confidence = getattr(result, 'sharpe_ci_confidence', 0.95)
            print(f"  Bootstrap Sharpe {confidence:.0%} CI: "
                  f"[{result.sharpe_ci_low:.3f}, {result.sharpe_ci_high:.3f}]{caveat}")

        print("  Caveat: one asset, one window. If these parameters were chosen by")
        print("  optimising on this same data, this p-value is not corrected for that")
        print("  search and overstates the evidence.")

    def _print_backtest_results(self, result: BacktestResult, backtest_id: str,
                                metadata: Dict[str, Any] = None) -> None:
        """
        Print formatted backtest results to console.

        Args:
            result: BacktestResult to render
            backtest_id: Unique identifier for this backtest run
            metadata: Backtest metadata. Its ``provenance`` entry is condensed into
                a single ``Provenance:`` line, where a dirty working tree is marked
                explicitly - a result produced from uncommitted code cannot be
                reproduced from its recorded commit
        """
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Backtest ID: {backtest_id}")
        summary = format_provenance_summary((metadata or {}).get('provenance'))
        if summary:
            print(f"Provenance: {summary}")
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
        self._print_benchmark_comparison(result)
        self._print_significance(result)

        if result.trades:
            print(f"\nFIRST 5 TRADES:")
            for i, trade in enumerate(result.trades[:5]):
                print(f"  {i+1}. {trade.timestamp.strftime('%Y-%m-%d')} - "
                      f"{trade.side.value.upper()} {trade.quantity:.4f} @ ${trade.price:.2f}")
            
            if len(result.trades) > 5:
                print(f"  ... and {len(result.trades) - 5} more trades")
        
        print(f"{'='*60}\n")