"""
Unit tests for ConsoleExporter.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from io import StringIO

import pandas as pd

from niffler.exporters.base_exporter import ExportError
from niffler.exporters.console_exporter import ConsoleExporter
from niffler.backtesting.backtest_result import BacktestResult
from niffler.backtesting.trade import Trade, TradeSide


class TestConsoleExporter(unittest.TestCase):
    """Test cases for ConsoleExporter."""

    def setUp(self):
        """Set up test fixtures."""
        self.exporter = ConsoleExporter()

        self.result = Mock(spec=BacktestResult)
        self.result.strategy_name = "Simple MA Crossover"
        self.result.symbol = "BTC/USDT"
        self.result.start_date = datetime(2024, 1, 1)
        self.result.end_date = datetime(2024, 1, 3)
        self.result.initial_capital = 10000.0
        self.result.final_capital = 10200.0
        self.result.total_return = 200.0
        self.result.total_return_pct = 2.0
        self.result.max_drawdown = -1.5
        self.result.sharpe_ratio = 1.25
        self.result.win_rate = 100.0
        self.result.total_trades = 1
        self.result.total_commission = 12.5
        self.result.total_slippage = 3.75
        self.result.benchmark_name = "buy_and_hold"
        self.result.benchmark_return_pct = 5.0
        self.result.benchmark_sharpe_ratio = 0.9
        self.result.benchmark_max_drawdown = -4.0
        self.result.benchmark_total_cost = 10.0
        self.result.benchmark_error = None
        self.result.excess_return_pct = -3.0
        self.result.information_ratio = -0.4
        self.result.round_trip_count = 1
        self.result.mean_trade_return_pct = 2.0
        self.result.t_statistic = None
        self.result.p_value = None
        self.result.sharpe_ci_low = None
        self.result.sharpe_ci_high = None
        self.result.sharpe_ci_confidence = 0.95
        self.result.significance_min_trades = 30
        self.result.is_sample_sufficient = False
        self.result.significance_verdict = (
            "SAMPLE TOO SMALL: 1 round trip(s), 30 required."
        )
        self.result.portfolio_values = pd.Series(
            [10000.0, 10100.0, 10200.0],
            index=[datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
        )
        self.result.trades = [
            Trade(
                timestamp=datetime(2024, 1, 2),
                symbol="BTC/USDT",
                side=TradeSide.BUY,
                quantity=0.5,
                price=20000.0,
                value=10000.0
            )
        ]

    def test_export_prints_results(self):
        """A valid result is printed to stdout."""
        with patch('sys.stdout', new_callable=StringIO) as stdout:
            self.exporter.export_backtest_result(self.result, "id-123", {})

        output = stdout.getvalue()
        self.assertIn("BACKTEST RESULTS", output)
        self.assertIn("id-123", output)
        self.assertIn("BTC/USDT", output)

    def test_export_invalid_result_raises(self):
        """An unexportable result raises instead of reporting a silent success."""
        with patch.object(self.exporter, 'validate_result', return_value=False):
            with self.assertRaises(ExportError) as context:
                self.exporter.export_backtest_result(self.result, "id-123", {})

        self.assertIn("console", str(context.exception))

    def test_export_invalid_result_prints_nothing(self):
        """No partial report is printed when the export is refused."""
        with patch.object(self.exporter, 'validate_result', return_value=False):
            with patch('sys.stdout', new_callable=StringIO) as stdout:
                with self.assertRaises(ExportError):
                    self.exporter.export_backtest_result(self.result, "id-123", {})

        self.assertNotIn("BACKTEST RESULTS", stdout.getvalue())


class TestConsoleBenchmarkAndSignificance(unittest.TestCase):
    """The console must never print a confident number it has not earned."""

    def setUp(self):
        self.exporter = ConsoleExporter()

    def _result(self, **overrides):
        """A real BacktestResult so the printed block matches production."""
        index = pd.date_range('2024-01-01', periods=3, freq='D')
        defaults = {
            'strategy_name': "Simple MA Crossover",
            'symbol': "BTC/USDT",
            'start_date': index[0],
            'end_date': index[-1],
            'initial_capital': 10000.0,
            'final_capital': 14000.0,
            'total_return': 4000.0,
            'total_return_pct': 40.0,
            'trades': [],
            'portfolio_values': pd.Series([10000.0, 12000.0, 14000.0], index=index),
            'max_drawdown': -10.0,
            'sharpe_ratio': 1.1,
            'win_rate': 55.0,
            'total_trades': 80,
            'profit_factor': 1.4,
            'benchmark_name': "buy_and_hold",
            'benchmark_return_pct': 120.0,
            'benchmark_sharpe_ratio': 1.8,
            'benchmark_max_drawdown': -25.0,
            'benchmark_total_cost': 15.0,
            'excess_return_pct': -80.0,
            'information_ratio': -0.6,
            'round_trip_count': 40,
            'mean_trade_return_pct': 0.8,
            't_statistic': 2.4,
            'p_value': 0.021,
            'sharpe_ci_low': 0.2,
            'sharpe_ci_high': 2.0,
            'significance_min_trades': 30,
            'is_sample_sufficient': True,
            'significance_verdict': "Mean round-trip return +0.800% differs from zero.",
        }
        defaults.update(overrides)
        return BacktestResult(**defaults)

    def _render(self, result):
        with patch('sys.stdout', new_callable=StringIO) as stdout:
            self.exporter.export_backtest_result(result, "id-123", {})
        return stdout.getvalue()

    def test_a_winning_return_that_lost_to_the_asset_says_so(self):
        """+40% next to +120% must not read as a success."""
        output = self._render(self._result())

        self.assertIn("BENCHMARK COMPARISON (buy_and_hold)", output)
        self.assertIn("120.00", output)
        self.assertIn("-80.00 percentage points", output)
        self.assertIn("behind", output)

    def test_the_benchmark_costs_are_stated(self):
        """The reader must be able to see the comparison was not rigged."""
        output = self._render(self._result())

        self.assertIn("$15.00", output)
        self.assertIn("same cost model", output)

    def test_a_small_sample_gets_a_banner_and_no_verdict(self):
        """The gate has to be impossible to skim past."""
        output = self._render(self._result(
            round_trip_count=6,
            is_sample_sufficient=False,
            p_value=0.0031,
            significance_verdict="SAMPLE TOO SMALL: 6 round trip(s), 30 required.",
        ))

        self.assertIn("SAMPLE TOO SMALL", output)
        self.assertIn("!!!!", output)
        # The p-value is shown, but never without the label that kills it.
        self.assertIn("0.0031", output)
        self.assertIn("NOT meaningful at n=6", output)

    def test_a_sufficient_sample_carries_the_multiple_testing_caveat(self):
        output = self._render(self._result())

        self.assertNotIn("NOT meaningful", output)
        self.assertIn("optimising on this same data", output)

    def test_no_benchmark_requested_is_stated_not_hidden(self):
        output = self._render(self._result(
            benchmark_name=None, benchmark_return_pct=None,
            benchmark_sharpe_ratio=None, benchmark_max_drawdown=None,
            benchmark_total_cost=None, excess_return_pct=None,
            information_ratio=None,
        ))

        self.assertIn("none requested", output)
        self.assertIn("unmeasured", output)

    def test_an_unavailable_benchmark_prints_its_reason(self):
        output = self._render(self._result(
            benchmark_name=None, benchmark_return_pct=None,
            benchmark_sharpe_ratio=None, benchmark_max_drawdown=None,
            benchmark_total_cost=None, excess_return_pct=None,
            information_ratio=None,
            benchmark_error="no bar could absorb the order",
        ))

        self.assertIn("NO BENCHMARK", output)
        self.assertIn("no bar could absorb the order", output)


if __name__ == '__main__':
    unittest.main()
