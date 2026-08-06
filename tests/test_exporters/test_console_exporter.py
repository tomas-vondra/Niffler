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


if __name__ == '__main__':
    unittest.main()
