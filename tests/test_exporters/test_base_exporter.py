"""
Unit tests for BaseExporter.

The behaviour under test here is the contract that keeps export failures visible: an
exporter that cannot export must raise, because a silent ``return`` is indistinguishable
from success to ExporterManager and would let the CLI exit 0 with nothing written.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from niffler.exporters.base_exporter import BaseExporter, ExportError
from niffler.backtesting.backtest_result import BacktestResult


class _RecordingExporter(BaseExporter):
    """Minimal concrete exporter that records what it was asked to export."""

    def __init__(self):
        super().__init__()
        self.exported = []

    def export_backtest_result(self, result: BacktestResult, backtest_id: str,
                               metadata: Dict[str, Any]) -> None:
        self.require_valid_result(result, "recorder")
        self.exported.append(backtest_id)


class TestRequireValidResult(unittest.TestCase):
    """Tests for BaseExporter.require_valid_result."""

    def setUp(self):
        self.exporter = _RecordingExporter()

        self.result = Mock(spec=BacktestResult)
        self.result.portfolio_values = pd.Series(
            [10000.0, 10100.0],
            index=[datetime(2024, 1, 1), datetime(2024, 1, 2)]
        )
        self.result.trades = []

    def test_valid_result_passes_through(self):
        """A valid result does not raise and the export proceeds."""
        self.exporter.export_backtest_result(self.result, "id-1", {})

        self.assertEqual(self.exporter.exported, ["id-1"])

    def test_invalid_result_raises_export_error(self):
        """An invalid result raises ExportError rather than returning silently."""
        with patch.object(self.exporter, 'validate_result', return_value=False):
            with self.assertRaises(ExportError):
                self.exporter.export_backtest_result(self.result, "id-1", {})

        self.assertEqual(self.exporter.exported, [])

    def test_error_message_names_the_destination(self):
        """The error identifies which destination refused the export."""
        with patch.object(self.exporter, 'validate_result', return_value=False):
            with self.assertRaises(ExportError) as context:
                self.exporter.require_valid_result(self.result, "Elasticsearch")

        self.assertIn("Elasticsearch", str(context.exception))

    def test_export_error_is_an_exception(self):
        """ExportError is catchable as a normal exception by ExporterManager."""
        self.assertTrue(issubclass(ExportError, Exception))

    def test_missing_portfolio_values_is_rejected(self):
        """A result without portfolio values cannot be exported."""
        self.result.portfolio_values = pd.Series(dtype=float)

        with self.assertRaises(ExportError):
            self.exporter.export_backtest_result(self.result, "id-1", {})


if __name__ == '__main__':
    unittest.main()
