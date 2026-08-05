"""
Unit tests for CSVExporter.
"""

import unittest
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
import pandas as pd
import tempfile
import shutil
import os
import json
from pathlib import Path

from niffler.exporters.base_exporter import ExportError
from niffler.exporters.csv_exporter import CSVExporter, sanitize_path_component
from niffler.backtesting.backtest_result import BacktestResult
from niffler.backtesting.trade import Trade, TradeSide


class TestCSVExporter(unittest.TestCase):
    """Test cases for CSVExporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = CSVExporter(self.temp_dir)
        
        # Create mock BacktestResult
        self.mock_result = Mock(spec=BacktestResult)
        self.mock_result.strategy_name = "Simple_MA_Strategy"
        self.mock_result.symbol = "BTC-USD"
        self.mock_result.start_date = datetime(2024, 1, 1)
        self.mock_result.end_date = datetime(2024, 3, 31)
        
        # Mock portfolio values
        portfolio_values = pd.Series(
            [10000.0, 10100.0, 10200.0],
            index=[datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
        )
        self.mock_result.portfolio_values = portfolio_values
        
        # Create mock trades
        mock_trade1 = Mock(spec=Trade)
        mock_trade1.timestamp = datetime(2024, 1, 15)
        mock_trade1.symbol = "BTC-USD"
        mock_trade1.side = TradeSide.BUY
        mock_trade1.quantity = 0.25
        mock_trade1.price = 45000.0
        mock_trade1.value = 11250.0
        mock_trade1.commission = 0.0

        mock_trade2 = Mock(spec=Trade)
        mock_trade2.timestamp = datetime(2024, 2, 1)
        mock_trade2.symbol = "BTC-USD"
        mock_trade2.side = TradeSide.SELL
        mock_trade2.quantity = 0.15
        mock_trade2.price = 48000.0
        mock_trade2.value = 7200.0
        mock_trade2.commission = 0.0


        self.mock_result.trades = [mock_trade1, mock_trade2]
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init_default_directory(self):
        """Test initialization with default directory."""
        exporter = CSVExporter()
        self.assertEqual(str(exporter.output_dir), ".")
        self.assertEqual(exporter.config, {})
    
    def test_init_custom_directory(self):
        """Test initialization with custom directory."""
        exporter = CSVExporter(self.temp_dir)
        self.assertEqual(str(exporter.output_dir), self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_init_with_config(self):
        """Test initialization with config."""
        config = {'option': 'value'}
        exporter = CSVExporter(self.temp_dir, config)
        self.assertEqual(exporter.config, config)
    
    def test_generate_filename(self):
        """Test filename generation."""
        backtest_id = "12345678-1234-1234-1234-123456789abc"
        filename = self.exporter._generate_filename(self.mock_result, backtest_id)
        
        expected = "BTC-USD_Simple_MA_Strategy_20240101_20240331_12345678"
        self.assertEqual(filename, expected)
    
    def test_export_metadata(self):
        """Test metadata export to JSON."""
        metadata = {
            'strategy_name': 'Simple MA Strategy',
            'symbol': 'BTC-USD',
            'total_return': 1500.0
        }
        backtest_id = "test-id-123"
        base_filename = "test_filename"
        
        result_file = self.exporter._export_metadata(metadata, backtest_id, base_filename)
        
        # Check file was created
        expected_file = os.path.join(self.temp_dir, f"{base_filename}_metadata.json")
        self.assertEqual(result_file, expected_file)
        self.assertTrue(os.path.exists(expected_file))
        
        # Check file contents
        with open(expected_file, 'r') as f:
            saved_metadata = json.load(f)
        
        expected_metadata = {**metadata, 'backtest_id': backtest_id}
        self.assertEqual(saved_metadata, expected_metadata)
    
    def test_export_portfolio_values(self):
        """Test portfolio values export to CSV."""
        backtest_id = "test-id-123"
        base_filename = "test_filename"
        
        result_file = self.exporter._export_portfolio_values(self.mock_result, backtest_id, base_filename)
        
        # Check file was created
        expected_file = os.path.join(self.temp_dir, f"{base_filename}_portfolio.csv")
        self.assertEqual(result_file, expected_file)
        self.assertTrue(os.path.exists(expected_file))
        
        # Check file contents
        df = pd.read_csv(expected_file)
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df.columns), ['timestamp', 'portfolio_value', 'backtest_id'])
        self.assertTrue(all(df['backtest_id'] == backtest_id))
        self.assertListEqual(list(df['portfolio_value']), [10000.0, 10100.0, 10200.0])
    
    def test_export_trades_with_trades(self):
        """Test trades export with existing trades."""
        backtest_id = "test-id-123"
        base_filename = "test_filename"
        
        result_file = self.exporter._export_trades(self.mock_result, backtest_id, base_filename)
        
        # Check file was created
        expected_file = os.path.join(self.temp_dir, f"{base_filename}_trades.csv")
        self.assertEqual(result_file, expected_file)
        self.assertTrue(os.path.exists(expected_file))
        
        # Check file contents
        df = pd.read_csv(expected_file)
        self.assertEqual(len(df), 2)
        expected_columns = ['timestamp', 'symbol', 'side', 'price', 'quantity', 'value',
                            'commission', 'slippage_cost', 'backtest_id']
        self.assertListEqual(list(df.columns), expected_columns)
        
        # Check first trade
        self.assertEqual(df.iloc[0]['symbol'], 'BTC-USD')
        self.assertEqual(df.iloc[0]['side'], 'buy')
        self.assertEqual(df.iloc[0]['price'], 45000.0)
        self.assertEqual(df.iloc[0]['quantity'], 0.25)
        self.assertEqual(df.iloc[0]['value'], 11250.0)
    
    def test_export_trades_no_trades(self):
        """Test trades export with no trades."""
        self.mock_result.trades = []
        backtest_id = "test-id-123"
        base_filename = "test_filename"
        
        with patch.object(self.exporter.logger, 'info') as mock_logger:
            result_file = self.exporter._export_trades(self.mock_result, backtest_id, base_filename)
            
            self.assertEqual(result_file, "")
            mock_logger.assert_called_once_with("No trades to export")
    
    def test_export_backtest_result_success(self):
        """Test full backtest result export."""
        backtest_id = "test-backtest-123"
        metadata = {'strategy_name': 'Simple MA', 'total_return': 1500.0}
        
        with patch.object(self.exporter, 'validate_result', return_value=True):
            with patch.object(self.exporter.logger, 'info') as mock_logger:
                self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)
        
        # Check that info logs were called
        self.assertTrue(mock_logger.called)
        
        # Check that files were created
        base_filename = self.exporter._generate_filename(self.mock_result, backtest_id)
        
        metadata_file = os.path.join(self.temp_dir, f"{base_filename}_metadata.json")
        portfolio_file = os.path.join(self.temp_dir, f"{base_filename}_portfolio.csv")
        trades_file = os.path.join(self.temp_dir, f"{base_filename}_trades.csv")
        
        self.assertTrue(os.path.exists(metadata_file))
        self.assertTrue(os.path.exists(portfolio_file))
        self.assertTrue(os.path.exists(trades_file))
    
    def test_export_backtest_result_invalid_result(self):
        """An unexportable result raises instead of reporting a silent success."""
        backtest_id = "test-backtest-123"
        metadata = {'test': 'metadata'}

        with patch.object(self.exporter, 'validate_result', return_value=False):
            with patch.object(self.exporter.logger, 'error') as mock_logger:
                with self.assertRaises(ExportError) as context:
                    self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)
                self.assertIn("CSV", str(context.exception))
                mock_logger.assert_called_once_with(
                    "Invalid backtest result, cannot export to CSV"
                )

    def test_export_backtest_result_invalid_result_writes_nothing(self):
        """A refused export must not leave partial files behind."""
        backtest_id = "test-backtest-123"
        metadata = {'test': 'metadata'}

        with patch.object(self.exporter, 'validate_result', return_value=False):
            with self.assertRaises(ExportError):
                self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)

        self.assertEqual(list(Path(self.temp_dir).glob("*")), [])
    
    def test_export_backtest_result_exception(self):
        """Test export with exception during processing."""
        backtest_id = "test-backtest-123"
        metadata = {'test': 'metadata'}
        
        with patch.object(self.exporter, 'validate_result', return_value=True):
            with patch.object(self.exporter, '_export_metadata', side_effect=Exception("Test error")):
                with patch.object(self.exporter.logger, 'error') as mock_logger:
                    with self.assertRaises(Exception):
                        self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)
                    mock_logger.assert_called_once_with("Failed to export CSV files: Test error")
    
    def test_set_output_directory(self):
        """Test setting output directory."""
        new_dir = os.path.join(self.temp_dir, "new_output")
        self.exporter.set_output_directory(new_dir)
        
        self.assertEqual(str(self.exporter.output_dir), new_dir)
        self.assertTrue(os.path.exists(new_dir))
    
    def test_create_directory_if_not_exists(self):
        """Test that directory is created if it doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, "does_not_exist")
        self.assertFalse(os.path.exists(non_existent_dir))

        exporter = CSVExporter(non_existent_dir)
        self.assertTrue(os.path.exists(non_existent_dir))

    def test_export_trades_commission_defaults_when_absent(self):
        """Trades without a commission attribute export a 0.0 commission."""
        legacy_trade = Mock(spec=['timestamp', 'symbol', 'side', 'price', 'quantity', 'value'])
        legacy_trade.timestamp = datetime(2024, 1, 15)
        legacy_trade.symbol = "BTC-USD"
        legacy_trade.side = TradeSide.BUY
        legacy_trade.quantity = 0.25
        legacy_trade.price = 45000.0
        legacy_trade.value = 11250.0
        self.mock_result.trades = [legacy_trade]

        result_file = self.exporter._export_trades(self.mock_result, "test-id-123", "test_filename")

        df = pd.read_csv(result_file)
        self.assertIn('commission', df.columns)
        self.assertListEqual(list(df['commission']), [0.0])

    def test_export_trades_uses_commission_when_present(self):
        """Trades carrying a commission field export that value."""
        for trade, commission in zip(self.mock_result.trades, [1.25, 2.5]):
            trade.commission = commission

        result_file = self.exporter._export_trades(self.mock_result, "test-id-123", "test_filename")

        df = pd.read_csv(result_file)
        self.assertListEqual(list(df['commission']), [1.25, 2.5])

    def test_export_metadata_with_non_finite_values(self):
        """Metadata containing inf/NaN is written as valid JSON with nulls."""
        metadata = {
            'strategy_name': 'Simple MA Strategy',
            'profit_factor': float('inf'),
            'sharpe_ratio': float('nan'),
            'nested': {'largest_loss': float('-inf')},
            'series': [1.0, float('inf')]
        }

        result_file = self.exporter._export_metadata(metadata, "test-id-123", "test_filename")

        with open(result_file, 'r') as f:
            raw = f.read()

        # No non-standard JSON literals should be present
        self.assertNotIn('Infinity', raw)
        self.assertNotIn('NaN', raw)

        saved = json.loads(raw)
        self.assertIsNone(saved['profit_factor'])
        self.assertIsNone(saved['sharpe_ratio'])
        self.assertIsNone(saved['nested']['largest_loss'])
        self.assertEqual(saved['series'], [1.0, None])


class TestSanitizePathComponent(unittest.TestCase):
    """Test cases for the filename slug helper."""

    def test_replaces_path_separators(self):
        """Path separators never survive into a filename component."""
        self.assertEqual(sanitize_path_component("BTC/USDT"), "BTC_USDT")
        self.assertEqual(sanitize_path_component("BTC\\USDT"), "BTC_USDT")

    def test_collapses_whitespace(self):
        """Whitespace runs collapse into single underscores."""
        self.assertEqual(sanitize_path_component("Simple MA Crossover"), "Simple_MA_Crossover")
        self.assertEqual(sanitize_path_component("  Simple   MA  "), "Simple_MA")

    def test_replaces_windows_illegal_characters(self):
        """Characters illegal on Windows are replaced."""
        self.assertEqual(sanitize_path_component('a<b>c:d"e|f?g*h'), "a_b_c_d_e_f_g_h")

    def test_keeps_readable_values_unchanged(self):
        """Already-safe values are preserved verbatim."""
        self.assertEqual(sanitize_path_component("BTC-USD"), "BTC-USD")
        self.assertEqual(sanitize_path_component("Simple_MA_Strategy"), "Simple_MA_Strategy")

    def test_fallback_for_empty_and_reserved(self):
        """Empty, whitespace-only and reserved names fall back."""
        self.assertEqual(sanitize_path_component(""), "unknown")
        self.assertEqual(sanitize_path_component("   "), "unknown")
        self.assertEqual(sanitize_path_component(None), "unknown")
        self.assertEqual(sanitize_path_component("..."), "unknown")
        self.assertEqual(sanitize_path_component("CON"), "unknown")
        self.assertEqual(sanitize_path_component("", fallback="unknown_symbol"), "unknown_symbol")

    def test_truncates_long_values(self):
        """Very long values are truncated to a safe length."""
        result = sanitize_path_component("A" * 200)
        self.assertEqual(len(result), 64)


class TestCSVExporterUnsafeNames(unittest.TestCase):
    """Regression tests for symbols/strategy names that are not filesystem safe."""

    def setUp(self):
        """Set up a backtest result whose symbol contains a path separator."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = CSVExporter(self.temp_dir)

        self.mock_result = Mock(spec=BacktestResult)
        self.mock_result.strategy_name = "Simple MA Crossover"
        self.mock_result.symbol = "BTC/USDT"
        self.mock_result.start_date = datetime(2024, 1, 1)
        self.mock_result.end_date = datetime(2024, 3, 31)
        self.mock_result.portfolio_values = pd.Series(
            [10000.0, 10100.0],
            index=[datetime(2024, 1, 1), datetime(2024, 1, 2)]
        )

        mock_trade = Mock(spec=Trade)
        mock_trade.timestamp = datetime(2024, 1, 15)
        mock_trade.symbol = "BTC/USDT"
        mock_trade.side = TradeSide.BUY
        mock_trade.quantity = 0.25
        mock_trade.price = 45000.0
        mock_trade.value = 11250.0
        mock_trade.commission = 0.0
        self.mock_result.trades = [mock_trade]

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_generate_filename_sanitizes_symbol_and_strategy(self):
        """Slashes and spaces are slugified instead of creating directories."""
        filename = self.exporter._generate_filename(
            self.mock_result, "12345678-1234-1234-1234-123456789abc"
        )

        self.assertEqual(
            filename,
            "BTC_USDT_Simple_MA_Crossover_20240101_20240331_12345678"
        )
        self.assertNotIn('/', filename)
        self.assertNotIn('\\', filename)

    def test_export_writes_files_for_symbol_with_slash(self):
        """Exporting 'BTC/USDT' actually writes files to disk (F12 regression)."""
        backtest_id = "12345678-1234-1234-1234-123456789abc"
        metadata = {'strategy_name': 'Simple MA Crossover', 'symbol': 'BTC/USDT'}

        self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)

        base_filename = self.exporter._generate_filename(self.mock_result, backtest_id)
        for suffix in ('_metadata.json', '_portfolio.csv', '_trades.csv'):
            expected_file = Path(self.temp_dir) / f"{base_filename}{suffix}"
            self.assertTrue(expected_file.exists(), f"Missing export file: {expected_file}")

        # Everything must land directly in the output directory, not in a nested tree
        written = sorted(p.name for p in Path(self.temp_dir).iterdir())
        self.assertEqual(len(written), 3)
        self.assertTrue(all((Path(self.temp_dir) / name).is_file() for name in written))


if __name__ == '__main__':
    unittest.main()