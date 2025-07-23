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

from niffler.exporters.csv_exporter import CSVExporter
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
        
        mock_trade2 = Mock(spec=Trade)
        mock_trade2.timestamp = datetime(2024, 2, 1)
        mock_trade2.symbol = "BTC-USD"
        mock_trade2.side = TradeSide.SELL
        mock_trade2.quantity = 0.15
        mock_trade2.price = 48000.0
        mock_trade2.value = 7200.0
        
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
        expected_columns = ['timestamp', 'symbol', 'side', 'price', 'quantity', 'value', 'backtest_id']
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
        """Test export with invalid result."""
        backtest_id = "test-backtest-123"
        metadata = {'test': 'metadata'}
        
        with patch.object(self.exporter, 'validate_result', return_value=False):
            with patch.object(self.exporter.logger, 'error') as mock_logger:
                self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)
                mock_logger.assert_called_once_with("Invalid backtest result, skipping CSV export")
    
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


if __name__ == '__main__':
    unittest.main()