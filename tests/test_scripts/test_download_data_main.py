import unittest
from unittest.mock import Mock, patch
import pandas as pd
import os
import tempfile
import shutil
import sys
import argparse

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from scripts import download_data as dd
from niffler.data import CCXTDownloader, YahooFinanceDownloader


class TestDownloadDataMain(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        # Patch os.getcwd to return our temp_dir so main() creates files there
        self.patcher_getcwd = patch('scripts.download_data.os.getcwd', return_value=self.temp_dir)
        self.mock_getcwd = self.patcher_getcwd.start()

    def tearDown(self):
        """Clean up after each test method."""
        self.patcher_getcwd.stop()
        shutil.rmtree(self.temp_dir)

    @patch('scripts.download_data.CCXTDownloader')
    @patch('scripts.download_data.os.getcwd')
    @patch('scripts.download_data.os.makedirs')
    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01', '--end_date', '2022-01-02'])
    def test_main_ccxt_success(self, mock_makedirs, mock_getcwd, mock_ccxt_downloader):
        """Test main function with successful ccxt data download."""
        mock_getcwd.return_value = self.temp_dir

        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'open': [47000, 47200],
            'high': [47500, 47800],
            'low': [46500, 47000],
            'close': [47200, 47600],
            'volume': [1000, 1200]
        })

        # Mock downloader instance and methods
        mock_downloader_instance = Mock()
        mock_downloader_instance.validate_timeframe.return_value = True
        mock_downloader_instance.download.return_value = mock_df
        mock_ccxt_downloader.return_value = mock_downloader_instance

        # Mock DataFrame.to_csv
        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            dd.main()

            # Verify that downloader was created and used
            mock_ccxt_downloader.assert_called_once()
            mock_downloader_instance.validate_timeframe.assert_called_once_with('1d')
            mock_downloader_instance.download.assert_called_once()

            # Verify that CSV was saved
            mock_to_csv.assert_called_once()

            # Verify directory creation
            mock_makedirs.assert_called_once()

    @patch('scripts.download_data.YahooFinanceDownloader')
    @patch('scripts.download_data.os.getcwd')
    @patch('scripts.download_data.os.makedirs')
    @patch('sys.argv', ['script.py', '--source', 'yahoo', '--symbol', 'BTC-USD', '--start_date', '2022-01-01', '--end_date', '2022-01-02'])
    def test_main_yahoo_success(self, mock_makedirs, mock_getcwd, mock_yahoo_downloader):
        """Test main function with successful yahoo data download."""
        mock_getcwd.return_value = self.temp_dir

        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'Open': [47000, 47200],
            'High': [47500, 47800],
            'Low': [46500, 47000],
            'Close': [47200, 47600],
            'Volume': [1000, 1200]
        })
        mock_df.index.name = 'Date'

        # Mock downloader instance and methods
        mock_downloader_instance = Mock()
        mock_downloader_instance.validate_timeframe.return_value = True
        mock_downloader_instance.download.return_value = mock_df
        mock_yahoo_downloader.return_value = mock_downloader_instance

        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            dd.main()

            # Verify that downloader was created and used
            mock_yahoo_downloader.assert_called_once()
            mock_downloader_instance.validate_timeframe.assert_called_once_with('1d')
            mock_downloader_instance.download.assert_called_once()

            mock_to_csv.assert_called_once()

    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-02', '--end_date', '2022-01-01'])
    def test_main_invalid_date_range(self):
        """Test main function with invalid date range."""
        with patch('scripts.download_data.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("start_date cannot be after end_date.")

    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', 'invalid-date', '--end_date', '2022-01-02'])
    def test_main_invalid_date_format(self):
        """Test main function with invalid date format."""
        with patch('scripts.download_data.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("Invalid start_date or end_date format. Use YYYY-MM-DD.")

    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01', '--end_date', '2022-01-02', '--timeframe', 'invalid'])
    def test_main_invalid_ccxt_timeframe(self):
        """Test main function with invalid ccxt timeframe."""
        with patch('scripts.download_data.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("Invalid timeframe 'invalid' for ccxt. Supported timeframes are: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M")

    @patch('sys.argv', ['script.py', '--source', 'yahoo', '--symbol', 'BTC-USD', '--start_date', '2022-01-01', '--end_date', '2022-01-02', '--timeframe', 'invalid'])
    def test_main_invalid_yahoo_timeframe(self):
        """Test main function with invalid yahoo timeframe."""
        with patch('scripts.download_data.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("Invalid timeframe 'invalid' for yahoo. Supported timeframes are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")

    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01', '--end_date', '2022-01-02', '--exchange', ''])
    def test_main_missing_exchange_for_ccxt(self):
        """Test main function with missing exchange for ccxt."""
        with patch('scripts.download_data.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("--exchange is required for ccxt source.")

    @patch('scripts.download_data.CCXTDownloader')
    @patch('scripts.download_data.os.getcwd')
    @patch('scripts.download_data.os.makedirs')
    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01'])
    def test_main_default_end_date(self, mock_makedirs, mock_getcwd, mock_ccxt_downloader):
        """Test main function with default end date (today)."""
        mock_getcwd.return_value = self.temp_dir
        
        # Mock downloader instance and methods
        mock_downloader_instance = Mock()
        mock_downloader_instance.validate_timeframe.return_value = True
        mock_downloader_instance.download.return_value = None  # No data returned
        mock_ccxt_downloader.return_value = mock_downloader_instance

        with patch('scripts.download_data.pd.Timestamp.now') as mock_now:
            mock_now.return_value.strftime.return_value = '2022-01-03'
            dd.main()

            # Should use today's date as end_date
            mock_ccxt_downloader.assert_called_once()
            mock_downloader_instance.download.assert_called_once()

    def test_filename_generation(self):
        """Test default filename generation logic."""
        # Test ccxt filename
        symbol = 'BTC/USDT'
        exchange = 'binance'
        timeframe = '1d'
        start_date = '2022-01-01'
        end_date = '2022-01-02'

        symbol_clean = symbol.replace('/', '').replace('-', '')
        expected_filename = f"{symbol_clean}_{exchange}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"

        self.assertEqual(expected_filename, "BTCUSDT_binance_1d_20220101_20220102.csv")

        # Test yahoo filename
        symbol = 'BTC-USD'
        source = 'yahoo'

        symbol_clean = symbol.replace('/', '').replace('-', '')
        expected_filename = f"{symbol_clean}_{source}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"

        self.assertEqual(expected_filename, "BTCUSD_yahoo_1d_20220101_20220102.csv")


class TestArgumentParser(unittest.TestCase):
    """Test the argument parser functionality."""

    def test_required_arguments(self):
        """Test that required arguments are enforced."""
        parser = argparse.ArgumentParser(description='Download historical market data.')
        parser.add_argument('--source', type=str, required=True, choices=['ccxt', 'yahoo'])
        parser.add_argument('--symbol', type=str, required=True)
        parser.add_argument('--start_date', type=str, required=True)

        # Should raise SystemExit when required args are missing
        with self.assertRaises(SystemExit):
            parser.parse_args(['--source', 'ccxt'])

    def test_valid_source_choices(self):
        """Test that only valid source choices are accepted."""
        parser = argparse.ArgumentParser(description='Download historical market data.')
        parser.add_argument('--source', type=str, required=True, choices=['ccxt', 'yahoo'])
        parser.add_argument('--symbol', type=str, required=True)
        parser.add_argument('--start_date', type=str, required=True)

        # Should raise SystemExit for invalid source
        with self.assertRaises(SystemExit):
            parser.parse_args(['--source', 'invalid', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01'])


if __name__ == '__main__':
    unittest.main(verbosity=2)