import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import os
import tempfile
import shutil
import sys
from datetime import datetime
import argparse

# Adjust the path to import the download_data script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
import download_data as mdd

class TestMarketDataDownloader(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        # Patch os.getcwd to return our temp_dir so main() creates files there
        self.patcher_getcwd = patch('download_data.os.getcwd', return_value=self.temp_dir)
        self.mock_getcwd = self.patcher_getcwd.start()

    def tearDown(self):
        """Clean up after each test method."""
        self.patcher_getcwd.stop()
        shutil.rmtree(self.temp_dir)

    @patch('market_data_downloader.ccxt')
    def test_download_ccxt_data_success(self, mock_ccxt):
        """Test successful data download from ccxt exchange."""
        # Mock exchange setup
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 47000, 47500, 46500, 47200, 1000],  # 2022-01-01
            [1641081600000, 47200, 47800, 47000, 47600, 1200],  # 2022-01-02
        ]
        mock_ccxt.binance.return_value = mock_exchange

        start_ms = 1640995200000  # 2022-01-01
        end_ms = 1641081600000    # 2022-01-02

        result = dd.download_ccxt_data('binance', 'BTC/USDT', '1d', start_ms, end_ms)

        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result.columns), ['open', 'high', 'low', 'close', 'volume'])
        mock_exchange.fetch_ohlcv.assert_called_once_with('BTC/USDT', '1d', start_ms, 1000)

    @patch('market_data_downloader.ccxt')
    def test_download_ccxt_data_with_pagination(self, mock_ccxt):
        """Test ccxt data download with pagination."""
        mock_exchange = Mock()

        # Mock multiple calls to simulate pagination
        mock_exchange.fetch_ohlcv.side_effect = [
            # First call returns full limit
            [[1640995200000, 47000, 47500, 46500, 47200, 1000] for _ in range(1000)],
            # Second call returns remaining data
            [[1641081600000, 47200, 47800, 47000, 47600, 1200]],
        ]
        mock_ccxt.binance.return_value = mock_exchange

        start_ms = 1640995200000
        end_ms = 1641081600000

        result = dd.download_ccxt_data('binance', 'BTC/USDT', '1d', start_ms, end_ms, limit=1000)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1001)
        self.assertEqual(mock_exchange.fetch_ohlcv.call_count, 2)

    @patch('market_data_downloader.ccxt', None)
    def test_download_ccxt_data_no_ccxt_library(self):
        """Test ccxt data download when ccxt library is not available."""
        result = dd.download_ccxt_data('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)
        self.assertIsNone(result)

    @patch('market_data_downloader.ccxt')
    def test_download_ccxt_data_exception(self, mock_ccxt):
        """Test ccxt data download with exception handling."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("Connection error")
        mock_ccxt.binance.return_value = mock_exchange

        result = dd.download_ccxt_data('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)

        self.assertIsNone(result)

    @patch('market_data_downloader.yf')
    def test_download_yfinance_data_success(self, mock_yf):
        """Test successful data download from yfinance."""
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'Open': [47000, 47200],
            'High': [47500, 47800],
            'Low': [46500, 47000],
            'Close': [47200, 47600],
            'Volume': [1000, 1200]
        })
        mock_yf.download.return_value = mock_df

        result = dd.download_yfinance_data('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        mock_yf.download.assert_called_once_with('BTC-USD', start='2022-01-01', end='2022-01-02', interval='1d')

    @patch('market_data_downloader.yf')
    def test_download_yfinance_data_empty_result(self, mock_yf):
        """Test yfinance data download with empty result."""
        mock_yf.download.return_value = pd.DataFrame()

        result = dd.download_yfinance_data('INVALID-TICKER', '2022-01-01', '2022-01-02', '1d')

        self.assertIsNone(result)

    @patch('market_data_downloader.yf', None)
    def test_download_yfinance_data_no_yfinance_library(self):
        """Test yfinance data download when yfinance library is not available."""
        result = dd.download_yfinance_data('BTC-USD', '2022-01-01', '2022-01-02', '1d')
        self.assertIsNone(result)

    @patch('market_data_downloader.yf')
    def test_download_yfinance_data_exception(self, mock_yf):
        """Test yfinance data download with exception handling."""
        mock_yf.download.side_effect = Exception("Network error")

        result = dd.download_yfinance_data('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertIsNone(result)

    @patch('market_data_downloader.download_ccxt_data')
    @patch('market_data_downloader.os.getcwd')
    @patch('market_data_downloader.os.makedirs')
    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01', '--end_date', '2022-01-02'])
    def test_main_ccxt_success(self, mock_makedirs, mock_getcwd, mock_download_ccxt):
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
        mock_download_ccxt.return_value = mock_df

        # Mock DataFrame.to_csv
        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            dd.main()

            # Verify that download function was called
            mock_download_ccxt.assert_called_once()

            # Verify that CSV was saved
            mock_to_csv.assert_called_once()

            # Verify directory creation
            mock_makedirs.assert_called_once()

    @patch('market_data_downloader.download_yfinance_data')
    @patch('market_data_downloader.os.getcwd')
    @patch('market_data_downloader.os.makedirs')
    @patch('sys.argv', ['script.py', '--source', 'yahoo', '--symbol', 'BTC-USD', '--start_date', '2022-01-01', '--end_date', '2022-01-02'])
    def test_main_yahoo_success(self, mock_makedirs, mock_getcwd, mock_download_yahoo):
        """Test main function with successful yahoo data download."""
        mock_getcwd.return_value = self.temp_dir

        # Create mock DataFrame with MultiIndex columns (simulating yfinance output)
        mock_df = pd.DataFrame({
            ('Open', 'BTC-USD'): [47000, 47200],
            ('High', 'BTC-USD'): [47500, 47800],
            ('Low', 'BTC-USD'): [46500, 47000],
            ('Close', 'BTC-USD'): [47200, 47600],
            ('Volume', 'BTC-USD'): [1000, 1200],
            ('Adj Close', 'BTC-USD'): [47200, 47600]
        })
        mock_df.columns = pd.MultiIndex.from_tuples(mock_df.columns)
        mock_df.index.name = 'Date'
        mock_download_yahoo.return_value = mock_df

        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            dd.main()

            mock_download_yahoo.assert_called_once()
            mock_to_csv.assert_called_once()

    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-02', '--end_date', '2022-01-01'])
    def test_main_invalid_date_range(self):
        """Test main function with invalid date range."""
        with patch('market_data_downloader.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("start_date cannot be after end_date.")

    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', 'invalid-date'])
    def test_main_invalid_date_format(self):
        """Test main function with invalid date format."""
        with patch('market_data_downloader.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("Invalid start_date or end_date format. Use YYYY-MM-DD.")

    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01', '--timeframe', 'invalid'])
    def test_main_invalid_ccxt_timeframe(self):
        """Test main function with invalid ccxt timeframe."""
        with patch('market_data_downloader.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("Invalid timeframe 'invalid' for ccxt. Supported timeframes are: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M")

    @patch('sys.argv', ['script.py', '--source', 'yahoo', '--symbol', 'BTC-USD', '--start_date', '2022-01-01', '--timeframe', 'invalid'])
    def test_main_invalid_yahoo_timeframe(self):
        """Test main function with invalid yahoo timeframe."""
        with patch('market_data_downloader.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("Invalid timeframe 'invalid' for yahoo. Supported timeframes are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")

    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01', '--exchange', ''])
    def test_main_missing_exchange_for_ccxt(self):
        """Test main function with missing exchange for ccxt."""
        with patch('market_data_downloader.logging.error') as mock_log_error:
            dd.main()
            mock_log_error.assert_called_with("--exchange is required for ccxt source.")

    @patch('market_data_downloader.download_ccxt_data')
    @patch('market_data_downloader.os.getcwd')
    @patch('market_data_downloader.os.makedirs')
    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01'])
    def test_main_default_end_date(self, mock_makedirs, mock_getcwd, mock_download_ccxt):
        """Test main function with default end date (today)."""
        mock_getcwd.return_value = self.temp_dir
        mock_download_ccxt.return_value = None  # No data returned

        with patch('market_data_downloader.pd.Timestamp.now') as mock_now:
            mock_now.return_value.strftime.return_value = '2022-01-03'
            dd.main()

            # Should use today's date as end_date
            mock_download_ccxt.assert_called_once()

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
    # Create a test suite
    suite = unittest.TestSuite()

    # Add all test methods
    suite.addTest(unittest.makeSuite(TestMarketDataDownloader))
    suite.addTest(unittest.makeSuite(TestArgumentParser))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for test, traceback in result.failures:
            print(f"FAILED: {test}")
            print(traceback)
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
            print(traceback)