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
from niffler.data.exceptions import (
    DownloadError,
    InvalidTimeframeError,
    NoDataAvailableError,
)


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

    @patch('scripts.download_data.CCXTDownloader')
    @patch('scripts.download_data.os.getcwd')
    @patch('scripts.download_data.os.makedirs')
    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01', '--end_date', '2022-01-02'])
    def test_main_returns_non_zero_when_downloader_returns_none(self, mock_makedirs, mock_getcwd,
                                                                mock_ccxt_downloader):
        """A downloader that returns None must not look like a success."""
        mock_getcwd.return_value = self.temp_dir

        mock_downloader_instance = Mock()
        mock_downloader_instance.validate_timeframe.return_value = True
        mock_downloader_instance.download.return_value = None
        mock_ccxt_downloader.return_value = mock_downloader_instance

        with patch('scripts.download_data.logging.error') as mock_log_error:
            exit_code = dd.main()

        self.assertEqual(exit_code, 1)
        mock_log_error.assert_called_once()

    @patch('scripts.download_data.CCXTDownloader')
    @patch('scripts.download_data.os.getcwd')
    @patch('scripts.download_data.os.makedirs')
    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01', '--end_date', '2022-01-02'])
    def test_main_returns_non_zero_when_downloader_raises(self, mock_makedirs, mock_getcwd,
                                                          mock_ccxt_downloader):
        """A downloader raising a typed error is reported, not propagated."""
        mock_getcwd.return_value = self.temp_dir

        class DownloadError(Exception):
            """Stand-in for the downloaders' typed exceptions."""

        mock_downloader_instance = Mock()
        mock_downloader_instance.validate_timeframe.return_value = True
        mock_downloader_instance.download.side_effect = DownloadError("exchange unreachable")
        mock_ccxt_downloader.return_value = mock_downloader_instance

        with patch('scripts.download_data.logging.error') as mock_log_error:
            exit_code = dd.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("exchange unreachable", str(mock_log_error.call_args_list))

    @patch('scripts.download_data.CCXTDownloader')
    @patch('scripts.download_data.os.getcwd')
    @patch('scripts.download_data.os.makedirs')
    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-01', '--end_date', '2022-01-02'])
    def test_main_returns_zero_on_success(self, mock_makedirs, mock_getcwd, mock_ccxt_downloader):
        """A successful download returns exit code 0."""
        mock_getcwd.return_value = self.temp_dir

        mock_downloader_instance = Mock()
        mock_downloader_instance.validate_timeframe.return_value = True
        mock_downloader_instance.download.return_value = pd.DataFrame({
            'open': [1.0], 'high': [2.0], 'low': [0.5], 'close': [1.5], 'volume': [10.0]
        })
        mock_ccxt_downloader.return_value = mock_downloader_instance

        with patch.object(pd.DataFrame, 'to_csv'):
            exit_code = dd.main()

        self.assertEqual(exit_code, 0)

    @patch('sys.argv', ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT', '--start_date', '2022-01-02', '--end_date', '2022-01-01'])
    def test_main_invalid_date_range_returns_non_zero(self):
        """Argument validation errors exit non-zero."""
        with patch('scripts.download_data.logging.error'):
            self.assertEqual(dd.main(), 1)

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


class TestTypedDownloaderExceptions(unittest.TestCase):
    """The CLI must distinguish the typed errors raised by the downloaders.

    Downloaders no longer return None on failure: they raise InvalidTimeframeError,
    NoDataAvailableError or DownloadError. Each must produce a clean, distinguishable
    message and a non-zero exit code rather than a traceback or a silent success.
    """

    def _run_with(self, error):
        """Run main() against a downloader whose download() raises `error`."""
        downloader = Mock()
        downloader.validate_timeframe.return_value = True
        downloader.download.side_effect = error

        argv = ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT',
                '--start_date', '2022-01-01', '--end_date', '2022-01-02']

        with patch('sys.argv', argv):
            with patch('scripts.download_data.CCXTDownloader', return_value=downloader):
                with patch('scripts.download_data.os.makedirs'):
                    with patch('scripts.download_data.logging.error') as mock_error:
                        exit_code = dd.main()

        messages = " | ".join(str(call.args[0]) for call in mock_error.call_args_list)
        return exit_code, messages

    def test_invalid_timeframe_lists_supported_values(self):
        """An unsupported timeframe tells the user what is supported."""
        exit_code, messages = self._run_with(
            InvalidTimeframeError('3s', ['1m', '1h', '1d'], source='binance')
        )

        self.assertEqual(exit_code, 1)
        self.assertIn('3s', messages)
        self.assertIn('1m, 1h, 1d', messages)

    def test_no_data_available_is_reported_as_empty_not_broken(self):
        """An empty venue response is reported distinctly from a broken download."""
        exit_code, messages = self._run_with(
            NoDataAvailableError('no candles in range', source='binance', symbol='BTC/USDT')
        )

        self.assertEqual(exit_code, 1)
        self.assertIn('No data available', messages)

    def test_download_error_is_reported_as_a_failure(self):
        """A genuine transport/venue failure is reported as a failed download."""
        exit_code, messages = self._run_with(
            DownloadError('connection reset', source='binance', symbol='BTC/USDT')
        )

        self.assertEqual(exit_code, 1)
        self.assertIn('Failed to download', messages)
        self.assertIn('connection reset', messages)

    def test_unexpected_error_does_not_escape_as_a_traceback(self):
        """An untyped failure is still caught and reported, not raised at the user."""
        exit_code, messages = self._run_with(RuntimeError('boom'))

        self.assertEqual(exit_code, 1)
        self.assertIn('Unexpected error', messages)


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


class TestPartialDownloadHandling(unittest.TestCase):
    """A truncated download must never be reported as a successful one."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    @staticmethod
    def _frame(partial=False, reason=None):
        df = pd.DataFrame({
            'open': [1.0, 2.0], 'high': [2.0, 3.0], 'low': [0.5, 1.5],
            'close': [1.5, 2.5], 'volume': [10.0, 20.0]
        })
        if partial:
            df.attrs['partial'] = True
            df.attrs['partial_reason'] = reason or 'exchange stopped advancing the cursor'
        return df

    def _run_main(self, df):
        """Run main() against a downloader that returns `df`."""
        written = {}

        def fake_to_csv(self_df, path, *args, **kwargs):
            written['path'] = path

        downloader = Mock()
        downloader.validate_timeframe.return_value = True
        downloader.download.return_value = df

        argv = ['script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT',
                '--start_date', '2022-01-01', '--end_date', '2022-01-02']
        with patch('sys.argv', argv), \
                patch('scripts.download_data.os.getcwd', return_value=self.temp_dir), \
                patch('scripts.download_data.CCXTDownloader', return_value=downloader), \
                patch.object(pd.DataFrame, 'to_csv', fake_to_csv), \
                patch('scripts.download_data.logging.error') as mock_error:
            exit_code = dd.main()

        return exit_code, written.get('path'), mock_error

    def test_partial_download_exits_non_zero(self):
        exit_code, _, _ = self._run_main(self._frame(partial=True))
        self.assertEqual(exit_code, 1)

    def test_partial_download_reports_the_reason(self):
        _, _, mock_error = self._run_main(
            self._frame(partial=True, reason='exchange died after page 3')
        )
        logged = str(mock_error.call_args_list)
        self.assertIn('exchange died after page 3', logged)
        self.assertIn('INCOMPLETE', logged)

    def test_partial_download_is_written_to_a_marked_path(self):
        _, path, _ = self._run_main(self._frame(partial=True))
        self.assertIsNotNone(path)
        self.assertTrue(path.endswith('.partial.csv'), path)

    def test_complete_download_still_exits_zero_on_the_normal_path(self):
        exit_code, path, _ = self._run_main(self._frame(partial=False))
        self.assertEqual(exit_code, 0)
        self.assertTrue(path.endswith('.csv'))
        self.assertNotIn('.partial.', path)


class TestDownloadDataLogLevel(unittest.TestCase):
    """--log-level must reach setup_logging, and default to INFO."""

    @patch('scripts.download_data.CCXTDownloader')
    @patch('scripts.download_data.os.getcwd')
    @patch('scripts.download_data.os.makedirs')
    def _run_main(self, argv, mock_makedirs, mock_getcwd, mock_ccxt):
        """Run main() with a stubbed download and report the level requested."""
        mock_getcwd.return_value = tempfile.mkdtemp()

        mock_downloader = Mock()
        mock_downloader.validate_timeframe.return_value = True
        mock_downloader.download.return_value = pd.DataFrame({
            'open': [1.0], 'high': [2.0], 'low': [0.5],
            'close': [1.5], 'volume': [100],
        })
        mock_ccxt.return_value = mock_downloader

        with patch('sys.argv', argv):
            with patch('scripts.download_data.setup_logging') as mock_setup:
                with patch.object(pd.DataFrame, 'to_csv'):
                    dd.main()
        return mock_setup

    def test_log_level_defaults_to_info(self):
        """Omitting --log-level configures logging at INFO."""
        mock_setup = self._run_main([
            'script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT',
            '--start_date', '2022-01-01', '--end_date', '2022-01-02',
        ])
        mock_setup.assert_called_once_with(level='INFO')

    def test_log_level_is_forwarded(self):
        """--log-level DEBUG reaches setup_logging instead of a hardcoded INFO."""
        mock_setup = self._run_main([
            'script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT',
            '--start_date', '2022-01-01', '--end_date', '2022-01-02',
            '--log-level', 'DEBUG',
        ])
        mock_setup.assert_called_once_with(level='DEBUG')

    def test_invalid_log_level_is_rejected(self):
        """An unknown level fails at argument parsing, matching the other scripts."""
        with patch('sys.argv', [
            'script.py', '--source', 'ccxt', '--symbol', 'BTC/USDT',
            '--start_date', '2022-01-01', '--log-level', 'TRACE',
        ]):
            with self.assertRaises(SystemExit):
                dd.main()


if __name__ == '__main__':
    unittest.main(verbosity=2)