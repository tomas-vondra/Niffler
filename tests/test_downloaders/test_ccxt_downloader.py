import unittest
from unittest.mock import Mock, patch
import pandas as pd
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data import CCXTDownloader
from niffler.data.exceptions import (
    DownloadError,
    InvalidTimeframeError,
    NoDataAvailableError,
)


class TestCCXTDownloader(unittest.TestCase):

    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_download_ccxt_data_success(self, mock_ccxt):
        """Test successful data download from ccxt exchange."""
        # Mock exchange setup
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 47000, 47500, 46500, 47200, 1000],  # 2022-01-01
            [1641081600000, 47200, 47800, 47000, 47600, 1200],  # 2022-01-02
        ]
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        start_ms = 1640995200000  # 2022-01-01
        end_ms = 1641081600000    # 2022-01-02

        downloader = CCXTDownloader()
        result = downloader.download('binance', 'BTC/USDT', '1d', start_ms, end_ms)

        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result.columns), ['open', 'high', 'low', 'close', 'volume'])
        self.assertFalse(result.attrs['partial'])
        mock_exchange.fetch_ohlcv.assert_called_once_with('BTC/USDT', '1d', start_ms, 1000)

    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
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
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        start_ms = 1640995200000
        end_ms = 1641081600000

        downloader = CCXTDownloader()
        result = downloader.download('binance', 'BTC/USDT', '1d', start_ms, end_ms, limit=1000)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1001)
        self.assertEqual(mock_exchange.fetch_ohlcv.call_count, 2)

    @patch('niffler.data.downloaders.ccxt_downloader.ccxt', None)
    def test_download_ccxt_data_no_ccxt_library(self):
        """Missing ccxt library raises DownloadError instead of returning None."""
        downloader = CCXTDownloader()
        with self.assertRaises(DownloadError):
            downloader.download('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)

    @patch('niffler.data.downloaders.ccxt_downloader.time.sleep')
    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_download_ccxt_data_exception(self, mock_ccxt, mock_sleep):
        """A permanently failing fetch raises DownloadError, never returns None."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("Connection error")
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        downloader = CCXTDownloader(max_retries=2)

        with self.assertRaises(DownloadError) as ctx:
            downloader.download('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)

        self.assertIn('Connection error', str(ctx.exception))
        # Initial attempt + 2 retries
        self.assertEqual(mock_exchange.fetch_ohlcv.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('niffler.data.downloaders.ccxt_downloader.time.sleep')
    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_transient_error_is_retried_then_succeeds(self, mock_ccxt, mock_sleep):
        """A transient network error is retried with exponential backoff."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = [
            Exception("temporary network error"),
            [
                [1640995200000, 47000, 47500, 46500, 47200, 1000],
                [1641081600000, 47200, 47800, 47000, 47600, 1200],
            ],
        ]
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        downloader = CCXTDownloader(max_retries=3, backoff_base_seconds=2.0)
        result = downloader.download('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)

        self.assertEqual(len(result), 2)
        self.assertFalse(result.attrs['partial'])
        mock_sleep.assert_called_once_with(2.0)

    @patch('niffler.data.downloaders.ccxt_downloader.time.sleep')
    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_backoff_is_exponential_and_capped(self, mock_ccxt, mock_sleep):
        """Backoff doubles per retry and is capped by backoff_max_seconds."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("boom")
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        downloader = CCXTDownloader(max_retries=4, backoff_base_seconds=1.0,
                                    backoff_max_seconds=4.0)

        with self.assertRaises(DownloadError):
            downloader.download('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 4.0])

    @patch('niffler.data.downloaders.ccxt_downloader.time.sleep')
    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_partial_results_are_preserved_on_give_up(self, mock_ccxt, mock_sleep):
        """Candles already fetched survive a mid-download failure, flagged partial."""
        page = [[1640995200000 + i * 86400000, 1, 2, 0.5, 1.5, 10] for i in range(3)]
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = [page, Exception("connection reset")]
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        start_ms = 1640995200000
        end_ms = start_ms + 30 * 86400000

        downloader = CCXTDownloader(max_retries=0)
        result = downloader.download('binance', 'BTC/USDT', '1d', start_ms, end_ms, limit=3)

        self.assertEqual(len(result), 3)
        self.assertTrue(result.attrs['partial'])
        self.assertIn('connection reset', result.attrs['partial_reason'])

    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_stalled_cursor_does_not_loop_forever(self, mock_ccxt):
        """An exchange that never advances the cursor cannot hang the download."""
        # Every page is identical: last timestamp never moves past `since`.
        stalled_page = [[1640995200000, 1, 2, 0.5, 1.5, 10] for _ in range(2)]
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = stalled_page
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        start_ms = 1640995200000
        end_ms = start_ms + 30 * 86400000

        downloader = CCXTDownloader()
        result = downloader.download('binance', 'BTC/USDT', '1d', start_ms, end_ms, limit=2)

        # Second page advances since to ts+1, third page returns the same ts -> stop
        self.assertLessEqual(mock_exchange.fetch_ohlcv.call_count, 3)
        self.assertTrue(result.attrs['partial'])
        self.assertIn('did not advance', result.attrs['partial_reason'])

    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_no_data_raises_no_data_available(self, mock_ccxt):
        """An empty (but successful) response raises NoDataAvailableError."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = []
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        downloader = CCXTDownloader()
        with self.assertRaises(NoDataAvailableError):
            downloader.download('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)

    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_no_data_error_is_not_a_download_error(self, mock_ccxt):
        """'No data' must stay distinguishable from 'the download broke'."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = []
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        downloader = CCXTDownloader()
        try:
            downloader.download('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)
        except NoDataAvailableError as e:
            self.assertNotIsInstance(e, DownloadError)
        else:
            self.fail("NoDataAvailableError not raised")

    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_invalid_timeframe_raises(self, mock_ccxt):
        """An unsupported timeframe raises InvalidTimeframeError."""
        downloader = CCXTDownloader()
        with self.assertRaises(InvalidTimeframeError):
            downloader.download('binance', 'BTC/USDT', '7h', 1640995200000, 1641081600000)

    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_unknown_exchange_raises_download_error(self, mock_ccxt):
        """A missing/broken exchange class raises DownloadError."""
        del mock_ccxt.nosuchexchange

        downloader = CCXTDownloader()
        with self.assertRaises(DownloadError):
            downloader.download('nosuchexchange', 'BTC/USDT', '1d', 1640995200000, 1641081600000)

    def test_validate_timeframe_valid(self):
        """Test timeframe validation with valid timeframes."""
        downloader = CCXTDownloader()

        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

        for timeframe in valid_timeframes:
            with self.subTest(timeframe=timeframe):
                self.assertTrue(downloader.validate_timeframe(timeframe))

    def test_validate_timeframe_invalid(self):
        """Test timeframe validation with invalid timeframes."""
        downloader = CCXTDownloader()

        invalid_timeframes = ['invalid', '2m', '10m', '45m', '3h', '7h', '2d', '1y']

        for timeframe in invalid_timeframes:
            with self.subTest(timeframe=timeframe):
                self.assertFalse(downloader.validate_timeframe(timeframe))

    def test_get_supported_timeframes(self):
        """Test getting supported timeframes."""
        downloader = CCXTDownloader()
        supported = downloader.get_supported_timeframes()

        expected = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        self.assertEqual(supported, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
