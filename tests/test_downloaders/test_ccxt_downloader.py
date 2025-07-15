import unittest
from unittest.mock import Mock, patch
import pandas as pd
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data import CCXTDownloader


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
        """Test ccxt data download when ccxt library is not available."""
        downloader = CCXTDownloader()
        result = downloader.download('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)
        self.assertIsNone(result)

    @patch('niffler.data.downloaders.ccxt_downloader.ccxt')
    def test_download_ccxt_data_exception(self, mock_ccxt):
        """Test ccxt data download with exception handling."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("Connection error")
        mock_ccxt.binance = Mock(return_value=mock_exchange)

        downloader = CCXTDownloader()
        result = downloader.download('binance', 'BTC/USDT', '1d', 1640995200000, 1641081600000)

        self.assertIsNone(result)

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