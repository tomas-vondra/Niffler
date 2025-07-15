import unittest
from unittest.mock import Mock, patch
import pandas as pd
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data import YahooFinanceDownloader


class TestYahooFinanceDownloader(unittest.TestCase):

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
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

        downloader = YahooFinanceDownloader()
        result = downloader.download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        mock_yf.download.assert_called_once_with('BTC-USD', start='2022-01-01', end='2022-01-02', interval='1d')

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_download_yfinance_data_empty_result(self, mock_yf):
        """Test yfinance data download with empty result."""
        mock_yf.download.return_value = pd.DataFrame()

        downloader = YahooFinanceDownloader()
        result = downloader.download('INVALID-TICKER', '2022-01-01', '2022-01-02', '1d')

        self.assertIsNone(result)

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf', None)
    def test_download_yfinance_data_no_yfinance_library(self):
        """Test yfinance data download when yfinance library is not available."""
        downloader = YahooFinanceDownloader()
        result = downloader.download('BTC-USD', '2022-01-01', '2022-01-02', '1d')
        self.assertIsNone(result)

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_download_yfinance_data_exception(self, mock_yf):
        """Test yfinance data download with exception handling."""
        mock_yf.download.side_effect = Exception("Network error")

        downloader = YahooFinanceDownloader()
        result = downloader.download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertIsNone(result)

    def test_validate_timeframe_valid(self):
        """Test timeframe validation with valid timeframes."""
        downloader = YahooFinanceDownloader()
        
        valid_timeframes = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        
        for timeframe in valid_timeframes:
            with self.subTest(timeframe=timeframe):
                self.assertTrue(downloader.validate_timeframe(timeframe))

    def test_validate_timeframe_invalid(self):
        """Test timeframe validation with invalid timeframes."""
        downloader = YahooFinanceDownloader()
        
        invalid_timeframes = ['invalid', '3m', '10m', '45m', '2h', '3h', '2d', '1y']
        
        for timeframe in invalid_timeframes:
            with self.subTest(timeframe=timeframe):
                self.assertFalse(downloader.validate_timeframe(timeframe))

    def test_get_supported_timeframes(self):
        """Test getting supported timeframes."""
        downloader = YahooFinanceDownloader()
        supported = downloader.get_supported_timeframes()
        
        expected = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        self.assertEqual(supported, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)