import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data import YahooFinanceDownloader
from niffler.data.exceptions import (
    DownloadError,
    InvalidTimeframeError,
    NoDataAvailableError,
)


def _ohlcv_frame(prefix: float = 0.0) -> pd.DataFrame:
    """Build a small single-ticker OHLCV frame for mocking yf.download."""
    return pd.DataFrame({
        'Open': [47000 + prefix, 47200 + prefix],
        'High': [47500 + prefix, 47800 + prefix],
        'Low': [46500 + prefix, 47000 + prefix],
        'Close': [47200 + prefix, 47600 + prefix],
        'Volume': [1000, 1200],
    }, index=pd.date_range('2022-01-01', periods=2, freq='D'))


class TestYahooFinanceDownloader(unittest.TestCase):

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_download_yfinance_data_success(self, mock_yf):
        """Test successful data download from yfinance."""
        mock_yf.download.return_value = _ohlcv_frame()

        downloader = YahooFinanceDownloader()
        result = downloader.download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        mock_yf.download.assert_called_once_with(
            'BTC-USD', start='2022-01-01', end='2022-01-02', interval='1d',
            auto_adjust=True,
        )

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_auto_adjust_is_always_passed_explicitly(self, mock_yf):
        """auto_adjust must never be left to the installed yfinance default."""
        mock_yf.download.return_value = _ohlcv_frame()

        downloader = YahooFinanceDownloader(auto_adjust=False)
        result = downloader.download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertFalse(mock_yf.download.call_args.kwargs['auto_adjust'])
        self.assertFalse(result.attrs['auto_adjust'])

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_auto_adjust_can_be_overridden_per_call(self, mock_yf):
        """The per-call auto_adjust parameter wins over the instance default."""
        mock_yf.download.return_value = _ohlcv_frame()

        downloader = YahooFinanceDownloader(auto_adjust=True)
        result = downloader.download('BTC-USD', '2022-01-01', '2022-01-02', '1d',
                                     auto_adjust=False)

        self.assertFalse(mock_yf.download.call_args.kwargs['auto_adjust'])
        self.assertFalse(result.attrs['auto_adjust'])

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_download_yfinance_data_empty_result(self, mock_yf):
        """An empty result raises NoDataAvailableError, not a falsy return."""
        mock_yf.download.return_value = pd.DataFrame()

        downloader = YahooFinanceDownloader()
        with self.assertRaises(NoDataAvailableError):
            downloader.download('INVALID-TICKER', '2022-01-01', '2022-01-02', '1d')

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_no_data_error_is_not_a_download_error(self, mock_yf):
        """'No data' must stay distinguishable from 'the download broke'."""
        mock_yf.download.return_value = pd.DataFrame()

        downloader = YahooFinanceDownloader()
        try:
            downloader.download('INVALID-TICKER', '2022-01-01', '2022-01-02', '1d')
        except NoDataAvailableError as e:
            self.assertNotIsInstance(e, DownloadError)
        else:
            self.fail("NoDataAvailableError not raised")

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf', None)
    def test_download_yfinance_data_no_yfinance_library(self):
        """Missing yfinance library raises DownloadError instead of returning None."""
        downloader = YahooFinanceDownloader()
        with self.assertRaises(DownloadError):
            downloader.download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_download_yfinance_data_exception(self, mock_yf):
        """A broken download raises DownloadError instead of returning None."""
        mock_yf.download.side_effect = Exception("Network error")

        downloader = YahooFinanceDownloader()
        with self.assertRaises(DownloadError) as ctx:
            downloader.download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertIn('Network error', str(ctx.exception))

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_invalid_timeframe_raises(self, mock_yf):
        """An unsupported interval raises InvalidTimeframeError."""
        downloader = YahooFinanceDownloader()
        with self.assertRaises(InvalidTimeframeError):
            downloader.download('BTC-USD', '2022-01-01', '2022-01-02', '3h')
        mock_yf.download.assert_not_called()

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_multiindex_single_ticker_is_flattened(self, mock_yf):
        """A (field, ticker) MultiIndex collapses to plain field names."""
        df = _ohlcv_frame()
        df.columns = pd.MultiIndex.from_product([df.columns, ['BTC-USD']])
        mock_yf.download.return_value = df

        result = YahooFinanceDownloader().download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertEqual(list(result.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_multiindex_reversed_level_order_is_flattened(self, mock_yf):
        """The field level is located by name, not by assuming level 0."""
        df = _ohlcv_frame()
        df.columns = pd.MultiIndex.from_product([['BTC-USD'], df.columns])
        mock_yf.download.return_value = df

        result = YahooFinanceDownloader().download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertEqual(list(result.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_multiindex_with_empty_placeholder_level(self, mock_yf):
        """The ('Adj Close', '') placeholder layout still flattens correctly."""
        df = _ohlcv_frame()
        df['Adj Close'] = df['Close']
        df.columns = pd.MultiIndex.from_tuples([
            ('Open', 'BTC-USD'), ('High', 'BTC-USD'), ('Low', 'BTC-USD'),
            ('Close', 'BTC-USD'), ('Volume', 'BTC-USD'), ('Adj Close', ''),
        ])
        mock_yf.download.return_value = df

        result = YahooFinanceDownloader().download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertEqual(list(result.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_multiindex_multiple_tickers_selects_requested(self, mock_yf):
        """When several tickers come back, only the requested one is kept."""
        wanted = _ohlcv_frame()
        other = _ohlcv_frame(prefix=1.0)
        df = pd.concat({'BTC-USD': wanted, 'ETH-USD': other}, axis=1)
        # yfinance orders columns as (field, ticker)
        df.columns = pd.MultiIndex.from_tuples([(f, t) for t, f in df.columns])
        mock_yf.download.return_value = df

        result = YahooFinanceDownloader().download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

        self.assertEqual(list(result.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])
        self.assertEqual(result['Open'].iloc[0], 47000.0)

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_multiindex_unrelated_tickers_raises(self, mock_yf):
        """Several tickers, none matching the request, is a DownloadError."""
        df = pd.concat({'ETH-USD': _ohlcv_frame(), 'SOL-USD': _ohlcv_frame(1.0)}, axis=1)
        df.columns = pd.MultiIndex.from_tuples([(f, t) for t, f in df.columns])
        mock_yf.download.return_value = df

        with self.assertRaises(DownloadError):
            YahooFinanceDownloader().download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

    @patch('niffler.data.downloaders.yahoo_finance_downloader.yf')
    def test_unrecognised_column_layout_raises(self, mock_yf):
        """A MultiIndex without any recognisable price field is an error."""
        df = _ohlcv_frame()
        df.columns = pd.MultiIndex.from_product([['a', 'b', 'c', 'd', 'e'], ['x']])
        mock_yf.download.return_value = df

        with self.assertRaises(DownloadError):
            YahooFinanceDownloader().download('BTC-USD', '2022-01-01', '2022-01-02', '1d')

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
