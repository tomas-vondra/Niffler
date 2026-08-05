import pandas as pd
import logging
import yfinance as yf
from typing import Optional
from .base_downloader import BaseDownloader
from ..exceptions import DownloadError, InvalidTimeframeError, NoDataAvailableError


class YahooFinanceDownloader(BaseDownloader):
    """
    Downloader for traditional financial data using Yahoo Finance.

    Price convention
    ----------------
    Niffler stores **back-adjusted** prices by default (``auto_adjust=True``):
    Open/High/Low/Close are adjusted for splits *and* dividends, and no separate
    ``Adj Close`` column is produced. This is passed to ``yfinance`` explicitly
    on every call because upstream changed its own default (raw before 0.2.51,
    adjusted after), which silently made the same download reproduce different
    numbers depending on the installed version.

    Pass ``auto_adjust=False`` (constructor or per call) to store raw,
    split/dividend-unadjusted prices instead.
    """

    SUPPORTED_TIMEFRAMES = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    #: Column names yfinance uses for the price/volume fields, lower-cased.
    _PRICE_FIELDS = frozenset({'open', 'high', 'low', 'close', 'adj close', 'volume'})

    def __init__(self, normalize_columns: bool = True, auto_adjust: bool = True):
        """
        Initialize Yahoo Finance downloader.

        Args:
            normalize_columns: Whether to normalize column names to standard format
            auto_adjust: Whether to request split/dividend-adjusted OHLC prices.
                Always forwarded explicitly to yfinance so the stored convention
                does not depend on the installed yfinance version.
        """
        super().__init__("Yahoo Finance Downloader")
        self.normalize_columns = normalize_columns
        self.auto_adjust = auto_adjust

    def download(self, ticker: str, start_date: str, end_date: str,
                interval: str, auto_adjust: Optional[bool] = None) -> pd.DataFrame:
        """
        Download historical data using Yahoo Finance.

        Args:
            ticker: Stock/crypto ticker symbol (e.g., 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (e.g., '1d', '1h')
            auto_adjust: Override the instance-level adjustment convention for
                this call. ``None`` (default) uses ``self.auto_adjust``.

        Returns:
            DataFrame with OHLCV data. ``df.attrs['auto_adjust']`` records which
            price convention the frame was downloaded under.

        Raises:
            InvalidTimeframeError: The interval is not supported by Yahoo Finance.
            NoDataAvailableError: Yahoo Finance returned an empty result.
            DownloadError: The download broke, or the response could not be
                interpreted (e.g. several tickers came back for one request).
        """
        if not self.validate_timeframe(interval):
            raise InvalidTimeframeError(interval, self.get_supported_timeframes(), source="Yahoo Finance")

        if yf is None:
            raise DownloadError("yfinance library is not available", source="yahoo", symbol=ticker)

        effective_auto_adjust = self.auto_adjust if auto_adjust is None else auto_adjust

        try:
            logging.info(
                f"Fetching {ticker} data from Yahoo Finance "
                f"(auto_adjust={effective_auto_adjust})..."
            )
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval,
                             auto_adjust=effective_auto_adjust)
        except Exception as e:
            raise DownloadError(
                f"Error downloading data from Yahoo Finance for {ticker}: {e}",
                source="yahoo", symbol=ticker,
            ) from e

        if df is None or df.empty:
            logging.info("No data fetched.")
            raise NoDataAvailableError(
                f"Yahoo Finance returned no data for {ticker} {interval} "
                f"between {start_date} and {end_date}",
                source="yahoo", symbol=ticker,
            )

        # Ensure the index is named 'Date'
        if df.index.name != 'Date':
            df.index.name = 'Date'

        df = self._flatten_columns(df, ticker)

        if self.normalize_columns:
            # Select and reorder columns as desired: Open, High, Low, Close, Volume
            desired_order = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Filter desired_order to only include columns actually present in df
            final_columns = [col for col in desired_order if col in df.columns]
            df = df[final_columns]

        df.attrs['auto_adjust'] = effective_auto_adjust
        logging.info(f"Successfully fetched {len(df)} candles.")
        return df

    def _flatten_columns(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Flatten the MultiIndex columns yfinance returns, without assuming a
        single ticker or a fixed level order.

        yfinance returns ``(field, ticker)`` for multi-ticker requests but the
        level order is not guaranteed, and a single-ticker request may still be
        returned as a MultiIndex with an empty second level (e.g.
        ``('Adj Close', '')``). This locates the level that actually holds the
        OHLCV field names and reduces the other level to the requested ticker.

        Args:
            df: Raw frame as returned by ``yf.download``.
            ticker: The ticker that was requested, used to pick the right slice.

        Returns:
            DataFrame with single-level columns.

        Raises:
            DownloadError: The columns could not be interpreted, or several
                tickers were returned and none of them matches ``ticker``.
        """
        if not isinstance(df.columns, pd.MultiIndex):
            return df

        field_level = None
        for level in range(df.columns.nlevels):
            values = {str(v).lower() for v in df.columns.get_level_values(level)}
            if values & self._PRICE_FIELDS:
                field_level = level
                break

        if field_level is None:
            raise DownloadError(
                f"Unrecognised Yahoo Finance column layout for {ticker}: {list(df.columns)}",
                source="yahoo", symbol=ticker,
            )

        def ticker_labels(col: tuple) -> list:
            """Non-field, non-placeholder labels of a single column key."""
            return [str(v) for i, v in enumerate(col) if i != field_level and str(v) != '']

        tickers = list(dict.fromkeys(
            label for col in df.columns for label in ticker_labels(col)
        ))

        keep = [True] * len(df.columns)
        if len(tickers) > 1:
            match = [t for t in tickers if t.upper() == str(ticker).upper()]
            if not match:
                raise DownloadError(
                    f"Yahoo Finance returned {len(tickers)} tickers {tickers} for the "
                    f"single-ticker request '{ticker}'; cannot flatten columns",
                    source="yahoo", symbol=ticker,
                )
            selected = match[0]
            logging.warning(
                f"Yahoo Finance returned {len(tickers)} tickers {tickers} for request "
                f"'{ticker}'; keeping only '{selected}'"
            )
            keep = [
                not ticker_labels(col) or selected in ticker_labels(col)
                for col in df.columns
            ]

        kept_indices = [i for i, k in enumerate(keep) if k]
        field_labels = [df.columns[i][field_level] for i in kept_indices]
        df = df.iloc[:, kept_indices]
        df.columns = field_labels

        # Drop duplicate columns that can survive flattening (e.g. an
        # 'Adj Close' carried on the placeholder level).
        if df.columns.duplicated().any():
            duplicated = df.columns[df.columns.duplicated()].tolist()
            logging.warning(f"Dropping duplicate Yahoo Finance columns for {ticker}: {duplicated}")
            df = df.loc[:, ~df.columns.duplicated()]

        return df

    def get_supported_timeframes(self) -> list:
        """Get list of supported timeframes for Yahoo Finance."""
        return self.SUPPORTED_TIMEFRAMES.copy()
