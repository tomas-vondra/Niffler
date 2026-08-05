import pandas as pd
import logging
import time
import ccxt
from typing import List, Optional
from .base_downloader import BaseDownloader
from ..exceptions import DownloadError, InvalidTimeframeError, NoDataAvailableError


class CCXTDownloader(BaseDownloader):
    """
    Downloader for cryptocurrency exchange data using CCXT library.

    Multi-hour paginated downloads are resilient: every page fetch is retried
    with exponential backoff, the cursor is checked for forward progress so a
    misbehaving exchange cannot spin forever, and if the download gives up
    after partial success the candles fetched so far are returned flagged as
    partial (``df.attrs['partial'] is True``) instead of being discarded.
    """

    SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

    #: Hard cap on pagination iterations, so a pathological exchange response
    #: can never turn the ``while`` loop into an infinite loop.
    MAX_PAGES = 100_000

    def __init__(self, enable_rate_limit: bool = True, max_retries: int = 3,
                 backoff_base_seconds: float = 1.0, backoff_max_seconds: float = 60.0,
                 progress_every: int = 10):
        """
        Initialize CCXT downloader.

        Args:
            enable_rate_limit: Whether to enable rate limiting for API calls
            max_retries: Number of retries per page after the initial attempt.
                0 disables retrying.
            backoff_base_seconds: Initial sleep between retries; doubled on each
                subsequent retry (exponential backoff).
            backoff_max_seconds: Upper bound for a single backoff sleep.
            progress_every: Log a progress line every N pages.
        """
        super().__init__("CCXT Downloader")
        self.enable_rate_limit = enable_rate_limit
        self.max_retries = max(0, max_retries)
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.progress_every = max(1, progress_every)

    def download(self, exchange_id: str, symbol: str, timeframe: str,
                start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
        """
        Download historical data using CCXT within a specified date range.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'bybit')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1d', '1h')
            start_ms: Start timestamp in milliseconds
            end_ms: End timestamp in milliseconds
            limit: Maximum number of candles per request

        Returns:
            DataFrame indexed by timestamp with OHLCV columns. If the download
            broke part-way but candles had already been fetched, the partial
            frame is returned with ``df.attrs['partial'] = True`` and
            ``df.attrs['partial_reason']`` describing why it stopped.

        Raises:
            InvalidTimeframeError: The timeframe is not supported by CCXT.
            NoDataAvailableError: The exchange returned no candles at all.
            DownloadError: The download broke before any candle was fetched, or
                the exchange/library is unusable.
        """
        if not self.validate_timeframe(timeframe):
            raise InvalidTimeframeError(timeframe, self.get_supported_timeframes(), source="CCXT")

        if ccxt is None:
            raise DownloadError("ccxt library is not available", source=exchange_id, symbol=symbol)

        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({'enableRateLimit': self.enable_rate_limit})
        except Exception as e:
            raise DownloadError(
                f"Could not initialise exchange '{exchange_id}': {e}",
                source=exchange_id, symbol=symbol,
            ) from e

        all_ohlcv: List[list] = []
        current_since = start_ms
        partial_reason: Optional[str] = None

        logging.info(
            f"Fetching {symbol} {timeframe} data from {exchange_id} from "
            f"{pd.to_datetime(start_ms, unit='ms')} to {pd.to_datetime(end_ms, unit='ms')}..."
        )

        for page in range(1, self.MAX_PAGES + 1):
            try:
                ohlcv = self._fetch_page(exchange, exchange_id, symbol, timeframe, current_since, limit)
            except DownloadError as e:
                # Give up on this page but keep whatever we already have.
                partial_reason = str(e)
                logging.error(f"Download stopped after {len(all_ohlcv)} candles: {partial_reason}")
                break

            if not ohlcv:
                break

            # Filter out data beyond the end_ms
            filtered_ohlcv = [candle for candle in ohlcv if candle[0] <= end_ms]
            all_ohlcv.extend(filtered_ohlcv)

            if page % self.progress_every == 0:
                logging.info(
                    f"{exchange_id} {symbol} {timeframe}: page {page}, "
                    f"{len(all_ohlcv)} candles, cursor at {pd.to_datetime(ohlcv[-1][0], unit='ms')}"
                )

            # If the last fetched candle is already past the end_ms, or if we got less than 'limit' candles, we are done
            if ohlcv[-1][0] >= end_ms or len(ohlcv) < limit:
                break

            next_since = ohlcv[-1][0] + 1
            if next_since <= current_since:
                # The exchange returned a page that does not advance the cursor;
                # continuing would loop forever over the same candles.
                partial_reason = (
                    f"exchange {exchange_id} did not advance the pagination cursor "
                    f"(still at {current_since}); stopping to avoid an infinite loop"
                )
                logging.error(partial_reason)
                break
            current_since = next_since
        else:
            partial_reason = f"pagination exceeded MAX_PAGES={self.MAX_PAGES}"
            logging.error(partial_reason)

        if not all_ohlcv:
            if partial_reason is not None:
                raise DownloadError(
                    f"Download from {exchange_id} failed before any candle was fetched: {partial_reason}",
                    source=exchange_id, symbol=symbol,
                )
            logging.info("No data fetched.")
            raise NoDataAvailableError(
                f"No candles available for {symbol} {timeframe} on {exchange_id} "
                f"between {pd.to_datetime(start_ms, unit='ms')} and {pd.to_datetime(end_ms, unit='ms')}",
                source=exchange_id, symbol=symbol,
            )

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Ensure data is within the requested range (inclusive of start_ms and end_ms)
        df = df[(df.index.astype('int64') // 10**6 >= start_ms) & (df.index.astype('int64') // 10**6 <= end_ms)]

        df.attrs['partial'] = partial_reason is not None
        if partial_reason is not None:
            df.attrs['partial_reason'] = partial_reason
            logging.warning(
                f"PARTIAL DOWNLOAD: returning {len(df)} candles for {symbol} {timeframe} "
                f"on {exchange_id} - the series is incomplete ({partial_reason})"
            )
        else:
            logging.info(f"Successfully fetched {len(df)} candles.")

        return df

    def _fetch_page(self, exchange, exchange_id: str, symbol: str, timeframe: str,
                    since: int, limit: int) -> list:
        """
        Fetch a single page of candles, retrying with exponential backoff.

        Args:
            exchange: Instantiated CCXT exchange client.
            exchange_id: Exchange identifier, used for logging/errors.
            symbol: Trading pair.
            timeframe: Data timeframe.
            since: Cursor timestamp in milliseconds.
            limit: Maximum number of candles for this request.

        Returns:
            The raw OHLCV list returned by CCXT (possibly empty).

        Raises:
            DownloadError: Every attempt (initial + retries) failed.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                delay = min(self.backoff_base_seconds * (2 ** attempt), self.backoff_max_seconds)
                logging.warning(
                    f"fetch_ohlcv failed for {symbol} on {exchange_id} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay:.1f}s"
                )
                time.sleep(delay)

        raise DownloadError(
            f"fetch_ohlcv failed after {self.max_retries + 1} attempt(s): {last_error}",
            source=exchange_id, symbol=symbol,
        )

    def get_supported_timeframes(self) -> list:
        """Get list of supported timeframes for CCXT."""
        return self.SUPPORTED_TIMEFRAMES.copy()

    def get_supported_exchanges(self) -> list:
        """Get list of available exchanges."""
        return ccxt.exchanges
