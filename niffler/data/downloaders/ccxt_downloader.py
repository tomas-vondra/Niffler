import pandas as pd
import logging
import ccxt
from typing import Optional
from .base_downloader import BaseDownloader


class CCXTDownloader(BaseDownloader):
    """
    Downloader for cryptocurrency exchange data using CCXT library.
    """
    
    SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    
    def __init__(self, enable_rate_limit: bool = True):
        """
        Initialize CCXT downloader.
        
        Args:
            enable_rate_limit: Whether to enable rate limiting for API calls
        """
        super().__init__("CCXT Downloader")
        self.enable_rate_limit = enable_rate_limit
        
    def download(self, exchange_id: str, symbol: str, timeframe: str, 
                start_ms: int, end_ms: int, limit: int = 1000) -> Optional[pd.DataFrame]:
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
            DataFrame with OHLCV data or None if failed
        """
        if not self.validate_timeframe(timeframe):
            logging.error(f"Invalid timeframe '{timeframe}' for CCXT. Supported: {self.get_supported_timeframes()}")
            return None
            
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({'enableRateLimit': self.enable_rate_limit})
            
            all_ohlcv = []
            current_since = start_ms

            logging.info(f"Fetching {symbol} {timeframe} data from {exchange_id} from {pd.to_datetime(start_ms, unit='ms')} to {pd.to_datetime(end_ms, unit='ms')}...")

            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit)
                if not ohlcv:
                    break
                
                # Filter out data beyond the end_ms
                filtered_ohlcv = [candle for candle in ohlcv if candle[0] <= end_ms]
                all_ohlcv.extend(filtered_ohlcv)

                # If the last fetched candle is already past the end_ms, or if we got less than 'limit' candles, we are done
                if ohlcv[-1][0] >= end_ms or len(ohlcv) < limit:
                    break
                
                current_since = ohlcv[-1][0] + 1  # Move to the next candle after the last one fetched

            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                # Ensure data is within the requested range (inclusive of start_ms and end_ms)
                df = df[(df.index.astype(int) // 10**6 >= start_ms) & (df.index.astype(int) // 10**6 <= end_ms)]
                logging.info(f"Successfully fetched {len(df)} candles.")
                return df
            else:
                logging.info("No data fetched.")
                return None
        except Exception as e:
            logging.error(f"Error downloading data from {exchange_id}: {e}")
            return None
            
    def get_supported_timeframes(self) -> list:
        """Get list of supported timeframes for CCXT."""
        return self.SUPPORTED_TIMEFRAMES.copy()
        
    def get_supported_exchanges(self) -> list:
        """Get list of available exchanges."""
        return ccxt.exchanges