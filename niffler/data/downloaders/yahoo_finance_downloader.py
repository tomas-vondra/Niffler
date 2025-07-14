import pandas as pd
import logging
import yfinance as yf
from typing import Optional
from .base_downloader import BaseDownloader


class YahooFinanceDownloader(BaseDownloader):
    """
    Downloader for traditional financial data using Yahoo Finance.
    """
    
    SUPPORTED_TIMEFRAMES = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    def __init__(self, normalize_columns: bool = True):
        """
        Initialize Yahoo Finance downloader.
        
        Args:
            normalize_columns: Whether to normalize column names to standard format
        """
        super().__init__("Yahoo Finance Downloader")
        self.normalize_columns = normalize_columns
        
    def download(self, ticker: str, start_date: str, end_date: str, 
                interval: str) -> Optional[pd.DataFrame]:
        """
        Download historical data using Yahoo Finance.
        
        Args:
            ticker: Stock/crypto ticker symbol (e.g., 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (e.g., '1d', '1h')
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.validate_timeframe(interval):
            logging.error(f"Invalid timeframe '{interval}' for Yahoo Finance. Supported: {self.get_supported_timeframes()}")
            return None
            
        try:
            logging.info(f"Fetching {ticker} data from Yahoo Finance...")
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            
            if not df.empty:
                # Ensure the index is named 'Date'
                if df.index.name != 'Date':
                    df.index.name = 'Date'

                # yfinance can sometimes return a MultiIndex for columns, flatten it
                if isinstance(df.columns, pd.MultiIndex):
                    # If there's an 'Adj Close' column at the second level, drop it
                    if ('Adj Close', '') in df.columns:
                        df = df.drop(columns=[('Adj Close', '')])
                    # Flatten the MultiIndex columns, taking the first level
                    df.columns = [col[0] for col in df.columns.values]

                if self.normalize_columns:
                    # Select and reorder columns as desired: Open, High, Low, Close, Volume
                    desired_order = ['Open', 'High', 'Low', 'Close', 'Volume']
                    # Filter desired_order to only include columns actually present in df
                    final_columns = [col for col in desired_order if col in df.columns]
                    df = df[final_columns]
                
                logging.info(f"Successfully fetched {len(df)} candles.")
                return df
            else:
                logging.info("No data fetched.")
                return None
        except Exception as e:
            logging.error(f"Error downloading data from Yahoo Finance for {ticker}: {e}")
            return None
            
    def get_supported_timeframes(self) -> list:
        """Get list of supported timeframes for Yahoo Finance."""
        return self.SUPPORTED_TIMEFRAMES.copy()