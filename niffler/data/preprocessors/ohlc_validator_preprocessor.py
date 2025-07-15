import pandas as pd
import logging
from .base_preprocessor import BasePreprocessor


class OhlcValidatorPreprocessor(BasePreprocessor):
    """
    Preprocessor that validates OHLC (Open, High, Low, Close) data integrity.
    """
    
    def __init__(self):
        super().__init__("OhlcValidatorPreprocessor")
    
    def can_process(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has OHLC columns."""
        if df.empty:
            return False
            
        # Find OHLC columns (case-insensitive)
        ohlc_cols = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'high', 'low', 'close']:
                ohlc_cols[col_lower] = col
        
        # Check if we have the required OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in ohlc_cols]
        
        return len(missing_cols) == 0
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLC (Open, High, Low, Close) data integrity.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with invalid OHLC rows removed
        """
        if df.empty:
            logging.warning("DataFrame is empty for OHLC validation")
            return df
        
        # Find OHLC columns (case-insensitive)
        ohlc_cols = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'high', 'low', 'close']:
                ohlc_cols[col_lower] = col
        
        # Check if we have the required OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in ohlc_cols]
        
        if missing_cols:
            logging.info(f"OHLC validation skipped - missing columns: {missing_cols}")
            return df
        
        logging.info("Validating OHLC data integrity")
        
        # Get actual column names
        open_col = ohlc_cols['open']
        high_col = ohlc_cols['high']
        low_col = ohlc_cols['low']
        close_col = ohlc_cols['close']
        
        original_rows = len(df)
        invalid_rows = pd.Series([False] * len(df), index=df.index)
        
        # Rule 1: High should be >= Low
        high_low_invalid = df[high_col] < df[low_col]
        if high_low_invalid.any():
            count = high_low_invalid.sum()
            logging.warning(f"Found {count} rows where High < Low")
            invalid_rows |= high_low_invalid
        
        # Rule 2: High should be >= Open
        high_open_invalid = df[high_col] < df[open_col]
        if high_open_invalid.any():
            count = high_open_invalid.sum()
            logging.warning(f"Found {count} rows where High < Open")
            invalid_rows |= high_open_invalid
        
        # Rule 3: High should be >= Close
        high_close_invalid = df[high_col] < df[close_col]
        if high_close_invalid.any():
            count = high_close_invalid.sum()
            logging.warning(f"Found {count} rows where High < Close")
            invalid_rows |= high_close_invalid
        
        # Rule 4: Low should be <= Open
        low_open_invalid = df[low_col] > df[open_col]
        if low_open_invalid.any():
            count = low_open_invalid.sum()
            logging.warning(f"Found {count} rows where Low > Open")
            invalid_rows |= low_open_invalid
        
        # Rule 5: Low should be <= Close
        low_close_invalid = df[low_col] > df[close_col]
        if low_close_invalid.any():
            count = low_close_invalid.sum()
            logging.warning(f"Found {count} rows where Low > Close")
            invalid_rows |= low_close_invalid
        
        # Remove invalid rows
        if invalid_rows.any():
            invalid_count = invalid_rows.sum()
            logging.warning(f"Removing {invalid_count} rows with invalid OHLC data")
            df = df[~invalid_rows]
        else:
            logging.info("All OHLC data is valid")
        
        final_rows = len(df)
        logging.info(f"OHLC validation completed. Rows: {original_rows} -> {final_rows}")
        
        return df