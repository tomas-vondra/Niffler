import pandas as pd
import logging
from .base_preprocessor import BasePreprocessor


class TimeGapDetectorPreprocessor(BasePreprocessor):
    """
    Preprocessor that detects time gaps in trading data sequence.
    """
    
    def __init__(self):
        super().__init__("TimeGapDetectorPreprocessor")
    
    def can_process(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has datetime index and sufficient data."""
        return (not df.empty and 
                len(df) >= 2 and 
                isinstance(df.index, pd.DatetimeIndex))
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect time gaps in the trading data sequence.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Original DataFrame (gaps are logged but not filled)
        """
        if df.empty or len(df) < 2:
            logging.info("Insufficient data for time gap detection")
            return df
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.warning("Index is not DatetimeIndex, skipping time gap detection")
            return df
        
        logging.info("Detecting time gaps in data")
        
        # Calculate time differences between consecutive rows
        time_diffs = df.index.to_series().diff().dropna()
        
        # Determine expected frequency
        # Use the most common time difference as the expected frequency
        mode_diff = time_diffs.mode()
        
        if mode_diff.empty:
            logging.warning("Could not determine expected frequency")
            return df
        
        expected_freq = mode_diff.iloc[0]
        logging.info(f"Expected frequency: {expected_freq}")
        
        # Define gap threshold (e.g., 1.5x the expected frequency)
        gap_threshold = expected_freq * 1.5
        
        # Find gaps
        gaps = time_diffs[time_diffs > gap_threshold]
        
        if not gaps.empty:
            logging.warning(f"Found {len(gaps)} time gaps in data:")
            for timestamp, gap_size in gaps.items():
                gap_start = df.index[df.index < timestamp][-1] if len(df.index[df.index < timestamp]) > 0 else None
                gap_end = timestamp
                logging.warning(f"  Gap from {gap_start} to {gap_end} (duration: {gap_size})")
        else:
            logging.info("No significant time gaps detected")
        
        # Calculate data completeness
        if len(df) > 1:
            total_expected_periods = (df.index[-1] - df.index[0]) / expected_freq + 1
            completeness = len(df) / total_expected_periods * 100
            logging.info(f"Data completeness: {completeness:.1f}%")
        
        return df