import pandas as pd
import numpy as np
import logging
from .base_preprocessor import BasePreprocessor


class DataQualityValidatorPreprocessor(BasePreprocessor):
    """
    Preprocessor that validates data quality and ensures trading data integrity.
    """
    
    def __init__(self):
        super().__init__("DataQualityValidatorPreprocessor")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and ensure trading data integrity.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
        """
        if df.empty:
            logging.warning("DataFrame is empty after cleaning")
            return df
        
        # Check for negative values in price and volume columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        price_cols = [col for col in numeric_cols if col.lower() in ['open', 'high', 'low', 'close']]
        volume_cols = [col for col in numeric_cols if col.lower() in ['volume']]
        
        # Validate price columns (should be positive)
        for col in price_cols:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    logging.warning(f"Column '{col}': {negative_count} non-positive values found")
                    # Remove rows with non-positive prices
                    df = df[df[col] > 0]
        
        # Validate volume columns (should be non-negative)
        for col in volume_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    logging.warning(f"Column '{col}': {negative_count} negative values found")
                    # Remove rows with negative volume
                    df = df[df[col] >= 0]
        
        # Check for duplicate timestamps/index
        if df.index.duplicated().any():
            duplicate_count = df.index.duplicated().sum()
            logging.warning(f"Found {duplicate_count} duplicate timestamps, removing duplicates")
            df = df[~df.index.duplicated(keep='first')]
        
        # Ensure data is sorted by timestamp
        if not df.index.is_monotonic_increasing:
            logging.info("Sorting data by timestamp")
            df = df.sort_index()
        
        # Final validation summary
        logging.info(f"Data validation completed. Final shape: {df.shape}")
        
        return df