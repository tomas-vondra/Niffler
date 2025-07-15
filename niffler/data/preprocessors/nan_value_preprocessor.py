import pandas as pd
import logging
from .base_preprocessor import BasePreprocessor


class NanValuePreprocessor(BasePreprocessor):
    """
    Preprocessor that handles NaN values using forward-fill method.
    """
    
    def __init__(self):
        super().__init__("NanValuePreprocessor")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values using forward-fill method.
        
        Args:
            df: DataFrame with potential NaN values
            
        Returns:
            DataFrame with NaN values handled
        """
        if df.empty:
            logging.warning("Empty DataFrame provided for NaN value processing")
            return df
            
        # Check for NaN values
        nan_count = df.isnull().sum().sum()
        
        if nan_count > 0:
            logging.warning(f"Found {nan_count} NaN values, applying forward-fill")
            
            # Log which columns have NaN values
            nan_columns = df.isnull().sum()
            nan_columns = nan_columns[nan_columns > 0]
            for col, count in nan_columns.items():
                logging.info(f"Column '{col}': {count} NaN values")
            
            # Apply forward-fill
            df_clean = df.ffill()
            
            # Check if any NaN values remain (at the beginning of the series)
            remaining_nan = df_clean.isnull().sum().sum()
            if remaining_nan > 0:
                logging.warning(f"{remaining_nan} NaN values remain at the beginning of series")
                # For leading NaN values, use backward fill
                df_clean = df_clean.bfill()
                
                # If still NaN values remain, drop those rows
                final_nan = df_clean.isnull().sum().sum()
                if final_nan > 0:
                    logging.warning(f"Dropping {final_nan} rows with persistent NaN values")
                    df_clean = df_clean.dropna()
        else:
            logging.info("No NaN values found")
            df_clean = df
        
        return df_clean