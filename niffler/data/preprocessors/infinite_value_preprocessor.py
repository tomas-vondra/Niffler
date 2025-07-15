import pandas as pd
import numpy as np
import logging
from .base_preprocessor import BasePreprocessor


class InfiniteValuePreprocessor(BasePreprocessor):
    """
    Preprocessor that removes infinite values from DataFrame.
    """
    
    def __init__(self):
        super().__init__("InfiniteValuePreprocessor")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove infinite values from the DataFrame.
        
        Args:
            df: DataFrame with potential infinite values
            
        Returns:
            DataFrame with infinite values removed
        """
        if df.empty:
            logging.warning("Empty DataFrame provided for infinite value processing")
            return df
            
        # Check for infinite values
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
        inf_count = inf_mask.sum().sum()
        
        if inf_count > 0:
            logging.warning(f"Found {inf_count} infinite values, replacing with NaN")
            
            # Replace infinite values with NaN
            df_clean = df.replace([np.inf, -np.inf], np.nan)
            
            # Log which columns had infinite values
            inf_columns = inf_mask.sum()
            inf_columns = inf_columns[inf_columns > 0]
            for col, count in inf_columns.items():
                logging.info(f"Column '{col}': {count} infinite values replaced")
        else:
            logging.info("No infinite values found")
            df_clean = df
        
        return df_clean