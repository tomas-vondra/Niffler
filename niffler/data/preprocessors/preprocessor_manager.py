import pandas as pd
import logging
from typing import List, Optional
from .base_preprocessor import BasePreprocessor


class PreprocessorManager:
    """
    Service class that manages and orchestrates multiple preprocessors in sequence.
    This manager iterates over all registered preprocessors and applies them
    to the data in the specified order.
    """
    
    def __init__(self, name: str = "PreprocessorManager"):
        self.name = name
        self.preprocessors: List[BasePreprocessor] = []
        
    def add_preprocessor(self, preprocessor: BasePreprocessor) -> 'PreprocessorManager':
        """
        Add a preprocessor to the processing pipeline.
        
        Args:
            preprocessor: Preprocessor instance to add
            
        Returns:
            Self for method chaining
        """
        # Check if this preprocessor is already in the list
        if preprocessor not in self.preprocessors:
            self.preprocessors.append(preprocessor)
            logging.info(f"Added preprocessor: {preprocessor}")
        else:
            logging.warning(f"Preprocessor already exists, skipping: {preprocessor}")
        return self
        
    def remove_preprocessor(self, preprocessor_name: str) -> bool:
        """
        Remove a preprocessor by name.
        
        Args:
            preprocessor_name: Name of the preprocessor to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, preprocessor in enumerate(self.preprocessors):
            if preprocessor.name == preprocessor_name:
                removed = self.preprocessors.pop(i)
                logging.info(f"Removed preprocessor: {removed}")
                return True
        return False
        
    def clear_preprocessors(self) -> None:
        """Remove all preprocessors from the pipeline."""
        self.preprocessors.clear()
        logging.info("Cleared all preprocessors")
        
    def list_preprocessors(self) -> List[str]:
        """
        Get list of preprocessor names in the pipeline.
        
        Returns:
            List of preprocessor names
        """
        return [p.name for p in self.preprocessors]
        
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the DataFrame through all registered preprocessors in sequence.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        if df is None:
            logging.warning("None DataFrame provided for preprocessing")
            return None
            
        if df.empty:
            logging.warning("Empty DataFrame provided for preprocessing")
            return df
            
        if not self.preprocessors:
            logging.warning("No preprocessors registered in PreprocessorManager")
            return df
            
        original_rows = len(df)
        logging.info(f"Starting preprocessing pipeline with {len(self.preprocessors)} preprocessors")
        logging.info(f"Initial data shape: {df.shape}")
        
        current_df = df.copy()
        
        for i, preprocessor in enumerate(self.preprocessors):
            try:
                # Check if preprocessor can handle this data
                if hasattr(preprocessor, 'can_process') and not preprocessor.can_process(current_df):
                    preprocessor_name = getattr(preprocessor, 'name', str(preprocessor))
                    logging.warning(f"Preprocessor {preprocessor_name} cannot process data, skipping")
                    continue
                    
                rows_before = len(current_df)
                preprocessor_name = getattr(preprocessor, 'name', str(preprocessor))
                logging.info(f"Step {i+1}/{len(self.preprocessors)}: Running {preprocessor_name}")
                
                # Apply the preprocessor
                current_df = preprocessor.process(current_df)
                
                rows_after = len(current_df)
                if rows_before != rows_after:
                    logging.info(f"{preprocessor_name}: {rows_before} -> {rows_after} rows")
                    
                # Check if data became empty
                if current_df.empty:
                    logging.error(f"Data became empty after {preprocessor_name}")
                    break
                    
            except Exception as e:
                preprocessor_name = getattr(preprocessor, 'name', str(preprocessor))
                logging.error(f"Error in preprocessor {preprocessor_name}: {e}")
                # Continue with remaining preprocessors
                continue
                
        final_rows = len(current_df)
        logging.info(f"Preprocessing pipeline completed")
        logging.info(f"Final data shape: {current_df.shape}")
        logging.info(f"Total rows: {original_rows} -> {final_rows}")
        
        return current_df
        
    def can_run(self, df: pd.DataFrame) -> bool:
        """
        Check if any preprocessor can process the given DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if at least one preprocessor can process the data
        """
        if df is None or df.empty:
            return False
            
        return any(hasattr(p, 'can_process') and p.can_process(df) for p in self.preprocessors)


def create_default_manager() -> PreprocessorManager:
    """
    Create a PreprocessorManager with the default preprocessing pipeline.
    
    Returns:
        PreprocessorManager with standard preprocessing steps
    """
    from .infinite_value_preprocessor import InfiniteValuePreprocessor
    from .nan_value_preprocessor import NanValuePreprocessor
    from .ohlc_validator_preprocessor import OhlcValidatorPreprocessor
    from .data_quality_validator_preprocessor import DataQualityValidatorPreprocessor
    from .time_gap_detector_preprocessor import TimeGapDetectorPreprocessor
    
    manager = PreprocessorManager("DefaultManager")
    
    # Add preprocessors in logical order
    manager.add_preprocessor(InfiniteValuePreprocessor())
    manager.add_preprocessor(NanValuePreprocessor())
    manager.add_preprocessor(OhlcValidatorPreprocessor())
    manager.add_preprocessor(DataQualityValidatorPreprocessor())
    manager.add_preprocessor(TimeGapDetectorPreprocessor())
    
    return manager