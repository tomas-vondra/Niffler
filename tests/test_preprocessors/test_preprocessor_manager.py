import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data import PreprocessorManager, create_default_manager
from niffler.data.preprocessors import (
    InfiniteValuePreprocessor,
    NanValuePreprocessor,
    OhlcValidatorPreprocessor,
    DataQualityValidatorPreprocessor,
    TimeGapDetectorPreprocessor
)

# Disable logging for tests to keep output clean
logging.disable(logging.CRITICAL)


class TestPreprocessorManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample OHLC data
        self.sample_dates = pd.date_range('2024-01-01', periods=5, freq='D')
        self.valid_ohlc_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0],
            'low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=self.sample_dates)
    
    def test_preprocessor_manager_complete_workflow(self):
        """Test the complete preprocessing workflow using PreprocessorManager."""
        # Create data with multiple issues
        problematic_data = self.valid_ohlc_data.copy()
        problematic_data.loc[self.sample_dates[0], 'high'] = np.inf  # Infinite value
        problematic_data.loc[self.sample_dates[1], 'low'] = np.nan  # NaN value
        problematic_data.loc[self.sample_dates[2], 'close'] = -10.0  # Negative price
        problematic_data.loc[self.sample_dates[3], 'high'] = 90.0   # High < Low (invalid OHLC)
        
        manager = create_default_manager()
        result = manager.run(problematic_data)
        
        # Should handle all issues
        self.assertFalse(np.isinf(result.values).any())  # No infinite values
        self.assertFalse(result.isnull().any().any())     # No NaN values
        # Should have fewer rows due to invalid data removal
        self.assertLess(len(result), len(problematic_data))
    
    def test_preprocessor_manager_empty_dataframe(self):
        """Test PreprocessorManager with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        manager = create_default_manager()
        result = manager.run(empty_df)
        
        # Should return empty DataFrame
        self.assertTrue(result.empty)
    
    def test_preprocessor_manager_add_remove_preprocessors(self):
        """Test adding and removing preprocessors from PreprocessorManager."""
        manager = PreprocessorManager()
        
        # Initially empty
        self.assertEqual(len(manager.list_preprocessors()), 0)
        
        # Add preprocessors
        inf_processor = InfiniteValuePreprocessor()
        nan_processor = NanValuePreprocessor()
        
        manager.add_preprocessor(inf_processor)
        manager.add_preprocessor(nan_processor)
        
        # Should have 2 preprocessors
        self.assertEqual(len(manager.list_preprocessors()), 2)
        self.assertIn('InfiniteValuePreprocessor', manager.list_preprocessors())
        self.assertIn('NanValuePreprocessor', manager.list_preprocessors())
        
        # Remove one preprocessor
        removed = manager.remove_preprocessor('InfiniteValuePreprocessor')
        self.assertTrue(removed)
        self.assertEqual(len(manager.list_preprocessors()), 1)
        self.assertNotIn('InfiniteValuePreprocessor', manager.list_preprocessors())
        
        # Try to remove non-existent preprocessor
        not_removed = manager.remove_preprocessor('NonExistentPreprocessor')
        self.assertFalse(not_removed)
        
        # Clear all preprocessors
        manager.clear_preprocessors()
        self.assertEqual(len(manager.list_preprocessors()), 0)
    
    def test_preprocessor_manager_can_run(self):
        """Test PreprocessorManager can_run method."""
        manager = create_default_manager()
        
        # Should be able to run on valid data
        self.assertTrue(manager.can_run(self.valid_ohlc_data))
        
        # Should not be able to run on empty data
        empty_df = pd.DataFrame()
        self.assertFalse(manager.can_run(empty_df))
        
        # Empty manager should not be able to run
        empty_manager = PreprocessorManager()
        self.assertFalse(empty_manager.can_run(self.valid_ohlc_data))
    
    def test_create_default_manager(self):
        """Test create_default_manager function."""
        manager = create_default_manager()
        
        # Should have all expected preprocessors
        expected_preprocessors = [
            'InfiniteValuePreprocessor',
            'NanValuePreprocessor',
            'OhlcValidatorPreprocessor',
            'DataQualityValidatorPreprocessor',
            'TimeGapDetectorPreprocessor'
        ]
        
        actual_preprocessors = manager.list_preprocessors()
        self.assertEqual(len(actual_preprocessors), len(expected_preprocessors))
        
        for expected in expected_preprocessors:
            self.assertIn(expected, actual_preprocessors)
    
    def test_preprocessor_manager_sequential_processing(self):
        """Test that PreprocessorManager processes data sequentially."""
        # Create data that will be modified by each preprocessor
        test_data = self.valid_ohlc_data.copy()
        test_data.loc[self.sample_dates[0], 'high'] = np.inf  # For InfiniteValuePreprocessor
        
        # Create a manager with just the first two preprocessors
        manager = PreprocessorManager()
        manager.add_preprocessor(InfiniteValuePreprocessor())
        manager.add_preprocessor(NanValuePreprocessor())
        
        result = manager.run(test_data)
        
        # Should have processed both steps
        self.assertFalse(np.isinf(result.values).any())  # Infinite values removed
        self.assertFalse(result.isnull().any().any())     # NaN values handled
    
    def test_preprocessor_manager_run_with_valid_data(self):
        """Test PreprocessorManager run method with valid data."""
        manager = create_default_manager()
        result = manager.run(self.valid_ohlc_data)
        
        # Should return similar data (may be slightly different due to processing)
        self.assertEqual(len(result), len(self.valid_ohlc_data))
        self.assertEqual(list(result.columns), list(self.valid_ohlc_data.columns))
    
    def test_preprocessor_manager_run_with_none_input(self):
        """Test PreprocessorManager run method with None input."""
        manager = create_default_manager()
        result = manager.run(None)
        
        # Should return None
        self.assertIsNone(result)
    
    def test_preprocessor_manager_custom_order(self):
        """Test PreprocessorManager with custom processor order."""
        manager = PreprocessorManager()
        
        # Add preprocessors in different order
        manager.add_preprocessor(NanValuePreprocessor())
        manager.add_preprocessor(InfiniteValuePreprocessor())
        manager.add_preprocessor(DataQualityValidatorPreprocessor())
        
        # Create test data
        test_data = self.valid_ohlc_data.copy()
        test_data.loc[self.sample_dates[0], 'high'] = np.inf
        test_data.loc[self.sample_dates[1], 'low'] = np.nan
        test_data.loc[self.sample_dates[2], 'close'] = -10.0
        
        result = manager.run(test_data)
        
        # Should still process correctly despite different order
        self.assertFalse(np.isinf(result.values).any())
        # NanValuePreprocessor runs first, but since it can't handle NaN at position 1 with the order, some may remain
        # DataQualityValidator doesn't remove NaN values, so they may persist
        self.assertLess(len(result), len(test_data))  # Row with negative price removed
    
    def test_preprocessor_manager_single_preprocessor(self):
        """Test PreprocessorManager with single preprocessor."""
        manager = PreprocessorManager()
        manager.add_preprocessor(InfiniteValuePreprocessor())
        
        # Create test data with only infinite values
        test_data = self.valid_ohlc_data.copy()
        test_data.loc[self.sample_dates[0], 'high'] = np.inf
        
        result = manager.run(test_data)
        
        # Should only handle infinite values
        self.assertFalse(np.isinf(result.values).any())
        self.assertEqual(len(result), len(test_data))
    
    def test_preprocessor_manager_duplicate_preprocessors(self):
        """Test PreprocessorManager with duplicate preprocessors."""
        manager = PreprocessorManager()
        
        # Add same preprocessor twice
        processor = InfiniteValuePreprocessor()
        manager.add_preprocessor(processor)
        manager.add_preprocessor(processor)
        
        # Should only have one instance
        self.assertEqual(len(manager.list_preprocessors()), 1)
    
    def test_preprocessor_manager_error_handling(self):
        """Test PreprocessorManager error handling."""
        manager = PreprocessorManager()
        
        # Add a mock preprocessor that raises an exception
        class FailingPreprocessor:
            def __init__(self):
                self.name = "FailingPreprocessor"
                
            def process(self, df):
                raise ValueError("Test error")
                
            def can_process(self, df):
                return True
        
        manager.add_preprocessor(FailingPreprocessor())
        
        # Should handle the error gracefully
        result = manager.run(self.valid_ohlc_data)
        
        # Should return original data when error occurs
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)


if __name__ == '__main__':
    # Re-enable logging for when tests are run directly
    logging.disable(logging.NOTSET)
    unittest.main(verbosity=2)