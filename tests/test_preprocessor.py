import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import core modules
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

class TestPreprocessor(unittest.TestCase):
    
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
    
    def test_infinite_value_preprocessor(self):
        """Test InfiniteValuePreprocessor."""
        # Create data with infinite values
        df_with_inf = self.valid_ohlc_data.copy()
        df_with_inf.loc[self.sample_dates[0], 'high'] = np.inf
        df_with_inf.loc[self.sample_dates[1], 'low'] = -np.inf
        
        preprocessor = InfiniteValuePreprocessor()
        result = preprocessor.process(df_with_inf)
        
        # Check that infinite values are replaced with NaN
        self.assertTrue(pd.isna(result.loc[self.sample_dates[0], 'high']))
        self.assertTrue(pd.isna(result.loc[self.sample_dates[1], 'low']))
        
        # Check that other values remain unchanged
        self.assertEqual(result.loc[self.sample_dates[2], 'high'], 107.0)
    
    def test_infinite_value_preprocessor_no_inf(self):
        """Test InfiniteValuePreprocessor when no infinite values exist."""
        preprocessor = InfiniteValuePreprocessor()
        result = preprocessor.process(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_nan_value_preprocessor(self):
        """Test NanValuePreprocessor with forward-fill."""
        # Create data with NaN values
        df_with_nan = self.valid_ohlc_data.copy()
        df_with_nan.loc[self.sample_dates[1], 'high'] = np.nan
        df_with_nan.loc[self.sample_dates[2], 'close'] = np.nan
        
        preprocessor = NanValuePreprocessor()
        result = preprocessor.process(df_with_nan)
        
        # Check that NaN values are forward-filled
        self.assertEqual(result.loc[self.sample_dates[1], 'high'], 105.0)  # Forward-filled from previous day
        self.assertEqual(result.loc[self.sample_dates[2], 'close'], 103.0)  # Forward-filled from previous day
    
    def test_nan_value_preprocessor_leading_nan(self):
        """Test NanValuePreprocessor with leading NaN values."""
        df_with_leading_nan = self.valid_ohlc_data.copy()
        df_with_leading_nan.loc[self.sample_dates[0], 'open'] = np.nan
        
        preprocessor = NanValuePreprocessor()
        result = preprocessor.process(df_with_leading_nan)
        
        # Leading NaN should be backward-filled
        self.assertEqual(result.loc[self.sample_dates[0], 'open'], 101.0)  # Backward-filled
    
    def test_ohlc_validator_preprocessor_valid(self):
        """Test OhlcValidatorPreprocessor with valid data."""
        preprocessor = OhlcValidatorPreprocessor()
        result = preprocessor.process(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_ohlc_validator_preprocessor_invalid_high_low(self):
        """Test OhlcValidatorPreprocessor with High < Low."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'high'] = 90.0  # High < Low
        
        preprocessor = OhlcValidatorPreprocessor()
        result = preprocessor.process(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[0], result.index)
    
    def test_ohlc_validator_preprocessor_invalid_high_open(self):
        """Test OhlcValidatorPreprocessor with High < Open."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[1], 'high'] = 90.0  # High < Open
        
        preprocessor = OhlcValidatorPreprocessor()
        result = preprocessor.process(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[1], result.index)
    
    def test_ohlc_validator_preprocessor_invalid_low_close(self):
        """Test OhlcValidatorPreprocessor with Low > Close."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[2], 'low'] = 110.0  # Low > Close
        
        preprocessor = OhlcValidatorPreprocessor()
        result = preprocessor.process(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[2], result.index)
    
    def test_ohlc_validator_preprocessor_missing_columns(self):
        """Test OhlcValidatorPreprocessor with missing columns."""
        incomplete_data = self.valid_ohlc_data[['open', 'high', 'volume']].copy()  # Missing 'low' and 'close'
        
        preprocessor = OhlcValidatorPreprocessor()
        result = preprocessor.process(incomplete_data)
        
        # Should return original data when OHLC columns are missing
        pd.testing.assert_frame_equal(result, incomplete_data)
    
    def test_ohlc_validator_preprocessor_case_insensitive(self):
        """Test OhlcValidatorPreprocessor with different case column names."""
        case_data = self.valid_ohlc_data.copy()
        case_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        preprocessor = OhlcValidatorPreprocessor()
        result = preprocessor.process(case_data)
        
        # Should work with different case
        pd.testing.assert_frame_equal(result, case_data)
    
    def test_time_gap_detector_preprocessor_no_gaps(self):
        """Test TimeGapDetectorPreprocessor with no gaps."""
        preprocessor = TimeGapDetectorPreprocessor()
        result = preprocessor.process(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_time_gap_detector_preprocessor_with_gaps(self):
        """Test TimeGapDetectorPreprocessor with gaps."""
        # Create data with a gap (skip day 3)
        gapped_dates = [self.sample_dates[0], self.sample_dates[1], self.sample_dates[3], self.sample_dates[4]]
        gapped_data = self.valid_ohlc_data.loc[gapped_dates].copy()
        
        preprocessor = TimeGapDetectorPreprocessor()
        result = preprocessor.process(gapped_data)
        
        # Should return original data (gaps are logged, not filled)
        pd.testing.assert_frame_equal(result, gapped_data)
    
    def test_time_gap_detector_preprocessor_non_datetime_index(self):
        """Test TimeGapDetectorPreprocessor with non-datetime index."""
        non_datetime_data = self.valid_ohlc_data.reset_index()
        
        preprocessor = TimeGapDetectorPreprocessor()
        result = preprocessor.process(non_datetime_data)
        
        # Should return original data
        pd.testing.assert_frame_equal(result, non_datetime_data)
    
    def test_time_gap_detector_preprocessor_insufficient_data(self):
        """Test TimeGapDetectorPreprocessor with insufficient data."""
        single_row_data = self.valid_ohlc_data.iloc[:1].copy()
        
        preprocessor = TimeGapDetectorPreprocessor()
        result = preprocessor.process(single_row_data)
        
        # Should return original data
        pd.testing.assert_frame_equal(result, single_row_data)
    
    def test_data_quality_validator_preprocessor_negative_prices(self):
        """Test DataQualityValidatorPreprocessor with negative prices."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'open'] = -10.0
        
        preprocessor = DataQualityValidatorPreprocessor()
        result = preprocessor.process(invalid_data)
        
        # Row with negative price should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[0], result.index)
    
    def test_data_quality_validator_preprocessor_negative_volume(self):
        """Test DataQualityValidatorPreprocessor with negative volume."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[1], 'volume'] = -100
        
        preprocessor = DataQualityValidatorPreprocessor()
        result = preprocessor.process(invalid_data)
        
        # Row with negative volume should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[1], result.index)
    
    def test_data_quality_validator_preprocessor_duplicate_timestamps(self):
        """Test DataQualityValidatorPreprocessor with duplicate timestamps."""
        duplicate_data = self.valid_ohlc_data.copy()
        # Add duplicate row
        duplicate_row = duplicate_data.iloc[0:1].copy()
        duplicate_data = pd.concat([duplicate_data, duplicate_row])
        
        preprocessor = DataQualityValidatorPreprocessor()
        result = preprocessor.process(duplicate_data)
        
        # Should remove duplicate
        self.assertEqual(len(result), 5)  # Original length
    
    def test_data_quality_validator_preprocessor_unsorted_data(self):
        """Test DataQualityValidatorPreprocessor with unsorted data."""
        unsorted_data = self.valid_ohlc_data.copy()
        # Reverse the order
        unsorted_data = unsorted_data.iloc[::-1]
        
        preprocessor = DataQualityValidatorPreprocessor()
        result = preprocessor.process(unsorted_data)
        
        # Should be sorted
        self.assertTrue(result.index.is_monotonic_increasing)
    
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

if __name__ == '__main__':
    # Re-enable logging for when tests are run directly
    logging.disable(logging.NOTSET)
    unittest.main()