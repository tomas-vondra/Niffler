import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data.preprocessors import DataQualityValidatorPreprocessor

# Disable logging for tests to keep output clean
logging.disable(logging.CRITICAL)


class TestDataQualityValidatorPreprocessor(unittest.TestCase):
    
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
        
        self.preprocessor = DataQualityValidatorPreprocessor()
    
    def test_process_valid_data(self):
        """Test DataQualityValidatorPreprocessor with valid data."""
        result = self.preprocessor.process(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_process_negative_prices(self):
        """Test DataQualityValidatorPreprocessor with negative prices."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'open'] = -10.0
        invalid_data.loc[self.sample_dates[1], 'high'] = -5.0
        
        result = self.preprocessor.process(invalid_data)
        
        # Rows with negative prices should be removed
        self.assertEqual(len(result), 3)
        self.assertNotIn(self.sample_dates[0], result.index)
        self.assertNotIn(self.sample_dates[1], result.index)
    
    def test_process_zero_prices(self):
        """Test DataQualityValidatorPreprocessor with zero prices."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'open'] = 0.0
        invalid_data.loc[self.sample_dates[1], 'close'] = 0.0
        
        result = self.preprocessor.process(invalid_data)
        
        # Rows with zero prices should be removed
        self.assertEqual(len(result), 3)
        self.assertNotIn(self.sample_dates[0], result.index)
        self.assertNotIn(self.sample_dates[1], result.index)
    
    def test_process_negative_volume(self):
        """Test DataQualityValidatorPreprocessor with negative volume."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[1], 'volume'] = -100
        
        result = self.preprocessor.process(invalid_data)
        
        # Row with negative volume should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[1], result.index)
    
    def test_process_zero_volume(self):
        """Test DataQualityValidatorPreprocessor with zero volume."""
        valid_data = self.valid_ohlc_data.copy()
        valid_data.loc[self.sample_dates[1], 'volume'] = 0
        
        result = self.preprocessor.process(valid_data)
        
        # Zero volume should be allowed
        self.assertEqual(len(result), 5)
        self.assertIn(self.sample_dates[1], result.index)
        self.assertEqual(result.loc[self.sample_dates[1], 'volume'], 0)
    
    def test_process_duplicate_timestamps(self):
        """Test DataQualityValidatorPreprocessor with duplicate timestamps."""
        duplicate_data = self.valid_ohlc_data.copy()
        # Add duplicate row
        duplicate_row = duplicate_data.iloc[0:1].copy()
        duplicate_data = pd.concat([duplicate_data, duplicate_row])
        
        result = self.preprocessor.process(duplicate_data)
        
        # Should remove duplicate
        self.assertEqual(len(result), 5)  # Original length
        
        # Should keep first occurrence
        self.assertIn(self.sample_dates[0], result.index)
    
    def test_process_unsorted_data(self):
        """Test DataQualityValidatorPreprocessor with unsorted data."""
        unsorted_data = self.valid_ohlc_data.copy()
        # Reverse the order
        unsorted_data = unsorted_data.iloc[::-1]
        
        result = self.preprocessor.process(unsorted_data)
        
        # Should be sorted
        self.assertTrue(result.index.is_monotonic_increasing)
        
        # Should have same data, just sorted
        self.assertEqual(len(result), 5)
    
    def test_process_empty_dataframe(self):
        """Test DataQualityValidatorPreprocessor with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.preprocessor.process(empty_df)
        
        # Should return empty DataFrame
        self.assertTrue(result.empty)
    
    def test_process_without_volume_column(self):
        """Test DataQualityValidatorPreprocessor without volume column."""
        no_volume_data = self.valid_ohlc_data[['open', 'high', 'low', 'close']].copy()
        
        result = self.preprocessor.process(no_volume_data)
        
        # Should work without volume column
        pd.testing.assert_frame_equal(result, no_volume_data)
    
    def test_process_without_price_columns(self):
        """Test DataQualityValidatorPreprocessor without price columns."""
        no_price_data = self.valid_ohlc_data[['volume']].copy()
        
        result = self.preprocessor.process(no_price_data)
        
        # Should work without price columns
        pd.testing.assert_frame_equal(result, no_price_data)
    
    def test_process_mixed_case_columns(self):
        """Test DataQualityValidatorPreprocessor with mixed case column names."""
        mixed_case_data = self.valid_ohlc_data.copy()
        mixed_case_data.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        
        result = self.preprocessor.process(mixed_case_data)
        
        # Should work with mixed case
        pd.testing.assert_frame_equal(result, mixed_case_data)
    
    def test_process_multiple_invalid_rows(self):
        """Test DataQualityValidatorPreprocessor with multiple invalid rows."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'open'] = -10.0  # Negative price
        invalid_data.loc[self.sample_dates[1], 'volume'] = -100  # Negative volume
        invalid_data.loc[self.sample_dates[2], 'close'] = 0.0   # Zero price
        
        result = self.preprocessor.process(invalid_data)
        
        # All invalid rows should be removed
        self.assertEqual(len(result), 2)
        self.assertNotIn(self.sample_dates[0], result.index)
        self.assertNotIn(self.sample_dates[1], result.index)
        self.assertNotIn(self.sample_dates[2], result.index)
    
    def test_process_with_nan_values(self):
        """Test DataQualityValidatorPreprocessor with NaN values."""
        data_with_nan = self.valid_ohlc_data.copy()
        data_with_nan.loc[self.sample_dates[0], 'open'] = np.nan
        data_with_nan.loc[self.sample_dates[1], 'volume'] = np.nan
        
        result = self.preprocessor.process(data_with_nan)
        
        # DataQualityValidator doesn't explicitly remove NaN values, only invalid data types
        # NaN values are passed through unchanged
        self.assertEqual(len(result), 5)
        self.assertIn(self.sample_dates[0], result.index)
        self.assertIn(self.sample_dates[1], result.index)
        self.assertTrue(pd.isna(result.loc[self.sample_dates[0], 'open']))
        self.assertTrue(pd.isna(result.loc[self.sample_dates[1], 'volume']))
    
    def test_process_non_datetime_index(self):
        """Test DataQualityValidatorPreprocessor with non-datetime index."""
        non_datetime_data = self.valid_ohlc_data.reset_index()
        
        result = self.preprocessor.process(non_datetime_data)
        
        # Should work with non-datetime index
        self.assertEqual(len(result), 5)
        
        # DataQualityValidator sorts by index regardless of type
        self.assertTrue(result.index.is_monotonic_increasing)
    
    def test_process_complex_duplicates(self):
        """Test DataQualityValidatorPreprocessor with complex duplicate scenarios."""
        # Create data with multiple duplicates
        complex_data = pd.concat([
            self.valid_ohlc_data.iloc[0:1],  # First row
            self.valid_ohlc_data.iloc[0:1],  # Duplicate of first row
            self.valid_ohlc_data.iloc[1:3],  # Second and third rows
            self.valid_ohlc_data.iloc[1:2],  # Duplicate of second row
            self.valid_ohlc_data.iloc[3:5]   # Fourth and fifth rows
        ])
        
        result = self.preprocessor.process(complex_data)
        
        # Should remove all duplicates, keeping only first occurrence
        self.assertEqual(len(result), 5)
        
        # Should be sorted
        self.assertTrue(result.index.is_monotonic_increasing)


if __name__ == '__main__':
    # Re-enable logging for when tests are run directly
    logging.disable(logging.NOTSET)
    unittest.main(verbosity=2)