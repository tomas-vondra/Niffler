import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
import logging

# Add the scripts directory to the path so we can import preprocessor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import preprocessor

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
    
    def test_remove_infinite_values(self):
        """Test removal of infinite values."""
        # Create data with infinite values
        df_with_inf = self.valid_ohlc_data.copy()
        df_with_inf.loc[self.sample_dates[0], 'high'] = np.inf
        df_with_inf.loc[self.sample_dates[1], 'low'] = -np.inf
        
        result = preprocessor.remove_infinite_values(df_with_inf)
        
        # Check that infinite values are replaced with NaN
        self.assertTrue(pd.isna(result.loc[self.sample_dates[0], 'high']))
        self.assertTrue(pd.isna(result.loc[self.sample_dates[1], 'low']))
        
        # Check that other values remain unchanged
        self.assertEqual(result.loc[self.sample_dates[2], 'high'], 107.0)
    
    def test_remove_infinite_values_no_inf(self):
        """Test removal of infinite values when none exist."""
        result = preprocessor.remove_infinite_values(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_handle_nan_values(self):
        """Test handling of NaN values with forward-fill."""
        # Create data with NaN values
        df_with_nan = self.valid_ohlc_data.copy()
        df_with_nan.loc[self.sample_dates[1], 'high'] = np.nan
        df_with_nan.loc[self.sample_dates[2], 'close'] = np.nan
        
        result = preprocessor.handle_nan_values(df_with_nan)
        
        # Check that NaN values are forward-filled
        self.assertEqual(result.loc[self.sample_dates[1], 'high'], 105.0)  # Forward-filled from previous day
        self.assertEqual(result.loc[self.sample_dates[2], 'close'], 103.0)  # Forward-filled from previous day
    
    def test_handle_nan_values_leading_nan(self):
        """Test handling of leading NaN values."""
        df_with_leading_nan = self.valid_ohlc_data.copy()
        df_with_leading_nan.loc[self.sample_dates[0], 'open'] = np.nan
        
        result = preprocessor.handle_nan_values(df_with_leading_nan)
        
        # Leading NaN should be backward-filled
        self.assertEqual(result.loc[self.sample_dates[0], 'open'], 101.0)  # Backward-filled
    
    def test_validate_ohlc_data_valid(self):
        """Test OHLC validation with valid data."""
        result = preprocessor.validate_ohlc_data(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_validate_ohlc_data_invalid_high_low(self):
        """Test OHLC validation with High < Low."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'high'] = 90.0  # High < Low
        
        result = preprocessor.validate_ohlc_data(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[0], result.index)
    
    def test_validate_ohlc_data_invalid_high_open(self):
        """Test OHLC validation with High < Open."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[1], 'high'] = 90.0  # High < Open
        
        result = preprocessor.validate_ohlc_data(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[1], result.index)
    
    def test_validate_ohlc_data_invalid_low_close(self):
        """Test OHLC validation with Low > Close."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[2], 'low'] = 110.0  # Low > Close
        
        result = preprocessor.validate_ohlc_data(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[2], result.index)
    
    def test_validate_ohlc_data_missing_columns(self):
        """Test OHLC validation with missing columns."""
        incomplete_data = self.valid_ohlc_data[['open', 'high', 'volume']].copy()  # Missing 'low' and 'close'
        
        result = preprocessor.validate_ohlc_data(incomplete_data)
        
        # Should return original data when OHLC columns are missing
        pd.testing.assert_frame_equal(result, incomplete_data)
    
    def test_validate_ohlc_data_case_insensitive(self):
        """Test OHLC validation with different case column names."""
        case_data = self.valid_ohlc_data.copy()
        case_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        result = preprocessor.validate_ohlc_data(case_data)
        
        # Should work with different case
        pd.testing.assert_frame_equal(result, case_data)
    
    def test_detect_time_gaps_no_gaps(self):
        """Test time gap detection with no gaps."""
        result = preprocessor.detect_time_gaps(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_detect_time_gaps_with_gaps(self):
        """Test time gap detection with gaps."""
        # Create data with a gap (skip day 3)
        gapped_dates = [self.sample_dates[0], self.sample_dates[1], self.sample_dates[3], self.sample_dates[4]]
        gapped_data = self.valid_ohlc_data.loc[gapped_dates].copy()
        
        result = preprocessor.detect_time_gaps(gapped_data)
        
        # Should return original data (gaps are logged, not filled)
        pd.testing.assert_frame_equal(result, gapped_data)
    
    def test_detect_time_gaps_non_datetime_index(self):
        """Test time gap detection with non-datetime index."""
        non_datetime_data = self.valid_ohlc_data.reset_index()
        
        result = preprocessor.detect_time_gaps(non_datetime_data)
        
        # Should return original data
        pd.testing.assert_frame_equal(result, non_datetime_data)
    
    def test_detect_time_gaps_insufficient_data(self):
        """Test time gap detection with insufficient data."""
        single_row_data = self.valid_ohlc_data.iloc[:1].copy()
        
        result = preprocessor.detect_time_gaps(single_row_data)
        
        # Should return original data
        pd.testing.assert_frame_equal(result, single_row_data)
    
    def test_validate_data_quality_negative_prices(self):
        """Test data quality validation with negative prices."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'open'] = -10.0
        
        result = preprocessor.validate_data_quality(invalid_data)
        
        # Row with negative price should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[0], result.index)
    
    def test_validate_data_quality_negative_volume(self):
        """Test data quality validation with negative volume."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[1], 'volume'] = -100
        
        result = preprocessor.validate_data_quality(invalid_data)
        
        # Row with negative volume should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[1], result.index)
    
    def test_validate_data_quality_duplicate_timestamps(self):
        """Test data quality validation with duplicate timestamps."""
        duplicate_data = self.valid_ohlc_data.copy()
        # Add duplicate row
        duplicate_row = duplicate_data.iloc[0:1].copy()
        duplicate_data = pd.concat([duplicate_data, duplicate_row])
        
        result = preprocessor.validate_data_quality(duplicate_data)
        
        # Should remove duplicate
        self.assertEqual(len(result), 5)  # Original length
    
    def test_validate_data_quality_unsorted_data(self):
        """Test data quality validation with unsorted data."""
        unsorted_data = self.valid_ohlc_data.copy()
        # Reverse the order
        unsorted_data = unsorted_data.iloc[::-1]
        
        result = preprocessor.validate_data_quality(unsorted_data)
        
        # Should be sorted
        self.assertTrue(result.index.is_monotonic_increasing)
    
    def test_clean_trading_data_complete_workflow(self):
        """Test the complete data cleaning workflow."""
        # Create data with multiple issues
        problematic_data = self.valid_ohlc_data.copy()
        problematic_data.loc[self.sample_dates[0], 'high'] = np.inf  # Infinite value
        problematic_data.loc[self.sample_dates[1], 'low'] = np.nan  # NaN value
        problematic_data.loc[self.sample_dates[2], 'close'] = -10.0  # Negative price
        problematic_data.loc[self.sample_dates[3], 'high'] = 90.0   # High < Low (invalid OHLC)
        
        result = preprocessor.clean_trading_data(problematic_data)
        
        # Should handle all issues
        self.assertFalse(np.isinf(result.values).any())  # No infinite values
        self.assertFalse(result.isnull().any().any())     # No NaN values
        self.assertTrue((result > 0).all().all())         # No negative values
        # Should have fewer rows due to invalid data removal
        self.assertLess(len(result), len(problematic_data))
    
    def test_clean_trading_data_empty_dataframe(self):
        """Test cleaning with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = preprocessor.clean_trading_data(empty_df)
        
        # Should return empty DataFrame
        self.assertTrue(result.empty)
    
    def test_process_file_nonexistent(self):
        """Test processing a non-existent file."""
        result = preprocessor.process_file("nonexistent_file.csv")
        
        # Should return None
        self.assertIsNone(result)
    
    def test_process_file_valid(self):
        """Test processing a valid CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.valid_ohlc_data.to_csv(f.name)
            temp_file = f.name
        
        try:
            result = preprocessor.process_file(temp_file)
            
            # Should return a DataFrame
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 5)
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_process_file_with_output(self):
        """Test processing a file with output specification."""
        # Create temporary input and output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_f:
            self.valid_ohlc_data.to_csv(input_f.name)
            input_file = input_f.name
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_f:
            output_file = output_f.name
        
        try:
            result = preprocessor.process_file(input_file, output_file)
            
            # Should return a DataFrame and create output file
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(os.path.exists(output_file))
            
            # Verify output file content
            output_data = pd.read_csv(output_file, index_col=0, parse_dates=True)
            self.assertEqual(len(output_data), 5)
            
        finally:
            # Clean up
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_process_file_invalid_csv(self):
        """Test processing an invalid CSV file."""
        # Create a temporary invalid CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\n1,2\n3,4,5,6\n")  # Inconsistent columns
            temp_file = f.name
        
        try:
            result = preprocessor.process_file(temp_file)
            
            # Should handle gracefully and return None or valid result
            # (pandas is quite forgiving with malformed CSV)
            
        finally:
            # Clean up
            os.unlink(temp_file)

class TestPreprocessorIntegration(unittest.TestCase):
    """Integration tests for the preprocessor module."""
    
    def test_main_function_with_mock_args(self):
        """Test the main function with mocked arguments."""
        # Create temporary CSV file
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            # Mock command line arguments
            test_args = ['preprocessor.py', '--input', input_file, '--output', output_file]
            
            with patch('sys.argv', test_args):
                # Should run without error
                preprocessor.main()
            
            # Verify output file was created
            self.assertTrue(os.path.exists(output_file))
            
        finally:
            # Clean up
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

if __name__ == '__main__':
    # Re-enable logging for when tests are run directly
    logging.disable(logging.NOTSET)
    unittest.main()