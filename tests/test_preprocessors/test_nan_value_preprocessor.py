import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data.preprocessors import NanValuePreprocessor

# Disable logging for tests to keep output clean
logging.disable(logging.CRITICAL)


class TestNanValuePreprocessor(unittest.TestCase):
    
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
        
        self.preprocessor = NanValuePreprocessor()
    
    def test_process_with_nan_values(self):
        """Test NanValuePreprocessor with forward-fill."""
        # Create data with NaN values
        df_with_nan = self.valid_ohlc_data.copy()
        df_with_nan.loc[self.sample_dates[1], 'high'] = np.nan
        df_with_nan.loc[self.sample_dates[2], 'close'] = np.nan
        
        result = self.preprocessor.process(df_with_nan)
        
        # Check that NaN values are forward-filled
        self.assertEqual(result.loc[self.sample_dates[1], 'high'], 105.0)  # Forward-filled from previous day
        self.assertEqual(result.loc[self.sample_dates[2], 'close'], 103.0)  # Forward-filled from previous day
    
    def test_process_with_leading_nan(self):
        """Test NanValuePreprocessor with leading NaN values."""
        df_with_leading_nan = self.valid_ohlc_data.copy()
        df_with_leading_nan.loc[self.sample_dates[0], 'open'] = np.nan
        df_with_leading_nan.loc[self.sample_dates[0], 'high'] = np.nan
        
        result = self.preprocessor.process(df_with_leading_nan)
        
        # Leading NaN should be backward-filled
        self.assertEqual(result.loc[self.sample_dates[0], 'open'], 101.0)  # Backward-filled
        self.assertEqual(result.loc[self.sample_dates[0], 'high'], 106.0)  # Backward-filled
    
    def test_process_no_nan_values(self):
        """Test NanValuePreprocessor when no NaN values exist."""
        result = self.preprocessor.process(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_process_with_trailing_nan(self):
        """Test NanValuePreprocessor with trailing NaN values."""
        df_with_trailing_nan = self.valid_ohlc_data.copy()
        df_with_trailing_nan.loc[self.sample_dates[4], 'close'] = np.nan
        df_with_trailing_nan.loc[self.sample_dates[4], 'volume'] = np.nan
        
        result = self.preprocessor.process(df_with_trailing_nan)
        
        # Trailing NaN should be forward-filled
        self.assertEqual(result.loc[self.sample_dates[4], 'close'], 105.0)  # Forward-filled
        self.assertEqual(result.loc[self.sample_dates[4], 'volume'], 1300)  # Forward-filled
    
    def test_process_empty_dataframe(self):
        """Test NanValuePreprocessor with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.preprocessor.process(empty_df)
        
        # Should return empty DataFrame
        self.assertTrue(result.empty)
    
    def test_process_all_nan_values(self):
        """Test NanValuePreprocessor with all NaN values."""
        df_all_nan = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [np.nan, np.nan, np.nan],
            'col3': [np.nan, np.nan, np.nan]
        })
        
        result = self.preprocessor.process(df_all_nan)
        
        # All NaN values should be dropped since there's nothing to fill with
        self.assertTrue(result.empty)
    
    def test_process_mixed_nan_pattern(self):
        """Test NanValuePreprocessor with mixed NaN patterns."""
        df_mixed = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0, np.nan, 5.0],
            'col2': [np.nan, 2.0, np.nan, 4.0, np.nan],
            'col3': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        result = self.preprocessor.process(df_mixed)
        
        # Check forward-fill behavior
        self.assertEqual(result.iloc[1, 0], 1.0)  # Forward-filled
        self.assertEqual(result.iloc[3, 0], 3.0)  # Forward-filled
        
        # Check backward-fill for leading NaN
        self.assertEqual(result.iloc[0, 1], 2.0)  # Backward-filled
        
        # Check that non-NaN values remain unchanged
        self.assertEqual(result.iloc[2, 2], 30.0)
        
        # Check forward-fill for trailing NaN
        self.assertEqual(result.iloc[4, 1], 4.0)  # Forward-filled
    
    def test_process_single_row_with_nan(self):
        """Test NanValuePreprocessor with single row containing NaN."""
        single_row_nan = pd.DataFrame({
            'col1': [np.nan],
            'col2': [5.0],
            'col3': [np.nan]
        })
        
        result = self.preprocessor.process(single_row_nan)
        
        # The row should be dropped due to persistent NaN values
        self.assertTrue(result.empty)


if __name__ == '__main__':
    # Re-enable logging for when tests are run directly
    logging.disable(logging.NOTSET)
    unittest.main(verbosity=2)