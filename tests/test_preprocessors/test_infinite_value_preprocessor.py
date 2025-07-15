import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data.preprocessors import InfiniteValuePreprocessor

# Disable logging for tests to keep output clean
logging.disable(logging.CRITICAL)


class TestInfiniteValuePreprocessor(unittest.TestCase):
    
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
        
        self.preprocessor = InfiniteValuePreprocessor()
    
    def test_process_with_infinite_values(self):
        """Test InfiniteValuePreprocessor with infinite values."""
        # Create data with infinite values
        df_with_inf = self.valid_ohlc_data.copy()
        df_with_inf.loc[self.sample_dates[0], 'high'] = np.inf
        df_with_inf.loc[self.sample_dates[1], 'low'] = -np.inf
        
        result = self.preprocessor.process(df_with_inf)
        
        # Check that infinite values are replaced with NaN
        self.assertTrue(pd.isna(result.loc[self.sample_dates[0], 'high']))
        self.assertTrue(pd.isna(result.loc[self.sample_dates[1], 'low']))
        
        # Check that other values remain unchanged
        self.assertEqual(result.loc[self.sample_dates[2], 'high'], 107.0)
    
    def test_process_no_infinite_values(self):
        """Test InfiniteValuePreprocessor when no infinite values exist."""
        result = self.preprocessor.process(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_process_with_positive_infinity(self):
        """Test InfiniteValuePreprocessor with only positive infinity."""
        df_with_pos_inf = self.valid_ohlc_data.copy()
        df_with_pos_inf.loc[self.sample_dates[0], 'high'] = np.inf
        df_with_pos_inf.loc[self.sample_dates[2], 'close'] = np.inf
        
        result = self.preprocessor.process(df_with_pos_inf)
        
        # Check that positive infinite values are replaced with NaN
        self.assertTrue(pd.isna(result.loc[self.sample_dates[0], 'high']))
        self.assertTrue(pd.isna(result.loc[self.sample_dates[2], 'close']))
        
        # Check that other values remain unchanged
        self.assertEqual(result.loc[self.sample_dates[1], 'high'], 106.0)
    
    def test_process_with_negative_infinity(self):
        """Test InfiniteValuePreprocessor with only negative infinity."""
        df_with_neg_inf = self.valid_ohlc_data.copy()
        df_with_neg_inf.loc[self.sample_dates[1], 'low'] = -np.inf
        df_with_neg_inf.loc[self.sample_dates[3], 'open'] = -np.inf
        
        result = self.preprocessor.process(df_with_neg_inf)
        
        # Check that negative infinite values are replaced with NaN
        self.assertTrue(pd.isna(result.loc[self.sample_dates[1], 'low']))
        self.assertTrue(pd.isna(result.loc[self.sample_dates[3], 'open']))
        
        # Check that other values remain unchanged
        self.assertEqual(result.loc[self.sample_dates[0], 'low'], 95.0)
    
    def test_process_empty_dataframe(self):
        """Test InfiniteValuePreprocessor with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.preprocessor.process(empty_df)
        
        # Should return empty DataFrame
        self.assertTrue(result.empty)
    
    def test_process_all_infinite_values(self):
        """Test InfiniteValuePreprocessor with all infinite values."""
        df_all_inf = pd.DataFrame({
            'col1': [np.inf, -np.inf, np.inf],
            'col2': [-np.inf, np.inf, -np.inf],
            'col3': [np.inf, np.inf, np.inf]
        })
        
        result = self.preprocessor.process(df_all_inf)
        
        # All values should be NaN
        self.assertTrue(result.isna().all().all())
        
        # Shape should be preserved
        self.assertEqual(result.shape, df_all_inf.shape)


if __name__ == '__main__':
    # Re-enable logging for when tests are run directly
    logging.disable(logging.NOTSET)
    unittest.main(verbosity=2)