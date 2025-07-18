import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data.preprocessors import TimeGapDetectorPreprocessor

# Disable logging for tests to keep output clean
logging.disable(logging.CRITICAL)


class TestTimeGapDetectorPreprocessor(unittest.TestCase):
    
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
        
        self.preprocessor = TimeGapDetectorPreprocessor()
    
    def test_process_no_gaps(self):
        """Test TimeGapDetectorPreprocessor with no gaps."""
        result = self.preprocessor.process(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_process_with_gaps(self):
        """Test TimeGapDetectorPreprocessor with gaps."""
        # Create data with a gap (skip day 3)
        gapped_dates = [self.sample_dates[0], self.sample_dates[1], self.sample_dates[3], self.sample_dates[4]]
        gapped_data = self.valid_ohlc_data.loc[gapped_dates].copy()
        
        result = self.preprocessor.process(gapped_data)
        
        # Should return original data (gaps are logged, not filled)
        pd.testing.assert_frame_equal(result, gapped_data)
    
    def test_process_non_datetime_index(self):
        """Test TimeGapDetectorPreprocessor with non-datetime index."""
        non_datetime_data = self.valid_ohlc_data.reset_index()
        
        result = self.preprocessor.process(non_datetime_data)
        
        # Should return original data
        pd.testing.assert_frame_equal(result, non_datetime_data)
    
    def test_process_insufficient_data(self):
        """Test TimeGapDetectorPreprocessor with insufficient data."""
        single_row_data = self.valid_ohlc_data.iloc[:1].copy()
        
        result = self.preprocessor.process(single_row_data)
        
        # Should return original data
        pd.testing.assert_frame_equal(result, single_row_data)
    
    def test_process_empty_dataframe(self):
        """Test TimeGapDetectorPreprocessor with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.preprocessor.process(empty_df)
        
        # Should return empty DataFrame
        self.assertTrue(result.empty)
    
    def test_process_two_rows_no_gap(self):
        """Test TimeGapDetectorPreprocessor with two consecutive rows."""
        two_row_data = self.valid_ohlc_data.iloc[:2].copy()
        
        result = self.preprocessor.process(two_row_data)
        
        # Should return original data
        pd.testing.assert_frame_equal(result, two_row_data)
    
    def test_process_two_rows_with_gap(self):
        """Test TimeGapDetectorPreprocessor with two rows having a gap."""
        # Create data with first and third dates (gap in between)
        gapped_dates = [self.sample_dates[0], self.sample_dates[2]]
        gapped_data = self.valid_ohlc_data.loc[gapped_dates].copy()
        
        result = self.preprocessor.process(gapped_data)
        
        # Should return original data (gap is logged, not filled)
        pd.testing.assert_frame_equal(result, gapped_data)
    
    def test_process_hourly_data(self):
        """Test TimeGapDetectorPreprocessor with hourly data."""
        hourly_dates = pd.date_range('2024-01-01', periods=24, freq='h')
        hourly_data = pd.DataFrame({
            'open': range(100, 124),
            'high': range(105, 129),
            'low': range(95, 119),
            'close': range(102, 126),
            'volume': range(1000, 1024)
        }, index=hourly_dates)
        
        result = self.preprocessor.process(hourly_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, hourly_data)
    
    def test_process_hourly_data_with_gaps(self):
        """Test TimeGapDetectorPreprocessor with hourly data having gaps."""
        hourly_dates = pd.date_range('2024-01-01', periods=24, freq='h')
        hourly_data = pd.DataFrame({
            'open': range(100, 124),
            'high': range(105, 129),
            'low': range(95, 119),
            'close': range(102, 126),
            'volume': range(1000, 1024)
        }, index=hourly_dates)
        
        # Remove some hours to create gaps
        gapped_hours = hourly_data.drop(index=[hourly_dates[5], hourly_dates[10], hourly_dates[15]])
        
        result = self.preprocessor.process(gapped_hours)
        
        # Should return original data (gaps are logged, not filled)
        pd.testing.assert_frame_equal(result, gapped_hours)
    
    def test_process_minute_data(self):
        """Test TimeGapDetectorPreprocessor with minute data."""
        minute_dates = pd.date_range('2024-01-01 09:00', periods=60, freq='min')
        minute_data = pd.DataFrame({
            'open': range(100, 160),
            'high': range(105, 165),
            'low': range(95, 155),
            'close': range(102, 162),
            'volume': range(1000, 1060)
        }, index=minute_dates)
        
        result = self.preprocessor.process(minute_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, minute_data)
    
    def test_process_irregular_frequency(self):
        """Test TimeGapDetectorPreprocessor with irregular frequency."""
        # Create data with irregular intervals
        irregular_dates = [
            pd.Timestamp('2024-01-01 09:00'),
            pd.Timestamp('2024-01-01 09:05'),
            pd.Timestamp('2024-01-01 09:07'),
            pd.Timestamp('2024-01-01 09:15'),
            pd.Timestamp('2024-01-01 09:20')
        ]
        irregular_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=irregular_dates)
        
        result = self.preprocessor.process(irregular_data)
        
        # Should return original data
        pd.testing.assert_frame_equal(result, irregular_data)
    
    def test_process_unsorted_datetime_index(self):
        """Test TimeGapDetectorPreprocessor with unsorted datetime index."""
        unsorted_data = self.valid_ohlc_data.copy()
        # Reverse the order
        unsorted_data = unsorted_data.iloc[::-1]
        
        result = self.preprocessor.process(unsorted_data)
        
        # Should return original data (unsorted)
        pd.testing.assert_frame_equal(result, unsorted_data)
    
    def test_process_large_gap(self):
        """Test TimeGapDetectorPreprocessor with a large gap."""
        # Create data with a large gap (skip multiple days)
        large_gap_dates = [self.sample_dates[0], self.sample_dates[4]]  # Skip days 2, 3, 4
        large_gap_data = self.valid_ohlc_data.loc[large_gap_dates].copy()
        
        result = self.preprocessor.process(large_gap_data)
        
        # Should return original data (gap is logged, not filled)
        pd.testing.assert_frame_equal(result, large_gap_data)


if __name__ == '__main__':
    # Re-enable logging for when tests are run directly
    logging.disable(logging.NOTSET)
    unittest.main(verbosity=2)