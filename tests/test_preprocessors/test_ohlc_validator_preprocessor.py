import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from niffler.data.preprocessors import OhlcValidatorPreprocessor

# Disable logging for tests to keep output clean
logging.disable(logging.CRITICAL)


class TestOhlcValidatorPreprocessor(unittest.TestCase):
    
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
        
        self.preprocessor = OhlcValidatorPreprocessor()
    
    def test_process_valid_data(self):
        """Test OhlcValidatorPreprocessor with valid data."""
        result = self.preprocessor.process(self.valid_ohlc_data)
        
        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
    
    def test_process_invalid_high_low(self):
        """Test OhlcValidatorPreprocessor with High < Low."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'high'] = 90.0  # High < Low
        
        result = self.preprocessor.process(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[0], result.index)
    
    def test_process_invalid_high_open(self):
        """Test OhlcValidatorPreprocessor with High < Open."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[1], 'high'] = 90.0  # High < Open
        
        result = self.preprocessor.process(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[1], result.index)
    
    def test_process_invalid_high_close(self):
        """Test OhlcValidatorPreprocessor with High < Close."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[2], 'high'] = 90.0  # High < Close
        
        result = self.preprocessor.process(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[2], result.index)
    
    def test_process_invalid_low_open(self):
        """Test OhlcValidatorPreprocessor with Low > Open."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[3], 'low'] = 110.0  # Low > Open
        
        result = self.preprocessor.process(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[3], result.index)
    
    def test_process_invalid_low_close(self):
        """Test OhlcValidatorPreprocessor with Low > Close."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[2], 'low'] = 110.0  # Low > Close
        
        result = self.preprocessor.process(invalid_data)
        
        # Invalid row should be removed
        self.assertEqual(len(result), 4)
        self.assertNotIn(self.sample_dates[2], result.index)
    
    def test_process_missing_columns(self):
        """Test OhlcValidatorPreprocessor with missing columns."""
        incomplete_data = self.valid_ohlc_data[['open', 'high', 'volume']].copy()  # Missing 'low' and 'close'
        
        result = self.preprocessor.process(incomplete_data)
        
        # Should return original data when OHLC columns are missing
        pd.testing.assert_frame_equal(result, incomplete_data)
    
    def test_process_case_insensitive(self):
        """Test OhlcValidatorPreprocessor with different case column names."""
        case_data = self.valid_ohlc_data.copy()
        case_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        result = self.preprocessor.process(case_data)
        
        # Should work with different case
        pd.testing.assert_frame_equal(result, case_data)
    
    def test_process_mixed_case_columns(self):
        """Test OhlcValidatorPreprocessor with mixed case column names."""
        mixed_case_data = self.valid_ohlc_data.copy()
        mixed_case_data.columns = ['OPEN', 'high', 'Low', 'CLOSE', 'volume']
        
        result = self.preprocessor.process(mixed_case_data)
        
        # Should work with mixed case
        pd.testing.assert_frame_equal(result, mixed_case_data)
    
    def test_process_multiple_invalid_rows(self):
        """Test OhlcValidatorPreprocessor with multiple invalid rows."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'high'] = 90.0  # High < Low
        invalid_data.loc[self.sample_dates[1], 'low'] = 110.0  # Low > High
        invalid_data.loc[self.sample_dates[2], 'high'] = 90.0  # High < Close
        
        result = self.preprocessor.process(invalid_data)
        
        # All invalid rows should be removed
        self.assertEqual(len(result), 2)
        self.assertNotIn(self.sample_dates[0], result.index)
        self.assertNotIn(self.sample_dates[1], result.index)
        self.assertNotIn(self.sample_dates[2], result.index)
    
    def test_process_empty_dataframe(self):
        """Test OhlcValidatorPreprocessor with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.preprocessor.process(empty_df)
        
        # Should return empty DataFrame
        self.assertTrue(result.empty)
    
    def test_process_edge_case_equal_values(self):
        """Test OhlcValidatorPreprocessor with edge case where values are equal."""
        edge_case_data = pd.DataFrame({
            'open': [100.0, 100.0],
            'high': [100.0, 100.0],
            'low': [100.0, 100.0],
            'close': [100.0, 100.0],
            'volume': [1000, 1000]
        })
        
        result = self.preprocessor.process(edge_case_data)
        
        # Should pass validation (all values equal is valid)
        pd.testing.assert_frame_equal(result, edge_case_data)
    
    def test_process_with_nan_values(self):
        """Test OhlcValidatorPreprocessor with NaN values in OHLC columns."""
        data_with_nan = self.valid_ohlc_data.copy()
        data_with_nan.loc[self.sample_dates[0], 'high'] = np.nan
        data_with_nan.loc[self.sample_dates[1], 'low'] = np.nan

        result = self.preprocessor.process(data_with_nan)

        # OhlcValidator doesn't remove rows with NaN values, only invalid OHLC relationships
        # NaN values are passed through unchanged
        self.assertEqual(len(result), 5)
        self.assertIn(self.sample_dates[0], result.index)
        self.assertIn(self.sample_dates[1], result.index)
        self.assertTrue(pd.isna(result.loc[self.sample_dates[0], 'high']))
        self.assertTrue(pd.isna(result.loc[self.sample_dates[1], 'low']))

    def test_stats_report_dropped_rows_and_rules(self):
        """Dropping rows must be reported, not silent."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'high'] = 90.0  # High < Low and High < Open/Close

        result = self.preprocessor.process(invalid_data)
        stats = result.attrs['ohlc_validation']

        self.assertEqual(stats['mode'], 'drop')
        self.assertEqual(stats['original_rows'], 5)
        self.assertEqual(stats['final_rows'], 4)
        self.assertEqual(stats['invalid_rows'], 1)
        self.assertEqual(stats['dropped_rows'], 1)
        self.assertEqual(stats['repaired_rows'], 0)
        self.assertAlmostEqual(stats['invalid_ratio'], 0.2)
        self.assertEqual(stats['rule_violations']['high_lt_low'], 1)
        self.assertEqual(stats['rule_violations']['low_gt_close'], 0)
        self.assertEqual(self.preprocessor.last_stats, stats)

    def test_stats_on_valid_data(self):
        """Valid data reports zero violations without touching the frame."""
        result = self.preprocessor.process(self.valid_ohlc_data)
        stats = result.attrs['ohlc_validation']

        self.assertEqual(stats['invalid_rows'], 0)
        self.assertEqual(stats['dropped_rows'], 0)
        self.assertEqual(stats['final_rows'], 5)
        # The caller's frame must not be annotated
        self.assertNotIn('ohlc_validation', self.valid_ohlc_data.attrs)

    def test_repair_mode_keeps_rows(self):
        """Repair mode clamps High/Low instead of punching a hole in the series."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'high'] = 90.0  # High < Low/Open/Close

        preprocessor = OhlcValidatorPreprocessor(mode='repair')
        result = preprocessor.process(invalid_data)

        self.assertEqual(len(result), 5)
        self.assertIn(self.sample_dates[0], result.index)
        # High clamped to max(open, high, low, close) = max(100, 90, 95, 102)
        self.assertEqual(result.loc[self.sample_dates[0], 'high'], 102.0)
        # Low clamped to min(...) = 90
        self.assertEqual(result.loc[self.sample_dates[0], 'low'], 90.0)
        self.assertEqual(result.attrs['ohlc_validation']['repaired_rows'], 1)
        self.assertEqual(result.attrs['ohlc_validation']['dropped_rows'], 0)

    def test_repair_mode_output_passes_validation(self):
        """A repaired frame contains no remaining OHLC violations."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[1], 'low'] = 110.0
        invalid_data.loc[self.sample_dates[3], 'high'] = 50.0

        repaired = OhlcValidatorPreprocessor(mode='repair').process(invalid_data)
        rechecked = OhlcValidatorPreprocessor(mode='flag').process(repaired)

        self.assertEqual(rechecked.attrs['ohlc_validation']['invalid_rows'], 0)

    def test_flag_mode_changes_nothing(self):
        """Flag mode reports violations but leaves the data untouched."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'high'] = 90.0

        result = OhlcValidatorPreprocessor(mode='flag').process(invalid_data)

        pd.testing.assert_frame_equal(result, invalid_data)
        self.assertEqual(result.attrs['ohlc_validation']['invalid_rows'], 1)
        self.assertEqual(result.attrs['ohlc_validation']['dropped_rows'], 0)

    def test_repair_mode_does_not_mutate_input(self):
        """Repair mode works on a copy of the caller's DataFrame."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'high'] = 90.0
        before = invalid_data.copy()

        OhlcValidatorPreprocessor(mode='repair').process(invalid_data)

        pd.testing.assert_frame_equal(invalid_data, before)

    def test_large_drop_fraction_emits_warning(self):
        """A large fraction of the data can never vanish without a WARNING."""
        invalid_data = self.valid_ohlc_data.copy()
        invalid_data.loc[self.sample_dates[0], 'high'] = 90.0
        invalid_data.loc[self.sample_dates[1], 'high'] = 90.0
        invalid_data.loc[self.sample_dates[2], 'high'] = 90.0

        logging.disable(logging.NOTSET)
        try:
            with self.assertLogs(level='WARNING') as captured:
                result = self.preprocessor.process(invalid_data)
        finally:
            logging.disable(logging.CRITICAL)

        self.assertEqual(len(result), 2)
        messages = ' '.join(captured.output)
        self.assertIn('HIGH OHLC INVALID RATE', messages)
        self.assertIn('60.00%', messages)

    def test_invalid_mode_rejected(self):
        """An unknown mode is rejected at construction time."""
        with self.assertRaises(ValueError):
            OhlcValidatorPreprocessor(mode='delete-everything')


if __name__ == '__main__':
    # Re-enable logging for when tests are run directly
    logging.disable(logging.NOTSET)
    unittest.main(verbosity=2)