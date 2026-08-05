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
        """Test NanValuePreprocessor forward-fills price columns."""
        # Create data with NaN values
        df_with_nan = self.valid_ohlc_data.copy()
        df_with_nan.loc[self.sample_dates[1], 'high'] = np.nan
        df_with_nan.loc[self.sample_dates[2], 'close'] = np.nan

        result = self.preprocessor.process(df_with_nan)

        # Check that NaN values are forward-filled
        self.assertEqual(result.loc[self.sample_dates[1], 'high'], 105.0)  # Forward-filled from previous day
        self.assertEqual(result.loc[self.sample_dates[2], 'close'], 103.0)  # Forward-filled from previous day

    def test_process_with_leading_nan_drops_row(self):
        """Leading NaN must NOT be backward-filled: the row is dropped instead.

        Backward-filling would copy a later bar into an earlier timestamp,
        which is look-ahead bias.
        """
        df_with_leading_nan = self.valid_ohlc_data.copy()
        df_with_leading_nan.loc[self.sample_dates[0], 'open'] = np.nan
        df_with_leading_nan.loc[self.sample_dates[0], 'high'] = np.nan

        result = self.preprocessor.process(df_with_leading_nan)

        self.assertNotIn(self.sample_dates[0], result.index)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.attrs['nan_fill']['dropped_rows'], 1)
        self.assertEqual(result.attrs['nan_fill']['filled_backward'], 0)

    def test_backward_fill_requires_explicit_opt_in(self):
        """Backward-fill is available but must be requested explicitly."""
        df_with_leading_nan = self.valid_ohlc_data.copy()
        df_with_leading_nan.loc[self.sample_dates[0], 'open'] = np.nan
        df_with_leading_nan.loc[self.sample_dates[0], 'high'] = np.nan

        preprocessor = NanValuePreprocessor(allow_backward_fill=True)
        result = preprocessor.process(df_with_leading_nan)

        self.assertEqual(len(result), 5)
        self.assertEqual(result.loc[self.sample_dates[0], 'open'], 101.0)  # Backward-filled
        self.assertEqual(result.loc[self.sample_dates[0], 'high'], 106.0)  # Backward-filled
        self.assertEqual(result.attrs['nan_fill']['filled_backward'], 2)

    def test_process_no_nan_values(self):
        """Test NanValuePreprocessor when no NaN values exist."""
        result = self.preprocessor.process(self.valid_ohlc_data)

        # Should return identical data
        pd.testing.assert_frame_equal(result, self.valid_ohlc_data)
        self.assertEqual(result.attrs['nan_fill']['synthetic_rows'], 0)

    def test_missing_volume_is_zero_filled_not_forward_filled(self):
        """Volume must never be forward-filled - that invents trades."""
        df_with_trailing_nan = self.valid_ohlc_data.copy()
        df_with_trailing_nan['volume'] = df_with_trailing_nan['volume'].astype(float)
        df_with_trailing_nan.loc[self.sample_dates[4], 'close'] = np.nan
        df_with_trailing_nan.loc[self.sample_dates[4], 'volume'] = np.nan

        result = self.preprocessor.process(df_with_trailing_nan)

        self.assertEqual(result.loc[self.sample_dates[4], 'close'], 105.0)  # price: forward-filled
        self.assertEqual(result.loc[self.sample_dates[4], 'volume'], 0.0)   # volume: zero-filled
        self.assertEqual(result.attrs['nan_fill']['filled_zero'], 1)
        self.assertEqual(result.attrs['nan_fill']['filled_forward'], 1)

    def test_volume_leading_nan_is_zero_not_dropped(self):
        """A leading NaN volume resolves to 0 without dropping the bar."""
        df = self.valid_ohlc_data.copy()
        df['volume'] = df['volume'].astype(float)
        df.loc[self.sample_dates[0], 'volume'] = np.nan

        result = self.preprocessor.process(df)

        self.assertEqual(len(result), 5)
        self.assertEqual(result.loc[self.sample_dates[0], 'volume'], 0.0)

    def test_synthetic_bars_are_counted(self):
        """Fabricated bars must be counted so consumers can see the damage."""
        df = self.valid_ohlc_data.copy()
        df.loc[self.sample_dates[1], 'close'] = np.nan
        df.loc[self.sample_dates[3], 'close'] = np.nan

        result = self.preprocessor.process(df)
        stats = result.attrs['nan_fill']

        self.assertEqual(stats['synthetic_rows'], 2)
        self.assertAlmostEqual(stats['synthetic_row_ratio'], 2 / 5)
        self.assertEqual(stats['per_column']['close'], 2)
        self.assertEqual(self.preprocessor.last_stats, stats)

    def test_add_synthetic_column_marks_rows(self):
        """The optional is_synthetic column marks fabricated bars per row."""
        df = self.valid_ohlc_data.copy()
        df.loc[self.sample_dates[2], 'close'] = np.nan

        preprocessor = NanValuePreprocessor(add_synthetic_column=True)
        result = preprocessor.process(df)

        self.assertIn('is_synthetic', result.columns)
        self.assertTrue(result.loc[self.sample_dates[2], 'is_synthetic'])
        self.assertFalse(result.loc[self.sample_dates[0], 'is_synthetic'])

    def test_max_fill_gap_limits_forward_fill(self):
        """A long dead stretch is not silently bridged when max_fill_gap is set."""
        df = self.valid_ohlc_data.copy()
        df.loc[self.sample_dates[1:4], 'close'] = np.nan

        preprocessor = NanValuePreprocessor(max_fill_gap=1)
        result = preprocessor.process(df)

        # Only the first missing bar is bridged; the rest stay unresolved and drop
        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[self.sample_dates[1], 'close'], 102.0)

    def test_invalid_policy_rejected(self):
        """An unknown fill policy is rejected at construction time."""
        with self.assertRaises(ValueError):
            NanValuePreprocessor(price_fill='magic')

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

        # Row 0 has a leading NaN in col2 that cannot be forward-filled -> dropped
        self.assertEqual(len(result), 4)
        self.assertNotIn(0, result.index)

        # Check forward-fill behavior
        self.assertEqual(result.loc[1, 'col1'], 1.0)  # Forward-filled
        self.assertEqual(result.loc[3, 'col1'], 3.0)  # Forward-filled
        self.assertEqual(result.loc[2, 'col2'], 2.0)  # Forward-filled
        self.assertEqual(result.loc[4, 'col2'], 4.0)  # Forward-filled

        # Check that non-NaN values remain unchanged
        self.assertEqual(result.loc[2, 'col3'], 30.0)

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

    def test_input_frame_is_not_mutated(self):
        """process() must not modify the caller's DataFrame."""
        df = self.valid_ohlc_data.copy()
        df.loc[self.sample_dates[2], 'close'] = np.nan
        before = df.copy()

        self.preprocessor.process(df)

        pd.testing.assert_frame_equal(df, before)


if __name__ == '__main__':
    # Re-enable logging for when tests are run directly
    logging.disable(logging.NOTSET)
    unittest.main(verbosity=2)
