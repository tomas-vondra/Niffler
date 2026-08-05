import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.common import REQUIRED_OHLCV_COLUMNS, load_ohlcv_csv


class TestLoadOhlcvCsv(unittest.TestCase):
    """Tests for the single CSV loader shared by all scripts."""

    def setUp(self):
        """Create a temporary directory for CSV fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _frame(self, periods: int = 5, start: str = '2024-01-01') -> pd.DataFrame:
        """Build a small valid OHLCV frame with a datetime index."""
        return pd.DataFrame({
            'open': [100.0] * periods,
            'high': [105.0] * periods,
            'low': [95.0] * periods,
            'close': [102.0] * periods,
            'volume': [1000.0] * periods
        }, index=pd.date_range(start, periods=periods, freq='D'))

    def _write(self, name: str, content: str) -> str:
        """Write raw CSV content and return its path."""
        path = os.path.join(self.temp_dir, name)
        with open(path, 'w', newline='') as handle:
            handle.write(content)
        return path

    def test_missing_file_raises_file_not_found(self):
        """A missing file raises FileNotFoundError, not a pandas error."""
        with self.assertRaises(FileNotFoundError):
            load_ohlcv_csv(os.path.join(self.temp_dir, 'nope.csv'))

    def test_timestamp_column(self):
        """A 'timestamp' column becomes the datetime index."""
        path = os.path.join(self.temp_dir, 'ts.csv')
        self._frame().rename_axis('timestamp').reset_index().to_csv(path, index=False)

        data = load_ohlcv_csv(path)

        self.assertIsInstance(data.index, pd.DatetimeIndex)
        self.assertEqual(data.index.name, 'timestamp')
        self.assertEqual(len(data), 5)

    def test_date_column_with_mixed_case_headers(self):
        """A 'Date' column is detected and all headers are lowercased."""
        path = os.path.join(self.temp_dir, 'date.csv')
        frame = self._frame().rename_axis('Date').reset_index()
        frame.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        frame.to_csv(path, index=False)

        data = load_ohlcv_csv(path)

        self.assertIsInstance(data.index, pd.DatetimeIndex)
        self.assertEqual(list(data.columns), list(REQUIRED_OHLCV_COLUMNS))

    def test_unnamed_index_column_is_used_as_timestamp(self):
        """A CSV written with a datetime index (no header) is parsed correctly."""
        path = os.path.join(self.temp_dir, 'unnamed.csv')
        self._frame().to_csv(path)  # index written without a name

        data = load_ohlcv_csv(path)

        self.assertIsInstance(data.index, pd.DatetimeIndex)
        # Regression: this used to be parsed as epoch nanoseconds (1970 dates).
        self.assertEqual(data.index[0], pd.Timestamp('2024-01-01'))

    def test_numeric_index_column_is_not_treated_as_timestamp(self):
        """A positional index column must not silently become 1970 dates."""
        path = os.path.join(self.temp_dir, 'positional.csv')
        frame = self._frame().reset_index(drop=True)
        frame.to_csv(path)  # writes 0..4 as the unnamed leading column

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path)
        self.assertIn("Could not determine the timestamp column", str(context.exception))

    def test_no_timestamp_information_raises(self):
        """A CSV without any timestamp information raises a clear error."""
        path = os.path.join(self.temp_dir, 'no_ts.csv')
        self._frame().reset_index(drop=True).to_csv(path, index=False)

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path)
        self.assertIn("Could not determine the timestamp column", str(context.exception))

    def test_index_is_sorted(self):
        """Rows are returned in chronological order regardless of file order."""
        path = os.path.join(self.temp_dir, 'unsorted.csv')
        frame = self._frame().rename_axis('timestamp').reset_index()
        frame = frame.iloc[::-1]
        frame.to_csv(path, index=False)

        data = load_ohlcv_csv(path)

        self.assertTrue(data.index.is_monotonic_increasing)
        self.assertEqual(data.index[0], pd.Timestamp('2024-01-01'))

    def test_duplicate_timestamps_raise_by_default(self):
        """Duplicate timestamps are detected instead of silently backtested."""
        path = os.path.join(self.temp_dir, 'dupes.csv')
        frame = self._frame(periods=3).rename_axis('timestamp').reset_index()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame.to_csv(path, index=False)

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path)
        self.assertIn("duplicate timestamp", str(context.exception))

    def test_duplicate_timestamps_can_warn_instead(self):
        """on_duplicates='warn' keeps the data and only logs."""
        path = os.path.join(self.temp_dir, 'dupes_warn.csv')
        frame = self._frame(periods=3).rename_axis('timestamp').reset_index()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame.to_csv(path, index=False)

        # Other test modules globally disable logging at import time.
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, previous_disable_level)

        with self.assertLogs('scripts.common', level='WARNING') as logs:
            data = load_ohlcv_csv(path, on_duplicates='warn')

        self.assertEqual(len(data), 4)
        self.assertIn('duplicate timestamp', ' '.join(logs.output))

    def test_missing_required_columns(self):
        """Missing OHLCV columns produce the documented error message."""
        path = os.path.join(self.temp_dir, 'incomplete.csv')
        frame = self._frame().drop(columns=['volume']).rename_axis('timestamp').reset_index()
        frame.to_csv(path, index=False)

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path)
        self.assertIn("Missing required columns: ['volume']", str(context.exception))

    def test_required_columns_can_be_disabled(self):
        """An empty required_columns skips OHLCV validation."""
        path = os.path.join(self.temp_dir, 'other.csv')
        pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
            'price': [1.0, 2.0, 3.0]
        }).to_csv(path, index=False)

        data = load_ohlcv_csv(path, required_columns=())

        self.assertEqual(list(data.columns), ['price'])

    def test_explicit_timestamp_column_is_case_insensitive(self):
        """An explicitly named timestamp column is matched case insensitively."""
        path = os.path.join(self.temp_dir, 'explicit.csv')
        frame = self._frame().rename_axis('Date').reset_index()
        frame.to_csv(path, index=False)

        data = load_ohlcv_csv(path, timestamp_column='Date')

        self.assertIsInstance(data.index, pd.DatetimeIndex)

    def test_unknown_explicit_timestamp_column_raises(self):
        """An unknown explicit timestamp column reports the available columns."""
        path = os.path.join(self.temp_dir, 'explicit_missing.csv')
        self._frame().rename_axis('timestamp').reset_index().to_csv(path, index=False)

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path, timestamp_column='when')
        self.assertIn("Timestamp column 'when' not found", str(context.exception))

    def test_unparsable_timestamps_raise_clear_error(self):
        """Non-date values in the timestamp column give a clear error."""
        path = self._write(
            'bad_ts.csv',
            "timestamp,open,high,low,close,volume\n"
            "not-a-date,1,2,0.5,1.5,10\n"
            "also-not-a-date,1,2,0.5,1.5,10\n"
        )

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path)
        self.assertIn("as timestamps", str(context.exception))

    def test_duplicate_headers_after_lowercasing_raise(self):
        """Columns that collide once lowercased are reported, not silently merged."""
        path = self._write(
            'dupe_headers.csv',
            "timestamp,Open,open,high,low,close,volume\n"
            "2024-01-01,1,1,2,0.5,1.5,10\n"
        )

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path)
        self.assertIn("Duplicate column names", str(context.exception))

    def test_empty_file_raises_value_error(self):
        """A completely empty file raises a clear ValueError."""
        path = self._write('empty.csv', "")

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path)
        self.assertIn("empty", str(context.exception))

    def test_header_only_file_raises_value_error(self):
        """A file with headers but no rows raises a clear ValueError."""
        path = self._write('header_only.csv', "timestamp,open,high,low,close,volume\n")

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path)
        self.assertIn("no rows", str(context.exception))

    def test_no_datetime_index_allowed_when_not_required(self):
        """require_datetime_index=False keeps the positional index."""
        path = self._write('free_form.csv', "a,b\n1,2\n3,4\n")

        data = load_ohlcv_csv(path, required_columns=(), require_datetime_index=False)

        self.assertEqual(len(data), 2)
        self.assertNotIsInstance(data.index, pd.DatetimeIndex)

    @patch('niffler.data.create_default_manager')
    def test_clean_runs_default_pipeline(self, mock_create_manager):
        """clean=True runs the default preprocessing pipeline."""
        path = os.path.join(self.temp_dir, 'clean.csv')
        self._frame().rename_axis('timestamp').reset_index().to_csv(path, index=False)

        cleaned = self._frame(periods=4)
        mock_manager = MagicMock()
        mock_manager.run.return_value = cleaned
        mock_create_manager.return_value = mock_manager

        data = load_ohlcv_csv(path, clean=True)

        self.assertEqual(len(data), 4)
        mock_manager.run.assert_called_once()

    @patch('niffler.data.create_default_manager')
    def test_clean_that_empties_data_raises(self, mock_create_manager):
        """Cleaning away every row is an error, not an empty success."""
        path = os.path.join(self.temp_dir, 'clean_empty.csv')
        self._frame().rename_axis('timestamp').reset_index().to_csv(path, index=False)

        mock_manager = MagicMock()
        mock_manager.run.return_value = pd.DataFrame()
        mock_create_manager.return_value = mock_manager

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path, clean=True)
        self.assertIn("Data cleaning removed all rows", str(context.exception))

    def test_invalid_on_duplicates_value(self):
        """An unknown duplicate policy is rejected up front."""
        path = os.path.join(self.temp_dir, 'policy.csv')
        self._frame().rename_axis('timestamp').reset_index().to_csv(path, index=False)

        with self.assertRaises(ValueError) as context:
            load_ohlcv_csv(path, on_duplicates='ignore')
        self.assertIn("Invalid on_duplicates value", str(context.exception))


class TestLoaderConsistencyAcrossScripts(unittest.TestCase):
    """The same CSV must behave identically in every script."""

    def setUp(self):
        """Write one CSV that used to be read differently by each script."""
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, 'shared.csv')
        frame = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=6, freq='D'),
            'Open': [100.0] * 6,
            'High': [105.0] * 6,
            'Low': [95.0] * 6,
            'Close': [102.0] * 6,
            'Volume': [1000.0] * 6
        })
        # Deliberately out of order and mixed case - the three scripts used to
        # disagree about both.
        frame.iloc[::-1].to_csv(self.path, index=False)

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_scripts_produce_the_same_frame(self):
        """backtest, analyze and optimize load identical data."""
        from scripts import analyze, backtest, optimize

        from_backtest = backtest.load_data(self.path)
        from_analyze = analyze.load_data(self.path)
        from_optimize = optimize.load_and_validate_data(self.path)

        pd.testing.assert_frame_equal(from_backtest, from_analyze)
        pd.testing.assert_frame_equal(from_backtest, from_optimize)
        self.assertTrue(from_backtest.index.is_monotonic_increasing)
        self.assertEqual(list(from_backtest.columns), list(REQUIRED_OHLCV_COLUMNS))


if __name__ == '__main__':
    unittest.main()
