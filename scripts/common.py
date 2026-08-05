"""Shared helpers for the Niffler command line scripts.

The CSV loader lives here so that ``backtest.py``, ``analyze.py`` and
``optimize.py`` all interpret the same file in exactly the same way: identical
timestamp-column detection, lowercase column names, a sorted datetime index and
identical validation errors.
"""

import logging
import os
import warnings
from typing import List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Columns every OHLCV dataset used by the trading scripts must provide.
REQUIRED_OHLCV_COLUMNS: Tuple[str, ...] = ('open', 'high', 'low', 'close', 'volume')

# Column names (already lowercased) that are treated as the timestamp column,
# in priority order.
TIMESTAMP_COLUMN_CANDIDATES: Tuple[str, ...] = ('timestamp', 'date', 'datetime', 'time')

# pandas writes a DataFrame with an unnamed index as a leading column without a
# header; read_csv then names it "Unnamed: 0" (or leaves it blank).
_UNNAMED_INDEX_COLUMNS: Tuple[str, ...] = ('unnamed: 0', '')

# Canonical name given to the datetime index produced by the loader.
INDEX_NAME = 'timestamp'

_DUPLICATE_POLICIES: Tuple[str, ...] = ('raise', 'warn')


def load_ohlcv_csv(file_path: str,
                   clean: bool = False,
                   timestamp_column: Optional[str] = None,
                   required_columns: Sequence[str] = REQUIRED_OHLCV_COLUMNS,
                   require_datetime_index: bool = True,
                   on_duplicates: str = 'raise') -> pd.DataFrame:
    """Load an OHLCV CSV file into a normalised DataFrame.

    The file is normalised the same way for every script:

    * column names are stripped and lowercased,
    * the timestamp column (explicit, well-known name, or the unnamed index
      column written by pandas) becomes a sorted ``DatetimeIndex``,
    * required columns are validated,
    * duplicate timestamps are reported.

    Args:
        file_path: Path to the CSV file.
        clean: Whether to run the default preprocessing pipeline on the data.
        timestamp_column: Explicit timestamp column name (case insensitive).
            Auto-detected when None.
        required_columns: Columns that must be present. Pass an empty sequence
            to skip the check.
        require_datetime_index: Raise when no timestamp information can be
            found instead of keeping the positional index.
        on_duplicates: ``"raise"`` or ``"warn"`` - what to do when the same
            timestamp appears more than once.

    Returns:
        DataFrame with lowercase columns and (normally) a sorted DatetimeIndex.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or unparsable, the timestamp column is
            missing or unparsable, required columns are missing, duplicate
            timestamps are found with ``on_duplicates="raise"``, or cleaning
            removes every row.
    """
    if on_duplicates not in _DUPLICATE_POLICIES:
        raise ValueError(
            f"Invalid on_duplicates value: {on_duplicates!r}. "
            f"Valid values are: {', '.join(_DUPLICATE_POLICIES)}"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = _read_csv(file_path)

    # Normalise column names before anything else so detection and validation
    # never depend on the casing used by the data source.
    df.columns = [str(column).strip().lower() for column in df.columns]

    duplicated_headers = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated_headers:
        raise ValueError(
            f"Duplicate column names in {file_path} after normalisation: {duplicated_headers}"
        )

    resolved_column = _resolve_timestamp_column(df, timestamp_column, file_path)
    if resolved_column is not None:
        df[resolved_column] = _parse_timestamps(df[resolved_column], resolved_column, file_path)
        df = df.set_index(resolved_column)
        df.index.name = INDEX_NAME
    elif require_datetime_index:
        raise ValueError(
            f"Could not determine the timestamp column of {file_path}. "
            f"Expected one of {list(TIMESTAMP_COLUMN_CANDIDATES)} "
            f"but found columns: {list(df.columns)}"
        )

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if isinstance(df.index, pd.DatetimeIndex):
        _check_duplicate_timestamps(df, file_path, on_duplicates)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

    if clean:
        df = _clean(df, file_path)

    return df


def _read_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV file, translating parser failures into clear ValueErrors."""
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Data file is empty: {file_path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Could not parse CSV file {file_path}: {e}") from e
    except OSError as e:
        raise ValueError(f"Could not read data file {file_path}: {e}") from e

    if df.empty:
        raise ValueError(f"Data file contains no rows: {file_path}")

    return df


def _resolve_timestamp_column(df: pd.DataFrame, timestamp_column: Optional[str],
                              file_path: str) -> Optional[str]:
    """Find the column holding timestamps, or None when there is none."""
    if timestamp_column is not None:
        wanted = str(timestamp_column).strip().lower()
        if wanted not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found in {file_path}. "
                f"Available columns: {list(df.columns)}"
            )
        return wanted

    for candidate in TIMESTAMP_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    for candidate in _UNNAMED_INDEX_COLUMNS:
        if candidate in df.columns and _looks_like_timestamps(df[candidate]):
            return candidate

    return None


def _looks_like_timestamps(series: pd.Series) -> bool:
    """Report whether a series can be interpreted as timestamps.

    Numeric series are rejected on purpose: a positional index written to CSV
    parses as epoch nanoseconds and would silently produce 1970 dates.
    """
    if pd.api.types.is_numeric_dtype(series):
        return False

    try:
        # This is only a probe, so pandas' format-inference warnings are noise.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pd.to_datetime(series)
    except (ValueError, TypeError, OverflowError):
        return False
    return True


def _parse_timestamps(series: pd.Series, column: str, file_path: str) -> pd.Series:
    """Convert a column to datetimes with a clear error message on failure."""
    try:
        return pd.to_datetime(series)
    except (ValueError, TypeError, OverflowError) as e:
        raise ValueError(
            f"Could not parse column '{column}' of {file_path} as timestamps: {e}"
        ) from e


def _check_duplicate_timestamps(df: pd.DataFrame, file_path: str, on_duplicates: str) -> None:
    """Detect duplicate index entries and raise or warn about them."""
    duplicated = df.index.duplicated()
    if not duplicated.any():
        return

    count = int(duplicated.sum())
    examples: List[str] = [str(value) for value in df.index[duplicated].unique()[:3]]
    message = (
        f"Found {count} duplicate timestamp(s) in {file_path} "
        f"(e.g. {', '.join(examples)})"
    )

    if on_duplicates == 'raise':
        raise ValueError(message)

    logger.warning(message)


def _clean(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """Run the default preprocessing pipeline over the loaded data."""
    from niffler.data import create_default_manager

    manager = create_default_manager()
    cleaned = manager.run(df)

    if cleaned is None or cleaned.empty:
        raise ValueError(f"Data cleaning removed all rows from {file_path}")

    return cleaned
