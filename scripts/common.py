"""Shared helpers for the Niffler command line scripts.

The CSV loader lives here so that ``backtest.py``, ``analyze.py`` and
``optimize.py`` all interpret the same file in exactly the same way: identical
timestamp-column detection, lowercase column names, a sorted datetime index and
identical validation errors.

The transaction-cost command line lives here for the same reason: all three
scripts must be able to express the *same* market assumption, or a strategy gets
optimised in one market and traded in another.
"""

import argparse
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from niffler.backtesting.cost_model import (
    CostModel,
    FixedSlippageModel,
    VolumeShareSlippageModel,
    ZeroCostModel,
)

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


# ---------------------------------------------------------------------------
# Transaction cost model CLI
#
# Shared by backtest.py, optimize.py and analyze.py so that a strategy is
# optimised, validated and backtested under one and the same market assumption.
# Optimising frictionlessly and then backtesting with costs is the trap this
# section exists to close.
# ---------------------------------------------------------------------------

#: Cost models selectable from the command line.
COST_MODEL_CHOICES: Tuple[str, ...] = ('none', 'fixed', 'volume')

#: Which tuning flags each model actually reads. A flag supplied for a model
#: that ignores it is an error rather than a silently dropped argument.
_COST_MODEL_FLAGS: Dict[str, Tuple[str, ...]] = {
    'none': (),
    'fixed': ('slippage_bps', 'half_spread_bps'),
    'volume': ('half_spread_bps', 'impact_coefficient', 'max_participation'),
}

#: Values used when a model is selected but a flag it reads was not given.
#: They are plausible defaults for a liquid instrument, not measurements of
#: anyone's execution, which is why the chosen configuration is always printed.
_COST_MODEL_DEFAULTS: Dict[str, float] = {
    'slippage_bps': 5.0,
    'half_spread_bps': 1.0,
    'impact_coefficient': 0.1,
    'max_participation': 0.1,
}

_SEPARATOR = '=' * 72

#: Printed whenever a run's fills cost nothing, so a frictionless number is
#: never presented as if it described a real market.
FRICTIONLESS_WARNING = (
    f"{_SEPARATOR}\n"
    "WARNING: this run assumes FRICTIONLESS FILLS.\n"
    "  Every order fills at the exact reference price in unlimited size: no\n"
    "  bid/ask spread, no slippage, no market impact, no participation limit.\n"
    "  Commission is the only cost charged.\n"
    "  These results describe a market that does not exist. Re-run with\n"
    "  --cost-model fixed or --cost-model volume before believing them.\n"
    f"{_SEPARATOR}"
)


def add_cost_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--cost-model`` flags to a script's parser.

    Args:
        parser: Parser to extend.
    """
    group = parser.add_argument_group('transaction costs')
    group.add_argument(
        '--cost-model', choices=list(COST_MODEL_CHOICES), default='none',
        help=("Transaction cost model applied to every fill: 'none' (default, "
              "frictionless and loudly flagged as such), 'fixed' (constant "
              "slippage + half spread) or 'volume' (half spread plus "
              "square-root market impact, capped at a share of the bar's volume)")
    )
    group.add_argument(
        '--slippage-bps', type=float, default=None,
        help=f"Fixed model only: execution slippage in basis points "
             f"(default: {_COST_MODEL_DEFAULTS['slippage_bps']:g})"
    )
    group.add_argument(
        '--half-spread-bps', type=float, default=None,
        help=f"Fixed and volume models: half the bid/ask spread in basis points "
             f"(default: {_COST_MODEL_DEFAULTS['half_spread_bps']:g})"
    )
    group.add_argument(
        '--impact-coefficient', type=float, default=None,
        help=f"Volume model only: dimensionless coefficient on "
             f"sqrt(participation), as a fraction of price "
             f"(default: {_COST_MODEL_DEFAULTS['impact_coefficient']:g})"
    )
    group.add_argument(
        '--max-participation', type=float, default=None,
        help=f"Volume model only: largest share of a bar's volume one order may "
             f"take, in (0, 1] (default: {_COST_MODEL_DEFAULTS['max_participation']:g})"
    )


def build_cost_model(args: argparse.Namespace) -> CostModel:
    """Build the cost model a parsed command line asks for.

    Flags belonging to a different model are rejected rather than ignored: a
    silently dropped ``--impact-coefficient`` would mean the user believes they
    are paying market impact while the run charges none.

    Args:
        args: Parsed arguments carrying the flags added by
            :func:`add_cost_model_arguments`.

    Returns:
        The configured cost model.

    Raises:
        ValueError: If the model name is unknown, or a flag was supplied that
            the selected model does not read. Invalid parameter values raise
            from the model constructors themselves.
    """
    choice = getattr(args, 'cost_model', 'none') or 'none'
    if choice not in COST_MODEL_CHOICES:
        raise ValueError(
            f"Unknown cost model '{choice}'. Available: {', '.join(COST_MODEL_CHOICES)}"
        )

    accepted = _COST_MODEL_FLAGS[choice]
    ignored = sorted(
        f"--{name.replace('_', '-')}"
        for name in _COST_MODEL_DEFAULTS
        if getattr(args, name, None) is not None and name not in accepted
    )
    if ignored:
        raise ValueError(
            f"--cost-model {choice} does not use {', '.join(ignored)}. "
            f"Remove the flag(s), or select a cost model that reads them "
            f"(fixed: --slippage-bps/--half-spread-bps; volume: "
            f"--half-spread-bps/--impact-coefficient/--max-participation)."
        )

    values = {
        name: (getattr(args, name) if getattr(args, name, None) is not None
               else _COST_MODEL_DEFAULTS[name])
        for name in accepted
    }

    if choice == 'fixed':
        return FixedSlippageModel(**values)
    if choice == 'volume':
        return VolumeShareSlippageModel(**values)
    return ZeroCostModel()


def report_cost_model(cost_model: CostModel, stream=None) -> bool:
    """Print the cost model in force, warning loudly when it charges nothing.

    Args:
        cost_model: The model the run will use.
        stream: Stream the frictionless warning goes to (default ``sys.stderr``).
            The one-line description always goes to stdout, beside the results.

    Returns:
        True when the frictionless warning was emitted.
    """
    print(f"Cost model: {cost_model.description}")

    if not cost_model.is_frictionless:
        return False

    print(FRICTIONLESS_WARNING, file=stream if stream is not None else sys.stderr)
    return True
