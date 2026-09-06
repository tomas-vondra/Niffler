"""Shared helpers for the Niffler command line scripts.

The CSV loader lives here so that ``backtest.py``, ``analyze.py`` and
``optimize.py`` all interpret the same file in exactly the same way: identical
timestamp-column detection, lowercase column names, a sorted datetime index and
identical validation errors.

The transaction-cost command line lives here for the same reason: all three
scripts must be able to express the *same* market assumption, or a strategy gets
optimised in one market and traded in another.

So does the run configuration. ``build_run_config`` is the single place that
turns parsed arguments into a
:class:`~niffler.backtesting.run_config.RunConfig`; every CLI hands the result
straight to the optimizer or an analyzer, which hand it to the engine. Because
there is one builder, a knob added to the config reaches every script at once,
and no script can populate half of it.
"""

import argparse
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from niffler.backtesting.benchmark import BENCHMARK_BUY_AND_HOLD, BENCHMARK_CHOICES
from niffler.backtesting.cost_model import (
    CostModel,
    FixedSlippageModel,
    VolumeShareSlippageModel,
    ZeroCostModel,
)
from niffler.backtesting.run_config import (
    BOOTSTRAP_DISABLED,
    DEFAULT_COMMISSION,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MIN_ORDER_VALUE,
    RunConfig,
)
from niffler.backtesting.significance import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_MIN_TRADES,
)
from scripts.config_file import CONFIG_ORIGINS_ATTR

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
    # A value from a configuration file is indistinguishable from a typed
    # flag by the time it reaches here, so the rejection says where it came
    # from - otherwise the user hunts for a flag they never typed.
    origins = getattr(args, CONFIG_ORIGINS_ATTR, None) or {}
    ignored = sorted(
        f"--{name.replace('_', '-')}"
        + (f" (set in {origins[name]})" if name in origins else '')
        for name in _COST_MODEL_DEFAULTS
        if getattr(args, name, None) is not None and name not in accepted
    )
    if ignored:
        raise ValueError(
            f"--cost-model {choice} does not use {', '.join(ignored)}. "
            f"Remove the flag(s) or the configuration entry, or select a "
            f"cost model that reads them "
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


# ---------------------------------------------------------------------------
# Run configuration CLI
#
# The engine takes ten settings. Before RunConfig existed, five of the six
# places that built one passed three of them, so a benchmark or an
# annualisation factor chosen on the command line never reached the engine
# inside a walk-forward fold. There is now one builder and one object, so a
# knob added to RunConfig is reachable from every script that calls
# build_run_config.
# ---------------------------------------------------------------------------


def add_engine_arguments(parser: argparse.ArgumentParser,
                         bootstrap: bool = False,
                         benchmark: bool = True) -> None:
    """Add the shared engine-setting flags to a script's parser.

    ``--capital``/``--initial_capital`` and ``--commission`` are *not* added
    here: every script already declares them under its own established
    spelling. What they must all share is ``dest='initial_capital'`` so
    :func:`build_run_config` reads one attribute name.

    ``execution_timing`` is deliberately absent. ``same_bar_close`` is
    look-ahead biased and the repository's invariants forbid exposing it on a
    CLI; it stays a library-only field of ``RunConfig``.

    Args:
        parser: Parser to extend.
        bootstrap: Add the bootstrap Sharpe interval flags. Only the backtest
            CLI prints that interval, and the resample loop is the one
            expensive part of the assessment - putting it behind an
            optimisation or a thousand Monte Carlo paths would cost hours for a
            number nothing reads.
        benchmark: Add ``--benchmark``. ``compare.py`` passes False: every one
            of its rows is an excess over buy-and-hold on the same bars, so
            ``--benchmark none`` would empty the table rather than configure it.
    """
    group = parser.add_argument_group('engine settings')
    group.add_argument(
        '--min-order-value', type=float, default=DEFAULT_MIN_ORDER_VALUE,
        help=(f"Smallest order notional that executes; smaller orders are "
              f"skipped (default: {DEFAULT_MIN_ORDER_VALUE:g})")
    )
    group.add_argument(
        '--periods-per-year', type=float, default=None,
        help=("Annualisation factor for the Sharpe ratio. Omitted (default) "
              "means it is inferred from the index - daily crypto 365, daily "
              "equities 252, hourly crypto 8760. Set it only to override that "
              "inference; there is no safe fixed number")
    )
    if benchmark:
        group.add_argument(
            '--benchmark', choices=list(BENCHMARK_CHOICES),
            default=BENCHMARK_BUY_AND_HOLD,
            help=("Passive alternative every backtest is measured against: "
                  "'buy_and_hold' (default) pays the same commission and cost "
                  "model over the same bars; 'none' reports the strategy's "
                  "numbers with nothing to compare them to")
        )
    group.add_argument(
        '--min-trades-for-significance', type=int, default=DEFAULT_MIN_TRADES,
        help=(f"Round trips below which no significance verdict is rendered "
              f"(default: {DEFAULT_MIN_TRADES}). Below this the metrics are "
              f"still reported, labelled as not meaningful")
    )
    if bootstrap:
        group.add_argument(
            '--bootstrap-samples', type=int, default=DEFAULT_BOOTSTRAP_SAMPLES,
            help=f"Resamples for the bootstrap Sharpe confidence interval "
                 f"(default: {DEFAULT_BOOTSTRAP_SAMPLES}); 0 skips it"
        )
        group.add_argument(
            '--bootstrap-seed', type=int, default=DEFAULT_BOOTSTRAP_SEED,
            help=f"Seed for that bootstrap, so the interval is reproducible "
                 f"(default: {DEFAULT_BOOTSTRAP_SEED})"
        )


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Build the run configuration a parsed command line asks for.

    The single place arguments become engine settings. A script that did not
    declare a given flag falls back to that setting's documented default, which
    is ``RunConfig``'s own default and therefore the engine's - so adding a flag
    to one CLI never silently changes another.

    ``bootstrap_samples`` falls back to 0, not to the backtest CLI's 1000: only
    that script prints the interval, and defaulting it on would run a
    thousand-resample bootstrap inside every grid cell of every fold.

    Args:
        args: Parsed arguments. Cost-model and risk-manager flags are read
            through :func:`build_cost_model` and :func:`build_risk_manager`, so
            an unusable combination fails here.

    Returns:
        The configuration every backtest of this run will use.

    Raises:
        ValueError: If a setting is out of range, or the cost-model or
            risk-manager flags are inconsistent.
        TypeError: If the cost model is not a CostModel.
    """
    return RunConfig(
        initial_capital=getattr(args, 'initial_capital', DEFAULT_INITIAL_CAPITAL),
        commission=getattr(args, 'commission', DEFAULT_COMMISSION),
        min_order_value=getattr(args, 'min_order_value', DEFAULT_MIN_ORDER_VALUE),
        periods_per_year=getattr(args, 'periods_per_year', None),
        cost_model=build_cost_model(args),
        risk_manager=build_risk_manager(args),
        benchmark=getattr(args, 'benchmark', BENCHMARK_BUY_AND_HOLD),
        min_trades_for_significance=getattr(
            args, 'min_trades_for_significance', DEFAULT_MIN_TRADES),
        bootstrap_samples=getattr(args, 'bootstrap_samples', BOOTSTRAP_DISABLED),
        bootstrap_seed=getattr(args, 'bootstrap_seed', DEFAULT_BOOTSTRAP_SEED),
    )


def report_run_config(run_config: RunConfig, stream=None) -> bool:
    """Print the settings this run uses, cost model included.

    Args:
        run_config: The configuration the run will use.
        stream: Stream the frictionless warning goes to (default stderr).

    Returns:
        True when the frictionless warning was emitted.
    """
    print(f"Capital: ${run_config.initial_capital:,.2f}   "
          f"Commission: {run_config.commission:.4f}   "
          f"Benchmark: {run_config.benchmark}")
    annualisation = ('inferred from the index' if run_config.periods_per_year is None
                     else f"{run_config.periods_per_year:g} periods/year")
    print(f"Annualisation: {annualisation}   "
          f"Significance gate: {run_config.min_trades_for_significance} round trips")
    print(f"Risk management: {describe_risk_configuration(run_config.risk_manager)}")
    return report_cost_model(
        run_config.cost_model if run_config.cost_model is not None else ZeroCostModel(),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Risk management CLI
#
# --risk-manager existed only in backtest.py, and it configured the *strategy*
# object. The optimizer and both analyzers construct their own strategies, so
# optimisation, walk-forward and Monte Carlo ran with risk management off
# unconditionally - the pipeline tuned and validated a system nobody would
# trade. The manager is now a RunConfig field, which every one of those already
# carries into its worker processes, so building it here reaches all of them.
# ---------------------------------------------------------------------------

# Imported here rather than at the top of the module to keep this addition to
# one contiguous block.
from typing import Set  # noqa: E402

from niffler.risk import (  # noqa: E402
    NO_RISK_MANAGER,
    create_risk_manager,
    describe_risk_manager,
    get_available_risk_managers,
    get_risk_manager_parameter_names,
)

#: Flag destination -> constructor keyword. The two differ: --max-position-size
#: configures FixedRiskManager's ``position_size_pct``, and create_risk_manager
#: rejects a keyword its manager does not declare, so the translation is
#: required rather than cosmetic.
_RISK_FLAG_TO_PARAMETER: Dict[str, str] = {
    'max_position_size': 'position_size_pct',
    'stop_loss_pct': 'stop_loss_pct',
    'max_positions': 'max_positions',
    'max_risk_per_trade': 'max_risk_per_trade',
}

#: Values used when a manager is selected but a flag it reads was not given.
#: These are backtest.py's historical defaults, kept so moving that script onto
#: this section leaves its numbers unchanged - FixedRiskManager's own
#: position_size_pct default is 0.1, not 0.2.
_RISK_MANAGER_DEFAULTS: Dict[str, float] = {
    'max_position_size': 0.2,
    'stop_loss_pct': 0.05,
    'max_positions': 5,
    'max_risk_per_trade': 0.02,
}


def add_risk_manager_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--risk-manager`` flags to a script's parser.

    The tuning flags default to None so "not supplied" stays distinguishable
    from "supplied at the default": a flag the selected manager does not read is
    an error, and that can only be detected when the two differ.

    Args:
        parser: Parser to extend.
    """
    group = parser.add_argument_group('risk management')
    group.add_argument(
        '--risk-manager', choices=get_available_risk_managers(),
        default=NO_RISK_MANAGER,
        help=(f"Risk manager applied to every backtest of this run: "
              f"'{NO_RISK_MANAGER}' (default) runs with no position sizing, no "
              f"stops and no exposure cap; 'fixed' sizes every entry at a "
              f"constant fraction of the portfolio behind a percentage stop")
    )
    group.add_argument(
        '--max-position-size', type=float, default=None,
        help=f"Largest position as a fraction of the portfolio "
             f"(default: {_RISK_MANAGER_DEFAULTS['max_position_size']:g})"
    )
    group.add_argument(
        '--stop-loss-pct', type=float, default=None,
        help=f"Stop loss as a fraction of the entry price "
             f"(default: {_RISK_MANAGER_DEFAULTS['stop_loss_pct']:g})"
    )
    group.add_argument(
        '--max-positions', type=int, default=None,
        help=f"Largest number of concurrent positions "
             f"(default: {_RISK_MANAGER_DEFAULTS['max_positions']:g})"
    )
    group.add_argument(
        '--max-risk-per-trade', type=float, default=None,
        help=f"Largest risk per trade as a fraction of the portfolio "
             f"(default: {_RISK_MANAGER_DEFAULTS['max_risk_per_trade']:g})"
    )


def build_risk_manager(args: argparse.Namespace):
    """Build the risk manager a parsed command line asks for.

    Flags the selected manager does not read are rejected rather than ignored,
    for the same reason ``build_cost_model`` rejects them: a silently dropped
    ``--stop-loss-pct`` means the user believes stops are armed while the run
    trades without any.

    Args:
        args: Parsed arguments carrying the flags added by
            :func:`add_risk_manager_arguments`. A script that declared none gets
            no risk management, which is what it had before.

    Returns:
        The configured manager, or None for ``'none'``.

    Raises:
        ValueError: If the manager name is unknown, or a flag was supplied that
            the selected manager does not accept.
    """
    choice = getattr(args, 'risk_manager', NO_RISK_MANAGER) or NO_RISK_MANAGER
    if choice == NO_RISK_MANAGER:
        _reject_unused_risk_flags(args, choice, set())
        return None

    accepted = get_risk_manager_parameter_names(choice)
    _reject_unused_risk_flags(args, choice, accepted)

    parameters = {}
    for dest, parameter in _RISK_FLAG_TO_PARAMETER.items():
        if parameter not in accepted:
            continue
        supplied = getattr(args, dest, None)
        parameters[parameter] = (supplied if supplied is not None
                                 else _RISK_MANAGER_DEFAULTS[dest])

    return create_risk_manager(choice, parameters)


def _reject_unused_risk_flags(args: argparse.Namespace, choice: str,
                              accepted: Set[str]) -> None:
    """Raise when a supplied tuning flag is not read by the chosen manager."""
    ignored = sorted(
        f"--{dest.replace('_', '-')}"
        for dest, parameter in _RISK_FLAG_TO_PARAMETER.items()
        if getattr(args, dest, None) is not None and parameter not in accepted
    )
    if not ignored:
        return

    if choice == NO_RISK_MANAGER:
        raise ValueError(
            f"--risk-manager {NO_RISK_MANAGER} does not use "
            f"{', '.join(ignored)}. Remove the flag(s), or select a risk "
            f"manager that reads them."
        )
    raise ValueError(
        f"--risk-manager {choice} does not use {', '.join(ignored)}. "
        f"Remove the flag(s). Accepted: {', '.join(sorted(accepted))}."
    )


def describe_risk_configuration(risk_manager) -> str:
    """Render the risk manager in force as one console line.

    Args:
        risk_manager: The manager the run will use, or None.

    Returns:
        ``'none'``, or the registry name followed by its parameters.
    """
    if risk_manager is None:
        return NO_RISK_MANAGER

    description = describe_risk_manager(risk_manager)
    name = description['name'] or description['class']
    parameters = description['parameters']
    if not parameters:
        return str(name)

    rendered = ', '.join(f"{key}={value:g}" for key, value in parameters.items())
    return f"{name} ({rendered})"
