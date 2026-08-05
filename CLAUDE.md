# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Niffler is a Python-based trading application that helps identify profitable market opportunities. The project includes data acquisition, preprocessing, backtesting, strategy optimization, and advanced robustness analysis components.

### Read this before changing numerical behaviour

A correctness audit removed several sources of look-ahead bias and silent failure. The
full list, with the reasoning, is in
[README.md → What changed and why your old results differed](README.md#what-changed-and-why-your-old-results-differed).
The short version, because these are easy to "helpfully" undo:

| Invariant | Do not |
|-----------|--------|
| Signals fill at the **next bar's open** | Make `same_bar_close` the default or expose it on the CLI |
| Annualisation is **inferred** from the index | Reintroduce a hardcoded `sqrt(252)` |
| **One** FIFO pairing routine (`pair_trades`) | Add a second pairing loop in an exporter or analyzer |
| Walk-forward **trains and tests on disjoint bars** | Restore fixed-parameter "walk-forward" as the default |
| Monte Carlo bootstraps **returns**, in draw order | Call `sort_index()` on resampled blocks, or glue price levels |
| Volume gaps fill with **0**; backward fill is opt-in | Restore a blanket `ffill().bfill()` |
| `max_drawdown` is **negative** | Use `max()` for "worst" or treat it as lower-is-better |
| Failures **raise** and exit non-zero | Log a message and return normally |
| Transaction costs are **always adverse** and apply to stops | Let a model return a negative cost, price a stop exit at the raw stop, or size a buy on the pre-slippage price |

Scope limits that are deliberate, not oversights: long-only, no live trading, one strategy,
Kelly risk manager is a stub. Slippage/spread/market impact **are** modelled now
(`niffler/backtesting/cost_model.py`) but default to off, and a run with no cost model
labels itself as frictionless.

## Development Setup

### Package Management
- Uses `uv` for dependency management (modern, fast Python package manager)
- Install dependencies: `uv sync`
- Python version: ≥3.13

### Core Dependencies
- `pandas` (≥2.3.1) for data manipulation
- `ccxt` for cryptocurrency exchange data
- `yfinance` for traditional financial data
- `numpy` for numerical computations and statistical analysis
- `python-dateutil` for advanced date handling

## Common Commands

### Testing
- Run all tests: `python -m unittest discover -s tests -p "test_*.py"`
- Run specific test module: `python -m unittest tests.test_downloaders.test_ccxt_downloader`
- Run specific test class: `python -m unittest tests.test_backtesting.test_backtest_engine.TestBacktestEngine`
- Uses built-in unittest framework (no pytest) — keep it that way
- The suite size is quoted in exactly one place, [README.md](README.md#testing). Do not
  restate a test count anywhere else; it rots immediately.

### Linting and Type Checking
- `ruff check .` must pass — CI enforces it (`.github/workflows/ci.yml`)
- `ruff format` is configured but deliberately **not** enforced; do not run a repo-wide
  reformat
- `[tool.ruff.lint]` selects E4/E7/E9/F/C4. An "ADOPTION BACKLOG" ignore block in
  `pyproject.toml` suppresses E402/E722/F401/F541/F811/F841 with current counts; remove them
  one rule at a time rather than all at once
- `mypy` is scoped to `files = ['niffler']`, non-strict, and advisory only
  (`continue-on-error` in CI) — it currently hits an internal error on
  `niffler/optimization/base_optimizer.py`

### Data Download
Main functionality via `scripts/download_data.py`:

```bash
# Cryptocurrency data from Binance
python scripts/download_data.py --source ccxt --symbol BTC/USDT --timeframe 1d --start_date 2024-01-01 --end_date 2024-01-05 --exchange binance

# Traditional financial data from Yahoo Finance
python scripts/download_data.py --source yahoo --symbol BTC-USD --timeframe 1d --start_date 2024-01-01 --end_date 2024-01-05
```

### Data Preprocessing
Trading data cleaning and validation via `scripts/preprocessor.py`:

```bash
# Clean single file
python scripts/preprocessor.py --input data/BTCUSDT_binance_1d_20240101_20240105.csv --output data/BTCUSDT_cleaned.csv

# Process directory
python scripts/preprocessor.py --input data/ --output cleaned_data/
```

### Backtesting
Strategy backtesting via `scripts/backtest.py`:

```bash
# Run backtest with Simple MA strategy (console output)
# NB: the flag is --capital, not --initial_capital
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --capital 10000 --commission 0.001

# Run backtest with CSV export
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --exporters csv --csv-output-dir results/

# Run backtest with multiple exporters (console + CSV + Elasticsearch)
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --exporters console,csv,elasticsearch --csv-output-dir results/

# Run backtest with data cleaning and custom Elasticsearch settings
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --clean --exporters elasticsearch --es-host localhost --es-port 9200

# Run backtest with risk management
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --risk-manager fixed --max-position-size 0.1 --stop-loss-pct 0.05
```

### Strategy Optimization
Parameter optimization for trading strategies via `scripts/optimize.py`:

```bash
# Grid search optimization for Simple MA strategy
python scripts/optimize.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma --method grid

# Random search with 100 trials
python scripts/optimize.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma --method random --trials 100

# Sort results by Sharpe ratio and save to custom file
python scripts/optimize.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma --sort-by sharpe_ratio --output my_results.json

# Reproducible parallel random search
python scripts/optimize.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma --method random --trials 200 --seed 42 --jobs 4
```

Note: `--sort-by max_drawdown` now ranks the **shallowest** drawdown first. `max_drawdown`
is a negative percentage, and it used to be flagged "lower is better", which selected the
worst parameter set every time (including inside walk-forward folds).

### Strategy Analysis
Advanced robustness testing via `scripts/analyze.py`:

```bash
# Walk-forward analysis (parameters are re-optimised per fold, so --params is NOT needed)
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma

# Walk-forward with custom windows and a random per-fold optimizer selecting on Sharpe
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma \
  --train_window 12 --test_window 6 --step 6 --anchored \
  --optimization_method random --optimization_metric sharpe_ratio

# The old fixed-parameter behaviour - NOT a validation, requires --params, warns loudly
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma \
  --mode segmented_in_sample --params '{"short_window": 10, "long_window": 30}'

# Monte Carlo analysis with 1000 simulations
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params '{"short_window": 10, "long_window": 30}' --simulations 1000

# Parallel execution for faster analysis
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params_file optimization_results.json --n_jobs 8
```

## Architecture

### Core Components
- `niffler/data/downloaders/` - Data acquisition from exchanges and APIs
  - `base_downloader.py` - Abstract base class for data downloaders
  - `ccxt_downloader.py` - Cryptocurrency exchange data via CCXT
  - `yahoo_finance_downloader.py` - Traditional financial data via yfinance
- `niffler/data/preprocessors/` - Data cleaning and validation pipeline
  - `preprocessor_manager.py` - Orchestrates the preprocessing pipeline
  - Individual processors for infinite values, NaN handling, OHLC validation, etc.
- `niffler/data/exceptions.py` - Typed data-layer exceptions
  (`NifflerDataError` → `DownloadError`, `NoDataAvailableError`, `InvalidTimeframeError`),
  re-exported from `niffler.data`. `NoDataAvailableError` is deliberately **not** a subclass
  of `DownloadError`
- `niffler/backtesting/` - Strategy backtesting framework
  - `backtest_engine.py` - Core backtesting engine; `run_backtest` decomposes into
    `_extract_signal_columns`, `_apply_risk_management`, `_process_stop_loss`,
    `_process_buy`, `_process_sell`
  - `portfolio.py` - `Portfolio` owns cash / position / entry price / stop / side and the
    operations on them (`apply_buy`, `apply_sell`, `open_position`, `add_to_position`,
    `close_position`, `market_value`, `position_fraction`, `unrealized_pnl`, `is_flat`).
    The engine holds no parallel mutable state
  - `round_trip.py` - `pair_trades()` → `List[RoundTrip]`, the **single** FIFO trade-pairing
    routine (correct partial fills both directions, pro-rata commission, P&L net of entry
    and exit commission). Used by engine statistics *and* the Elasticsearch position export
  - `cost_model.py` - Transaction cost models: `FillRequest`, the abstract `CostModel`,
    and `ZeroCostModel` (the default) / `FixedSlippageModel` / `VolumeShareSlippageModel`.
    Subclasses implement `adverse_fraction`, **not** `fill_price`: the base class owns the
    sign, so no parameterisation can produce a favourable fill. `max_fillable_quantity`
    caps how much of a bar one order may take
  - `trade.py` - Trade execution and tracking (`Trade` carries optional `commission` and
    `slippage_cost` fields, defaulted, as its last positional fields)
  - `backtest_result.py` - Performance metrics and results
- `niffler/strategies/` - Trading strategy implementations
  - `base_strategy.py` - Abstract base class for strategies
  - `simple_ma_strategy.py` - Simple moving average crossover strategy
- `niffler/optimization/` - Parameter optimization framework
  - `base_optimizer.py` - Abstract base class for optimizers
  - `grid_search_optimizer.py` - Exhaustive grid search optimization
  - `random_search_optimizer.py` - Random parameter sampling optimization
  - `optimizer_factory.py` - Factory for creating optimizers and parameter spaces
  - `parameter_space.py` - Defines parameter ranges for strategies
  - `optimization_result.py` - Stores and analyzes optimization results
- `niffler/analysis/` - Advanced strategy validation framework
  - `walk_forward_analyzer.py` - Real anchored/rolling walk-forward: per-fold optimisation
    on a training window, evaluation on the following untouched test window
    (`WalkForwardWindow`, `WalkForwardFold`, `MODE_WALK_FORWARD`,
    `MODE_SEGMENTED_IN_SAMPLE`)
  - `monte_carlo_analyzer.py` - Market scenario robustness via **return-series** block
    bootstrap with a reconstructed synthetic price path
  - `analysis_result.py` - Unified result container with stability metrics, failure
    accounting (`attempted_runs`, `failed_runs`, `failure_rate`, `is_survivorship_biased`)
    and the shared `log_failure_rate()` / `FAILURE_RATE_WARNING_THRESHOLD`
- `niffler/risk/` - Risk management framework
  - `base_risk_manager.py` - Abstract base class for risk management systems
  - `fixed_risk_manager.py` - Fixed position sizing and stop-loss risk management
  - `kelly_risk_manager.py` - **Stub.** The class and constructor exist; all three abstract
    methods raise `NotImplementedError`. `FixedRiskManager` is the only usable one
- `niffler/exporters/` - Modular result export system
  - `base_exporter.py` - Abstract base class for result exporters
  - `console_exporter.py` - Human-readable console output
  - `csv_exporter.py` - CSV file export for analysis tools
  - `elasticsearch_exporter.py` - Elasticsearch integration for visualization
  - `exporter_manager.py` - Multi-exporter coordination, registry and `ExportSummary`
  - `json_utils.py` - Compatibility re-export of `niffler/utils/json_utils.py`
- `niffler/utils/` - Layer-neutral helpers. **Nothing here may import from the
  backtesting / optimization / exporters layers**, so importing a helper never drags an
  optional third-party dependency (the Elasticsearch client) along with it
  - `json_utils.py` - `sanitize_numeric_values`, `safe_json_dumps`, `safe_json_dump`
    (inf/NaN → `null`, numpy scalars → Python numbers, `allow_nan=False`)
- `niffler/config/logging.py` - Unified logging configuration (the real implementation).
  `setup_logging` validates the level against a whitelist and creates the log file's parent
  directory on demand
- `config/logging.py` - Thin backwards-compatible re-export shim for the old
  `from config.logging import setup_logging` path. New code imports
  `niffler.config.logging`
- `config/elasticsearch/mappings/` - Elasticsearch schema definitions
  (`backtests`, `portfolio`, `trades`, `positions`)
- `scripts/` - Command-line interfaces for core functionality
  - `common.py` - **The** shared OHLCV CSV loader (`load_ohlcv_csv`) *and* the shared
    transaction-cost CLI (`add_cost_model_arguments`, `build_cost_model`,
    `report_cost_model`), used by `backtest.py`, `analyze.py` and `optimize.py`. A cost
    flag belonging to a different `--cost-model` is an error, never silently ignored.Header normalisation, timestamp-column detection
    (`timestamp`/`date`/`datetime`/`time` plus pandas' unnamed index column), datetime
    parsing, required-column and duplicate-timestamp validation, index sorting, optional
    `--clean` pass. Do not add a fourth loader
  - Every `main()` returns an `int` exit code and is invoked as `sys.exit(main())`
  - Scripts insert into `sys.path` only under `if __package__ in (None, '')`, so importing
    them as `scripts.<name>` (tests, discovery) touches nothing

### Data Storage
- Format: CSV files with standardized columns (timestamp, open, high, low, close, volume)
- Naming: `{SYMBOL}_{SOURCE}_{TIMEFRAME}_{STARTDATE}_{ENDDATE}.csv`
- Location: `data/` directory

## Key Implementation Details

### Backtest Execution Model
- **Execution timing**: a signal from bar *i* fills at the **open of bar i+1**
  (`execution_timing='next_bar_open'`, the default). A signal on the final bar never fills.
  `same_bar_close` is the old, look-ahead-biased behaviour, opt-in only and not exposed on
  the CLI
- **Annualisation**: `periods_per_year` is inferred from the index (median bar spacing +
  weekend-presence check), not hardcoded to 252. Daily crypto → 365, daily equities → 252,
  hourly crypto → 8760, weekly → ~52, monthly → ~12. Override with
  `BacktestEngine(periods_per_year=...)`
- **Trade statistics** derive from `pair_trades()` only. `_calculate_win_rate` and
  `_calculate_profit_factor` are thin wrappers over it — do not reintroduce a second
  pairing loop
- **Exits**: `RiskDecision.position_size` sizes *entries* only. A sell uses the strategy's
  own `position_size` (a fraction of held units); the risk manager can veto an exit via
  `allow_trade` but cannot resize it
- **Scaling in**: a buy while already positioned takes a quantity-weighted average entry
  price and only ever tightens the stop (a `None` stop leaves the existing one armed)
- **Stops**: probed against the bar's traded range (low for longs, high for shorts) and
  filled at `min(open, stop)` for a long, so a gap-through fills at the open
- **Transaction costs**: every fill goes through `BacktestEngine.cost_model`, which
  defaults to `ZeroCostModel` so pre-existing numbers stay reproducible. Costs are always
  adverse (a buy pays up, a sell gives up), apply to **stop exits too** (the reference price
  is `min(open, stop)` for a long and the model may only worsen it), and the buy budget is
  solved against the **slipped** price so cash can never go negative. A model that caps
  participation truncates the order to a logged **partial fill** rather than dropping it; a
  bar with no usable volume is unfillable for the volume model and logs `ORDER NOT FILLED`.
  The impact term is evaluated on the pre-slippage order size, which over-charges slightly
  and never under-charges. `--cost-model` is threaded through `backtest.py`, `optimize.py`
  and `analyze.py` so a strategy is fitted and validated in the market it is traded in
- **Long only.** A `-1` signal with no open position is a no-op. No shorting, no live
  trading

### Data Sources
- **CCXT**: Cryptocurrency exchange data with pagination, bounded retries and exponential
  backoff. A give-up mid-download returns the candles already fetched flagged
  `df.attrs['partial']` / `['partial_reason']` rather than discarding them
- **Yahoo Finance**: Traditional financial data via yfinance. `auto_adjust` is passed
  **explicitly on every call** (default `True` = back-adjusted for splits and dividends)
  because yfinance flipped its own default in 0.2.51. The convention used is recorded on
  `df.attrs['auto_adjust']`

### Data Preprocessing Pipeline
- **Infinite Value Removal**: Replaces ±∞ with NaN for calculation safety
- **NaN Handling**: explicit per-column gap policy — price columns forward-fill, **volume /
  flow columns fill with 0** (never a stale value), everything else forward-fills.
  **Backward fill is opt-in** (`allow_backward_fill=True`) because it is look-ahead;
  unresolvable leading rows are dropped. Fabricated cells/rows are counted and reported on
  `self.last_stats` and `result.attrs['nan_fill']`
- **OHLC Validation**: High ≥ Low and Open/Close within High/Low range. `mode='drop'`
  (default), `'repair'` (clamp high/low, keep the sampling grid) or `'flag'`. Counts are
  reported on `result.attrs['ohlc_validation']` and any drop logs a warning with the
  affected percentage
- **Time Gap Detection**: Identifies missing periods and calculates data completeness
- **Data Quality Checks**: Validates positive prices, non-negative volume, removes duplicates

### Error Handling
- Downloaders **raise** typed exceptions rather than returning `None` (see
  `niffler/data/exceptions.py`); `BaseDownloader.download` is annotated `-> pd.DataFrame`
- Scripts report and exit non-zero. Broad `except Exception` is acceptable only at a CLI
  boundary that reports a message and exits; library code catches specific types
- `ExporterManager.export_backtest_result` returns an `ExportSummary`; exporters raise
  `ExportError` rather than logging "skipping" and returning normally
- Analyzers count failed folds/simulations and warn about survivorship bias above 5%

### Analysis Framework Architecture
The analysis framework provides two main approaches for testing strategy robustness:

#### Walk-Forward Analysis
- **Purpose**: Measures how much of a fitted edge survives on data the optimizer never saw
- **Process**: each fold optimises on a training window and backtests the chosen parameters
  on the immediately following test window. `train_end == test_start`, so no bar is ever
  both trained and tested on. The per-fold optimizer always runs with `n_jobs=1` so
  fold-level and optimizer-level pools never nest
- **Key Metrics**: per-fold in-sample vs out-of-sample return, **walk-forward efficiency
  ratio**, temporal stability, failure accounting, out-of-sample overlap
- **Modes**: `walk_forward` (default, genuinely out-of-sample; raises `ValueError` without a
  `parameter_space`) and `segmented_in_sample` (the old fixed-parameter behaviour — it
  validates nothing, requires `optimal_parameters`, logs a warning and marks every fold
  in-sample)
- **Overlap caveat**: with the defaults (`test_window_months=6`, `step_months=3`)
  consecutive out-of-sample windows overlap 50%. Repeated bars are counted once in
  `combined_sharpe_ratio` and the overlap is reported/warned, but per-fold counters still
  treat each fold as one sample

#### Monte Carlo Analysis
- **Purpose**: Tests market scenario robustness using block bootstrap sampling
- **Process**: block-bootstraps the **return** series (not price levels), concatenates
  blocks in draw order (**never** `sort_index()` — that made the old version a no-op),
  compounds them from the real starting price into a synthetic close path, scales O/H/L
  around it and stamps a fresh evenly-spaced `DatetimeIndex`
- **Reproducibility**: each simulation derives `random_seed + simulation_id`, passed
  explicitly into the worker (spawn-based platforms do not inherit seeded generators), and
  results are re-ordered by simulation id before returning. Seeded `n_jobs=1` and
  `n_jobs=8` runs are byte-identical
- **Key Metrics**: Return distribution statistics, VaR/CVaR, percentile analysis,
  skewness/kurtosis
- **Shared discipline**: both analyzers return `(results, failed_count)` from their runner
  methods, report failure rates, and trim over-large result sets by keeping the **most
  recent** half — never the best-performing half, which amputates the left tail

### Risk Management Framework
The risk management system provides position sizing, stop-loss calculation, and portfolio-level risk controls:

#### Fixed Risk Manager
- **Purpose**: Simple, predictable risk management using fixed percentages
- **Position Sizing**: Fixed percentage of portfolio per trade (e.g., 10%)
- **Stop Loss**: Fixed percentage from entry price (e.g., 5%)
- **Portfolio Controls**: Maximum positions, exposure limits, risk per trade caps
- **Use Case**: Conservative risk management with predictable position sizes

#### Kelly Risk Manager (NOT IMPLEMENTED)
- **Purpose**: Optimal position sizing based on historical strategy performance
- **Method**: Would use the Kelly Criterion formula: f* = (bp - q) / b
- **Status**: **Stub.** `calculate_position_size()`, `calculate_stop_loss()` and
  `should_close_position()` all raise `NotImplementedError`. It is not selectable from the
  `backtest.py` CLI (`--risk-manager` accepts `none` and `fixed` only) and must not be
  described as working

#### Risk Management Features
- **Position Tracking**: Real-time monitoring of all open positions
- **Portfolio Risk Controls**: Total exposure limits, maximum concurrent positions
- **Stop Loss Management**: Automated stop-loss calculation and monitoring
- **Risk Metrics**: Comprehensive risk reporting and portfolio utilization tracking

### Export System Architecture
The modular export system enables flexible output of backtest results to multiple destinations:

#### Export Types
- **Console Exporter**: Human-readable formatted output for quick analysis
- **CSV Exporter**: Structured file export for external analysis tools (Excel, Python, R)
- **Elasticsearch Exporter**: Database integration for advanced visualization and dashboards

#### Export Features
- **Multi-Export Support**: Results can be exported to multiple destinations simultaneously
- **Unique Identification**: Each backtest receives a UUID for tracking and correlation
- **Metadata Integration**: Complete strategy parameters and performance metrics included
- **Configuration Management**: Environment-based configuration (.env) with command-line overrides
- **Error Resilience**: Individual exporter failures don't affect others or main process

#### Export Data Structure
- **Backtest Metadata**: Strategy details, parameters, performance metrics, execution info
- **Portfolio Values**: Time-series data of portfolio value evolution
- **Trade Details**: Individual trade records with timestamps, prices, quantities
- **Elasticsearch Integration**: Optimized bulk operations with configurable index mappings

#### Configuration
- **Environment Variables**: Default settings via `.env` file (an empty value counts as unset)
  - `ELASTICSEARCH_HOST` - Elasticsearch server hostname
  - `ELASTICSEARCH_PORT` - Elasticsearch server port  
  - `ELASTICSEARCH_INDEX_PREFIX` - Index naming prefix
  - `ELASTICSEARCH_SCHEME` - `http` (default) or `https`
  - `ELASTICSEARCH_API_KEY` - API key auth; takes precedence over basic auth
  - `ELASTICSEARCH_USERNAME` / `ELASTICSEARCH_PASSWORD` - basic auth (both required)
  - `ELASTICSEARCH_TIMEOUT` - Request timeout in seconds (default 30)
  - `ELASTICSEARCH_VERIFY_CERTS` - TLS certificate verification for https (default true)
- **Command-line Overrides**: Runtime configuration via `--es-host`, `--es-port`, `--es-index-prefix`
- **Mapping Files**: Elasticsearch schema definitions in `config/elasticsearch/mappings/`
- **Failure Reporting**: `ExporterManager.export_backtest_result` returns an `ExportSummary`
  (`successes`, `failures`, `backtest_id`, `ok`). Exporters raise rather than skipping
  silently, and `scripts/backtest.py` prints a per-exporter report and exits 1 on any failure

### Testing Approach
- Mock external dependencies (ccxt, yfinance, elasticsearch)
- Test both successful operations and error conditions
- Validate argument parsing and data output formats
- Test packages mirror the source layout: `tests/test_backtesting`, `test_analysis`,
  `test_downloaders`, `test_preprocessors`, `test_optimization`, `test_risk`,
  `test_exporters`, `test_scripts`
- Integration and functional testing to ensure end-to-end workflow reliability
- Isolated testing with proper mocking and teardown procedures — tests must not leave files
  in the repository root
- **When fixing a bug, add the test that would have caught it**, and verify by reverting the
  fix and watching that test fail. Every audit fix in this codebase was validated that way
- The suite size lives in [README.md](README.md#testing) and nowhere else