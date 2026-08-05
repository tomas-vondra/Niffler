# Niffler

Niffler is a Python-based trading application that helps you sniff out market opportunities — just like a Niffler can't resist shiny gold!

## What is Niffler?

Niffler is a personal quantitative trading framework that provides end-to-end functionality from data acquisition to strategy validation. It's designed for systematic development, testing, and validation of trading strategies with rigorous statistical analysis.

## Key Features

- **📈 Data Acquisition**: Download from cryptocurrency exchanges (CCXT) and traditional markets (Yahoo Finance)
- **🧹 Data Processing**: Comprehensive cleaning and validation pipeline
- **🎯 Strategy Framework**: Extensible strategy development with risk management integration  
- **⚡ Backtesting Engine**: Realistic simulation with commission handling and portfolio management
- **🔍 Parameter Optimization**: Grid search and random search with parallel processing
- **📊 Advanced Analysis**: Walk-forward and Monte Carlo robustness testing
- **🛡️ Risk Management**: Position sizing, stop-loss management, and portfolio controls
- **📤 Flexible Exports**: Multi-format result export (console, CSV, Elasticsearch) for analysis and visualization
- **✅ Comprehensive Testing**: see [Testing](#testing) for the current suite size and the command that reports it

> **Upgrading from an earlier revision?** Read
> [What changed and why your old results differed](#what-changed-and-why-your-old-results-differed)
> first. Backtest, walk-forward and Monte Carlo numbers all moved, deliberately.

## Quick Start

### Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <repository-url>
cd Niffler
uv sync

# Verify installation
python -m unittest discover -s tests -p "test_*.py"
```

### Basic Usage

```bash
# 1. Download market data
python scripts/download_data.py --source ccxt --symbol BTC/USDT --timeframe 1d --start_date 2024-01-01 --end_date 2024-12-31 --exchange binance

# 2. Clean the data
python scripts/preprocessor.py --input data/BTCUSDT_binance_1d_20240101_20241231.csv

# 3. Run backtest with export to CSV
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20241231_cleaned.csv --strategy simple_ma --symbol BTC/USDT --exporters console,csv --csv-output-dir results/

# 4. Optimize parameters  
python scripts/optimize.py --data data/BTCUSDT_binance_1d_20240101_20241231_cleaned.csv --strategy simple_ma --method grid --output data/optimization_results.json

# 5. Validate robustness (walk-forward re-optimises per fold, so it needs no --params)
python scripts/analyze.py --data data/BTCUSDT_binance_1d_20240101_20241231_cleaned.csv --analysis walk_forward --strategy simple_ma
```

Every script returns a **non-zero exit code on failure**, so they compose safely in shell
pipelines and CI. In particular `backtest.py` exits 1 if any exporter fails, and
`download_data.py` exits 1 if a download returned nothing or came back truncated.

## Framework Workflow

Niffler follows a systematic approach to quantitative trading strategy development:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Download  │ -> │ Data Processing │ -> │ Strategy Design │
│                 │    │                 │    │                 │
│ • Crypto (CCXT) │    │ • Clean data    │    │ • Signal logic  │  
│ • Traditional   │    │ • Validate      │    │ • Parameters    │
│   (Yahoo)       │    │ • Handle gaps   │    │ • Risk rules    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         v                        v                        v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Backtesting   │ -> │  Optimization   │ -> │  Validation     │
│                 │    │                 │    │                 │
│ • Portfolio mgmt│    │ • Grid search   │    │ • Walk-forward  │
│ • Commission    │    │ • Random search │    │ • Monte Carlo   │
│ • Risk controls │    │ • Parallel exec │    │ • Robustness    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         v                        v                        v
┌─────────────────┐
│   Export        │
│                 │
│ • Console       │
│ • CSV files     │
│ • Elasticsearch │
└─────────────────┘
```

### 1. Data Layer
**Download** market data from multiple sources, then **process** it through a comprehensive cleaning pipeline that handles missing values, validates OHLC relationships, and ensures data quality.

### 2. Strategy Development  
**Design** trading strategies using the extensible framework, implementing signal generation logic with risk management integration.

### 3. Backtesting
**Test** strategies against historical data with realistic simulation including commissions, position tracking, and portfolio management.

### 4. Optimization
**Find** optimal parameters using grid search or random search methods with parallel processing for efficiency.

### 5. Robustness Validation
**Validate** strategy robustness using advanced analysis methods:
- **Walk-Forward Analysis**: Tests temporal stability across rolling time windows
- **Monte Carlo Analysis**: Tests performance across thousands of market scenarios

### 6. Results Export
**Export** backtest results to multiple formats for analysis and monitoring:
- **Console**: Immediate human-readable feedback
- **CSV Files**: Structured data for external analysis tools
- **Elasticsearch**: Database integration for visualization dashboards

## What changed and why your old results differed

A correctness audit changed several defaults that were quietly flattering results. If you
re-run an old backtest and the numbers moved, this section is why. **In almost every case
the new number is the honest one and the old number was not achievable.**

### 1. Signals now fill on the next bar's open (was: same bar's close)

A signal computed from bar *i* used to be filled at the **close of bar i** — a price that
was not knowable at the moment the signal was computed. That is look-ahead bias, and it
inflates every strategy that reacts to intraday moves.

Fills now happen at the **open of bar i+1**. Consequences:

- Every fill price and fill timestamp shifts by one bar.
- A signal on the **final bar never fills** — there is nothing left to execute against — so
  trade counts can drop by one.
- Final capital, total return, Sharpe, drawdown and win rate all move.

The old behaviour survives as `BacktestEngine(execution_timing='same_bar_close')`, is
documented as biased, and exists only to reproduce historical figures. It is not exposed on
the CLI. See [Backtesting](docs/backtesting.md#execution-timing-no-look-ahead).

### 2. Sharpe is annualised from the data, not from a hardcoded 252

`np.sqrt(252)` was applied to everything, including 24/7 crypto and hourly bars. The
annualisation factor is now inferred from the index (median bar spacing plus a
weekend-presence check): daily crypto → 365, daily equities → 252, hourly crypto → 8760,
weekly → ~52, monthly → ~12.

Daily crypto Sharpe therefore grows by roughly `sqrt(365/252) ≈ 1.20x` versus the old
figure. Pass `BacktestEngine(periods_per_year=252)` to reproduce the old number.

### 3. Trade statistics use one FIFO pairing routine, net of commission

There were three inconsistent pairing implementations (engine metrics, win-rate helper,
Elasticsearch position export), and none handled partial fills correctly. They are now one
routine (`pair_trades()` → `RoundTrip`) with a FIFO queue of open lots, correct partial
fills in both directions, and commission apportioned pro rata.

Win rate, profit factor, average/largest win and loss, and the `niffler-positions` index
all now agree with each other. A thin gross win can correctly register as a **net loss**;
sequences with a second buy before the matching sell were previously wrong, sometimes
wildly so.

### 4. Walk-forward is a real train/test split (was: not a validation at all)

The old walk-forward took one pre-optimized parameter set and re-ran it over consecutive
slices of *the data those parameters were fitted on*. It validated nothing.

Each fold now **optimises on a training window and evaluates on the immediately following,
untouched test window** (`train_end == test_start`, so no bar is ever both trained and
tested on), and reports a per-fold walk-forward efficiency ratio.

- New parameters: `parameter_space`, `train_window_months`, `anchored`, `mode`,
  `optimization_method`, `optimization_metric`. New CLI flags: `--mode`, `--train_window`,
  `--anchored`, `--optimization_method`, `--optimization_metric`.
- `--params` / `--params_file` is **no longer required** for walk-forward; parameters are
  refit per fold, so any value passed is ignored with a warning.
- The old behaviour is still reachable as `--mode segmented_in_sample`, which requires
  `--params`, logs a warning that its results are not out-of-sample, and marks every fold
  in-sample.
- Constructing `WalkForwardAnalyzer` the old way (only `optimal_parameters`, no
  `parameter_space`) now raises `ValueError` by design.

**Walk-forward returns will typically look materially worse than before.** That is the
point — the previous figure was in-sample.

### 5. Monte Carlo bootstrap actually resamples now

The old block bootstrap glued raw **price levels** together and then called `sort_index()`,
which put the blocks back into nearly their original order — a near no-op that also
manufactured a huge artificial gap return at every block boundary.

It now block-bootstraps the **return** series, keeps blocks in draw order, compounds them
from the real starting price into a synthetic close path, scales each bar's O/H/L around
that close, and stamps a fresh evenly-spaced `DatetimeIndex`. Every synthetic return is a
real historical return; apart from the first bar, no source close or timestamp survives.

Also fixed: `random_seed` never reached the parallel workers (a silent no-op on Windows),
and results were collected in completion order. Seeded runs are now byte-identical between
`n_jobs=1` and `n_jobs=8`. Distributions shift accordingly.

### 6. Missing data is no longer invented

The preprocessor used to `ffill().bfill()` everything, which both fabricated volume and
copied future values backwards into the past.

- **Volume/flow columns fill with 0**, never a stale value — a bar with no trading did not
  trade at the previous bar's volume. Volume-dependent indicators and sizing will differ.
- **Backward fill is off by default** (it is look-ahead). Leading NaN rows are **dropped**
  instead, so series with leading gaps now start later and have fewer rows. Opt back in with
  `NanValuePreprocessor(allow_backward_fill=True)` for offline inspection only.
- Fabricated cells and rows are counted per column and reported, with a warning that
  escalates above a 5% fabricated fraction.
- OHLC validation still drops violating rows by default, but now **says how many** and
  offers `mode='repair'` (clamp high/low, keep the sampling grid) and `mode='flag'`.

Net effect: the default pipeline produces shorter, less fabricated frames. Measured
volatility rises and reported Sharpe ratios fall. That is the correction.

### 7. Yahoo Finance price convention is now explicit

`yfinance` flipped its own `auto_adjust` default in 0.2.51, which silently changed whether
downloaded prices were split/dividend adjusted depending on the installed version. Niffler
now passes `auto_adjust` **explicitly on every call** and records the convention used on
`df.attrs['auto_adjust']`.

**The convention is back-adjusted prices (`auto_adjust=True`)**: OHLC adjusted for splits
and dividends, no separate `Adj Close`. Pass `auto_adjust=False` (constructor or per call)
for raw prices. If your old data came from a pre-0.2.51 yfinance, it was raw and your
prices will differ.

### 8. Failures are no longer silent

- Downloaders **raise** instead of returning `None`: `InvalidTimeframeError`,
  `NoDataAvailableError` (request succeeded, venue had nothing) or `DownloadError`
  (something genuinely broke). `NoDataAvailableError` is deliberately *not* a subclass of
  `DownloadError` so callers can distinguish them.
- A CCXT download that gives up mid-pagination returns what it fetched flagged
  `df.attrs['partial']`; `download_data.py` writes it to `<name>.partial.csv` and exits 1
  rather than presenting it as a complete series.
- `ExporterManager.export_backtest_result` returns an `ExportSummary`
  (`successes`, `failures`, `backtest_id`, `ok`) instead of a bare id, exporters raise
  rather than logging "skipping", and `backtest.py` prints a per-exporter report and exits 1
  on any failure — including exporters whose constructor was rejected.
- Walk-forward and Monte Carlo count and report failed folds/simulations
  (`attempted_runs`, `failed_runs`, `failure_rate`, `is_survivorship_biased`), and warn by
  name about survivorship bias above a 5% failure rate. Result trimming keeps the **most
  recent** half, not the best-performing half.

### 9. Smaller corrections that still move numbers

- **Risk-managed exits actually exit.** A sell under a risk manager liquidated only a
  fraction of the position (a portfolio *value* fraction was reinterpreted as a fraction of
  held *units*), degenerating into buy-and-hold with exposure creep. Exits now use the
  strategy's own `position_size`; a risk manager can still veto an exit via `allow_trade`.
- **Scaling into a position keeps its cost basis.** A second buy used to overwrite the entry
  price and could replace the stop with `None`, permanently disarming it. The entry price is
  now a quantity-weighted average and a stop is only ever tightened.
- **Stops are probed against the bar's traded range**, not just its execution price, and
  fill at `min(open, stop)` for a long so a gap-through fills at the open. Stops fire
  earlier and at the stop price.
- **`worst_max_drawdown` is now the deepest drawdown.** `max_drawdown` is negative, so the
  old `max(...)` returned the *mildest* one in both analyzers.
- **Sorting by `max_drawdown` now ranks shallowest first.** It was marked "lower is better"
  while the values are negative, so walk-forward folds optimising on it selected the worst
  parameter set every time.
- **CSV filenames are sanitised.** `BTC/USDT` used to write nothing at all; it now yields
  `BTC_USDT_...`. Already-safe names are unchanged.
- **JSON output emits `null`** instead of the non-standard `Infinity`/`NaN` literals.

## What Niffler does *not* do

Being explicit, so nobody discovers these the expensive way:

- **No live trading.** There is no broker/exchange order path. Backtesting and analysis only.
- **No shorting.** The engine is long-only; a `-1` signal with no open position is a no-op.
  `Portfolio` is the right place to add it.
- **No slippage or spread model.** Fills happen at the exact bar open. Commission is
  modelled; market impact, bid/ask spread and partial-fill risk are not.
- **One strategy ships:** `simple_ma` (moving-average crossover). The framework is
  extensible; the library is not stocked.
- **Kelly risk manager is a stub.** `KellyRiskManager` exists as a class, but
  `calculate_position_size()`, `calculate_stop_loss()` and `should_close_position()` all
  raise `NotImplementedError`. `FixedRiskManager` is the only working risk manager.
- **Single-asset backtests.** The engine runs one symbol at a time; the portfolio-level
  risk limits (max concurrent positions, total exposure) are not exercised by it.
- **Walk-forward folds still overlap by default** (`test_window=6`, `step=3`). Repeated
  out-of-sample bars are counted once for the combined Sharpe and the overlap is reported
  and warned about, but per-fold counters still treat each fold as one sample.
- **Docker images are unverified at runtime.** The compose file validates and the Dockerfile
  is written for a non-root user, but no one has yet run `docker compose build && up` on a
  machine with a working daemon.

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[Installation Guide](docs/installation.md)** - Setup and dependencies
- **[Data Management](docs/data-management.md)** - Data download and preprocessing  
- **[Backtesting](docs/backtesting.md)** - Strategy testing and simulation
- **[Optimization](docs/optimization.md)** - Parameter optimization methods
- **[Analysis](docs/analysis.md)** - Advanced robustness testing
- **[Risk Management](docs/risk-management.md)** - Position sizing and risk controls
- **[Exporters](docs/exporters.md)** - Result export system and configuration

## Architecture

### Core Components

```
niffler/
├── data/           # Data acquisition (downloaders), preprocessing, typed exceptions
├── strategies/     # Trading strategy implementations
├── backtesting/    # BacktestEngine + Portfolio + Trade/RoundTrip pairing
├── optimization/   # Parameter optimization framework
├── analysis/       # Walk-forward and Monte Carlo validation
├── risk/           # Risk management systems
├── exporters/      # Console / CSV / Elasticsearch result export
├── config/         # Logging configuration
└── utils/          # Layer-neutral helpers (JSON sanitisation)

scripts/            # CLI entry points; scripts/common.py holds the shared CSV loader
config/             # Elasticsearch mappings, Grafana provisioning (+ logging.py shim)
visualization/      # Elasticsearch/Kibana maintenance scripts
```

Notable pieces:

- **`Portfolio`** (`niffler/backtesting/portfolio.py`) owns cash, position, entry price,
  stop and side, with `apply_buy` / `apply_sell` / `add_to_position` / `market_value` /
  `position_fraction`. Extracting it removed the parallel mutable locals that made
  `run_backtest` unreadable (and caused an unbound-variable crash on first-bar trades).
- **`pair_trades()` / `RoundTrip`** (`niffler/backtesting/round_trip.py`) is the single FIFO
  trade-pairing routine used by the engine's statistics *and* the Elasticsearch position
  export, so those can no longer disagree.
- **`scripts/common.py::load_ohlcv_csv`** is the one CSV loader for `backtest.py`,
  `analyze.py` and `optimize.py`: identical header normalisation, timestamp detection,
  index sorting, duplicate detection and error messages.
- **`niffler/config/logging.py`** is the real logging setup; the old `config/logging.py` is
  a thin re-export shim kept for backwards compatibility.
- **`niffler/utils/json_utils.py`** holds `safe_json_dump`/`sanitize_numeric_values`. It
  lives outside `exporters/` so that importing it never drags in the optional Elasticsearch
  client. `niffler/exporters/json_utils.py` re-exports it.

### Technology Stack

- **Python ≥3.13** with modern `uv` dependency management
- **pandas** for data manipulation and time series analysis
- **ccxt** for cryptocurrency exchange integration
- **yfinance** for traditional financial market data
- **numpy** for numerical computations
- **multiprocessing** for parallel optimization and analysis
- **ruff** and **mypy** as dev-only tooling, run in CI (`uv sync` installs the dev group)

## Testing

The suite is the source of truth for its own size. Run it:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

At the time of writing this reports **698 tests, 0 failures, 0 errors**. Treat that as a
sanity check, not a spec — if the command disagrees with this paragraph, believe the
command. It is the only place in the documentation that quotes a count.

The project uses the standard-library `unittest` framework. There is no pytest, and no
`conftest.py`; test discovery is plain `unittest discover`. CI
(`.github/workflows/ci.yml`) runs `ruff check .` and the same discovery command on every
push to `main`/`master` and every pull request, plus an advisory (non-blocking) `mypy` pass.

## Project Goals

This personal project aims to:

- **Systematic Approach**: Develop a structured methodology for trading strategy development
- **Rigorous Validation**: Implement advanced statistical testing to avoid overfitting
- **Realistic Testing**: Ensure backtesting reflects real-world trading constraints  
- **Risk Management**: Integrate proper risk controls from the ground up
- **Continuous Learning**: Provide a platform for experimenting with new ideas and techniques

## Development Philosophy

### Quality First
- **Comprehensive Testing**: see [Testing](#testing) — every audit fix is covered by a test
  that fails when the fix is reverted
- **Data Quality**: Built-in preprocessing ensures clean, reliable data — and reports how
  much of the data it had to fabricate or discard
- **Error Handling**: Failures raise typed exceptions and propagate to a non-zero exit code
  rather than being logged and swallowed

### Performance Focused
- **Realistic Simulation**: Commission handling, position tracking, risk management
- **Multi-core Processing**: Parallel optimization and analysis for efficiency
- **Memory Management**: Intelligent handling of large datasets and results

### Extensible Design
- **Abstract Base Classes**: Easy to add new strategies, risk managers, and analyzers
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Modern Python**: Uses latest features and best practices

---

*"Just like the magical creatures they're named after, Niffler helps discover hidden treasures in financial markets through systematic, rigorous analysis."*