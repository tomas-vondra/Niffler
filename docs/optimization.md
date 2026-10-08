# Strategy Optimization

## Optimization Script Usage

```bash
python scripts/optimize.py --data <data_file> --strategy <strategy_name> [--method <method>] [--trials <n>] [--sort-by <metric>] [--output <file>] [--clean] [...]
```

**Arguments:**
- `--data`: Path to CSV file containing historical market data (**required**)
- `--strategy`: Strategy to optimize (**required**; currently supports `simple_ma`)
- `--method`: Optimization method — `grid` (default) or `random`
- `--trials`: Number of trials for random search, default: **100**
- `--sort-by`: Metric to sort results by, default: `total_return` (see
  [sorting](#sorting-by-total_return-rediscovers-buy-and-hold))
- `--initial-capital`: Starting capital, default: 10000. (Spelled `--capital` in
  `backtest.py` and `--initial-capital` in `analyze.py`; the three CLIs have not been
  harmonised.)
- `--commission`: Commission rate per trade, default: 0.001
- `--clean`: Apply the data cleaning pipeline before optimization
- `--jobs`: Number of parallel worker processes, default: auto-detect
- `--seed`: Random seed for reproducible random search
- `--top-n`: How many best results to print, default: 10
- `--output`: Output JSON file for results, defaults to an auto-generated filename
- `--log-level`: `DEBUG` / `INFO` (default) / `WARNING` / `ERROR`

`optimize.py` exits non-zero when the data cannot be loaded or validated. It no longer
re-wraps every error (including `FileNotFoundError`) as a generic `ValueError`, so the
reported cause is the real one.

## Examples

**Grid search optimization for Simple Moving Average strategy:**
```bash
python scripts/optimize.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --method grid
```

**Random search with 100 trials, sorted by Sharpe ratio:**
```bash
python scripts/optimize.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --method random --trials 100 --sort-by sharpe_ratio
```

**Optimization with data cleaning and custom output file:**
```bash
python scripts/optimize.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --method grid --clean --output my_optimization_results.json
```

## Optimization Framework

### Core Components

#### OptimizerFactory
The `OptimizerFactory` creates optimizers and manages parameter spaces:

**Key Features:**
- Strategy-specific parameter space definition via `PARAMETER_SPACES` mapping
- Optimizer creation based on method selection via `OPTIMIZER_CLASSES` registry
- Strategy class lookup via `STRATEGY_CLASSES` mapping
- Parameter validation and bounds checking

**Available Methods:**
- `grid`: GridSearchOptimizer - Exhaustive grid search
- `random`: RandomSearchOptimizer - Random parameter sampling

#### Parameter Space Management
Parameter spaces are defined using the `ParameterSpace` class with validation:

**Simple MA Strategy Parameters (SIMPLE_MA_PARAMETER_SPACE):**
- `short_window`: Integer range [5, 20] with step 1 - Fast moving average period
- `long_window`: Integer range [20, 100] with step 5 - Slow moving average period  
- `position_size`: Float range [0.5, 1.0] with step 0.1 - Position size fraction

**Parameter Types Supported:**
- `int`: Integer parameters with min/max/step
- `float`: Float parameters with min/max/step  
- `choice`: Discrete choice parameters with list of options

#### Optimization Methods

##### Grid Search (`GridSearchOptimizer`)
Exhaustive search through all parameter combinations:

**Features:**
- Systematic exploration of entire parameter space
- Deterministic and reproducible results
- Guarantees finding best combination within search space
- Higher computational cost for large parameter spaces

##### Random Search (`RandomSearchOptimizer`) 
Stochastic sampling of parameter space:

**Features:**
- Configurable number of trials (default: 50)
- Faster exploration for large parameter spaces
- Good for initial parameter discovery
- Efficient for high-dimensional optimization problems

### Optimization Process

#### 1. Data Loading and Validation
- Loads CSV data with required columns: timestamp, open, high, low, close, volume
- Converts timestamp to datetime index and sorts data
- Optional data preprocessing if `--clean` flag is used
- Validates data format and completeness

#### 2. Strategy and Parameter Space Setup
- Retrieves strategy class from `STRATEGY_CLASSES` mapping
- Gets corresponding parameter space from `PARAMETER_SPACES` mapping
- Validates parameter configuration and bounds
- Creates optimizer instance with specified method

#### 3. Optimization Execution
- Creates optimizer using `create_optimizer()` factory function
- Configures optimization parameters (trials, sorting metric, etc.)
- Runs optimization with specified method
- Uses same backtesting engine as standalone backtests

#### 4. Result Collection and Analysis
- Collects comprehensive performance metrics for each parameter combination
- Sorts results by specified metric (default: `total_return`)
- Generates optimization summary and statistics
- Saves results to JSON file for further analysis

### Available Sorting Metrics

Results can be sorted by any of these performance metrics:
- `total_return`: Absolute profit/loss in currency units
- `total_return_pct`: Percentage return on initial capital
- `sharpe_ratio`: Risk-adjusted return measure
- `max_drawdown`: Maximum peak-to-trough decline percentage
- `win_rate`: Percentage of profitable trades
- `total_trades`: Number of trades executed
- `excess_return_pct`: Return over buy-and-hold on the same bars, charged the same costs

#### Sorting by `total_return` rediscovers buy-and-hold

`total_return` is still the default sort. In a bull market it systematically selects
whichever parameters keep you in the market longest, which is a rediscovery of
buy-and-hold reported as a strategy.

The top-N block now prints the benchmark return and the excess beside every result,
whatever the sort order, so the trap is visible without changing anything:

```
#1 - total_return: 63.20%
    Parameters: {'short_window': 5, 'long_window': 15}
    Total Return: $6,320.00 (63.20%)
    vs Buy-and-Hold: 121.40% (excess -58.20 pp)
```

`--sort-by excess_return_pct` is also available. Over a single dataset the benchmark is a
constant, so it produces the **same ordering** as `total_return` — what changes is that
the headline number tells you whether the winner beat doing nothing. A result with no
benchmark (`benchmark_error` set) sorts last rather than being treated as a zero excess.

The default was **not** changed. Changing a default silently is precisely what the
correctness audit was cleaning up; whether `excess_return_pct` should become the default
is a decision for the repository owner.

#### `max_drawdown` sort direction (behaviour change)

`max_drawdown` is expressed as a **negative** percentage (`-5.0` is a 5% drawdown), but it
was configured as "lower is better". Sorting `-5 / -40 / -12` therefore put `-40` first, so
`--sort-by max_drawdown` — and any walk-forward fold using
`--optimization-metric max_drawdown`, which takes `results[0]` — selected the **worst**
drawdown every time.

It is now sorted highest-first, so the **shallowest** drawdown ranks first. Sorting by any
other metric is unchanged.

### Search luck

A search keeps the best of N combinations, and the best of N estimates is biased upwards
even when none of them has any edge: flip ten coins 396 times and somebody gets nine heads.
Every `optimize.py` run therefore prints a `SEARCH LUCK` block after the plateau block.

It rests on one quantity from Bailey & López de Prado (2014): the **allowance** for
selection - how far above its starting point the best of N trials is expected to land by
chance, from N and the spread of the trial Sharpe ratios:
`sqrt(V[SR]) * ((1 - γ) Z⁻¹[1 - 1/N] + γ Z⁻¹[1 - 1/(N e)])`. A **luck line** is that
allowance added to a starting point, and the block reports the probability that the
winner's *true* Sharpe is above the line, given its bar count and the skewness and kurtosis
of its returns.

"By luck" needs a null, so there are two lines, in this order:

| Figure | Null | Luck line | Exported as |
|--------|------|-----------|-------------|
| **Grid-relative probability** - leads, and drives the verdict | every combination is as good as the average one, and tuning found nothing | the allowance above the **grid's mean Sharpe** | `grid_relative_probability`, `grid_relative_luck_line` |
| **Deflated Sharpe ratio** - the published statistic | no combination has any edge at all | the allowance above **zero** | `deflated_sharpe`, `expected_max_sharpe` |

Only the second is the deflated Sharpe ratio; the name is not used for the first. The
published figure does not lead because it is easy to clear for the wrong reason: every
combination of a long-only strategy on an asset that rose carries the same market exposure,
so the whole grid sits above zero. On the default `breakout` grid over BTCUSDT 2019-07 to
2024-07 the trial Sharpe ratios average 1.007 with a spread of 0.123, and the winner's is
1.263. The grid-relative line is 1.373 and the probability 40.1%; the deflated Sharpe ratio
puts its line at 0.366 and reads 97.9%. The family is above zero; the search is no evidence
for those particular parameters, and that is the verdict the block prints.

What it does not do:

- **N is every combination evaluated.** Neighbouring parameter sets are near-duplicates, so
  the number of independent trials is smaller. A smaller N lowers the luck line, so the
  default over-corrects rather than under-corrects. `--effective-trials N` overrides it and
  the block says which was used; estimating it from the trials is not implemented.
- **A combination that never traded counts as a trial but adds nothing to the spread.** It
  was tried, so it is in N; it has no Sharpe ratio, so it is not in the spread or the mean.
  The block says how many there were.
- **One search.** Other strategies and earlier grids on the same data are not counted.
- **Neither null is "no edge over the market".** Buy-and-hold stays a separate comparison.
- **A truncated result set gets no figure.** When the memory cap discarded results the
  survivors were selected by score, so their spread is not the grid's. The block says
  `NOT COMPUTED` and the exported fields are null.

The winner is whatever `--sort-by` ranked first. When that is not the Sharpe ratio, the
winner's Sharpe is at most the grid's best, so judging it against the expected *best* Sharpe
is the stricter reading. The probability is computed in per-bar Sharpe; the annualised
figures shown use the engine's own inferred annualisation.

`screen.py` prints and exports the grid-relative probability in its optimize stage, and
gates on it only when `--min-grid-relative-probability` is set. It is off by default
because the figure counts every combination as independent, which over-corrects, and the
funnel's later stages test the winner on data the search did not see; so it informs by
default and is strict only if asked. The breakout example above reads 40.1%: unset, the
funnel prints `not gated at optimize: grid-relative probability 0.40, no threshold set` and
carries on; with `--min-grid-relative-probability 0.5` it stops there. `screen.py` always
counts every combination; it has no `--effective-trials`. When the figure is
`NOT COMPUTED` the stage says so in a fenced block that calls it not a pass - a warning
when no threshold is set, a stop when one is.

### Output Format

#### JSON Output Structure
Results are saved in structured JSON format containing:
- **Optimization metadata**: Strategy name, method, data period
- **Parameter space definition**: Complete parameter ranges and types
- **Results array**: All parameter combinations tested with their performance metrics
- **Best results**: Top performing parameter combinations
- **`provenance`**: git SHA / branch / dirty flag, a SHA-256 of the input CSV and the
  library versions used — see the [README](../README.md#run-provenance). An optimisation
  run whose code and input cannot be identified is exactly as useless as an
  unreproducible backtest, and the chosen parameters usually outlive the run that
  produced them

The file is written with `safe_json_dump` (`niffler/utils/json_utils.py`), so non-finite
metrics are emitted as **`null`** rather than the non-standard `Infinity` / `NaN` literals
that most JSON parsers reject. This matters here in particular: degenerate parameter
combinations legitimately produce an infinite profit factor (no losing trades) or a NaN
Sharpe ratio (zero variance), and the resulting file used to be unparseable by anything
strict — including the `--params-file` path in `analyze.py`.

#### Console Output
- A `Progress: 120/396 combinations (30%) | elapsed 0:00:42 | ETA 0:01:37` log line at
  most every 10 seconds, at `INFO`, from the parent process - so it appears whatever
  `--jobs` is. A search that finishes inside 10 seconds prints none.
- Each parameter combination as it is evaluated, at `DEBUG` only
- The per-backtest lines (each fill, the data range, the benchmark) are held back during a
  search, a walk-forward and a Monte Carlo run; warnings still print, and
  `--log-level DEBUG` restores them
- Final summary with best results

### Integration Features

#### Data Preprocessing Integration
- Optional data cleaning via `--clean` flag using `PreprocessorManager`
- Applies full preprocessing pipeline before optimization
- Ensures consistent data quality across all optimization runs

#### Backtesting Integration
- Uses same `BacktestEngine` for consistent performance measurement
- Same commission rates and trading constraints
- Reliable comparison across parameter combinations
- Integration with risk management systems

#### Analysis Pipeline Integration
- Optimization results compatible with analysis scripts (`analyze.py`)
- JSON output can be used as `--params-file` input for robustness testing
- Seamless workflow from optimization to validation

### Parameter Space Configuration

The parameter space system supports flexible parameter definitions:

#### Parameter Types
- **Integer parameters**: `{'type': 'int', 'min': 5, 'max': 20, 'step': 1}`
- **Float parameters**: `{'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.1}`
- **Choice parameters**: `{'type': 'choice', 'choices': ['option1', 'option2']}`

#### Validation Rules
- Min value must be less than max value for numeric parameters
- Step size must be positive for numeric parameters
- Choice parameters must have non-empty choices list
- All parameters must specify valid type

#### Overriding the space for one project

The space above is a code constant on the strategy. `[optimize.parameter_space.STRATEGY]`
in `niffler.toml` replaces the entry for each parameter it names and leaves the rest of the
`PARAMETER_SPEC` alone, which is what a plateau that has run into the edge of the searched
range is asking for:

```toml
[optimize.parameter_space.simple_ma.long_window]
type = "int"
min  = 20
max  = 200
step = 5
```

The override is checked against the strategy's constructor signature and against
`ParameterSpace`, so a parameter the strategy does not accept, or an inverted range, is an
error rather than a silently ignored line. It applies to `scripts/optimize.py`; the
per-fold optimizers inside `analyze.py`, `compare.py` and `screen.py` still use the
strategy's own spec.

### Best Practices

#### Parameter Space Design
- Start with reasonable ranges based on domain knowledge
- Use step sizes that balance resolution with computational cost
- Consider parameter interactions and constraints
- Validate parameter ranges make sense for the strategy

#### Method Selection
- **Grid Search**: Use for comprehensive search of small parameter spaces
- **Random Search**: Use for initial exploration or large parameter spaces
- Consider computational budget and available time
- Grid search guarantees finding best combination within bounds

#### Result Interpretation
- Focus on risk-adjusted metrics (Sharpe ratio) over raw returns
- Consider multiple metrics to avoid overfitting
- Validate results with out-of-sample testing
- Use optimization results as input for robustness analysis