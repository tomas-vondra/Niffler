# Strategy Analysis

## Analysis Script Usage

```bash
python scripts/analyze.py --data <data_file> --analysis <analysis_type> --strategy <strategy_name> (--params <parameters_json> | --params_file <params_file>) [additional_options]
```

**Required Arguments:**
- `--data`: Path to CSV file containing historical market data
- `--analysis`: Analysis type (`walk_forward` or `monte_carlo`)
- `--strategy`: Strategy to analyze (currently supports `simple_ma`)

**Parameter Arguments** (`--params` or `--params_file`):
- `--params`: Strategy parameters as JSON string (e.g., `'{"short_window": 10, "long_window": 30}'`)
- `--params_file`: Path to JSON file containing parameters (can use optimization results file)

Required for `--analysis monte_carlo` and for `--mode segmented_in_sample`. **Not** required
for walk-forward analysis: it re-optimises the parameters on every training window, so a
fixed parameter set would be meaningless there (any value passed is ignored with a warning).

**Optional Arguments:**
- `--initial_capital`: Starting capital, default: 10000 (note: `backtest.py` spells this
  `--capital` and `optimize.py` spells it `--initial-capital`)
- `--commission`: Commission rate per trade, default: 0.001
- `--symbol`: Symbol identifier, default: "UNKNOWN"
- `--n_jobs`: Number of parallel jobs, default: auto-detect (max 4)
- `--output`: Save detailed results to JSON file
- `--verbose`, `-v`: Enable debug logging

`analyze.py` returns a **non-zero exit code** on failure: a missing `--params` where one is
required, an unloadable data file, or a results file that cannot be written. It no longer
prints "Analysis completed successfully!" after failing to save.

**Walk-Forward Specific Options:**
- `--mode`: `walk_forward` (default, genuinely out-of-sample) or `segmented_in_sample`
- `--train_window`: Training window size in months, default: 12
- `--test_window`: Test window size in months, default: 6
- `--step`: Step size in months between windows, default: 3
- `--anchored`: Anchor every training window to the first bar instead of rolling it
- `--optimization_method`: Optimizer used per training window (`grid` or `random`), default: `grid`
- `--optimization_metric`: Metric the per-fold optimizer selects by, default: `total_return`

**Monte Carlo Specific Options:**
- `--simulations`: Number of simulations, default: 1000
- `--bootstrap_pct`: Percentage of data to sample, default: 0.8 (80%)
- `--block_size`: Block size in days for bootstrap sampling, default: 30
- `--random_seed`: Random seed for reproducible results

## Examples

**Walk-forward analysis (parameters re-optimised on every training window):**
```bash
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma
```

**Re-run one fixed parameter set over consecutive in-sample slices (NOT a validation):**
```bash
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --mode segmented_in_sample --strategy simple_ma --params '{"short_window": 10, "long_window": 30}'
```

**Load parameters from optimization results (Monte Carlo):**
```bash
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params_file optimization_results.json
```

**Monte Carlo analysis with 1000 simulations:**
```bash
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params '{"short_window": 10, "long_window": 30}' --simulations 1000
```

**Parallel execution with custom settings:**
```bash
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params_file optimization_results.json --n_jobs 8 --bootstrap_pct 0.75 --output analysis_results.json
```

**Walk-forward with custom time windows:**
```bash
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --train_window 12 --test_window 6 --step 3
```

## Analysis Framework

### Walk-Forward Analysis

#### Purpose and Methodology
Walk-forward analysis measures how much of a strategy's fitted edge survives on data the
optimizer never saw. Each fold optimises the parameters on an **in-sample training window**
and then evaluates *those* parameters on the **immediately following out-of-sample test
window**. `train_end == test_start`, so no bar is ever both trained on and tested on.

The ratio of out-of-sample to in-sample performance — the **walk-forward efficiency ratio** —
is the real output of the exercise.

#### Modes

| Mode | What it does | Out-of-sample? |
|------|--------------|----------------|
| `walk_forward` (default) | Re-optimises parameters on each training window, evaluates on the next test window | Yes |
| `segmented_in_sample` | Re-runs one fixed parameter set over consecutive slices of the data it was fitted on | **No** |

`segmented_in_sample` is the old behaviour. It validates nothing — every slice is data the
parameters were already fitted on — so it must be requested explicitly, requires
`--params`, logs a warning, and marks every fold `in_sample=True`. Requesting
`walk_forward` without a parameter space raises a `ValueError` pointing at this mode.

#### Implementation Process
1. **Window Generation**: Emits `WalkForwardWindow(train_start, train_end, test_start, test_end)`
2. **Window Stepping**: Advances by step size between folds (default: 3 months)
3. **Per-Fold Optimization**: Runs the configured optimizer on the **training slice only**
   (always `n_jobs=1` so fold-level and optimizer-level pools never nest)
4. **Out-of-Sample Backtest**: Backtests the chosen parameters on the untouched test slice
5. **Efficiency & Aggregation**: Computes the per-fold efficiency ratio and aggregate metrics

#### Example Time Window Pattern
```
Fold 1: train Jan-Dec 2023  ->  test Jan-Jun 2024
Fold 2: train Apr 2023-Mar 2024  ->  test Apr-Sep 2024   (rolling, 3-month step)
Fold 3: train Jul 2023-Jun 2024  ->  test Jul-Dec 2024
```
With `--anchored`, every training window instead starts at the first bar of the dataset.

#### Configuration Parameters
- **Training Window**: 12 months (default) - In-sample data used to fit each fold
- **Test Window**: 6 months (default) - Out-of-sample data used to evaluate each fold
- **Step Size**: 3 months (default) - Time between fold start dates
- **Minimum Data**: 30 bars minimum for both the train and the test slice
- **Memory Management**: Keeps max 1000 folds; when exceeded it retains the **most recent**
  half and logs a warning that the reported aggregates cover only that subset
- **Timeout**: 600 seconds per fold (a large parameter space can exceed it)

#### Generated Metrics

**Performance Metrics:**
- `total_periods`: Number of periods analyzed
- `avg_return`, `median_return`, `std_return`: Return statistics across periods
- `avg_return_pct`, `median_return_pct`, `std_return_pct`: Percentage return statistics
- `avg_sharpe_ratio`, `combined_sharpe_ratio`: Risk-adjusted performance. Both are
  annualised with the factor inferred from the data's own bar frequency (365 for daily
  crypto, 252 for daily equities, 8760 for hourly crypto), so they are directly comparable
- `avg_max_drawdown`, `worst_max_drawdown`, `best_max_drawdown`: Drawdown analysis.
  `max_drawdown` is a **negative** percentage, so `worst_max_drawdown` is the *minimum*
  (deepest) value and `best_max_drawdown` the maximum (shallowest)
- `independent_oos_bars`, `overlapping_oos_bars`, `oos_overlap_pct`: How much of the pooled
  out-of-sample series was repeated across folds. With the defaults
  (`test_window_months=6`, `step_months=3`) consecutive test windows overlap by 50%;
  repeated bars are counted once in `combined_sharpe_ratio`, and a warning is logged
  whenever `step_months < test_window_months`. Set `step_months >= test_window_months` for
  folds that tile without overlap
- `avg_win_rate`, `avg_trades_per_period`: Trading statistics

**Efficiency Metrics (walk_forward mode):**
- `mean_efficiency_ratio`, `median_efficiency_ratio`, `std_efficiency_ratio`: Distribution of
  the per-bar-normalised out-of-sample / in-sample ratio (normalised so unequal window
  lengths stay comparable)
- `worst_efficiency_ratio`, `best_efficiency_ratio`: Range across folds
- `folds_with_efficiency_ratio`: How many folds produced a usable ratio. It is `None` when
  in-sample performance was <= 0, because dividing by a loss turns it into a flattering
  positive number
- `folds_above_half_efficiency_pct`: Share of folds retaining more than half their edge
- `avg_train_return_pct`: Mean in-sample return, for comparison against the OOS mean

**Failure Accounting:**
- `attempted_folds`, `failed_folds`, `failure_rate_pct`: How many folds were attempted and
  how many failed. A failure rate above 5% logs a warning naming survivorship bias, so a
  result aggregated from only the folds that happened to succeed can never look clean

**Consistency Analysis:**
- `positive_return_periods`, `positive_return_pct`: Profitable period count and percentage
- `profitable_periods`, `profitable_periods_pct`: Alternative profitability measures

**Per-Fold Detail:**
`AnalysisResult.metadata['folds']` carries each fold's train/test periods, chosen
parameters, bar counts, in-sample return, out-of-sample return and efficiency ratio.

**Stability Metrics:**
- `return_volatility`, `return_pct_volatility`: Cross-period return variability
- `return_consistency`, `return_pct_consistency`: Performance consistency ratios
- `temporal_stability`: Measures direction changes between consecutive periods
- `rolling_mean_stability`: Stability of rolling performance means (4+ periods)
- `trend_consistency`: Consistency of performance trends (4+ periods)

### Monte Carlo Analysis

#### Purpose and Methodology
Monte Carlo analysis tests **market scenario robustness** using block bootstrap sampling. It runs hundreds/thousands of simulations with different combinations of historical market conditions to assess strategy performance across various scenarios.

#### Implementation Process
1. **Return Bootstrap**: Resamples the **return** series in blocks of consecutive returns
2. **Path Reconstruction**: Compounds the resampled returns from the real starting price to
   build a synthetic close path, then scales each bar's O/H/L around the reconstructed
   close using its source bar's proportions (clamped so `high < low` is impossible)
3. **Fixed Parameters**: Uses same pre-optimized parameters for all simulations
4. **Parallel Simulation**: Runs multiple simulations concurrently
5. **Statistical Analysis**: Comprehensive distribution analysis of results

#### Block Bootstrap Process
Blocks are drawn from the **return** series, not the price series, and are concatenated in
**draw order** — they are deliberately *not* sorted back into chronological order. Sorting
resampled blocks would put them back in (nearly) their original sequence, which is what
made the old implementation a no-op; and gluing raw price *levels* end to end manufactured
enormous artificial gap returns at every block boundary. Every return in a synthetic path
is now a real historical return, and intra-block autocorrelation is preserved.

The result carries a fresh, strictly monotonic, evenly spaced `DatetimeIndex`: apart from
the first bar, no timestamp or close level from the source data survives.

#### Configuration Parameters
- **Simulations**: 1000 (default) - Number of Monte Carlo runs
- **Bootstrap Percentage**: 80% (default) - Fraction of the history length synthesised
- **Block Size**: 30 days (default) - Size of consecutive return blocks. A block size larger
  than half the available history is **clamped** with a warning: at `len(data) - 1` there is
  only one legal start offset, so every simulation would draw the identical path
- **Minimum Sample**: 50 bars minimum per simulation; shorter paths are discarded, logged
  and **counted as failures** rather than silently dropped
- **Memory Management**: When the result cap is exceeded the **most recent** half is kept
  (not the best-performing half, which used to amputate the left tail of the distribution)
  and a warning is logged
- **Timeout**: 30 seconds per simulation
- **Random Seed**: Each simulation derives its own seed as `random_seed + simulation_id`,
  which is passed explicitly into the worker process. Seeded runs are therefore reproducible
  on the parallel path too (on spawn-based platforms such as Windows, workers do not inherit
  the parent's seeded generators), and parallel results match sequential results exactly.
  Results are re-ordered by simulation id before being returned, so completion order cannot
  perturb the output.

#### Failure Accounting
`attempted_runs`, `failed_runs`, `failure_rate`, `successful_runs` and
`is_survivorship_biased` are exposed on `AnalysisResult`, and `failure_rate_pct` is injected
into `combined_metrics` and `analysis_parameters`. A failure rate above 5% logs a warning
naming survivorship bias, so a distribution built only from the simulations that happened to
succeed cannot be read as if it were complete.

#### Generated Metrics

**Performance Distribution:**
- `mean_return`, `median_return`, `std_return`: Return distribution statistics
- `mean_return_pct`, `median_return_pct`, `std_return_pct`: Percentage return statistics
- `mean_sharpe`, `median_sharpe`, `std_sharpe`: Sharpe ratio distribution
- `mean_max_drawdown`, `worst_max_drawdown`, `best_max_drawdown`: Drawdown analysis.
  `max_drawdown` is a **negative** percentage, so `worst_max_drawdown` is the *minimum*
  (deepest) value across simulations - it is the tail this analysis exists to expose
- `mean_win_rate`, `mean_trades_per_simulation`: Trading performance

**Scenario Analysis:**
- `positive_return_simulations`, `positive_return_pct`: Profitable simulation count
- `profitable_simulations`, `profitable_simulations_pct`: Alternative profitability measures
- `total_simulations`: Total number of completed simulations

**Risk Assessment (VaR/CVaR):**
- `return_var_5pct`, `return_var_1pct`: Value at Risk at 5% and 1% levels
- `return_cvar_5pct`, `return_cvar_1pct`: Conditional Value at Risk (Expected Shortfall)

**Distribution Shape Analysis:**
- `return_skewness`, `return_kurtosis`: Return distribution shape
- `return_pct_skewness`, `return_pct_kurtosis`: Percentage return distribution shape
- `sharpe_skewness`, `sharpe_kurtosis`: Sharpe ratio distribution shape

**Confidence Intervals:**
90%, 95%, and 99% confidence intervals for key metrics

**Percentile Analysis:**
5th, 25th, 50th, 75th, and 95th percentiles for:
- `total_return`, `total_return_pct`, `sharpe_ratio`, `max_drawdown`, `win_rate`

## Result Structure and Output

### AnalysisResult Object
Contains comprehensive analysis results with:
- **Core Data**: Analysis type, strategy info, date ranges
- **Individual Results**: List of `BacktestResult` objects from each period/simulation
- **Combined Metrics**: Aggregated performance statistics
- **Stability Metrics**: Statistical distribution and consistency measures
- **Analysis Parameters**: Configuration used for the analysis
- **Metadata**: Analysis-specific additional data

### Key Methods
- `to_dataframe()`: Converts results to pandas DataFrame for further analysis
- `get_summary_statistics()`: Provides mean, std, min, max for all metrics
- `get_performance_consistency()`: Calculates consistency ratios and success rates

### JSON Output Format
When `--output` is specified, results are saved in structured JSON containing:
- Complete analysis metadata and configuration
- All individual backtest results
- Aggregated metrics and statistics
- Percentile breakdowns and distribution analysis

## Parallel Processing

### Performance Optimization
- **Multi-core Processing**: Uses `ProcessPoolExecutor` for parallel execution
- **Automatic CPU Detection**: Defaults to min(CPU_count, 4) processes
- **Memory Management**: Intelligent result trimming to prevent memory overflow
- **Timeout Handling**: Prevents hanging on problematic periods/simulations

### Error Handling
- **Graceful Failures**: Continues analysis even if some periods/simulations fail
- **Success Rate Reporting**: Reports completion statistics
- **Timeout Management**: Handles long-running backtests appropriately

## Integration with Other Components

### Strategy Integration
- Uses `BaseStrategy` interface with pre-optimized parameters
- Currently supports `SimpleMAStrategy` implementation
- Validates strategy parameters before analysis begins

### Optimization Integration
- Compatible with optimization result files as parameter input
- Seamless workflow: optimize → analyze → validate
- JSON parameter files work directly as `--params_file` input

### Backtesting Integration
- Uses same `BacktestEngine` for consistent performance measurement
- Inherits commission rates, initial capital, and trading constraints
- Produces standard `BacktestResult` objects for analysis

## Use Cases and Interpretation

### When to Use Walk-Forward Analysis
- **Out-of-Sample Validation**: Measure how the strategy performs on data the optimizer
  never saw — the only honest answer to "will this work going forward?"
- **Efficiency Assessment**: See how much of the fitted edge survives out of sample; an
  efficiency ratio well below 1 means the optimizer was fitting noise
- **Stability Assessment**: Measure performance consistency across market regimes
- **Parameter Robustness**: Identify parameters that degrade over time

Walk-forward numbers are typically materially worse than a single in-sample backtest. That
is the point: the in-sample figure was never achievable.

### When to Use Monte Carlo Analysis
- **Scenario Testing**: Assess performance across various market conditions
- **Risk Assessment**: Calculate VaR and confidence intervals
- **Distribution Analysis**: Understand full range of possible outcomes
- **Robustness Testing**: Validate strategy across different data combinations

### Critical Questions Answered
Both analysis methods help answer:
- Will my optimized parameters work in the future?
- How robust is my strategy to different market conditions?
- What are realistic risk and return expectations?
- How consistent is strategy performance over time?
- What is the probability of loss over different time horizons?

## Known Limitations

Stated plainly, so results are not over-read:

- **Out-of-sample windows overlap by default.** With `test_window_months=6` and
  `step_months=3`, consecutive test windows share half their bars. Repeated bars are
  counted once when computing `combined_sharpe_ratio`, and `oos_overlap_pct` plus a warning
  report the overlap — but the per-fold counters (`positive_return_pct`,
  `profitable_periods_pct`, `return_consistency`) still treat each fold as one independent
  sample, which they are not. Set `--step` >= `--test_window` for folds that tile cleanly.
  The default was left alone deliberately, because changing it would silently alter every
  existing schedule
- **Walk-forward parallelism uses a fixed 600s per-fold timeout.** A large parameter space
  combined with a slow optimizer can exceed it and be counted as a failed fold
- **Monte Carlo inherits the backtest engine's limitations**: long-only, no slippage, no
  spread. Synthetic paths are resampled real returns, which preserves marginal distribution
  and intra-block autocorrelation but not regime structure, cross-asset relationships, or
  volatility clustering across block boundaries
- **Both analyzers trim over-large result sets** (keeping the most recent half) and log a
  warning saying so. Aggregates then describe the retained subset, not everything that ran
- **Only `simple_ma` is analysable today**, and only its parameter space is registered with
  the optimizer factory