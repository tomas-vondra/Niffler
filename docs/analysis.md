# Strategy Analysis

## Analysis Script Usage

```bash
python scripts/analyze.py --data <data_file> --analysis <analysis_type> --strategy <strategy_name> (--params <parameters_json> | --params_file <params_file>) [additional_options]
```

**Required Arguments:**
- `--data`: Path to CSV file containing historical market data
- `--analysis`: Analysis type (`walk_forward` or `monte_carlo`)
- `--strategy`: Strategy to analyze (currently supports `simple_ma`)
- `--params`: Strategy parameters as JSON string (e.g., `'{"short_window": 10, "long_window": 30}'`)
- `--params_file`: Path to JSON file containing parameters (can use optimization results file)

**Optional Arguments:**
- `--initial_capital`: Starting capital, default: 10000
- `--commission`: Commission rate per trade, default: 0.001
- `--symbol`: Symbol identifier, default: "UNKNOWN"
- `--n_jobs`: Number of parallel jobs, default: auto-detect (max 4)
- `--output`: Save detailed results to JSON file
- `--verbose`: Enable debug logging

**Walk-Forward Specific Options:**
- `--test_window`: Test window size in months, default: 6
- `--step`: Step size in months between windows, default: 3

**Monte Carlo Specific Options:**
- `--simulations`: Number of simulations, default: 1000
- `--bootstrap_pct`: Percentage of data to sample, default: 0.8 (80%)
- `--block_size`: Block size in days for bootstrap sampling, default: 30
- `--random_seed`: Random seed for reproducible results

## Examples

**Walk-forward analysis with specific parameters:**
```bash
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --params '{"short_window": 10, "long_window": 30}'
```

**Load parameters from optimization results:**
```bash
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --params_file optimization_results.json
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
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --params '{"short_window": 15, "long_window": 25}' --test_window 6 --step 3
```

## Analysis Framework

### Walk-Forward Analysis

#### Purpose and Methodology
Walk-forward analysis tests **temporal robustness** by validating pre-optimized parameters across rolling time windows. Unlike rolling optimization, this method uses **fixed parameters** obtained from prior optimization and tests their consistency over different time periods.

#### Implementation Process
1. **Period Generation**: Creates rolling time windows of specified size (default: 6 months)
2. **Window Stepping**: Advances by step size between periods (default: 3 months)
3. **Fixed Parameter Testing**: Uses same parameters for all time periods
4. **Sequential Backtesting**: Runs independent backtests on each time window
5. **Result Aggregation**: Combines results to measure temporal stability

#### Example Time Window Pattern
```
Period 1: Jan-Jun 2024 (6 months)
Period 2: Apr-Sep 2024 (starts 3 months later)
Period 3: Jul-Dec 2024
Period 4: Oct-Mar 2025
```

#### Configuration Parameters
- **Test Window**: 6 months (default) - Size of each analysis period
- **Step Size**: 3 months (default) - Time between period start dates  
- **Minimum Data**: 30 days minimum per period
- **Memory Management**: Keeps max 1000 results, trims to best 500 when exceeded
- **Timeout**: 60 seconds per period analysis

#### Generated Metrics

**Performance Metrics:**
- `total_periods`: Number of periods analyzed
- `avg_return`, `median_return`, `std_return`: Return statistics across periods
- `avg_return_pct`, `median_return_pct`, `std_return_pct`: Percentage return statistics
- `avg_sharpe_ratio`, `combined_sharpe_ratio`: Risk-adjusted performance
- `avg_max_drawdown`, `worst_max_drawdown`, `best_max_drawdown`: Drawdown analysis
- `avg_win_rate`, `avg_trades_per_period`: Trading statistics

**Consistency Analysis:**
- `positive_return_periods`, `positive_return_pct`: Profitable period count and percentage
- `profitable_periods`, `profitable_periods_pct`: Alternative profitability measures

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
1. **Block Bootstrap Sampling**: Samples 80% of data using 30-day consecutive blocks
2. **Time Series Preservation**: Maintains temporal structure within blocks
3. **Fixed Parameters**: Uses same pre-optimized parameters for all simulations
4. **Parallel Simulation**: Runs multiple simulations concurrently
5. **Statistical Analysis**: Comprehensive distribution analysis of results

#### Block Bootstrap Process
For 1000 days of data with 80% sampling and 30-day blocks:
- Target sample: ~800 days of data
- Required blocks: ~27 blocks (800/30)
- Block selection: Each block contains 30 consecutive days from random start position
- Block assembly: Blocks concatenated and sorted chronologically

#### Configuration Parameters
- **Simulations**: 1000 (default) - Number of Monte Carlo runs
- **Bootstrap Percentage**: 80% (default) - Fraction of data sampled per simulation
- **Block Size**: 30 days (default) - Size of consecutive data blocks
- **Minimum Sample**: 50 days minimum per simulation
- **Memory Management**: Keeps max 10,000 results
- **Timeout**: 30 seconds per simulation
- **Random Seed**: Optional for reproducible results

#### Generated Metrics

**Performance Distribution:**
- `mean_return`, `median_return`, `std_return`: Return distribution statistics
- `mean_return_pct`, `median_return_pct`, `std_return_pct`: Percentage return statistics
- `mean_sharpe`, `median_sharpe`, `std_sharpe`: Sharpe ratio distribution
- `mean_max_drawdown`, `worst_max_drawdown`, `best_max_drawdown`: Drawdown analysis
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
- **Temporal Validation**: Test if optimized parameters work consistently over time
- **Out-of-Sample Testing**: Validate parameters beyond optimization period
- **Stability Assessment**: Measure performance consistency across market regimes
- **Parameter Robustness**: Identify parameters that degrade over time

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