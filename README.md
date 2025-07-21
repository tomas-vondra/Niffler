# Niffler

Niffler is a Python-based trading application that helps you sniff out market opportunities — just like a Niffler can't resist shiny gold! 

## Features

- **Data Acquisition**: Download historical market data from cryptocurrency exchanges (via CCXT) and traditional financial markets (via yfinance)
- **Data Preprocessing**: Clean and validate trading data with comprehensive quality checks
- **Strategy Framework**: Implement and test custom trading strategies
- **Backtesting Engine**: Test strategies against historical data with realistic trading simulation
- **Risk Management**: Position sizing, stop-loss management, and portfolio risk controls
- **Strategy Optimization**: Find optimal strategy parameters using grid search and random search methods
- **Advanced Analysis**: Validate strategy robustness with Walk-forward analysis and Monte Carlo simulation
- **Comprehensive Testing**: Full test suite with 85+ unit tests covering all components

## Getting Started

### Dependencies

This project uses `uv` for dependency management. To install `uv`, follow the instructions [here](https://github.com/astral-sh/uv).

Once `uv` is installed, navigate to the project root directory and run:

```bash
uv sync
```

This will install all the necessary dependencies.

### Running Tests

Run the full test suite to ensure everything is working correctly:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Downloading Market Data

The `download_data.py` script is used to fetch historical market data. You can use either `ccxt` for cryptocurrency exchanges or `yfinance` for traditional financial data (e.g., stocks, forex).

#### Usage:

```bash
python scripts/download_data.py --source <source> --symbol <symbol> --timeframe <timeframe> --start_date <YYYY-MM-DD> [--end_date <YYYY-MM-DD>] [--exchange <exchange_id>] [--output <output_filename>]
```

**Arguments:**

*   `--source`: `ccxt` or `yahoo`.
*   `--symbol`: Trading pair (e.g., `BTC/USDT` for `ccxt`, `BTC-USD` for `yahoo`).
*   `--timeframe`: Interval (e.g., `1d`, `1h`, `1m`). Refer to the script for supported timeframes for each source.
*   `--start_date`: Start date in `YYYY-MM-DD` format.
*   `--end_date`: (Optional) End date in `YYYY-MM-DD` format. Defaults to today's date.
*   `--exchange`: (Required for `ccxt` source) Exchange ID (e.g., `binance`, `bybit`).
*   `--output`: (Optional) Output CSV file name. Defaults to a generated filename in the `data/` directory.

#### Examples:

**Download BTC/USDT 1-day data from Binance using ccxt:**

```bash
python scripts/download_data.py --source ccxt --symbol BTC/USDT --timeframe 1d --start_date 2024-01-01 --end_date 2024-01-05 --exchange binance
```

**Download BTC-USD 1-day data from Yahoo Finance using yfinance:**

```bash
python scripts/download_data.py --source yahoo --symbol BTC-USD --timeframe 1d --start_date 2024-01-01 --end_date 2024-01-05
```

### Data Preprocessing

The `preprocessor.py` script cleans and validates trading data for downstream analysis. It handles common data quality issues found in real market data.

#### Usage:

```bash
python scripts/preprocessor.py --input <input_file_or_directory> [--output <output_file_or_directory>] [--suffix <suffix>]
```

**Arguments:**

*   `--input`: Path to CSV file or directory containing CSV files to process.
*   `--output`: (Optional) Output file or directory path. Defaults to adding suffix to input filename.
*   `--suffix`: (Optional) Suffix for output files when processing directories. Default: `_cleaned`.

#### Features:

*   **Infinite Value Removal**: Detects and replaces ±∞ values with NaN
*   **NaN Handling**: Forward-fills missing values, with backward-fill fallback for leading NaN
*   **OHLC Validation**: Ensures High ≥ Low and Open/Close within High/Low range
*   **Time Gap Detection**: Identifies missing time periods and reports data completeness
*   **Data Quality Checks**: Validates positive prices, non-negative volume, removes duplicates

#### Examples:

**Clean a single file:**

```bash
python scripts/preprocessor.py --input data/BTCUSDT_binance_1d_20240101_20240105.csv --output data/BTCUSDT_cleaned.csv
```

**Process all CSV files in a directory:**

```bash
python scripts/preprocessor.py --input data/ --output cleaned_data/
```

### Strategy Backtesting

The `backtest.py` script allows you to test trading strategies against historical data.

#### Usage:

```bash
python scripts/backtest.py --data <data_file> --strategy <strategy_name> [--initial_capital <amount>] [--commission <rate>] [--clean]
```

**Arguments:**

*   `--data`: Path to CSV file containing historical market data
*   `--strategy`: Strategy to use (currently supports `simple_ma`)
*   `--initial_capital`: (Optional) Starting capital amount. Default: 10000
*   `--commission`: (Optional) Commission rate per trade. Default: 0.001 (0.1%)
*   `--clean`: (Optional) Apply data cleaning pipeline before backtesting

#### Examples:

**Run backtest with Simple Moving Average strategy:**

```bash
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --initial_capital 10000 --commission 0.001
```

**Run backtest with automatic data cleaning:**

```bash
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --clean
```

### Strategy Optimization

The `optimize.py` script helps you find optimal parameters for your trading strategies using various optimization methods.

#### Usage:

```bash
python scripts/optimize.py --data <data_file> --strategy <strategy_name> --method <optimization_method> [--trials <number>] [--sort-by <metric>] [--output <output_file>] [--clean]
```

**Arguments:**

*   `--data`: Path to CSV file containing historical market data
*   `--strategy`: Strategy to optimize (currently supports `simple_ma`)
*   `--method`: Optimization method (`grid` for grid search, `random` for random search)
*   `--trials`: (Optional) Number of trials for random search. Default: 50
*   `--sort-by`: (Optional) Metric to sort results by (`total_return`, `sharpe_ratio`, `max_drawdown`, etc.)
*   `--output`: (Optional) Output JSON file for results. Default: auto-generated filename
*   `--clean`: (Optional) Apply data cleaning pipeline before optimization

#### Examples:

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

### Strategy Analysis

The `analyze.py` script provides advanced robustness testing for trading strategies using two sophisticated methods: Walk-forward analysis and Monte Carlo simulation. This is typically used after optimization to validate that optimized parameters are robust across different time periods and market conditions.

#### Usage:

```bash
python scripts/analyze.py --data <data_file> --analysis <analysis_type> --strategy <strategy_name> (--params <parameters_json> | --params_file <params_file>) [additional_options]
```

**Arguments:**

*   `--data`: Path to CSV file containing historical market data
*   `--analysis`: Analysis type (`walk_forward` or `monte_carlo`)
*   `--strategy`: Strategy to analyze (currently supports `simple_ma`)
*   `--params`: Strategy parameters as JSON string (e.g., `'{"short_window": 10, "long_window": 30}'`)
*   `--params_file`: Path to JSON file containing parameters (can use optimization results file)
*   `--initial_capital`: (Optional) Starting capital. Default: 10000
*   `--commission`: (Optional) Commission rate per trade. Default: 0.001
*   `--output`: (Optional) Save detailed results to JSON file
*   `--n_jobs`: (Optional) Number of parallel jobs for analysis
*   `--verbose`: (Optional) Enable detailed logging

**Walk-forward Analysis Options:**
*   `--test_window`: Test window size in months. Default: 6
*   `--step`: Step size in months between test windows. Default: 3

**Monte Carlo Analysis Options:**
*   `--simulations`: Number of Monte Carlo simulations. Default: 1000
*   `--bootstrap_pct`: Percentage of data to sample in each simulation. Default: 0.8
*   `--block_size`: Block size in days for bootstrap sampling. Default: 30
*   `--random_seed`: Random seed for reproducible results

#### Analysis Methods:

**Walk-Forward Analysis:**
- Tests temporal robustness by validating pre-optimized parameters across rolling time windows
- Uses fixed parameters and tests them on sequential out-of-sample periods
- Provides period-by-period performance analysis and stability metrics
- Ideal for validating that optimized parameters work consistently over time

**Monte Carlo Analysis:**
- Tests market scenario robustness using bootstrap sampling of historical data
- Runs hundreds/thousands of simulations with block bootstrap sampling to preserve time series structure
- Provides return distribution statistics, VaR/CVaR analysis, and percentile breakdowns
- Ideal for assessing strategy performance across various market scenarios and estimating risk

#### Examples:

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

## Project Structure

*   `niffler/`: Core application logic
    *   `data/`: Data acquisition and preprocessing modules
    *   `strategies/`: Trading strategy implementations
    *   `backtesting/`: Backtesting engine and related components
    *   `risk/`: Risk management systems and position sizing
    *   `optimization/`: Parameter optimization framework
    *   `analysis/`: Advanced strategy validation and robustness testing
*   `scripts/`: Command-line interfaces for core functionality
*   `config/`: Configuration and logging setup
*   `data/`: (Ignored by Git) Downloaded market data storage
*   `tests/`: Comprehensive unit test suite with 85+ tests

## Architecture

Niffler follows a modular architecture:

- **Data Layer**: Handles data acquisition from multiple sources and preprocessing
- **Strategy Layer**: Abstract base classes and concrete strategy implementations  
- **Backtesting Layer**: Portfolio management, trade execution, and performance analysis
- **Risk Management Layer**: Position sizing, stop-loss calculation, and portfolio risk controls
- **Optimization Layer**: Parameter optimization using grid search and random search methods
- **Analysis Layer**: Advanced robustness testing with Walk-forward analysis and Monte Carlo simulation
- **Scripts Layer**: Command-line tools for easy interaction with core functionality

## Workflow

The typical workflow for using Niffler follows these steps:

1. **Data Acquisition**: Download historical market data using `download_data.py`
2. **Data Preprocessing**: Clean and validate the data using `preprocessor.py`
3. **Strategy Development**: Implement or customize trading strategies
4. **Backtesting**: Test strategies against historical data using `backtest.py`
5. **Optimization**: Find optimal parameters using `optimize.py`  
6. **Analysis**: Validate robustness using `analyze.py` with walk-forward or Monte Carlo methods
7. **Deployment**: Use validated strategies with confidence in their robustness

## Key Features of the Analysis Framework

### Walk-Forward Analysis
- **Temporal Validation**: Tests strategy performance across different time periods
- **Rolling Windows**: Uses configurable test windows that roll forward through historical data
- **Stability Metrics**: Calculates temporal stability, return consistency, and trend analysis
- **Out-of-Sample Testing**: Ensures parameters work beyond the optimization period

### Monte Carlo Analysis  
- **Scenario Testing**: Simulates thousands of different market scenarios
- **Block Bootstrap**: Preserves time series structure while creating synthetic data samples
- **Risk Assessment**: Provides comprehensive risk metrics including VaR and CVaR
- **Distribution Analysis**: Full statistical analysis of returns including skewness and kurtosis
- **Parallel Processing**: Efficient multi-core processing for large simulation batches

Both analysis methods help answer critical questions:
- Will my optimized parameters work in the future?
- How robust is my strategy to different market conditions?
- What are the realistic risk and return expectations?
- How consistent is the strategy's performance over time?

## Risk Management

Niffler includes a comprehensive risk management framework designed to control position sizing, manage stop losses, and enforce portfolio-level risk limits. The system is built around an abstract base class that allows for different risk management strategies.

### Available Risk Managers

#### Fixed Risk Manager
The `FixedRiskManager` provides simple, predictable risk management using fixed percentages:

**Features:**
- **Fixed Position Sizing**: Uses a constant percentage of portfolio for each trade (e.g., 10%)
- **Fixed Stop Loss**: Sets stop loss at a fixed percentage from entry price (e.g., 5%)
- **Portfolio Controls**: Enforces maximum number of concurrent positions
- **Risk Per Trade Limits**: Caps maximum risk per individual trade
- **Exposure Limits**: Controls total portfolio exposure across all positions

**Use Case:** Conservative risk management where position sizes are predictable and consistent across all trades.

**Example Configuration:**
```python
from niffler.risk.fixed_risk_manager import FixedRiskManager

risk_manager = FixedRiskManager(
    position_size_pct=0.1,      # 10% of portfolio per trade
    stop_loss_pct=0.05,         # 5% stop loss from entry
    max_positions=5,            # Maximum 5 concurrent positions
    max_risk_per_trade=0.02     # Maximum 2% portfolio risk per trade
)
```

#### Kelly Risk Manager (Planned)
The `KellyRiskManager` implements optimal position sizing based on the Kelly Criterion:

**Planned Features:**
- **Kelly Criterion Calculation**: Uses historical win/loss ratios for optimal position sizing
- **Fractional Kelly**: Supports conservative fractions of full Kelly (e.g., quarter-Kelly, half-Kelly)
- **Dynamic Stop Losses**: ATR-based stop losses that adapt to market volatility
- **Historical Analysis**: Analyzes recent trade performance to calculate optimal position sizes
- **Risk-Adjusted Sizing**: Automatically adjusts position sizes based on strategy performance

**Status:** Framework implemented, core calculations pending integration with backtest engine trade history.

### Risk Management Features

#### Position Tracking
- **Real-time Monitoring**: Tracks all open positions with entry prices, sizes, and stop losses
- **Portfolio State**: Maintains comprehensive portfolio state across all positions
- **Position Updates**: Automatically updates position information as trades are executed

#### Portfolio Risk Controls
- **Total Exposure Limits**: Controls maximum total exposure across all positions
- **Position Count Limits**: Enforces maximum number of concurrent positions
- **Individual Position Limits**: Caps maximum size of any single position
- **Risk Per Trade Controls**: Limits maximum risk on any individual trade

#### Stop Loss Management
- **Automated Calculation**: Automatically calculates stop loss prices based on chosen method
- **Stop Loss Monitoring**: Continuously monitors positions against stop loss levels
- **Position Closing Logic**: Determines when positions should be closed due to risk management rules

#### Risk Metrics and Reporting
- **Portfolio Summary**: Provides comprehensive overview of current portfolio state
- **Risk Exposure Analysis**: Calculates current and maximum potential risk exposure
- **Position Utilization**: Tracks how much of available position capacity is being used
- **Risk Efficiency Metrics**: Measures how effectively risk capacity is being utilized

### Integration with Backtesting

The risk management system integrates seamlessly with the backtesting engine:

1. **Trade Evaluation**: Before executing any trade, the backtest engine consults the risk manager
2. **Position Sizing**: Risk manager calculates appropriate position size based on current portfolio state
3. **Stop Loss Setting**: Risk manager determines stop loss price for new positions
4. **Position Monitoring**: Throughout the backtest, risk manager monitors positions for stop loss triggers
5. **Risk Reporting**: Post-backtest analysis includes comprehensive risk management metrics

### Risk Management Workflow

1. **Signal Generation**: Strategy generates buy/sell signals
2. **Risk Evaluation**: Risk manager evaluates trade against current portfolio state
3. **Position Sizing**: Risk manager calculates appropriate position size
4. **Stop Loss Calculation**: Risk manager determines stop loss price
5. **Portfolio Check**: Risk manager validates trade against portfolio limits
6. **Trade Execution**: If approved, trade is executed with risk management parameters
7. **Position Tracking**: Risk manager updates position tracking and portfolio state
8. **Ongoing Monitoring**: Risk manager continuously monitors positions for exit signals

This comprehensive risk management approach ensures that trading strategies operate within defined risk parameters, helping to protect capital and manage portfolio risk effectively.

