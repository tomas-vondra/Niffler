# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Niffler is a Python-based trading application that helps identify profitable market opportunities. The project includes data acquisition, preprocessing, backtesting, strategy optimization, and advanced robustness analysis components.

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
- Uses built-in unittest framework (no pytest)

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
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --initial_capital 10000 --commission 0.001

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
```

### Strategy Analysis
Advanced robustness testing via `scripts/analyze.py`:

```bash
# Walk-forward analysis with specific parameters
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --params '{"short_window": 10, "long_window": 30}'

# Load parameters from optimization results
python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --params_file optimization_results.json

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
- `niffler/backtesting/` - Strategy backtesting framework
  - `backtest_engine.py` - Core backtesting engine with portfolio management
  - `trade.py` - Trade execution and tracking
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
  - `walk_forward_analyzer.py` - Temporal robustness testing across rolling time windows
  - `monte_carlo_analyzer.py` - Market scenario robustness testing via bootstrap sampling
  - `analysis_result.py` - Unified result container with stability metrics
- `niffler/risk/` - Risk management framework
  - `base_risk_manager.py` - Abstract base class for risk management systems
  - `fixed_risk_manager.py` - Fixed position sizing and stop-loss risk management
  - `kelly_risk_manager.py` - Kelly criterion-based optimal position sizing (pending implementation)
- `niffler/exporters/` - Modular result export system
  - `base_exporter.py` - Abstract base class for result exporters
  - `console_exporter.py` - Human-readable console output
  - `csv_exporter.py` - CSV file export for analysis tools
  - `elasticsearch_exporter.py` - Elasticsearch integration for visualization
  - `exporter_manager.py` - Multi-exporter coordination and registry
- `config/logging.py` - Unified logging configuration
- `config/elasticsearch/mappings/` - Elasticsearch schema definitions
- `scripts/` - Command-line interfaces for core functionality

### Data Storage
- Format: CSV files with standardized columns (timestamp, open, high, low, close, volume)
- Naming: `{SYMBOL}_{SOURCE}_{TIMEFRAME}_{STARTDATE}_{ENDDATE}.csv`
- Location: `data/` directory

## Key Implementation Details

### Data Sources
- **CCXT**: Cryptocurrency exchange data with pagination support
- **Yahoo Finance**: Traditional financial data via yfinance library

### Data Preprocessing Pipeline
- **Infinite Value Removal**: Replaces ±∞ with NaN for calculation safety
- **NaN Handling**: Forward-fill with backward-fill fallback for missing values
- **OHLC Validation**: Ensures High ≥ Low and Open/Close within High/Low range
- **Time Gap Detection**: Identifies missing periods and calculates data completeness
- **Data Quality Checks**: Validates positive prices, non-negative volume, removes duplicates

### Error Handling
- Comprehensive error handling in download_data.py and preprocessor.py
- Proper logging and user feedback
- Graceful handling of network errors and invalid inputs
- Data validation prevents corrupted data from breaking downstream analysis

### Analysis Framework Architecture
The analysis framework provides two main approaches for testing strategy robustness:

#### Walk-Forward Analysis
- **Purpose**: Tests temporal robustness by validating pre-optimized parameters across rolling time windows
- **Process**: Uses fixed parameters obtained from optimization and tests them on sequential out-of-sample periods
- **Key Metrics**: Period-by-period performance, temporal stability, return consistency
- **Use Case**: Validate that optimized parameters work consistently across different time periods

#### Monte Carlo Analysis  
- **Purpose**: Tests market scenario robustness using bootstrap sampling of historical data
- **Process**: Runs hundreds/thousands of simulations with block bootstrap sampling to preserve time series structure
- **Key Metrics**: Return distribution statistics, VaR/CVaR, percentile analysis, skewness/kurtosis
- **Use Case**: Assess strategy performance across various market scenarios and estimate risk metrics

### Risk Management Framework
The risk management system provides position sizing, stop-loss calculation, and portfolio-level risk controls:

#### Fixed Risk Manager
- **Purpose**: Simple, predictable risk management using fixed percentages
- **Position Sizing**: Fixed percentage of portfolio per trade (e.g., 10%)
- **Stop Loss**: Fixed percentage from entry price (e.g., 5%)
- **Portfolio Controls**: Maximum positions, exposure limits, risk per trade caps
- **Use Case**: Conservative risk management with predictable position sizes

#### Kelly Risk Manager (Planned)
- **Purpose**: Optimal position sizing based on historical strategy performance
- **Method**: Uses Kelly Criterion formula: f* = (bp - q) / b
- **Features**: Fractional Kelly, lookback period analysis, volatility-based stops
- **Status**: Framework implemented, core calculations pending integration with backtest engine

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
- **Environment Variables**: Default settings via `.env` file
  - `ELASTICSEARCH_HOST` - Elasticsearch server hostname
  - `ELASTICSEARCH_PORT` - Elasticsearch server port  
  - `ELASTICSEARCH_INDEX_PREFIX` - Index naming prefix
- **Command-line Overrides**: Runtime configuration via `--es-host`, `--es-port`, `--es-index-prefix`
- **Mapping Files**: Elasticsearch schema definitions in `config/elasticsearch/mappings/`

### Testing Approach
- Mock external dependencies (ccxt, yfinance, elasticsearch)
- Test both successful operations and error conditions
- Validate argument parsing and data output formats
- Comprehensive testing: 452 total tests including:
  - 70 unit tests for exporters package (console, CSV, Elasticsearch, manager)
  - 36 tests for analysis framework (Monte Carlo, walk-forward)
  - 18 tests for backtesting engine
  - 48 tests for data management (downloaders, preprocessors)
  - 54 tests for optimization framework
  - 18 tests for risk management
  - 208 tests for scripts and CLI interfaces
- Integration and functional testing to ensure end-to-end workflow reliability
- Isolated testing with proper mocking and teardown procedures