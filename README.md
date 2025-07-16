# Niffler

Niffler is a Python-based trading application that helps you sniff out market opportunities — just like a Niffler can't resist shiny gold! 

## Features

- **Data Acquisition**: Download historical market data from cryptocurrency exchanges (via CCXT) and traditional financial markets (via yfinance)
- **Data Preprocessing**: Clean and validate trading data with comprehensive quality checks
- **Strategy Framework**: Implement and test custom trading strategies
- **Backtesting Engine**: Test strategies against historical data with realistic trading simulation
- **Comprehensive Testing**: Full test suite covering all components

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

## Project Structure

*   `niffler/`: Core application logic
    *   `data/`: Data acquisition and preprocessing modules
    *   `strategies/`: Trading strategy implementations
    *   `backtesting/`: Backtesting engine and related components
*   `scripts/`: Command-line interfaces for core functionality
*   `config/`: Configuration and logging setup
*   `data/`: (Ignored by Git) Downloaded market data storage
*   `tests/`: Comprehensive unit test suite

## Architecture

Niffler follows a modular architecture:

- **Data Layer**: Handles data acquisition from multiple sources and preprocessing
- **Strategy Layer**: Abstract base classes and concrete strategy implementations
- **Backtesting Layer**: Portfolio management, trade execution, and performance analysis
- **Scripts Layer**: Command-line tools for easy interaction with core functionality

