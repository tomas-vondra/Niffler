# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Niffler is a Python-based trading application that helps identify profitable market opportunities. The project is in early development with only the data acquisition component implemented.

## Development Setup

### Package Management
- Uses `uv` for dependency management (modern, fast Python package manager)
- Install dependencies: `uv sync`
- Python version: ≥3.13

### Core Dependencies
- `pandas` (≥2.3.1) for data manipulation
- `ccxt` for cryptocurrency exchange data
- `yfinance` for traditional financial data

## Common Commands

### Testing
- Run all tests: `python -m unittest discover -s tests -p "test_*.py"`
- Run specific test: `python -m unittest tests.test_download_data`
- Run preprocessor tests: `python -m unittest tests.test_preprocessor`
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

## Architecture

### Current Structure
- `scripts/download_data.py` - Market data acquisition from exchanges and APIs
- `scripts/preprocessor.py` - Data cleaning and validation pipeline
- `tests/test_download_data.py` - Test suite for data download functionality
- `tests/test_preprocessor.py` - Comprehensive test suite for data preprocessing (25 test cases)
- `data/` - CSV storage for downloaded market data (gitignored)

### Planned/Empty Structure
- `niffler/bot/` - Trading bot logic (empty)
- `niffler/strategies/` - Trading strategies (empty)
- `niffler/utils/` - Utility functions (empty)
- `config/` - Configuration management (empty)

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

### Testing Approach
- Mock external dependencies (ccxt, yfinance)
- Test both successful operations and error conditions
- Validate argument parsing and data output formats
- Comprehensive preprocessor testing with 25 test cases covering edge cases