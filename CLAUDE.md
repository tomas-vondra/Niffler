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
- Uses built-in unittest framework (no pytest)

### Data Download
Main functionality via `scripts/download_data.py`:

```bash
# Cryptocurrency data from Binance
python scripts/download_data.py --source ccxt --symbol BTC/USDT --timeframe 1d --start_date 2024-01-01 --end_date 2024-01-05 --exchange binance

# Traditional financial data from Yahoo Finance
python scripts/download_data.py --source yahoo --symbol BTC-USD --timeframe 1d --start_date 2024-01-01 --end_date 2024-01-05
```

## Architecture

### Current Structure
- `scripts/download_data.py` - Main functional component for market data acquisition
- `tests/test_download_data.py` - Comprehensive test suite
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

### Error Handling
- Comprehensive error handling in download_data.py
- Proper logging and user feedback
- Graceful handling of network errors and invalid inputs

### Testing Approach
- Mock external dependencies (ccxt, yfinance)
- Test both successful operations and error conditions
- Validate argument parsing and data output formats