# Niffler

Niffler is a Python-based trading app that helps you sniff out market opportunities — just like a Niffler can’t resist shiny gold!

## Getting Started

### Dependencies

This project uses `uv` for dependency management. To install `uv`, follow the instructions [here](https://github.com/astral-sh/uv).

Once `uv` is installed, navigate to the project root directory and run:

```bash
uv sync
```

This will install all the necessary dependencies.

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

## Project Structure

*   `niffler/`: Core application logic.
*   `scripts/`: Utility scripts, such as `download_data.py`.
*   `data/`: (Ignored by Git) Downloaded market data will be stored here.
*   `tests/`: Unit tests.

