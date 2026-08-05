# Data Management

## Data Acquisition

### Download Script Usage

```bash
python scripts/download_data.py --source <source> --symbol <symbol> --timeframe <timeframe> --start_date <YYYY-MM-DD> [--end_date <YYYY-MM-DD>] [--exchange <exchange_id>] [--output <filename>]
```

**Arguments:**
- `--source`: Data source - `ccxt` or `yahoo`
- `--symbol`: Trading pair (e.g., `BTC/USDT` for `ccxt`, `BTC-USD` for `yahoo`)
- `--timeframe`: Time interval (e.g., `1d`, `1h`, `1m`)
- `--start_date`: Start date in `YYYY-MM-DD` format
- `--end_date`: (Optional) End date in `YYYY-MM-DD` format, defaults to today
- `--exchange`: (Required for `ccxt`) Exchange ID, defaults to `binance` if not specified
- `--output`: (Optional) Custom output filename, defaults to auto-generated name

### Exit codes and partial downloads

`download_data.py` **exits non-zero on every failure path**. It previously logged
"No data to save." at INFO and exited 0, which made an empty download look like a success in
any script that checked the exit code.

| Situation | Exit code | Behaviour |
|-----------|-----------|-----------|
| Download succeeded | 0 | File written to the normal path |
| Unsupported timeframe | 1 | `InvalidTimeframeError`; the supported list is printed |
| Venue had no data for the range | 1 | `NoDataAvailableError`; reported as empty, not broken |
| Network / exchange / parse failure | 1 | `DownloadError` |
| **Truncated (partial) download** | 1 | Written to `<name>.partial.csv`, with the reason logged at ERROR |

The partial case matters: a CCXT download that gives up mid-pagination returns the candles
it did fetch, flagged `df.attrs['partial']` and `df.attrs['partial_reason']`. The script
writes that data to a clearly marked path so it is not mistaken for a complete series, and
still exits 1.

### Output Filename Generation

**Default Output Location:** Files are saved in `data/` directory (created automatically if it doesn't exist)

When `--output` is not specified, filenames are automatically generated using the pattern:
```
data/{SYMBOL}_{SOURCE/EXCHANGE}_{TIMEFRAME}_{STARTDATE}_{ENDDATE}.csv
```

Examples:
- `data/BTCUSDT_binance_1d_20240101_20240105.csv`
- `data/BTCUSD_yahoo_1d_20240101_20240331.csv`

### Examples

**Cryptocurrency data from Binance (default exchange):**
```bash
python scripts/download_data.py --source ccxt --symbol BTC/USDT --timeframe 1d --start_date 2024-01-01 --end_date 2024-01-05
```

**Cryptocurrency data from specific exchange:**
```bash
python scripts/download_data.py --source ccxt --symbol BTC/USDT --timeframe 1d --start_date 2024-01-01 --end_date 2024-01-05 --exchange bybit
```

**Traditional financial data:**
```bash
python scripts/download_data.py --source yahoo --symbol BTC-USD --timeframe 1d --start_date 2024-01-01 --end_date 2024-01-05
```

## Data Preprocessing

### Preprocessing Script Usage

```bash
python scripts/preprocessor.py --input <input_path> [--output <output_path>] [--suffix <suffix>]
```

**Arguments:**
- `--input`: Path to CSV file or directory containing CSV files to process
- `--output`: (Optional) Output file or directory path, defaults to input path with suffix applied
- `--suffix`: (Optional) Suffix for output files when processing directories, default: `_cleaned`

**Default Output Path:**
- For single files: Same directory as input file with suffix: `{original_filename}{suffix}.csv`
- For directories: Same directory structure with suffix applied to each file

`preprocessor.py` **exits non-zero** when a file cannot be read, parsed or written. It
previously returned `None` on every error path and therefore exited 0. Unlike the trading
scripts it only *warns* about duplicate timestamps rather than raising, because its job is
to clean arbitrary files rather than to refuse them.

### Preprocessing Pipeline

The preprocessing system applies multiple validators in sequence:

#### 1. Infinite Value Preprocessor
- **Purpose**: Replaces ±∞ values with NaN for safe mathematical calculations
- **Implementation**: Scans all numeric columns and converts infinite values
- **Impact**: Prevents calculation errors in downstream analysis

#### 2. NaN Value Preprocessor  
- **Purpose**: Handles missing data points in time series
- **Implementation**: Explicit **per-column** gap policy:
  - Price columns: forward-fill (configurable)
  - Volume / flow columns: filled with **0**, never a stale value — a bar with no trading
    did not trade at the previous bar's volume
  - Everything else: forward-fill
  - `max_fill_gap` caps how many consecutive bars a forward-fill may bridge
- **Backward fill is off by default.** It is look-ahead (it copies a future value into the
  past), so leading NaN rows are **dropped** instead. Enable it with
  `NanValuePreprocessor(allow_backward_fill=True)` for offline inspection only; it logs a
  LOOK-AHEAD warning when used.
- **Reporting**: Fabricated cells and rows are counted per column and exposed on
  `self.last_stats` and `result.attrs['nan_fill']` (`total_nan`, `filled_forward`,
  `filled_zero`, `filled_backward`, `dropped_rows`, `synthetic_rows`,
  `synthetic_row_ratio`, `per_column`), with a warning naming the fabricated fraction that
  escalates above 5%. `add_synthetic_column=True` adds a per-row `is_synthetic` marker.
- **Impact**: Series with gaps are now shorter and less fabricated than before. Measured
  volatility rises and reported Sharpe ratios fall accordingly — that is the correction.

#### 3. OHLC Validator Preprocessor
- **Purpose**: Validates price relationship consistency
- **Validation Rules**:
  - High ≥ Low (high price must be >= low price)
  - Low ≤ Open ≤ High (open price within daily range)
  - Low ≤ Close ≤ High (close price within daily range)
- **Modes** (`mode=` constructor argument):
  - `drop` (default): removes violating rows, as before
  - `repair`: clamps `high = max(o,h,l,c)` and `low = min(o,h,l,c)`, keeping the sampling
    grid intact
  - `flag`: reports only, changes nothing
- **Reporting**: Per-rule violation counts plus `invalid_rows`, `invalid_ratio`,
  `dropped_rows` and `repaired_rows` on `self.last_stats` and
  `result.attrs['ohlc_validation']`. Any drop or repair logs a warning with the affected
  percentage, and at or above `warn_threshold` (default 1%) an additional
  "HIGH OHLC INVALID RATE" warning fires — a large fraction can never vanish quietly.
- **Impact**: Ensures realistic price data for accurate backtesting, and makes the cost of
  that cleaning visible

#### 4. Time Gap Detector Preprocessor
- **Purpose**: Analyzes data completeness and identifies missing periods
- **Implementation**: 
  - Calculates expected vs actual data points
  - **Logs data completeness percentage only**
  - **Does not remove or modify any data**
- **Impact**: Provides visibility into data quality without altering dataset

#### 5. Data Quality Validator Preprocessor
- **Purpose**: Performs comprehensive data validation
- **Validations**:
  - Ensures positive prices (open, high, low, close > 0)
  - Validates non-negative volume
  - Removes duplicate timestamps
  - Checks for proper data types
- **Implementation**: Removes rows that fail validation criteria
- **Impact**: Ensures clean, consistent data for analysis

### Examples

**Clean single file:**
```bash
python scripts/preprocessor.py --input data/BTCUSDT_binance_1d_20240101_20240105.csv --output data/BTCUSDT_cleaned.csv
```

**Process directory with custom suffix:**
```bash
python scripts/preprocessor.py --input data/ --output cleaned_data/ --suffix _validated
```

## Data Format

### Standard CSV Structure
- **Columns**: timestamp, open, high, low, close, volume
- **Index**: timestamp (datetime index)
- **Storage**: `data/` directory (ignored by Git)

## Data Source Integrations

### CCXT Integration (Cryptocurrency Exchanges)

The CCXT integration provides access to cryptocurrency exchanges through a unified interface:

#### Implementation Details:
1. **Exchange Initialization**: Creates exchange instance with rate limiting enabled
2. **Pagination Handling**: Automatically handles large date ranges by fetching data in chunks (default 1000 candles per request)
3. **Data Filtering**: Ensures returned data is within requested time bounds using timestamp filtering
4. **Data Standardization**: Converts exchange-specific formats to standard OHLCV structure
5. **Timeframe Validation**: Validates requested timeframe against supported intervals

#### Supported Features:
- **Exchanges**: Access to all CCXT-supported exchanges (binance, bybit, coinbase, etc.)
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M (varies by exchange)
- **Automatic Rate Limiting**: Uses CCXT's built-in rate limiting (`enableRateLimit: true`)
- **Symbol Format**: Uses CCXT standard format (e.g., `BTC/USDT`)

#### Resilient Pagination

A single failed page used to abandon the whole download and discard everything already
fetched. Pagination is now bounded and recoverable:

- **Retries with exponential backoff**: `max_retries` (default 3) per page,
  `backoff_base_seconds` (default 1.0) doubling up to `backoff_max_seconds` (default 60)
- **Partial results are kept**: giving up mid-download returns the candles fetched so far as
  a DataFrame flagged `df.attrs['partial'] = True` with a `df.attrs['partial_reason']`,
  rather than throwing the work away
- **Cursor-advance guard**: if a page comes back whose last timestamp does not move the
  cursor forward, the loop stops with a partial result instead of spinning forever. A hard
  `MAX_PAGES = 100_000` cap backs it up
- **Progress logging**: a line every `progress_every` pages (default 10) with the candle
  count and the current cursor timestamp

All four are constructor arguments on `CCXTDownloader`.

#### Step-by-Step Process:
1. **Timeframe Validation**: Check if requested interval is supported
2. **Exchange Setup**: Initialize exchange class with rate limiting enabled
3. **Pagination Loop**: 
   - Fetch data in chunks starting from `start_ms`
   - Filter candles to stay within `end_ms` boundary
   - Continue until no more data or limit reached
4. **Data Assembly**: Combine all chunks into single list
5. **DataFrame Creation**: Convert to pandas DataFrame with datetime index
6. **Final Filtering**: Ensure data is within exact requested time bounds

### Yahoo Finance Integration (Traditional Markets)

The Yahoo Finance integration uses the yfinance library to access financial market data:

#### Price Adjustment Convention (important)

**Niffler stores back-adjusted prices by default.** `auto_adjust=True` means Open/High/Low/
Close are adjusted for splits *and* dividends, and no separate `Adj Close` column is
produced.

This is now passed to `yfinance` **explicitly on every call**, because upstream changed its
own default (raw before 0.2.51, adjusted after). Relying on that default meant the same
download reproduced different numbers depending on which yfinance happened to be installed —
a silent, version-dependent data corruption.

- Configure per instance: `YahooFinanceDownloader(auto_adjust=False)`
- Override per call: `downloader.download(..., auto_adjust=False)`
- The convention actually used is recorded on `result.attrs['auto_adjust']`

If data you downloaded previously came from a pre-0.2.51 yfinance, it was **raw** and its
prices will not match a fresh download.

#### Implementation Details:
1. **Direct API Call**: Uses `yf.download()` with ticker, date range, interval and an
   explicit `auto_adjust`
2. **Column Handling**: `_flatten_columns()` handles MultiIndex responses by locating the
   level that actually holds the OHLCV field names — it does not assume level 0
3. **Data Normalization**: Optionally reorders columns to standard OHLCV format
4. **Index Management**: Ensures datetime index is properly named
5. **Timeframe Validation**: Validates requested interval against supported timeframes

#### Supported Features:
- **Timeframes**: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
- **Symbol Format**: Uses Yahoo Finance format (e.g., `BTC-USD`, `AAPL`)
- **Data Types**: Whatever assets Yahoo Finance supports (stocks, crypto, forex, etc.)
- **Column Normalization**: Optional standardization to Open, High, Low, Close, Volume order

#### MultiIndex Handling

`_flatten_columns()` is defensive about the several shapes yfinance can return:

- keeps only the requested ticker when several come back
- raises `DownloadError` when several tickers come back and none matches
- raises `DownloadError` on an unrecognisable column layout, instead of silently producing
  a frame with the wrong columns
- handles the `('Adj Close', '')` placeholder layout
- de-duplicates surviving columns

### Error Handling

Downloaders **raise typed exceptions rather than returning `None`**. Silently returning
`None` on failure meant every caller had to remember to check, and the ones that forgot
carried on with no data. `BaseDownloader.download` is annotated `-> pd.DataFrame`.

```python
from niffler.data import (
    NifflerDataError,      # base class
    InvalidTimeframeError, # the requested timeframe is not supported
    NoDataAvailableError,  # the request succeeded, the venue had nothing
    DownloadError,         # the download genuinely broke
)
```

`NoDataAvailableError` is deliberately **not** a subclass of `DownloadError`, so callers can
distinguish "nothing to sell you" from "something is wrong" — catch it *before*
`DownloadError`. All four are importable from `niffler.data` or `niffler.data.exceptions`.

`DownloadError` also covers a missing optional client library (`ccxt`, `yfinance`), so an
uninstalled dependency surfaces as a download failure with a clear message rather than an
`ImportError` at an unexpected point.