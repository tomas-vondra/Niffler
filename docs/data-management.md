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

### Preprocessing Pipeline

The preprocessing system applies multiple validators in sequence:

#### 1. Infinite Value Preprocessor
- **Purpose**: Replaces ±∞ values with NaN for safe mathematical calculations
- **Implementation**: Scans all numeric columns and converts infinite values
- **Impact**: Prevents calculation errors in downstream analysis

#### 2. NaN Value Preprocessor  
- **Purpose**: Handles missing data points in time series
- **Implementation**: Forward-fill missing values with backward-fill fallback for leading NaN
- **Impact**: Ensures continuous data series for backtesting

#### 3. OHLC Validator Preprocessor
- **Purpose**: Validates price relationship consistency
- **Validation Rules**:
  - High ≥ Low (high price must be >= low price)
  - Low ≤ Open ≤ High (open price within daily range)
  - Low ≤ Close ≤ High (close price within daily range)
- **Implementation**: **Removes invalid rows** that violate OHLC relationships
- **Impact**: Ensures realistic price data for accurate backtesting

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

#### Implementation Details:
1. **Direct API Call**: Uses `yf.download()` with ticker, date range, and interval parameters
2. **Column Handling**: Manages potential MultiIndex columns from yfinance responses
3. **Data Normalization**: Optionally reorders columns to standard OHLCV format
4. **Index Management**: Ensures datetime index is properly named
5. **Timeframe Validation**: Validates requested interval against supported timeframes

#### Supported Features:
- **Timeframes**: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
- **Symbol Format**: Uses Yahoo Finance format (e.g., `BTC-USD`, `AAPL`)
- **Data Types**: Whatever assets Yahoo Finance supports (stocks, crypto, forex, etc.)
- **Column Normalization**: Optional standardization to Open, High, Low, Close, Volume order

#### Step-by-Step Process:
1. **Timeframe Validation**: Check if requested interval is in supported list
2. **Data Fetch**: Call `yf.download()` with parameters
3. **MultiIndex Handling**: Flatten column structure if MultiIndex is returned
4. **Column Processing**: Remove 'Adj Close' if present, reorder columns if normalization enabled
5. **Index Naming**: Ensure index is named 'Date'
6. **Empty Check**: Verify DataFrame contains data before returning

### Error Handling

Both integrations include basic error handling:
- **CCXT**: Catches exceptions during exchange operations and data fetching
- **Yahoo Finance**: Catches exceptions during yfinance download calls
- **Logging**: Both provide informational and error logging for debugging