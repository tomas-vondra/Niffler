
import argparse
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import ccxt
import yfinance as yf


ccxt_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
yahoo_timeframes = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

def download_ccxt_data(exchange_id, symbol, timeframe, start_ms, end_ms, limit=1000):
    """Downloads historical data using ccxt within a specified date range."""

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        
        all_ohlcv = []
        current_since = start_ms

        logging.info(f"Fetching {symbol} {timeframe} data from {exchange_id} from {pd.to_datetime(start_ms, unit='ms')} to {pd.to_datetime(end_ms, unit='ms')}...")

        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit)
            if not ohlcv:
                break
            
            # Filter out data beyond the end_ms
            filtered_ohlcv = [candle for candle in ohlcv if candle[0] <= end_ms]
            all_ohlcv.extend(filtered_ohlcv)

            # If the last fetched candle is already past the end_ms, or if we got less than 'limit' candles, we are done
            if ohlcv[-1][0] >= end_ms or len(ohlcv) < limit:
                break
            
            current_since = ohlcv[-1][0] + 1  # Move to the next candle after the last one fetched

        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            # Ensure data is within the requested range (inclusive of start_ms and end_ms)
            df = df[(df.index.astype(int) // 10**6 >= start_ms) & (df.index.astype(int) // 10**6 <= end_ms)]
            logging.info(f"Successfully fetched {len(df)} candles.")
            return df
        else:
            logging.info("No data fetched.")
            return None
    except Exception as e:
        logging.error(f"Error downloading data from {exchange_id}: {e}")
        return None

def download_yfinance_data(ticker, start_date, end_date, interval):
    """Downloads historical data using yfinance."""

    try:
        logging.info(f"Fetching {ticker} data from Yahoo Finance...")
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if not df.empty:
            logging.info(f"Successfully fetched {len(df)} candles.")
            return df
        else:
            logging.info("No data fetched.")
            return None
    except Exception as e:
        logging.error(f"Error downloading data from Yahoo Finance for {ticker}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download historical market data.')
    parser.add_argument('--source', type=str, required=True, choices=['ccxt', 'yahoo'],
                        help='Data source: "ccxt" for crypto exchanges or "yahoo" for Yahoo Finance.')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Trading pair (e.g., BTC/USDT for ccxt, BTC-USD for yahoo).')
    parser.add_argument('--timeframe', type=str, default='1d',
                        help='Timeframe/interval (e.g., 1m, 5m, 1h, 1d for ccxt; 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo for yahoo).')
    parser.add_argument('--start_date', type=str, required=True,
                        help='Start date for data download (YYYY-MM-DD). Required for ccxt and yahoo.')
    parser.add_argument('--end_date', type=str,
                        help='End date for data download (YYYY-MM-DD). Optional. Defaults to todays date if not provided.')
    parser.add_argument('--exchange', type=str, default='binance',
                        help='Exchange ID (e.g., binance, bybit). Default is "binance". Only required for ccxt source.')
    parser.add_argument('--output', type=str, default='',
                        help='Output CSV file name. Will be saved in the data/ directory. Default is generated based on symbol, source, timeframe, and dates.')

    args = parser.parse_args()

    # Set default end_date if not provided
    if not args.end_date:
        args.end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    output_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(output_dir, exist_ok=True)
    if not args.output:
        # Generate default output filename
        symbol_clean = args.symbol.replace('/', '').replace('-', '')
        if args.source == 'ccxt':
            filename = f"{symbol_clean}_{args.exchange}_{args.timeframe}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.csv"
        else: # yahoo
            filename = f"{symbol_clean}_{args.source}_{args.timeframe}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.csv"
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = os.path.join(output_dir, args.output)

    df = None

    try:
        start_date_ts = pd.Timestamp(args.start_date)
        end_date_ts = pd.Timestamp(args.end_date)
        if start_date_ts > end_date_ts:
            logging.error("start_date cannot be after end_date.")
            return
    except ValueError:
        logging.error("Invalid start_date or end_date format. Use YYYY-MM-DD.")
        return

    # Timeframe validation
    if args.source == 'ccxt':
        if args.timeframe not in ccxt_timeframes:
            logging.error(f"Invalid timeframe '{args.timeframe}' for ccxt. Supported timeframes are: {', '.join(ccxt_timeframes)}")
            return
        if not args.exchange:
            logging.error("--exchange is required for ccxt source.")
            return
        
        start_ms = int(start_date_ts.timestamp() * 1000)
        end_ms = int(end_date_ts.timestamp() * 1000)
        
        df = download_ccxt_data(args.exchange, args.symbol, args.timeframe, start_ms, end_ms)
    elif args.source == 'yahoo':
        if args.timeframe not in yahoo_timeframes:
            logging.error(f"Invalid timeframe '{args.timeframe}' for yahoo. Supported timeframes are: {', '.join(yahoo_timeframes)}")
            return
        df = download_yfinance_data(args.symbol, args.start_date, args.end_date, args.timeframe)
        if df is not None and not df.empty:
            # Ensure the index is named 'Date'
            if df.index.name != 'Date':
                df.index.name = 'Date'

            # yfinance can sometimes return a MultiIndex for columns, flatten it
            if isinstance(df.columns, pd.MultiIndex):
                # If there's an 'Adj Close' column at the second level, drop it
                if ('Adj Close', '') in df.columns:
                    df = df.drop(columns=[('Adj Close', '')])
                # Flatten the MultiIndex columns, taking the first level
                df.columns = [col[0] for col in df.columns.values]

            # Select and reorder columns as desired: Open, High, Low, Close, Volume
            desired_order = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Filter desired_order to only include columns actually present in df
            final_columns = [col for col in desired_order if col in df.columns]
            df = df[final_columns]

    if df is not None and not df.empty:
        df.to_csv(output_path)
        logging.info(f"Data successfully saved to {output_path}")
    else:
        logging.info("No data to save.")

if __name__ == '__main__':
    main()
