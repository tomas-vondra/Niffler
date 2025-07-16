
import argparse
import pandas as pd
import os
import logging
import sys
from pathlib import Path

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from niffler.data import CCXTDownloader, YahooFinanceDownloader
from config.logging import setup_logging

# Configure logging
setup_logging(level="INFO")


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

    # Initialize downloader and validate timeframe
    if args.source == 'ccxt':
        if not args.exchange:
            logging.error("--exchange is required for ccxt source.")
            return
            
        downloader = CCXTDownloader()
        if not downloader.validate_timeframe(args.timeframe):
            supported = downloader.get_supported_timeframes()
            logging.error(f"Invalid timeframe '{args.timeframe}' for ccxt. Supported timeframes are: {', '.join(supported)}")
            return
        
        start_ms = int(start_date_ts.timestamp() * 1000)
        end_ms = int(end_date_ts.timestamp() * 1000)
        
        df = downloader.download(args.exchange, args.symbol, args.timeframe, start_ms, end_ms)
        
    elif args.source == 'yahoo':
        downloader = YahooFinanceDownloader()
        if not downloader.validate_timeframe(args.timeframe):
            supported = downloader.get_supported_timeframes()
            logging.error(f"Invalid timeframe '{args.timeframe}' for yahoo. Supported timeframes are: {', '.join(supported)}")
            return
            
        df = downloader.download(args.symbol, args.start_date, args.end_date, args.timeframe)
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
