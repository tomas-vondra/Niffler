
import argparse
import pandas as pd
import os
import logging
import sys
from pathlib import Path
from typing import Optional

# Running "python scripts/download_data.py" puts scripts/ on sys.path but not
# the repository root, so the root has to be added for "import niffler" to work.
# When imported as scripts.download_data the root is already importable.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niffler.data import CCXTDownloader, YahooFinanceDownloader
from niffler.data.exceptions import (
    DownloadError,
    InvalidTimeframeError,
    NoDataAvailableError,
)
from niffler.config.logging import setup_logging


def _download(description: str, download_callable, *args, **kwargs) -> Optional[pd.DataFrame]:
    """Call a downloader and turn any failure into a user-facing error.

    The typed exceptions from :mod:`niffler.data.exceptions` are reported
    distinctly so the user can tell an unsupported timeframe from an empty venue
    response from a genuinely broken download. ``NoDataAvailableError`` is caught
    before ``DownloadError`` because it is deliberately not a subclass of it.

    Args:
        description: Human readable description of what is being downloaded.
        download_callable: The downloader's download method.
        *args: Positional arguments forwarded to the downloader.
        **kwargs: Keyword arguments forwarded to the downloader.

    Returns:
        The downloaded DataFrame, or None if the download failed or was empty.
    """
    try:
        return download_callable(*args, **kwargs)
    except InvalidTimeframeError as e:
        logging.error(f"{e}")
        logging.error(f"Supported timeframes: {', '.join(e.supported)}")
        return None
    except NoDataAvailableError as e:
        logging.error(f"No data available for {description}: {e}")
        return None
    except DownloadError as e:
        logging.error(f"Failed to download {description}: {e}")
        return None
    except Exception as e:
        # Unexpected provider failures: report them cleanly instead of dumping a
        # traceback on the user.
        logging.error(f"Unexpected error downloading {description}: {e}")
        return None


def main() -> int:
    """Download historical market data and save it as CSV.

    A download the downloader flagged as truncated (``df.attrs['partial']``) is
    written to a ``*.partial.csv`` file and reported as a failure, so a silently
    incomplete price series can never be mistaken for a complete one.

    Returns:
        Process exit code: 0 on success, 1 on failure or partial download.
    """
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

    # Logging options
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set logging level (default: INFO)')

    args = parser.parse_args()

    # Configure logging
    setup_logging(level=args.log_level)

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
            return 1
    except ValueError:
        logging.error("Invalid start_date or end_date format. Use YYYY-MM-DD.")
        return 1

    # Initialize downloader and validate timeframe
    if args.source == 'ccxt':
        if not args.exchange:
            logging.error("--exchange is required for ccxt source.")
            return 1

        downloader = CCXTDownloader()
        if not downloader.validate_timeframe(args.timeframe):
            supported = downloader.get_supported_timeframes()
            logging.error(f"Invalid timeframe '{args.timeframe}' for ccxt. Supported timeframes are: {', '.join(supported)}")
            return 1

        start_ms = int(start_date_ts.timestamp() * 1000)
        end_ms = int(end_date_ts.timestamp() * 1000)

        df = _download(
            f"{args.symbol} from {args.exchange}",
            downloader.download,
            args.exchange, args.symbol, args.timeframe, start_ms, end_ms
        )

    elif args.source == 'yahoo':
        downloader = YahooFinanceDownloader()
        if not downloader.validate_timeframe(args.timeframe):
            supported = downloader.get_supported_timeframes()
            logging.error(f"Invalid timeframe '{args.timeframe}' for yahoo. Supported timeframes are: {', '.join(supported)}")
            return 1

        df = _download(
            f"{args.symbol} from Yahoo Finance",
            downloader.download,
            args.symbol, args.start_date, args.end_date, args.timeframe
        )
        # No post-processing here: YahooFinanceDownloader already names the index
        # 'Date', flattens any MultiIndex columns (correctly handling multi-ticker
        # responses, which the old duplicate block here did not) and returns the
        # Open/High/Low/Close/Volume columns in order.

    if df is None or df.empty:
        logging.error(f"No data returned for {args.symbol} ({args.source}); nothing was saved.")
        return 1

    # Downloaders flag a truncated series with df.attrs['partial']. Writing it to
    # the normal output path and returning 0 would hand every downstream backtest a
    # silently incomplete price series.
    partial = bool(df.attrs.get('partial', False))
    if partial:
        reason = df.attrs.get('partial_reason', 'reason not reported by the downloader')
        stem, extension = os.path.splitext(output_path)
        output_path = f"{stem}.partial{extension or '.csv'}"
        logging.error(
            f"Download for {args.symbol} ({args.source}) is INCOMPLETE: {reason}. "
            f"Writing what was retrieved to {output_path}; do not backtest on it "
            f"without re-downloading the missing range."
        )

    try:
        df.to_csv(output_path)
    except OSError as e:
        logging.error(f"Could not write data to {output_path}: {e}")
        return 1

    if partial:
        logging.error(f"Partial data saved to {output_path}")
        return 1

    logging.info(f"Data successfully saved to {output_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
