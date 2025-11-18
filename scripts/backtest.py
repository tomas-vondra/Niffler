import argparse
import pandas as pd
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting import BacktestEngine, BacktestResult
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
from niffler.risk import FixedRiskManager
from niffler.exporters import ExporterManager
from config.logging import setup_logging


def extract_symbol_from_filename(file_path: str) -> str:
    """Extract symbol from filename.

    Expected formats:
    - BTCUSD_yahoo_1d_20240101_20241231_cleaned.csv -> BTCUSD
    - BTCUSDT_binance_1d_20240101_20240105.csv -> BTCUSDT
    - BTC-USD_data.csv -> BTC-USD
    - anything_else.csv -> filename without extension
    """
    filename = os.path.basename(file_path)
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]

    # Try to extract symbol (first part before underscore)
    parts = name_without_ext.split('_')
    if len(parts) > 0:
        return parts[0]

    return name_without_ext


def load_data(file_path: str, clean: bool = False) -> pd.DataFrame:
    """Load CSV data and optionally apply cleaning pipeline."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if clean:
        # Apply data cleaning pipeline
        from scripts.preprocessor import load_and_clean_csv
        df = load_and_clean_csv(file_path)

        if df is None:
            raise ValueError(f"Failed to load and clean data from {file_path}")
    else:
        # Load CSV file directly (assumes it's already cleaned)
        df = pd.read_csv(file_path)

        # Try to parse timestamp/date column as index
        timestamp_cols = ['timestamp', 'date', 'Date', 'Timestamp']
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break

    # Validate required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df



def main():
    parser = argparse.ArgumentParser(
        description='Backtest trading strategies on historical data with optional risk management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest without risk management
  python backtest.py --data data/BTC.csv --strategy simple_ma
  
  # Backtest with fixed risk management  
  python backtest.py --data data/BTC.csv --strategy simple_ma --risk-manager fixed \\
    --max-position-size 0.1 --stop-loss-pct 0.05 --max-positions 3
        """
    )
    
    # Required arguments
    parser.add_argument('--data', '-d', required=True,
                       help='Path to CSV data file')
    parser.add_argument('--strategy', '-s', default='simple_ma',
                       choices=['simple_ma'],
                       help='Strategy to backtest (default: simple_ma)')
    
    # Strategy parameters
    parser.add_argument('--short-window', type=int, default=10,
                       help='Short MA window for simple_ma strategy (default: 10)')
    parser.add_argument('--long-window', type=int, default=30,
                       help='Long MA window for simple_ma strategy (default: 30)')
    parser.add_argument('--position-size', type=float, default=1.0,
                       help='Position size as fraction of portfolio (default: 1.0)')
    
    # Backtest parameters
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital amount (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate per trade (default: 0.001)')
    
    # Output options
    # Get available exporters dynamically
    from niffler.exporters import ExporterManager
    available_exporters = ','.join(ExporterManager.get_available_exporter_names())
    parser.add_argument('--exporters', type=str, default='console',
                       help=f'Comma-separated list of exporters to use: {available_exporters} (default: console)')
    parser.add_argument('--csv-output-dir', default='.',
                       help='Directory for CSV output files (default: current directory)')
    parser.add_argument('--symbol', default=None,
                       help='Symbol identifier for the data (default: extracted from filename)')
    
    # Elasticsearch options (optional overrides for .env file configuration)
    parser.add_argument('--es-host',
                       help='Elasticsearch host (overrides ELASTICSEARCH_HOST env var)')
    parser.add_argument('--es-port', type=int,
                       help='Elasticsearch port (overrides ELASTICSEARCH_PORT env var)')
    parser.add_argument('--es-index-prefix',
                       help='Elasticsearch index prefix (overrides ELASTICSEARCH_INDEX_PREFIX env var)')
    
    # Data processing options
    parser.add_argument('--clean', action='store_true',
                       help='Apply data cleaning pipeline to the CSV file before backtesting')
    
    # Risk Management options
    parser.add_argument('--risk-manager', choices=['none', 'fixed'],
                       default='none',
                       help='Risk manager to use (default: none)')
    parser.add_argument('--max-position-size', type=float, default=0.2,
                       help='Maximum position size as fraction of portfolio (default: 0.2)')
    parser.add_argument('--stop-loss-pct', type=float, default=0.05,
                       help='Stop loss percentage (default: 0.05)')
    parser.add_argument('--max-positions', type=int, default=5,
                       help='Maximum number of concurrent positions (default: 5)')
    parser.add_argument('--max-risk-per-trade', type=float, default=0.02,
                       help='Maximum risk per trade as fraction of portfolio (default: 0.02)')
    
    # Logging options
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    parser.add_argument('--min-order-value', type=float, default=1.0,
                       help='Minimum order value to execute trades (default: 1.0)')
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging(level=args.log_level)
    
    try:
        # Load data
        print(f"Loading data from {args.data}...")
        data = load_data(args.data, clean=args.clean)
        print(f"Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")

        # Extract symbol from filename if not provided
        symbol = args.symbol
        if symbol is None:
            symbol = extract_symbol_from_filename(args.data)
            print(f"Symbol extracted from filename: {symbol}")

        # Initialize risk manager
        risk_manager = None
        if args.risk_manager == 'fixed':
            risk_manager = FixedRiskManager(
                position_size_pct=args.max_position_size,
                stop_loss_pct=args.stop_loss_pct,
                max_positions=args.max_positions,
                max_risk_per_trade=args.max_risk_per_trade
            )
            print(f"Risk Manager: {risk_manager.get_risk_metrics()['risk_management_type']}")
        
        # Initialize strategy
        if args.strategy == 'simple_ma':
            strategy = SimpleMAStrategy(
                short_window=args.short_window,
                long_window=args.long_window,
                position_size=args.position_size,
                risk_manager=risk_manager
            )
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        
        print(f"Strategy: {strategy.get_description()}")
        
        # Print risk management info
        if risk_manager is not None:
            risk_metrics = risk_manager.get_risk_metrics()
            print(f"Risk Management: {risk_metrics.get('risk_management_type', 'Unknown')}")
            print(f"  Max Position Size: {risk_metrics.get('max_position_size', 'N/A')}")
            print(f"  Stop Loss: {risk_metrics.get('stop_loss_pct', 'N/A')}")
            print(f"  Max Positions: {risk_metrics.get('max_positions', 'N/A')}")
        else:
            print("Risk Management: None")
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=args.capital,
            commission=args.commission,
            min_order_value=args.min_order_value
        )
        
        print("Running backtest...")

        # Run backtest
        result = engine.run_backtest(strategy, data, symbol)

        # Setup exporters
        exporter_manager = ExporterManager()

        # Parse exporters parameter
        exporter_names = [name.strip().lower() for name in args.exporters.split(',')]

        # Create exporters - pass all options, each exporter will use what it needs
        exporter_manager.create_exporters_from_list(
            exporter_names,
            output_dir=args.csv_output_dir,
            host=args.es_host,
            port=args.es_port,
            index_prefix=args.es_index_prefix
        )

        # Prepare strategy parameters for metadata (generic - gets from strategy object)
        strategy_params = strategy.parameters.copy()

        # Export results using all configured exporters
        backtest_id = exporter_manager.export_backtest_result(
            result=result,
            strategy_params=strategy_params,
            symbol=symbol,
            initial_capital=args.capital,
            commission=args.commission
        )
        
        print(f"Backtest completed with ID: {backtest_id}")
        if exporter_manager.get_exporter_count() > 1:
            print(f"Exported using: {', '.join(exporter_manager.get_exporter_names())}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()