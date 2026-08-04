import argparse
import os
import sys
from pathlib import Path
from typing import Any, List

import pandas as pd

# Running "python scripts/backtest.py" puts scripts/ on sys.path but not the
# repository root, so the root has to be added for "import niffler" to work.
# When imported as scripts.backtest the root is already importable.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niffler.backtesting import BacktestEngine
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
from niffler.risk import FixedRiskManager
from niffler.exporters import ExporterManager
from niffler.config.logging import setup_logging
from scripts.common import load_ohlcv_csv


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
    """Load CSV data and optionally apply the cleaning pipeline.

    Args:
        file_path: Path to the CSV file with OHLCV data.
        clean: Whether to run the default preprocessing pipeline.

    Returns:
        DataFrame with lowercase OHLCV columns and a sorted datetime index.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the file cannot be interpreted as OHLCV data.
    """
    return load_ohlcv_csv(file_path, clean=clean)


def report_export_outcome(export_result: Any, exporter_names: List[str]) -> int:
    """Print a per-exporter export report and derive the process exit code.

    Accepts either the ``ExportSummary`` returned by the current
    ``ExporterManager`` or the bare backtest id returned by older versions.

    Args:
        export_result: Value returned by ExporterManager.export_backtest_result.
        exporter_names: Names of the configured exporters, used as a fallback
            when the manager only returns a backtest id.

    Returns:
        0 when every exporter succeeded, 1 when at least one failed.
    """
    successes = getattr(export_result, 'successes', None)
    failures = getattr(export_result, 'failures', None)
    backtest_id = getattr(export_result, 'backtest_id', export_result)

    print(f"Backtest completed with ID: {backtest_id}")

    if successes is None or failures is None:
        # Legacy ExporterManager: no per-exporter outcome is available.
        print(f"Exported using: {', '.join(exporter_names)}")
        return 0

    print("Export report:")
    for name in successes:
        print(f"  OK     {name}")
    for name, error in failures:
        print(f"  FAILED {name}: {error}")

    if failures:
        total = len(successes) + len(failures)
        print(f"Error: {len(failures)} of {total} exporters failed", file=sys.stderr)
        return 1

    return 0


def main() -> int:
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

        if exporter_manager.get_exporter_count() == 0:
            print(f"Error: no usable exporters created from '{args.exporters}'", file=sys.stderr)
            return 1

        # Prepare strategy parameters for metadata (generic - gets from strategy object)
        strategy_params = strategy.parameters.copy()

        # Export results using all configured exporters
        export_result = exporter_manager.export_backtest_result(
            result=result,
            strategy_params=strategy_params,
            symbol=symbol,
            initial_capital=args.capital,
            commission=args.commission
        )

        return report_export_outcome(export_result, exporter_manager.get_exporter_names())

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())