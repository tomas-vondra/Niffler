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
from config.logging import setup_logging


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


def print_backtest_results(result: BacktestResult):
    """Print formatted backtest results."""
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Strategy: {result.strategy_name}")
    print(f"Symbol: {result.symbol}")
    print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Initial Capital: ${result.initial_capital:,.2f}")
    print(f"  Final Capital: ${result.final_capital:,.2f}")
    print(f"  Total Return: ${result.total_return:,.2f}")
    print(f"  Total Return %: {result.total_return_pct:.2f}%")
    print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"  Win Rate: {result.win_rate:.1f}%")
    print(f"  Total Trades: {result.total_trades}")
    
    if result.trades:
        print(f"\nFIRST 5 TRADES:")
        for i, trade in enumerate(result.trades[:5]):
            print(f"  {i+1}. {trade.timestamp.strftime('%Y-%m-%d')} - "
                  f"{trade.side.value.upper()} {trade.quantity:.4f} @ ${trade.price:.2f}")
        
        if len(result.trades) > 5:
            print(f"  ... and {len(result.trades) - 5} more trades")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Backtest trading strategies on historical data')
    
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
    parser.add_argument('--output', '-o',
                       help='Save results to CSV file')
    parser.add_argument('--symbol', default='UNKNOWN',
                       help='Symbol identifier for the data (default: UNKNOWN)')
    
    # Data processing options
    parser.add_argument('--clean', action='store_true',
                       help='Apply data cleaning pipeline to the CSV file before backtesting')
    
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
        
        # Initialize strategy
        if args.strategy == 'simple_ma':
            strategy = SimpleMAStrategy(
                short_window=args.short_window,
                long_window=args.long_window,
                position_size=args.position_size
            )
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        
        print(f"Strategy: {strategy.get_description()}")
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=args.capital,
            commission=args.commission,
            min_order_value=args.min_order_value
        )
        
        print("Running backtest...")
        
        # Run backtest
        result = engine.run_backtest(strategy, data, args.symbol)
        
        # Print results
        print_backtest_results(result)
        
        # Save results if requested
        if args.output:
            print(f"Saving results to {args.output}...")
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'timestamp': result.portfolio_values.index,
                'portfolio_value': result.portfolio_values.values
            })
            
            # Add trade information
            if result.trades:
                trades_df = pd.DataFrame([
                    {
                        'timestamp': trade.timestamp,
                        'side': trade.side.value,
                        'price': trade.price,
                        'quantity': trade.quantity,
                        'value': trade.value
                    }
                    for trade in result.trades
                ])
                
                # Save trades to separate file
                trades_file = args.output.replace('.csv', '_trades.csv')
                trades_df.to_csv(trades_file, index=False)
                print(f"Trade details saved to {trades_file}")
            
            results_df.to_csv(args.output, index=False)
            print(f"Portfolio values saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()