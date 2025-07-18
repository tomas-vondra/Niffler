#!/usr/bin/env python3
"""
Parameter optimization script for trading strategies.

This script allows you to optimize strategy parameters using various methods
like grid search and random search. It evaluates different parameter combinations
using backtesting and finds the best performing parameters based on the chosen
objective function.

Examples:
    # Grid search optimization for Simple MA strategy
    python scripts/optimize.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma --method grid

    # Random search with 100 trials
    python scripts/optimize.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma --method random --trials 100

    # Sort results by Sharpe ratio
    python scripts/optimize.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma --sort-by sharpe_ratio

    # Save results to custom file
    python scripts/optimize.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma --output my_results.json
"""

import argparse
import sys
import os
import pandas as pd
import logging
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.logging import setup_logging
from niffler.optimization.optimizer_factory import (
    create_optimizer,
    get_strategy_class, 
    get_parameter_space,
    get_available_optimizers,
    STRATEGY_CLASSES
)
from niffler.data.preprocessors.preprocessor_manager import PreprocessorManager


def load_and_validate_data(file_path: str, clean_data: bool = False) -> pd.DataFrame:
    """
    Load and validate price data for optimization.
    
    Args:
        file_path: Path to CSV file with OHLCV data
        clean_data: Whether to apply data preprocessing
        
    Returns:
        Validated DataFrame with datetime index
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime and set as index
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        data.sort_index(inplace=True)
        
        # Clean data if requested
        if clean_data:
            logging.info("Applying data preprocessing...")
            from niffler.data.preprocessors.preprocessor_manager import create_default_manager
            manager = create_default_manager()
            data = manager.run(data)
        
        # Validate data quality
        if data.empty:
            raise ValueError("Data file is empty")
        
        if data.isnull().any().any():
            logging.warning("Data contains NaN values - consider using --clean flag")
        
        logging.info(f"Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")
        return data
        
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")


def main():
    """Main function for parameter optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize trading strategy parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--data', required=True,
                       help='Path to CSV file with OHLCV data')
    parser.add_argument('--strategy', required=True, 
                       choices=list(STRATEGY_CLASSES.keys()),
                       help='Trading strategy to optimize')
    
    # Optimization method
    parser.add_argument('--method', default='grid',
                       choices=get_available_optimizers(),
                       help='Optimization method (default: grid)')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials for random search (default: 100)')
    
    # Optimization parameters
    parser.add_argument('--sort-by', default='total_return',
                       choices=['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades'],
                       help='Metric to sort top results by (default: total_return)')
    
    # Backtest parameters
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                       help='Initial capital for backtesting (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate per trade (default: 0.001)')
    
    # Data processing
    parser.add_argument('--clean', action='store_true',
                       help='Apply data preprocessing before optimization')
    
    # Performance options
    parser.add_argument('--jobs', type=int, default=None,
                       help='Number of parallel jobs (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results')
    
    # Output options
    parser.add_argument('--output', default=None,
                       help='Output file for results (default: auto-generated)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top results to display (default: 10)')
    
    # Logging
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate data
        logger.info(f"Loading data from {args.data}")
        data = load_and_validate_data(args.data, args.clean)
        
        # Get strategy class and parameter space
        strategy_class = get_strategy_class(args.strategy)
        parameter_space = get_parameter_space(args.strategy)
        
        # Create optimizer
        optimizer = create_optimizer(
            method=args.method,
            strategy_class=strategy_class,
            parameter_space=parameter_space,
            data=data,
            initial_capital=args.initial_capital,
            commission=args.commission,
            sort_by=args.sort_by,
            n_jobs=args.jobs
        )
        
        # Run optimization
        logger.info(f"Starting {args.method} optimization for {args.strategy} strategy")
        logger.info(f"Sorting by: {args.sort_by}")
        logger.info(f"Initial capital: ${args.initial_capital:,.2f}")
        logger.info(f"Commission: {args.commission:.4f}")
        
        start_time = datetime.now()
        
        # Run optimization with method-specific parameters
        if args.method == 'random':
            results = optimizer.optimize(n_trials=args.trials, seed=args.seed)
        else:
            results = optimizer.optimize()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if not results:
            logger.error("No valid optimization results found")
            return 1
        
        # Display results
        logger.info(f"Optimization completed in {duration}")
        logger.info(f"Evaluated {len(results)} parameter combinations")
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION RESULTS - {args.strategy.upper()} STRATEGY")
        print(f"{'='*80}")
        print(f"Sorted By: {args.sort_by}")
        print(f"Total Combinations: {len(results)}")
        print(f"Duration: {duration}")
        print()
        
        # Show top results
        print(f"TOP {min(args.top_n, len(results))} RESULTS:")
        print("-" * 80)
        
        for i, result in enumerate(results[:args.top_n], 1):
            # Get the sort value for display using the optimizer's metrics config
            from niffler.optimization.base_optimizer import BaseOptimizer
            _, accessor_func = BaseOptimizer.METRICS_CONFIG[args.sort_by]
            sort_value = accessor_func(result)
            
            if args.sort_by in ['total_return', 'max_drawdown', 'win_rate']:
                print(f"#{i} - {args.sort_by}: {sort_value:.2%}")
            elif args.sort_by == 'sharpe_ratio':
                print(f"#{i} - {args.sort_by}: {sort_value:.3f}")
            else:
                print(f"#{i} - {args.sort_by}: {sort_value}")
            print(f"    Parameters: {result.parameters}")
            print(f"    Total Return: {result.backtest_result.total_return:.2%}")
            print(f"    Sharpe Ratio: {result.backtest_result.sharpe_ratio:.3f}")
            print(f"    Max Drawdown: {result.backtest_result.max_drawdown:.2%}")
            print(f"    Total Trades: {result.backtest_result.total_trades}")
            print(f"    Win Rate: {result.backtest_result.win_rate:.1%}")
            print()
        
        # Show best parameters for each metric
        print(f"BEST PARAMETERS BY METRIC:")
        print("-" * 80)
        
        try:
            best_metrics = optimizer.analyze_best_metrics(results)
        except Exception as e:
            logger.warning(f"Could not analyze best metrics: {e}")
            best_metrics = {}
        for metric_name, metric_data in best_metrics.items():
            value = metric_data['value']
            params = metric_data['parameters']
            
            # Format value based on metric type
            if metric_name in ['total_return', 'max_drawdown', 'win_rate']:
                formatted_value = f"{value:.2%}"
            elif metric_name == 'sharpe_ratio':
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = f"{value}"
            
            print(f"Best {metric_name.replace('_', ' ').title()}: {formatted_value}")
            print(f"    Parameters: {params}")
            print()
        
        # Generate output filename if not provided
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"optimization_results_{args.strategy}_{args.method}_{timestamp}.json"
        
        # Save results
        optimizer.save_results(results, args.output)
        print(f"Full results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())