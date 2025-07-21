#!/usr/bin/env python3
"""
Advanced analysis script for Niffler trading strategies.

Provides Walk-forward analysis and Monte Carlo analysis for strategy validation.
This script takes pre-optimized parameters and tests their robustness.
"""

import argparse
import pandas as pd
import logging
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.logging import setup_logging
from niffler.analysis import WalkForwardAnalyzer, MonteCarloAnalyzer
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run advanced analysis on trading strategies using pre-optimized parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Walk-forward analysis with specific parameters
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --params '{"short_window": 10, "long_window": 30}'

  # Load parameters from optimization results
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --params_file optimization_results.json

  # Monte Carlo analysis with specific parameters
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params '{"short_window": 10, "long_window": 30}' --simulations 500

  # Walk-forward with custom windows
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --params '{"short_window": 10, "long_window": 30}' --test_window 6 --step 3
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data', 
        required=True,
        help='Path to CSV file with OHLCV data'
    )
    
    parser.add_argument(
        '--analysis',
        required=True,
        choices=['walk_forward', 'monte_carlo'],
        help='Type of analysis to perform'
    )
    
    parser.add_argument(
        '--strategy',
        required=True,
        choices=['simple_ma'],
        help='Trading strategy to analyze'
    )
    
    # Parameter specification (one of these is required)
    param_group = parser.add_mutually_exclusive_group(required=True)
    param_group.add_argument(
        '--params',
        help='Strategy parameters as JSON string (e.g., \'{"short_window": 10, "long_window": 30}\')'
    )
    param_group.add_argument(
        '--params_file',
        help='Path to JSON file containing optimization results or parameters'
    )
    
    # Analysis configuration
    parser.add_argument(
        '--initial_capital',
        type=float,
        default=10000.0,
        help='Initial capital for backtests (default: 10000.0)'
    )
    
    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate for trades (default: 0.001)'
    )
    
    # Walk-forward specific arguments
    parser.add_argument(
        '--test_window',
        type=int,
        default=6,
        help='Test window in months for walk-forward analysis (default: 6)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=3,
        help='Step size in months for walk-forward analysis (default: 3)'
    )
    
    # Monte Carlo specific arguments
    parser.add_argument(
        '--simulations',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )
    
    parser.add_argument(
        '--bootstrap_pct',
        type=float,
        default=0.8,
        help='Percentage of data to sample in each simulation (default: 0.8)'
    )
    
    
    parser.add_argument(
        '--block_size',
        type=int,
        default=30,
        help='Block size in days for block bootstrap sampling (default: 30)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        help='Random seed for reproducible Monte Carlo results'
    )
    
    parser.add_argument(
        '--n_jobs',
        type=int,
        help='Number of parallel jobs for analysis (default: auto-detect)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        help='Output file for detailed results (JSON format)'
    )
    
    parser.add_argument(
        '--symbol',
        default='UNKNOWN',
        help='Symbol identifier for the data (default: UNKNOWN)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate data from CSV file."""
    try:
        data = pd.read_csv(file_path)
        
        # Convert timestamp column to datetime index
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
        elif data.index.name == 'timestamp':
            data.index = pd.to_datetime(data.index)
        else:
            # Try to parse index as datetime
            try:
                data.index = pd.to_datetime(data.index)
            except:
                raise ValueError("Data must have a 'timestamp' column or datetime index")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by index
        data = data.sort_index()
        
        logging.info(f"Loaded {len(data)} rows of data from {file_path}")
        logging.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        return data
        
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise


def load_parameters(args) -> dict:
    """Load strategy parameters from command line arguments."""
    if args.params:
        # Parse JSON string
        try:
            params = json.loads(args.params)
            logging.info(f"Loaded parameters from command line: {params}")
            return params
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --params: {e}")
    
    elif args.params_file:
        # Load from file
        try:
            with open(args.params_file, 'r') as f:
                data = json.load(f)
            
            # Handle different file formats
            if 'results' in data and len(data['results']) > 0:
                # Optimization results file - use best result
                best_result = data['results'][0]
                params = best_result['parameters']
                logging.info(f"Loaded best parameters from optimization file: {params}")
                return params
            elif 'parameters' in data:
                # Direct parameters file
                params = data['parameters']
                logging.info(f"Loaded parameters from file: {params}")
                return params
            else:
                # Assume the file itself contains the parameters
                logging.info(f"Loaded parameters from file: {data}")
                return data
                
        except FileNotFoundError:
            raise ValueError(f"Parameters file not found: {args.params_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in parameters file: {e}")
    
    else:
        raise ValueError("Either --params or --params_file must be specified")


def get_strategy_class(strategy_name: str):
    """Get strategy class by name."""
    strategy_map = {
        'simple_ma': SimpleMAStrategy
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_map[strategy_name]


def validate_parameters(strategy_class, parameters: dict):
    """Validate that parameters are compatible with the strategy."""
    try:
        # Try to create strategy instance to validate parameters
        strategy = strategy_class(**parameters)
        logging.info("Parameter validation successful")
    except Exception as e:
        raise ValueError(f"Invalid parameters for {strategy_class.__name__}: {e}")


def run_walk_forward_analysis(args, data: pd.DataFrame, parameters: dict):
    """Run walk-forward analysis."""
    logging.info("Running Walk-forward Analysis")
    
    # Get strategy class
    strategy_class = get_strategy_class(args.strategy)
    
    # Validate parameters
    validate_parameters(strategy_class, parameters)
    
    # Create analyzer
    analyzer = WalkForwardAnalyzer(
        strategy_class=strategy_class,
        optimal_parameters=parameters,
        test_window_months=args.test_window,
        step_months=args.step,
        initial_capital=args.initial_capital,
        commission=args.commission,
        n_jobs=args.n_jobs
    )
    
    # Run analysis
    result = analyzer.analyze(data, args.symbol)
    
    # Print summary
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("="*60)
    print(f"Strategy: {result.strategy_name}")
    print(f"Symbol: {result.symbol}")
    print(f"Analysis Period: {result.analysis_start_date.date()} to {result.analysis_end_date.date()}")
    print(f"Number of Periods: {result.n_periods}")
    
    print(f"\nParameters Used: {parameters}")
    print(f"Test Windows: {args.test_window} months")
    print(f"Step Size: {args.step} months")
    
    print(f"\nCombined Metrics:")
    for metric, value in result.combined_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print(f"\nStability Metrics:")
    for metric, value in result.stability_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Show period-by-period results
    df = result.to_dataframe()
    print(f"\nPeriod-by-Period Results:")
    display_cols = ['start_date', 'end_date', 'total_return', 'total_return_pct', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    available_cols = [col for col in display_cols if col in df.columns]
    if available_cols:
        print(df[available_cols].round(4))
    else:
        print(df.round(4))
    
    return result


def run_monte_carlo_analysis(args, data: pd.DataFrame, parameters: dict):
    """Run Monte Carlo analysis."""
    logging.info("Running Monte Carlo Analysis")
    
    # Get strategy class
    strategy_class = get_strategy_class(args.strategy)
    
    # Validate parameters
    validate_parameters(strategy_class, parameters)
    
    # Create analyzer
    analyzer = MonteCarloAnalyzer(
        strategy_class=strategy_class,
        optimal_parameters=parameters,
        n_simulations=args.simulations,
        bootstrap_sample_pct=args.bootstrap_pct,
        block_size_days=args.block_size,
        initial_capital=args.initial_capital,
        commission=args.commission,
        n_jobs=args.n_jobs,
        random_seed=args.random_seed
    )
    
    # Run analysis
    result = analyzer.analyze(data, args.symbol)
    
    # Print summary
    print("\n" + "="*60)
    print("MONTE CARLO ANALYSIS RESULTS")
    print("="*60)
    print(f"Strategy: {result.strategy_name}")
    print(f"Symbol: {result.symbol}")
    print(f"Analysis Period: {result.analysis_start_date.date()} to {result.analysis_end_date.date()}")
    print(f"Successful Simulations: {len(result.individual_results)}")
    
    print(f"\nUsing Parameters: {parameters}")
    print(f"\nSimulation Parameters:")
    print(f"  Target Simulations: {args.simulations}")
    print(f"  Bootstrap Sample: {args.bootstrap_pct*100:.1f}%")
    print(f"  Block Bootstrap: Yes (preserves time series structure)")
    print(f"  Block Size: {args.block_size} days")
    
    print(f"\nCombined Metrics:")
    for metric, value in result.combined_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print(f"\nDistribution Statistics:")
    for metric, value in result.stability_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Show percentile analysis
    percentile_results = analyzer.get_percentile_results(result.individual_results)
    print(f"\nPercentile Analysis:")
    for metric, percentiles in percentile_results.items():
        print(f"  {metric}:")
        for p_name, p_value in percentiles.items():
            if isinstance(p_value, (int, float)):
                print(f"    {p_name}: {p_value:.4f}")
            else:
                print(f"    {p_name}: {p_value}")
    
    return result




def save_results(result, output_file: str):
    """Save analysis results to JSON file."""
    try:
        # Convert result to dictionary
        output_data = {
            'analysis_type': result.analysis_type,
            'strategy_name': result.strategy_name,
            'symbol': result.symbol,
            'analysis_start_date': result.analysis_start_date.isoformat(),
            'analysis_end_date': result.analysis_end_date.isoformat(),
            'n_periods': result.n_periods,
            'combined_metrics': result.combined_metrics,
            'stability_metrics': result.stability_metrics,
            'analysis_parameters': result.analysis_parameters,
            'summary_statistics': result.get_summary_statistics(),
            'performance_consistency': result.get_performance_consistency()
        }
        
        # Add period/simulation results
        df = result.to_dataframe()
        if result.analysis_type == 'walk_forward':
            output_data['period_results'] = df.to_dict('records')
        else:  # monte_carlo
            output_data['simulation_results'] = df.to_dict('records')
        
        # Add metadata if available
        if result.metadata:
            output_data['metadata'] = result.metadata
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")


def main():
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    try:
        # Load data
        data = load_data(args.data)
        
        # Load parameters
        parameters = load_parameters(args)
        
        # Run analysis
        if args.analysis == 'walk_forward':
            result = run_walk_forward_analysis(args, data, parameters)
        elif args.analysis == 'monte_carlo':
            result = run_monte_carlo_analysis(args, data, parameters)
        else:
            raise ValueError(f"Unknown analysis type: {args.analysis}")
        
        # Save results if output file specified
        if args.output:
            save_results(result, args.output)
        
        print(f"\nAnalysis completed successfully!")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()