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

# Running "python scripts/analyze.py" puts scripts/ on sys.path but not the
# repository root, so the root has to be added for "import niffler" to work.
# When imported as scripts.analyze the root is already importable.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niffler.config.logging import setup_logging
from niffler.analysis import (
    WalkForwardAnalyzer,
    MonteCarloAnalyzer,
    MODE_WALK_FORWARD,
    MODE_SEGMENTED_IN_SAMPLE,
)
from niffler.optimization.base_optimizer import BaseOptimizer
from niffler.optimization.optimizer_factory import (
    get_available_optimizers,
    get_parameter_space,
)
from niffler.strategies.registry import get_available_strategies, get_strategy_class
from niffler.utils.json_utils import safe_json_dump
from niffler.utils.provenance import collect_provenance
from scripts.common import (
    add_cost_model_arguments,
    add_engine_arguments,
    build_run_config,
    load_ohlcv_csv,
    report_run_config,
)
from scripts.config_file import add_config_arguments, apply_config, report_config


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run advanced analysis on trading strategies using pre-optimized parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Walk-forward analysis (parameters are re-optimised on every training window)
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma

  # Walk-forward with custom windows
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --train_window 12 --test_window 6 --step 3

  # Re-run one fixed parameter set over consecutive in-sample slices (NOT a validation)
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --mode segmented_in_sample --strategy simple_ma --params '{"short_window": 10, "long_window": 30}'

  # Monte Carlo analysis with specific parameters
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params '{"short_window": 10, "long_window": 30}' --simulations 500

  # Load parameters from optimization results
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params_file optimization_results.json
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
        choices=get_available_strategies(),
        help='Trading strategy to analyze'
    )
    
    # Parameter specification.
    #
    # Required for monte_carlo and for walk-forward's segmented_in_sample mode. Real
    # walk-forward re-optimises the parameters on every training window, so a fixed
    # parameter set is meaningless there and must not be demanded from the user.
    param_group = parser.add_mutually_exclusive_group(required=False)
    param_group.add_argument(
        '--params',
        help='Strategy parameters as JSON string (e.g., \'{"short_window": 10, "long_window": 30}\'). '
             'Required for --analysis monte_carlo and for --mode segmented_in_sample.'
    )
    param_group.add_argument(
        '--params_file',
        help='Path to JSON file containing optimization results or parameters'
    )
    
    # Analysis configuration
    parser.add_argument(
        '--initial_capital', '--capital', '--initial-capital',
        dest='initial_capital',
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

    # Transaction costs (slippage, spread, liquidity)
    add_cost_model_arguments(parser)

    # Benchmark, annualisation, order floor and the significance gate. These
    # reach the engine inside every fold and every simulated path now that the
    # analyzers carry a RunConfig rather than three loose numbers.
    add_engine_arguments(parser)


    # Walk-forward specific arguments
    parser.add_argument(
        '--mode',
        choices=[MODE_WALK_FORWARD, MODE_SEGMENTED_IN_SAMPLE],
        default=MODE_WALK_FORWARD,
        help=("Walk-forward mode. 'walk_forward' (default) re-optimises the parameters on "
              "each training window and reports genuinely out-of-sample results. "
              "'segmented_in_sample' re-runs one fixed --params set over consecutive "
              "slices of the same data and validates nothing.")
    )

    parser.add_argument(
        '--train_window',
        type=int,
        default=12,
        help='Training window in months for walk-forward analysis (default: 12)'
    )

    parser.add_argument(
        '--anchored',
        action='store_true',
        help='Anchor every training window to the first bar instead of rolling it forward'
    )

    parser.add_argument(
        '--optimization_method',
        choices=get_available_optimizers(),
        default='grid',
        help='Optimizer used on each walk-forward training window (default: grid)'
    )

    parser.add_argument(
        '--optimization_metric',
        choices=list(BaseOptimizer.METRICS_CONFIG.keys()),
        default='total_return',
        help='Metric the per-fold optimizer selects parameters by (default: total_return)'
    )

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
        '--random_seed', '--seed',
        dest='seed',
        type=int,
        help='Random seed for reproducible Monte Carlo results'
    )
    
    parser.add_argument(
        '--n_jobs', '--jobs',
        dest='n_jobs',
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
        help='Enable verbose logging (shorthand for --log-level DEBUG)'
    )

    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level (default: INFO)'
    )

    add_config_arguments(parser)

    return parser


def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate OHLCV data from a CSV file.

    Args:
        file_path: Path to the CSV file with OHLCV data.

    Returns:
        DataFrame with lowercase OHLCV columns and a sorted datetime index.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the file cannot be interpreted as OHLCV data.
    """
    try:
        data = load_ohlcv_csv(file_path)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

    logging.info(f"Loaded {len(data)} rows of data from {file_path}")
    logging.info(f"Date range: {data.index[0]} to {data.index[-1]}")

    return data


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


def validate_parameters(strategy_class, parameters: dict):
    """Validate that parameters are compatible with the strategy."""
    try:
        # Try to create strategy instance to validate parameters
        strategy_class(**parameters)
        logging.info("Parameter validation successful")
    except Exception as e:
        raise ValueError(f"Invalid parameters for {strategy_class.__name__}: {e}")


def run_walk_forward_analysis(args, data: pd.DataFrame, parameters: dict = None,
                              run_config=None):
    """Run walk-forward analysis.

    In the default ``walk_forward`` mode the strategy parameters are re-optimised on
    every training window, so ``parameters`` is unused; the search space comes from the
    strategy's registered parameter space. In ``segmented_in_sample`` mode the supplied
    fixed ``parameters`` are re-run over consecutive slices instead.

    Args:
        args: Parsed command line arguments.
        data: OHLCV data with a DatetimeIndex.
        parameters: Fixed strategy parameters, required only for segmented_in_sample mode.
        run_config: Engine settings used by the per-fold optimisation and by
            the out-of-sample evaluation alike.

    Returns:
        The AnalysisResult produced by WalkForwardAnalyzer.

    Raises:
        ValueError: If the configuration or the parameters are invalid.
    """
    logging.info("Running Walk-forward Analysis")

    # Get strategy class
    strategy_class = get_strategy_class(args.strategy)

    segmented = args.mode == MODE_SEGMENTED_IN_SAMPLE
    parameter_space = None

    if segmented:
        if not parameters:
            raise ValueError(
                f"--mode {MODE_SEGMENTED_IN_SAMPLE} requires --params or --params_file"
            )
        validate_parameters(strategy_class, parameters)
    else:
        parameter_space = get_parameter_space(args.strategy)
        if parameters:
            logging.warning(
                "Ignoring --params in walk_forward mode: parameters are re-optimised on "
                "every training window. Use --mode segmented_in_sample to pin them."
            )
            parameters = None

    # Create analyzer
    analyzer = WalkForwardAnalyzer(
        strategy_class=strategy_class,
        parameter_space=parameter_space,
        optimal_parameters=parameters,
        mode=args.mode,
        anchored=args.anchored,
        train_window_months=args.train_window,
        test_window_months=args.test_window,
        step_months=args.step,
        optimization_method=args.optimization_method,
        optimization_metric=args.optimization_metric,
        n_jobs=args.n_jobs,
        run_config=run_config
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

    print(f"\nMode: {args.mode}")
    if segmented:
        print("  WARNING: segmented_in_sample results are NOT out-of-sample.")
        print(f"Parameters Used: {parameters}")
    else:
        print(f"Training Windows: {args.train_window} months "
              f"({'anchored' if args.anchored else 'rolling'})")
        print(f"Optimizer: {args.optimization_method} on {args.optimization_metric}")
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
    
    # Show per-fold parameters and in-sample vs out-of-sample performance
    folds = (result.metadata or {}).get('folds') or []
    if folds:
        print(f"\nFold-by-Fold (parameters chosen on train, measured on test):")
        for fold in folds:
            efficiency = fold.get('efficiency_ratio')
            efficiency_text = 'n/a' if efficiency is None else f"{efficiency:.3f}"
            train_return = fold.get('train_return_pct')
            train_text = 'n/a' if train_return is None else f"{train_return:.2f}%"
            print(f"  #{fold.get('fold_number')}: {fold.get('parameters')} "
                  f"IS={train_text} OOS={fold.get('test_return_pct', 0.0):.2f}% "
                  f"efficiency={efficiency_text}")

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


def run_monte_carlo_analysis(args, data: pd.DataFrame, parameters: dict,
                            run_config=None):
    """Run Monte Carlo analysis.

    Args:
        args: Parsed command line arguments.
        data: OHLCV data with a DatetimeIndex.
        parameters: Fixed strategy parameters to simulate.
        run_config: Engine settings every simulated path is backtested under.

    Returns:
        The AnalysisResult produced by MonteCarloAnalyzer.
    """
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
        n_jobs=args.n_jobs,
        random_seed=args.seed,
        run_config=run_config
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




def save_results(result, output_file: str, provenance: dict = None) -> None:
    """Save analysis results to a JSON file.

    Args:
        result: Analysis result object to serialise.
        output_file: Path of the JSON file to write.
        provenance: Optional run provenance record (see
            ``niffler.utils.provenance.collect_provenance``), written under a
            top-level ``provenance`` key so a walk-forward or Monte Carlo verdict
            can be tied back to the code and data that produced it.

    Raises:
        OSError: If the file cannot be written.
        TypeError: If the result cannot be serialised to JSON.
    """
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

        if provenance is not None:
            output_data['provenance'] = provenance

        # Save to file
        with open(output_file, 'w') as f:
            safe_json_dump(output_data, f, indent=2, default=str)
        
        logging.info(f"Results saved to {output_file}")

    except (OSError, TypeError, ValueError) as e:
        # Never swallow this: main() reports the analysis as failed instead of
        # claiming success while no file was written.
        logging.error(f"Error saving results to {output_file}: {e}")
        raise


def main() -> int:
    """Run the requested analysis.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = create_parser()

    # Persisted defaults, folded in before parsing so a flag still wins.
    config = apply_config(parser, 'analyze')

    args = parser.parse_args()

    # Setup logging. --verbose stays a shorthand for the level, so the two
    # spellings cannot disagree.
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(level=log_level)
    report_config(config)

    try:
        # Load data
        data = load_data(args.data)

        # Load parameters. A fixed parameter set is only meaningful for Monte Carlo and
        # for the segmented in-sample mode; real walk-forward refits them per fold.
        if args.params or args.params_file:
            parameters = load_parameters(args)
        elif args.analysis == 'monte_carlo':
            raise ValueError(
                "--params or --params_file is required for --analysis monte_carlo"
            )
        elif args.mode == MODE_SEGMENTED_IN_SAMPLE:
            raise ValueError(
                f"--params or --params_file is required for --mode {MODE_SEGMENTED_IN_SAMPLE}"
            )
        else:
            parameters = None

        # Engine settings, shared by both analyses and by the per-fold
        # optimiser inside walk-forward.
        run_config = build_run_config(args)
        report_run_config(run_config)

        # Run analysis
        if args.analysis == 'walk_forward':
            result = run_walk_forward_analysis(args, data, parameters, run_config)
        elif args.analysis == 'monte_carlo':
            result = run_monte_carlo_analysis(args, data, parameters, run_config)
        else:
            raise ValueError(f"Unknown analysis type: {args.analysis}")
        
        # Save results if output file specified
        if args.output:
            save_results(result, args.output, provenance=collect_provenance(args.data))
        
        print(f"\nAnalysis completed successfully!")
        return 0

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
