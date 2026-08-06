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
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# Running "python scripts/optimize.py" puts scripts/ on sys.path but not the
# repository root, so the root has to be added for "import niffler" to work.
# When imported as scripts.optimize the root is already importable.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niffler.config.logging import setup_logging
from niffler.utils.provenance import collect_provenance
from niffler.optimization import plateau as plateau_analysis
from niffler.optimization.optimizer_factory import (
    create_optimizer,
    get_strategy_class,
    get_parameter_space,
    get_available_optimizers,
    STRATEGY_CLASSES
)
from scripts.common import (
    add_cost_model_arguments,
    build_cost_model,
    load_ohlcv_csv,
    report_cost_model,
)


# Results the CLI retains before the optimizer starts discarding the
# worst-scoring half. The library default (1000) is smaller than the default
# simple_ma grid (1632 combinations), so an unmodified grid search used to hand
# back a sample biased towards high scores - fine for reporting a winner, fatal
# for reporting what the rest of the grid did. This ceiling keeps whole grids of
# a realistic size intact while still bounding memory for pathological ones; a
# run that exceeds it says so and its distribution statistics are withheld.
CLI_MAX_RESULTS_IN_MEMORY = 20000


def add_plateau_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the parameter-plateau flags to the optimizer's parser.

    Distribution statistics and the winner's plateau score print on every run:
    they are read off scores the optimizer already computed, they cost nothing,
    and they are the honest counterweight to printing only the best row. The
    heatmap and the CSV surface are opt-in because they are output volume
    rather than information the reader always needs, and the plateau-centre
    recommendation is opt-in because it is *not* the winner and must never be
    mistaken for one.

    Args:
        parser: Parser to extend.
    """
    group = parser.add_argument_group('plateau analysis')
    group.add_argument('--plateau-metric', default=None,
                       choices=['total_return', 'sharpe_ratio', 'max_drawdown',
                                'win_rate', 'total_trades', 'excess_return_pct'],
                       help='Metric the surface is built from (default: --sort-by)')
    group.add_argument('--plateau-tolerance', type=float, default=plateau_analysis.DEFAULT_TOLERANCE,
                       help=(f'Plateau band width as a fraction of the winner\'s edge over the '
                             f'grid median, in [0, 1] (default: '
                             f'{plateau_analysis.DEFAULT_TOLERANCE:g}, i.e. cells retaining at '
                             f'least 75%% of that edge)'))
    group.add_argument('--plateau-heatmap', action='store_true',
                       help='Print an ASCII heatmap of the parameter surface')
    group.add_argument('--plateau-csv', default=None,
                       help='Write the full surface to this CSV file for external plotting')
    group.add_argument('--plateau-centre', action='store_true',
                       help=('Also report the centre of the plateau around the winner. It is '
                             'reported beside the winner and never replaces it'))
    group.add_argument('--no-plateau', action='store_true',
                       help='Skip plateau analysis and whole-grid distribution statistics')


def report_plateau(results, args, selection: str) -> None:
    """Print the whole-grid distribution and plateau blocks, and export the surface.

    Args:
        results: The optimisation results, exactly as returned by the optimizer.
        args: Parsed command line carrying the plateau flags.
        selection: How the evaluated combinations were chosen, one of the
            ``plateau.SELECTION_*`` constants. Only the caller knows whether a
            partial grid is an unbiased sample or a score-biased survivor set.
    """
    metric = args.plateau_metric or args.sort_by
    report = plateau_analysis.analyse_results(
        results,
        metric=metric,
        selection=selection,
        tolerance=args.plateau_tolerance,
    )

    print()
    print(plateau_analysis.render_report(
        report,
        show_heatmap=args.plateau_heatmap,
        show_centre=args.plateau_centre,
    ))

    if args.plateau_csv:
        cells = plateau_analysis.write_surface_csv(
            report.surface, args.plateau_csv, report.plateau)
        print(f"Parameter surface ({cells} cells) written to: {args.plateau_csv}")


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
    if clean_data:
        logging.info("Applying data preprocessing...")

    data = load_ohlcv_csv(file_path, clean=clean_data)

    if data.isnull().any().any():
        logging.warning("Data contains NaN values - consider using --clean flag")

    logging.info(f"Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")
    return data


def main() -> int:
    """Main function for parameter optimization.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
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
                       choices=['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
                                'total_trades', 'excess_return_pct'],
                       help=('Metric to sort top results by (default: total_return). '
                             'excess_return_pct ranks by return over buy-and-hold on the '
                             'same bars; over one dataset that is the same ORDER as '
                             'total_return, but the printed number tells you whether the '
                             'winner actually beat doing nothing'))
    
    # Backtest parameters
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                       help='Initial capital for backtesting (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate per trade (default: 0.001)')

    # Transaction costs (slippage, spread, liquidity)
    add_cost_model_arguments(parser)
    
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

    # Parameter plateau / surface analysis
    add_plateau_arguments(parser)

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

        # Transaction costs. Parameters fitted without them are fitted for a
        # market nobody trades in, so the model is reported up front.
        cost_model = build_cost_model(args)
        report_cost_model(cost_model)
        
        # Create optimizer
        optimizer = create_optimizer(
            method=args.method,
            strategy_class=strategy_class,
            parameter_space=parameter_space,
            data=data,
            initial_capital=args.initial_capital,
            commission=args.commission,
            sort_by=args.sort_by,
            n_jobs=args.jobs,
            cost_model=cost_model,
            max_results_in_memory=CLI_MAX_RESULTS_IN_MEMORY
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
            
            if args.sort_by in ['total_return', 'max_drawdown', 'win_rate', 'excess_return_pct']:
                print(f"#{i} - {args.sort_by}: {sort_value:.2f}%")
            elif args.sort_by == 'sharpe_ratio':
                print(f"#{i} - {args.sort_by}: {sort_value:.3f}")
            else:
                print(f"#{i} - {args.sort_by}: {sort_value}")
            print(f"    Parameters: {result.parameters}")
            print(f"    Total Return: ${result.backtest_result.total_return:,.2f} ({result.backtest_result.total_return_pct:.2f}%)")
            # Printed for every sort order, not just --sort-by excess_return_pct:
            # sorting on total_return in a bull market selects whatever stays
            # invested longest, and this line is what makes that visible.
            benchmark_pct = getattr(result.backtest_result, 'benchmark_return_pct', None)
            if benchmark_pct is not None:
                excess = result.backtest_result.excess_return_pct
                print(f"    vs Buy-and-Hold: {benchmark_pct:.2f}% "
                      f"(excess {excess:+.2f} pp)")
            print(f"    Sharpe Ratio: {result.backtest_result.sharpe_ratio:.3f}")
            print(f"    Max Drawdown: {result.backtest_result.max_drawdown:.2f}%")
            print(f"    Total Trades: {result.backtest_result.total_trades}")
            print(f"    Win Rate: {result.backtest_result.win_rate:.1f}%")
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
            if metric_name in ['total_return', 'max_drawdown', 'win_rate', 'excess_return_pct']:
                formatted_value = f"{value:.2f}%"
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
        
        # Save results, stamped with the code/data/environment that produced them.
        # Collected once here rather than inside save_results, which is also called
        # from library code that has no idea what the input file was.
        provenance = collect_provenance(args.data)
        optimizer.save_results(results, args.output, provenance=provenance)
        print(f"Full results saved to: {args.output}")

        # Plateau analysis last, and non-fatally: it reads scores the run
        # already produced, so a reporting bug must not throw away an
        # optimisation that has just been saved to disk.
        if not args.no_plateau:
            if optimizer.results_truncated:
                selection = plateau_analysis.SELECTION_TRUNCATED
            elif args.method == 'random':
                selection = plateau_analysis.SELECTION_SAMPLED
            else:
                selection = plateau_analysis.SELECTION_EXHAUSTIVE

            try:
                report_plateau(results, args, selection)
            except Exception as e:
                logger.warning(f"Could not run plateau analysis: {e}")

        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())