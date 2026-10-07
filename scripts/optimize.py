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
import math
import sys
import pandas as pd
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

# Running "python scripts/optimize.py" puts scripts/ on sys.path but not the
# repository root, so the root has to be added for "import niffler" to work.
# When imported as scripts.optimize the root is already importable.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.config.logging import setup_logging
from niffler.exporters import ExporterManager
from niffler.exporters.run_record import DETAIL_TRIAL
from niffler.utils.provenance import collect_provenance
from niffler.optimization import deflated_sharpe as deflated_sharpe_analysis
from niffler.optimization import plateau as plateau_analysis
from niffler.optimization.base_optimizer import BaseOptimizer
from niffler.optimization.optimizer_factory import (
    create_optimizer,
    get_parameter_space,
    get_available_optimizers,
)
from niffler.optimization.parameter_space import ParameterSpace
from niffler.strategies.registry import (
    get_available_strategies,
    get_parameter_spec,
    get_strategy_class,
    get_strategy_parameter_names,
)
from niffler.utils.run_identity import RUN_KIND_OPTIMIZE
from scripts.common import (
    add_cost_model_arguments,
    add_engine_arguments,
    add_experiment_arguments,
    add_exporter_arguments,
    add_risk_manager_arguments,
    build_run_config,
    build_run_identity,
    configure_exporters,
    load_ohlcv_csv,
    report_export_outcome,
    report_run_config,
    report_run_identity,
    symbol_from_data_path,
    warn_if_holdout_data,
)
from scripts.config_file import (
    add_config_arguments,
    apply_config,
    report_config,
    typed_on_command_line,
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
                       choices=list(BaseOptimizer.METRICS_CONFIG),
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


#: The plateau figures an exported optimization summary carries. All None when
#: they could not be honestly computed.
_NO_PLATEAU_SUMMARY = {
    'plateau_metric': None,
    'grid_baseline': None,
    'grid_median': None,
    'fraction_beating_baseline': None,
    'plateau_retention': None,
}


def plateau_summary(results, args, selection: str) -> Dict[str, Any]:
    """Whole-grid figures for the exported summary, or None where unknowable.

    The same analysis :func:`report_plateau` prints, reduced to the numbers
    that say whether the winner is typical of its grid. It reads scores the
    run already produced, so a failure here must not cost the export.

    Args:
        results: The optimisation results, as returned by the optimizer.
        args: Parsed command line carrying the plateau flags.
        selection: One of the ``plateau.SELECTION_*`` constants.

    Returns:
        ``plateau_metric``, ``grid_baseline``, ``grid_median``,
        ``fraction_beating_baseline`` and ``plateau_retention``. A truncated
        result set reports **no** grid statistic: its survivors were selected
        by score, so a median computed from them flatters the grid.
    """
    if args.no_plateau:
        return dict(_NO_PLATEAU_SUMMARY)

    metric = args.plateau_metric or args.sort_by
    try:
        report = plateau_analysis.analyse_results(
            results, metric=metric, selection=selection,
            tolerance=args.plateau_tolerance,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Plateau figures left out of the export: {e}")
        return dict(_NO_PLATEAU_SUMMARY)

    distribution = report.distribution
    if not distribution.reliable:
        return {**_NO_PLATEAU_SUMMARY, 'plateau_metric': metric}

    return {
        'plateau_metric': metric,
        'grid_baseline': distribution.baseline,
        'grid_median': distribution.median,
        'fraction_beating_baseline': distribution.fraction_beating_baseline,
        'plateau_retention': report.plateau.retention if report.plateau else None,
    }


def add_deflated_sharpe_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the deflated-Sharpe flag to the optimizer's parser.

    Args:
        parser: Parser to extend.
    """
    group = parser.add_argument_group('deflated Sharpe')
    group.add_argument('--effective-trials', '--effective_trials', dest='effective_trials',
                       type=float, default=None,
                       help=('Number of INDEPENDENT trials to deflate the winner by '
                             '(default: every combination evaluated, which over-counts '
                             'because neighbouring parameter sets are near-duplicates, '
                             'and so over-corrects rather than under-corrects)'))


def analyse_deflated_sharpe(results, args, selection: str,
                            run_config) -> Optional[deflated_sharpe_analysis.DeflatedSharpe]:
    """Deflate the winner's Sharpe by the search that selected it.

    It reads equity curves the run already produced, so a failure here must
    cost neither the export nor the report.

    Args:
        results: The optimisation results, as returned by the optimizer.
        args: Parsed command line carrying ``effective_trials``.
        selection: One of the ``plateau.SELECTION_*`` constants.
        run_config: The engine settings the search ran under, which decide the
            annualisation the figures are shown in.

    Returns:
        The analysis, or None when it could not be run.
    """
    try:
        engine = BacktestEngine.from_config(run_config)
        periods_per_year = engine.resolve_periods_per_year(
            results[0].backtest_result.portfolio_values.index)
        return deflated_sharpe_analysis.analyse_results(
            results, selection,
            effective_trials=getattr(args, 'effective_trials', None),
            periods_per_year=periods_per_year,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(f"Deflated Sharpe left out: {e}")
        return None


def build_export_views(document: Dict[str, Any], args, selection: str,
                       truncated: bool, plateau: Dict[str, Any],
                       deflated: Optional[Dict[str, Any]] = None):
    """Shape an optimization for a document store: one summary, one row per trial.

    Args:
        document: The value ``BaseOptimizer.results_document`` returned.
        args: Parsed command line.
        selection: One of the ``plateau.SELECTION_*`` constants.
        truncated: Whether the optimizer discarded results to cap memory.
        plateau: The value :func:`plateau_summary` returned.
        deflated: The value ``deflated_sharpe.summary_fields`` returned; all
            None when omitted.

    Returns:
        ``(summary, details)`` for ``ExporterManager.create_run_record``.
    """
    trials = document.get('results') or []
    rows = []
    for rank, trial in enumerate(trials, start=1):
        metrics = trial.get('metrics') or {}
        score = metrics.get(args.sort_by)
        # The same four states the plateau surface uses: a combination that
        # never traded or produced no finite score is not a mediocre result.
        if metrics.get('total_trades') == 0:
            status = plateau_analysis.CELL_NO_TRADES
        elif isinstance(score, float) and not math.isfinite(score):
            status = plateau_analysis.CELL_NON_FINITE
        else:
            status = plateau_analysis.CELL_OK
        rows.append({
            'rank': rank,
            'parameters': trial.get('parameters'),
            **metrics,
            'status': status,
        })

    winner = trials[0] if trials else {}
    summary = {
        'method': args.method,
        'sort_by': args.sort_by,
        'n_trials': len(trials),
        'results_truncated': truncated,
        'selection': selection,
        'best_parameters': winner.get('parameters'),
        # The winner's own metrics, under the names a backtest summary uses, so
        # one table can hold both. `kind` says which is which.
        **(winner.get('metrics') or {}),
        **plateau,
        **(deflated if deflated is not None
           else deflated_sharpe_analysis.summary_fields(None)),
    }
    return summary, {DETAIL_TRIAL: rows}


def build_parameter_space(strategy: str, config) -> ParameterSpace:
    """Build the search space, letting the configuration file widen or move it.

    A strategy declares its space as a ``PARAMETER_SPEC`` code constant, and a
    plateau that runs into the edge of that window is the run asking for a
    wider one. ``[optimize.parameter_space.<strategy>]`` replaces the entry for
    each parameter it names and leaves the rest of the spec alone, so widening
    one window cannot silently drop the others.

    Args:
        strategy: Registered strategy name.
        config: The loaded configuration file, or None.

    Returns:
        The search space this optimisation will use.

    Raises:
        ValueError: If the override names a parameter the strategy does not
            accept, or describes one in a way ParameterSpace rejects.
    """
    overrides = (config.tables.get('parameter_space') if config else None) or {}
    wanted = overrides.get(strategy)
    if not wanted:
        return get_parameter_space(strategy)

    accepted = get_strategy_parameter_names(strategy)
    unknown = sorted(set(wanted) - accepted)
    if unknown:
        raise ValueError(
            f"[optimize.parameter_space.{strategy}] in {config.path} sets "
            f"{', '.join(unknown)}, which '{strategy}' does not accept. "
            f"It accepts: {', '.join(sorted(accepted))}"
        )

    spec = get_parameter_spec(strategy)
    for name, entry in wanted.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"[optimize.parameter_space.{strategy}.{name}] in {config.path} "
                f"must be a table of the form "
                f"{{ type = 'int', min = 5, max = 40, step = 1 }}"
            )
        spec[name] = dict(entry)

    logging.info(f"Parameter space from {config.path}: "
                 f"overriding {', '.join(sorted(wanted))}")
    return ParameterSpace(spec)


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
                       choices=get_available_strategies(),
                       help='Trading strategy to optimize')
    
    # Optimization method
    parser.add_argument('--method', default='grid',
                       choices=get_available_optimizers(),
                       help='Optimization method (default: grid)')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials for random search (default: 100)')
    
    # Optimization parameters
    parser.add_argument('--sort-by', default='total_return',
                       choices=list(BaseOptimizer.METRICS_CONFIG),
                       help=('Metric to sort top results by (default: total_return). '
                             'excess_return_pct ranks by return over buy-and-hold on the '
                             'same bars; over one dataset that is the same ORDER as '
                             'total_return, but the printed number tells you whether the '
                             'winner actually beat doing nothing'))
    
    # Backtest parameters
    parser.add_argument('--initial-capital', '--capital', dest='initial_capital',
                       type=float, default=10000.0,
                       help='Initial capital for backtesting (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate per trade (default: 0.001)')

    # Transaction costs (slippage, spread, liquidity)
    add_cost_model_arguments(parser)

    # Benchmark, annualisation, order floor and the significance gate. The
    # benchmark in particular is what makes --sort-by excess_return_pct mean
    # anything, so it has to be settable here.
    add_engine_arguments(parser)

    # Position sizing, stops and exposure caps. Optimising without them tunes a
    # system that is not the one the flag would trade.
    add_risk_manager_arguments(parser)

    # Data processing
    parser.add_argument('--clean', action='store_true',
                       help='Apply data preprocessing before optimization')
    
    # Performance options
    parser.add_argument('--jobs', '--n-jobs', '--n_jobs', dest='n_jobs', type=int, default=None,
                       help='Number of parallel jobs (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results')
    
    # Output options. The JSON file is this script's durable record and what a
    # later --params-file step reads, so json is in the default list.
    add_exporter_arguments(parser, default='console,json')
    parser.add_argument('--output', default=None,
                       help='Path for the JSON result file; implies the json exporter '
                            '(default: auto-generated)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top results to display (default: 10)')

    # Parameter plateau / surface analysis
    add_plateau_arguments(parser)
    add_deflated_sharpe_arguments(parser)

    # Logging
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    # Persisted defaults. [optimize.parameter_space.<strategy>] is structured
    # data rather than a flag value, so it is handed over separately.
    add_experiment_arguments(parser)
    add_config_arguments(parser)
    config = apply_config(parser, 'optimize', tables=('parameter_space',))

    args = parser.parse_args()
    if args.effective_trials is not None and not args.effective_trials >= 1:
        parser.error(f"--effective-trials must be at least 1, got {args.effective_trials:g}")
    experiment_typed = typed_on_command_line(parser, 'experiment')

    # Setup logging
    setup_logging(level=args.log_level)
    report_config(config)
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate data
        warn_if_holdout_data(args.data)
        logger.info(f"Loading data from {args.data}")
        data = load_and_validate_data(args.data, args.clean)
        
        # Get strategy class and parameter space
        strategy_class = get_strategy_class(args.strategy)
        parameter_space = build_parameter_space(args.strategy, config)

        # Engine settings. Parameters fitted without transaction costs are
        # fitted for a market nobody trades in, so the whole configuration is
        # reported up front.
        run_config = build_run_config(args)
        report_run_config(run_config)

        # An optimization starts a chain, so it has no parent: its experiment
        # is whatever the profile or --experiment names, or none.
        identity, identity_note = build_run_identity(
            args, RUN_KIND_OPTIMIZE, config=config, experiment_typed=experiment_typed
        )
        report_run_identity(identity, identity_note)

        # Exporters are created before the search: an exporter that cannot
        # export an optimization must not be discovered after the grid has run.
        exporter_manager = ExporterManager()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        configure_exporters(
            exporter_manager, args, RUN_KIND_OPTIMIZE,
            default_output=f"optimization_results_{args.strategy}_{args.method}_{timestamp}.json"
        )

        # Create optimizer
        optimizer = create_optimizer(
            method=args.method,
            strategy_class=strategy_class,
            parameter_space=parameter_space,
            data=data,
            sort_by=args.sort_by,
            n_jobs=args.n_jobs,
            run_config=run_config,
            max_results_in_memory=CLI_MAX_RESULTS_IN_MEMORY
        )
        
        # Run optimization
        logger.info(f"Starting {args.method} optimization for {args.strategy} strategy")
        logger.info(f"Sorting by: {args.sort_by}")
        logger.info(f"Initial capital: ${run_config.initial_capital:,.2f}")
        logger.info(f"Commission: {run_config.commission:.4f}")
        
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
        
        # Export the results, stamped with the run and with the code, data and
        # environment that produced them. Provenance is collected once here and
        # shared by every exporter.
        # How the evaluated combinations were chosen. Only this script knows
        # whether a partial grid is an unbiased sample or score-biased survivors.
        if optimizer.results_truncated:
            selection = plateau_analysis.SELECTION_TRUNCATED
        elif args.method == 'random':
            selection = plateau_analysis.SELECTION_SAMPLED
        else:
            selection = plateau_analysis.SELECTION_EXHAUSTIVE

        deflated = analyse_deflated_sharpe(results, args, selection, run_config)
        document = optimizer.results_document(results)
        summary, details = build_export_views(
            document, args, selection, bool(optimizer.results_truncated),
            plateau_summary(results, args, selection),
            deflated_sharpe_analysis.summary_fields(deflated)
        )
        record = exporter_manager.create_run_record(
            identity, args.strategy, document,
            provenance=collect_provenance(args.data),
            settings=run_config.to_metadata(),
            symbol=symbol_from_data_path(args.data),
            summary=summary,
            details=details,
        )
        exit_code = report_export_outcome(
            exporter_manager.export_run(record), what='Optimization')

        # Plateau analysis last, and non-fatally: it reads scores the run
        # already produced, so a reporting bug must not throw away an
        # optimisation that has just been saved to disk.
        if not args.no_plateau:
            try:
                report_plateau(results, args, selection)
            except Exception as e:
                logger.warning(f"Could not run plateau analysis: {e}")

        # After the plateau block: that one says how typical the winner is of
        # its grid, this one how far above a lucky winner it sits.
        if deflated is not None:
            print()
            print(deflated_sharpe_analysis.render_report(deflated))

        return exit_code
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())