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
    OVERLAP_TAG,
    POOLED_METRICS,
    describe_fold_independence,
)
from niffler.optimization.base_optimizer import BaseOptimizer
from niffler.optimization.optimizer_factory import (
    get_available_optimizers,
    get_parameter_space,
)
from niffler.strategies.registry import get_available_strategies, get_strategy_class
from niffler.exporters import ExporterManager
from niffler.exporters.run_record import DETAIL_FOLD, DETAIL_SIMULATION
from niffler.utils.provenance import collect_provenance
from niffler.utils.run_identity import RUN_KIND_MONTE_CARLO, RUN_KIND_WALK_FORWARD
from scripts.common import (
    PARAMS_TABLE,
    StrategyParameters,
    add_cost_model_arguments,
    add_engine_arguments,
    add_experiment_arguments,
    add_risk_manager_arguments,
    add_strategy_parameter_arguments,
    add_exporter_arguments,
    build_run_config,
    build_run_identity,
    configure_exporters,
    load_ohlcv_csv,
    report_export_outcome,
    report_run_config,
    report_run_identity,
    resolve_strategy_parameters,
    symbol_from_data_path,
    warn_if_holdout_data,
)
from scripts.config_file import (
    add_config_arguments,
    apply_config,
    report_config,
    typed_on_command_line,
)


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
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --strategy simple_ma --train-window 12 --test-window 6 --step 6

  # Re-run one fixed parameter set over consecutive in-sample slices (NOT a validation)
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis walk_forward --mode segmented_in_sample --strategy simple_ma --params '{"short_window": 10, "long_window": 30}'

  # Monte Carlo analysis with specific parameters
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params '{"short_window": 10, "long_window": 30}' --simulations 500

  # Load parameters from optimization results
  python scripts/analyze.py --data data/BTCUSDT_binance_1d.csv --analysis monte_carlo --strategy simple_ma --params-file optimization_results.json
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
        choices=[RUN_KIND_WALK_FORWARD, RUN_KIND_MONTE_CARLO],
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
    # A fixed set is required for --analysis monte_carlo and for
    # --mode segmented_in_sample; the two flags combine, --params winning.
    add_strategy_parameter_arguments(parser)
    add_experiment_arguments(parser)

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

    # Position sizing, stops and exposure caps. A Monte Carlo distribution is a
    # risk measurement; taking it with the risk layer off measures nothing.
    add_risk_manager_arguments(parser)

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
        '--train-window', '--train_window',
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
        '--optimization-method', '--optimization_method',
        choices=get_available_optimizers(),
        default='grid',
        help='Optimizer used on each walk-forward training window (default: grid)'
    )

    parser.add_argument(
        '--optimization-metric', '--optimization_metric',
        choices=list(BaseOptimizer.METRICS_CONFIG.keys()),
        default='total_return',
        help='Metric the per-fold optimizer selects parameters by (default: total_return)'
    )

    parser.add_argument(
        '--test-window', '--test_window',
        type=int,
        default=6,
        help='Test window in months for walk-forward analysis (default: 6)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=None,
        help='Months between folds (default: --test-window, '
             'which keeps out-of-sample windows non-overlapping)'
    )
    
    # Monte Carlo specific arguments
    parser.add_argument(
        '--simulations',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )
    
    parser.add_argument(
        '--bootstrap-pct', '--bootstrap_pct',
        type=float,
        default=0.8,
        help='Percentage of data to sample in each simulation (default: 0.8)'
    )
    
    
    parser.add_argument(
        '--block-size', '--block_size',
        type=int,
        default=30,
        help='Block size in days for block bootstrap sampling (default: 30)'
    )
    
    parser.add_argument(
        '--seed', '--random-seed', '--random_seed',
        dest='seed',
        type=int,
        help='Random seed for reproducible Monte Carlo results'
    )
    
    parser.add_argument(
        '--jobs', '--n-jobs', '--n_jobs',
        dest='n_jobs',
        type=int,
        help='Number of parallel jobs for analysis (default: auto-detect)'
    )
    
    # Output arguments
    add_exporter_arguments(parser, default='console')
    parser.add_argument(
        '--output',
        help='Path for the JSON result file; implies the json exporter'
    )
    
    parser.add_argument(
        '--symbol',
        default=None,
        help='Symbol identifier for the data (default: extracted from the file name)'
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


def load_parameters(args, config=None) -> StrategyParameters:
    """Resolve strategy parameters and the run that produced them.

    The work is :func:`scripts.common.resolve_strategy_parameters`, shared with
    ``backtest.py``.

    Args:
        args: Parsed command line arguments.
        config: The value ``apply_config`` returned, or None.

    Returns:
        The resolved parameters, their parent run and whether any were supplied.
    """
    return resolve_strategy_parameters(args, args.strategy, config=config)


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
                f"--mode {MODE_SEGMENTED_IN_SAMPLE} requires --params or --params-file"
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
    step_origin = '' if args.step is not None else ' (default: equal to the test window)'
    print(f"Step Size: {analyzer.step_months} months{step_origin}")

    # Overlapping folds share bars, so a number that counts folds must not be
    # printed looking the same as one that counts independent observations.
    independence = describe_fold_independence(result.combined_metrics)
    overlapping = independence['folds_independent'] is False
    fold_tag = f"  {OVERLAP_TAG}" if overlapping else ''
    if overlapping:
        print(f"\n{'!' * 66}")
        print(f"FOLDS OVERLAP: {independence['oos_overlap_pct']:.1f}% of pooled "
              f"out-of-sample bars are repeats.")
        print("The folds are NOT independent observations. Every line tagged")
        print(f"{OVERLAP_TAG} counts a fold as one sample.")
        print("Leave --step unset (or >= --test-window) for independent folds.")
        print('!' * 66)

    print(f"\nCombined Metrics:")
    for metric, value in result.combined_metrics.items():
        tag = '' if metric in POOLED_METRICS else fold_tag
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}{tag}")
        else:
            print(f"  {metric}: {value}{tag}")

    print(f"\nStability Metrics:")
    for metric, value in result.stability_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}{fold_tag}")
        else:
            print(f"  {metric}: {value}{fold_tag}")

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




def build_results_document(result) -> dict:
    """Render an analysis result as the document every exporter is handed.

    It carries neither a ``run`` nor a ``provenance`` block: those are attached
    once, by ``ExporterManager.create_run_record``.

    Args:
        result: Analysis result object to render.

    Returns:
        A dict holding the summary and one record per fold or simulation.
    """
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
    if result.analysis_type == RUN_KIND_WALK_FORWARD:
        output_data['period_results'] = df.to_dict('records')
        output_data['fold_independence'] = describe_fold_independence(result.combined_metrics)
    else:  # monte_carlo
        output_data['simulation_results'] = df.to_dict('records')

    # Add metadata if available
    if result.metadata:
        output_data['metadata'] = result.metadata

    return output_data


#: The parts of the result document that describe the run as a whole.
_SUMMARY_FIELDS = (
    'analysis_type', 'strategy_name', 'analysis_start_date', 'analysis_end_date',
    'n_periods', 'combined_metrics', 'stability_metrics', 'analysis_parameters',
    'performance_consistency',
)

#: The parts of ``fold_independence`` a walk-forward summary carries at top level.
FOLD_INDEPENDENCE_SUMMARY_FIELDS = ('folds_independent', 'oos_overlap_pct')


def build_export_views(result, document: dict):
    """Shape an analysis for a document store: one summary, one row per fold or simulation.

    A walk-forward keeps its richest per-fold record on ``result.metadata['folds']``
    (in-sample against out-of-sample return, the efficiency ratio, the fitted
    parameters); that is what a fold row carries when it exists, because the
    in-sample/out-of-sample pair is the whole point of looking at a fold.

    Args:
        result: The analysis result.
        document: The value :func:`build_results_document` returned.

    Returns:
        ``(summary, details)`` for ``ExporterManager.create_run_record``.
    """
    summary = {name: document.get(name) for name in _SUMMARY_FIELDS}
    summary['attempted_runs'] = getattr(result, 'attempted_runs', None)
    summary['failed_runs'] = getattr(result, 'failed_runs', None)
    summary['failure_rate'] = getattr(result, 'failure_rate', None)

    if document.get('analysis_type') == RUN_KIND_WALK_FORWARD:
        # Flat and explicitly mapped, so a leaderboard can filter on them. An
        # unknown overlap stays None: null is not the same claim as false.
        independence = document.get('fold_independence') or {}
        for name in FOLD_INDEPENDENCE_SUMMARY_FIELDS:
            summary[name] = independence.get(name)
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        folds = metadata.get('folds')
        periods = list(document.get('period_results') or [])
        if not (isinstance(folds, list) and folds):
            return summary, {DETAIL_FOLD: periods}
        if len(folds) != len(periods):
            return summary, {DETAIL_FOLD: list(folds)}
        # One shape, not two: the period row has Sharpe and drawdown, the fold
        # record has the in-sample/out-of-sample pair. A fold row needs both.
        return summary, {DETAIL_FOLD: [
            {**period, **fold} for period, fold in zip(periods, folds)]}

    return summary, {DETAIL_SIMULATION: list(document.get('simulation_results') or [])}


def main() -> int:
    """Run the requested analysis.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = create_parser()

    # Persisted defaults, folded in before parsing so a flag still wins.
    config = apply_config(parser, 'analyze', tables=(PARAMS_TABLE,))

    args = parser.parse_args()
    experiment_typed = typed_on_command_line(parser, 'experiment')

    # Setup logging. --verbose stays a shorthand for the level, so the two
    # spellings cannot disagree.
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(level=log_level)
    report_config(config)

    # 'UNKNOWN' used to be the default, which is what every exported analysis
    # was then filed under. Derived from the file name, as backtest.py does.
    if args.symbol is None:
        args.symbol = symbol_from_data_path(args.data)

    try:
        # Load data
        warn_if_holdout_data(args.data)
        data = load_data(args.data)

        # Load parameters. A fixed parameter set is only meaningful for Monte Carlo and
        # for the segmented in-sample mode; real walk-forward refits them per fold.
        resolved = load_parameters(args, config)
        uses_fixed_parameters = (args.analysis == RUN_KIND_MONTE_CARLO
                                 or args.mode == MODE_SEGMENTED_IN_SAMPLE)
        if resolved.supplied:
            parameters = resolved.values
            logging.info(f"Strategy parameters: {parameters}")
        elif args.analysis == RUN_KIND_MONTE_CARLO:
            raise ValueError(
                "--params or --params-file is required for --analysis monte_carlo"
            )
        elif args.mode == MODE_SEGMENTED_IN_SAMPLE:
            raise ValueError(
                f"--params or --params-file is required for --mode {MODE_SEGMENTED_IN_SAMPLE}"
            )
        else:
            parameters = None

        # Engine settings, shared by both analyses and by the per-fold
        # optimiser inside walk-forward.
        run_config = build_run_config(args)
        report_run_config(run_config)

        # Real walk-forward refits its parameters per fold, so a params file
        # given alongside it fed nothing and is not this run's parent.
        identity, identity_note = build_run_identity(
            args,
            # --analysis takes the run kinds themselves, so it is the kind.
            args.analysis,
            config=config,
            parent=resolved.parent if uses_fixed_parameters else None,
            experiment_typed=experiment_typed,
        )
        report_run_identity(identity, identity_note)

        # Exporters are created before the analysis: one that cannot export
        # this kind of run must not be discovered after the folds have run.
        exporter_manager = ExporterManager()
        configure_exporters(exporter_manager, args, identity.kind)

        # Run analysis
        if args.analysis == RUN_KIND_WALK_FORWARD:
            result = run_walk_forward_analysis(args, data, parameters, run_config)
        elif args.analysis == RUN_KIND_MONTE_CARLO:
            result = run_monte_carlo_analysis(args, data, parameters, run_config)
        else:
            raise ValueError(f"Unknown analysis type: {args.analysis}")
        
        # Export the result through every configured exporter.
        document = build_results_document(result)
        summary, details = build_export_views(result, document)
        record = exporter_manager.create_run_record(
            identity, args.strategy, document,
            provenance=collect_provenance(args.data),
            settings=run_config.to_metadata(),
            symbol=args.symbol,
            summary=summary,
            details=details,
        )
        # A failed export is a failed run: it must not be reported as a
        # successful analysis while no file was written.
        if report_export_outcome(exporter_manager.export_run(record), what='Analysis'):
            return 1
        
        print(f"\nAnalysis completed successfully!")
        return 0

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
