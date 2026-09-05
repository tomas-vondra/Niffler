import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Running "python scripts/backtest.py" puts scripts/ on sys.path but not the
# repository root, so the root has to be added for "import niffler" to work.
# When imported as scripts.backtest the root is already importable.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niffler.backtesting import BacktestEngine
from niffler.backtesting.benchmark import BENCHMARK_BUY_AND_HOLD, BENCHMARK_CHOICES
from niffler.backtesting.significance import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_MIN_TRADES,
)
from niffler.strategies.registry import (
    create_strategy,
    get_available_strategies,
    get_strategy_parameter_names,
)
from niffler.risk import FixedRiskManager
from niffler.exporters import ExporterManager, get_available_exporters
from niffler.utils.provenance import collect_provenance
from niffler.config.logging import setup_logging
from scripts.common import add_cost_model_arguments, build_cost_model, load_ohlcv_csv, report_cost_model


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


# Convenience flags that map onto strategy constructor parameters. The flag name
# is only needed to phrase errors in the terms the user actually typed.
STRATEGY_PARAMETER_FLAGS = {
    'short_window': '--short-window',
    'long_window': '--long-window',
    'position_size': '--position-size',
}


def build_strategy_parameters(args) -> Dict[str, Any]:
    """Collect strategy parameters from --params and the convenience flags.

    An explicitly passed flag overrides the same key in ``--params``. A parameter
    the chosen strategy does not accept raises rather than being dropped, so
    ``--strategy rsi --short-window 5`` fails loudly instead of silently running
    an RSI backtest with default settings.

    Args:
        args: Parsed command line arguments.

    Returns:
        Keyword arguments for the strategy constructor.

    Raises:
        ValueError: If --params is not a JSON object, or a supplied parameter is
            not accepted by the chosen strategy.
    """
    parameters: Dict[str, Any] = {}

    if args.params:
        try:
            parsed = json.loads(args.params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --params: {e}") from e
        if not isinstance(parsed, dict):
            raise ValueError(
                f"--params must be a JSON object, got {type(parsed).__name__}"
            )
        parameters.update(parsed)

    for name in STRATEGY_PARAMETER_FLAGS:
        value = getattr(args, name, None)
        if value is not None:
            parameters[name] = value

    accepted = get_strategy_parameter_names(args.strategy)
    unknown = sorted(set(parameters) - accepted)
    if unknown:
        # Report the flag spelling where the parameter has one, since that is
        # what the user typed.
        rendered = ', '.join(STRATEGY_PARAMETER_FLAGS.get(name, name) for name in unknown)
        raise ValueError(
            f"Strategy '{args.strategy}' does not accept: {rendered}. "
            f"It accepts: {', '.join(sorted(accepted))}"
        )

    return parameters


# Convenience flags that map onto exporter constructor options, in the same shape
# as STRATEGY_PARAMETER_FLAGS: option name -> (argparse attribute, flag spelling).
# Each flag defaults to None so an explicitly passed one can be told apart from an
# unset one, and only the ones actually passed are forwarded - an option nobody
# asked for must not be broadcast to exporters that would reject it.
EXPORTER_OPTION_FLAGS = {
    'output_dir': ('csv_output_dir', '--csv-output-dir'),
    'host': ('es_host', '--es-host'),
    'port': ('es_port', '--es-port'),
    'index_prefix': ('es_index_prefix', '--es-index-prefix'),
}


def build_exporter_options(args) -> Dict[str, Any]:
    """Collect exporter options from --exporter-params and the convenience flags.

    An explicitly passed flag overrides the same key in ``--exporter-params``. The
    options are validated against the chosen exporters by
    ``ExporterManager.create_exporters_from_list``, which raises when none of them
    accepts an option, so ``--exporters console --csv-output-dir results/`` fails
    loudly instead of writing nothing anywhere.

    Args:
        args: Parsed command line arguments.

    Returns:
        Keyword arguments for the exporter constructors.

    Raises:
        ValueError: If --exporter-params is not a JSON object.
    """
    options: Dict[str, Any] = {}

    if args.exporter_params:
        try:
            parsed = json.loads(args.exporter_params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --exporter-params: {e}") from e
        if not isinstance(parsed, dict):
            raise ValueError(
                f"--exporter-params must be a JSON object, got {type(parsed).__name__}"
            )
        options.update(parsed)

    for option, (attribute, _flag) in EXPORTER_OPTION_FLAGS.items():
        value = getattr(args, attribute, None)
        if value is not None:
            options[option] = value

    return options


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Backtest trading strategies on historical data with optional risk management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest without risk management
  python backtest.py --data data/BTC.csv --strategy simple_ma

  # Any registered strategy is configured through --params
  python backtest.py --data data/BTC.csv --strategy rsi \\
    --params '{"rsi_period": 14, "oversold": 30, "overbought": 70}'

  python backtest.py --data data/BTC.csv --strategy breakout \\
    --params '{"entry_window": 20, "exit_window": 10}'

  # Backtest with fixed risk management
  python backtest.py --data data/BTC.csv --strategy simple_ma --risk-manager fixed \\
    --max-position-size 0.1 --stop-loss-pct 0.05 --max-positions 3
        """
    )
    
    # Required arguments
    parser.add_argument('--data', '-d', required=True,
                       help='Path to CSV data file')
    parser.add_argument('--strategy', '-s', default='simple_ma',
                       choices=get_available_strategies(),
                       help='Strategy to backtest (default: simple_ma)')

    # Strategy parameters.
    #
    # --params is the generic path and works for every registered strategy. The
    # named flags below are conveniences for parameters the strategies happen to
    # share; each defaults to None so an explicitly passed flag can be told apart
    # from an unset one. A flag the chosen strategy does not accept is an error,
    # never silently ignored - the same rule the cost-model flags follow.
    parser.add_argument('--params',
                       help='Strategy parameters as a JSON object, e.g. '
                            '\'{"rsi_period": 14, "oversold": 30}\'. Works for any '
                            'strategy; unknown names are reported with the accepted ones.')
    parser.add_argument('--short-window', type=int, default=None,
                       help='Short MA window (simple_ma; default: strategy default)')
    parser.add_argument('--long-window', type=int, default=None,
                       help='Long MA window (simple_ma; default: strategy default)')
    parser.add_argument('--position-size', type=float, default=None,
                       help='Position size as fraction of portfolio (default: 1.0)')
    
    # Backtest parameters
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital amount (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate per trade (default: 0.001)')

    # Transaction costs (slippage, spread, liquidity)
    add_cost_model_arguments(parser)

    # Benchmark and statistical significance
    benchmark_group = parser.add_argument_group('benchmark and significance')
    benchmark_group.add_argument(
        '--benchmark', choices=list(BENCHMARK_CHOICES), default=BENCHMARK_BUY_AND_HOLD,
        help=("Passive alternative the strategy is measured against: "
              "'buy_and_hold' (default) buys at the first executable bar and holds "
              "to the end, paying the same commission and the same cost model; "
              "'none' reports the strategy's numbers with nothing to compare them to")
    )
    benchmark_group.add_argument(
        '--min-trades-for-significance', type=int, default=DEFAULT_MIN_TRADES,
        help=(f"Round trips below which no significance verdict is rendered "
              f"(default: {DEFAULT_MIN_TRADES}). Below this the metrics are still "
              f"reported, labelled as not meaningful")
    )
    benchmark_group.add_argument(
        '--bootstrap-samples', type=int, default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=(f"Resamples for the bootstrap Sharpe confidence interval "
              f"(default: {DEFAULT_BOOTSTRAP_SAMPLES}); 0 skips it")
    )
    benchmark_group.add_argument(
        '--bootstrap-seed', type=int, default=DEFAULT_BOOTSTRAP_SEED,
        help=(f"Seed for that bootstrap, so the interval is reproducible "
              f"(default: {DEFAULT_BOOTSTRAP_SEED})")
    )


    # Output options
    #
    # The exporter choices come from niffler.exporters.registry, and --exporter-params
    # is the generic path that reaches any registered exporter's constructor. The named
    # flags below are conveniences for the options the shipped exporters happen to have;
    # each defaults to None so only explicitly passed ones are forwarded. An option no
    # chosen exporter accepts is an error, never silently ignored.
    available_exporters = ','.join(get_available_exporters())
    parser.add_argument('--exporters', type=str, default='console',
                       help=f'Comma-separated list of exporters to use: {available_exporters} (default: console)')
    parser.add_argument('--exporter-params',
                       help='Exporter options as a JSON object, e.g. \'{"output_dir": "results"}\'. '
                            'Works for any registered exporter; an option none of the chosen '
                            'exporters accepts is reported with the accepted ones.')
    parser.add_argument('--csv-output-dir', default=None,
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
        
        # Initialize strategy. Construction is generic: a strategy registered in
        # niffler.strategies.registry is usable here with no change to this file.
        strategy = create_strategy(
            args.strategy,
            build_strategy_parameters(args),
            risk_manager=risk_manager
        )


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
        
        # Transaction costs. Built before the engine so an unusable combination
        # of flags fails before any work is done.
        cost_model = build_cost_model(args)
        report_cost_model(cost_model)

        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=args.capital,
            commission=args.commission,
            min_order_value=args.min_order_value,
            cost_model=cost_model,
            benchmark=args.benchmark,
            min_trades_for_significance=args.min_trades_for_significance,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed
        )

        print(f"Benchmark: {args.benchmark}")

        print("Running backtest...")

        # Run backtest
        result = engine.run_backtest(strategy, data, symbol)

        # Setup exporters
        exporter_manager = ExporterManager()

        # Parse exporters parameter
        exporter_names = [name.strip().lower() for name in args.exporters.split(',')]

        # Create exporters. Construction is generic: an exporter registered in
        # niffler.exporters.registry is usable here with no change to this file, and
        # each one is handed the options its constructor declares.
        exporter_manager.create_exporters_from_list(
            exporter_names, **build_exporter_options(args)
        )

        if exporter_manager.get_exporter_count() == 0:
            print(f"Error: no usable exporters created from '{args.exporters}'", file=sys.stderr)
            return 1

        # Prepare strategy parameters for metadata (generic - gets from strategy object)
        strategy_params = strategy.parameters.copy()

        # Collect provenance once for the whole run: every exporter shares the record,
        # so the input file is hashed once no matter how many destinations are configured.
        provenance = collect_provenance(args.data)

        # Export results using all configured exporters
        export_result = exporter_manager.export_backtest_result(
            result=result,
            strategy_params=strategy_params,
            symbol=symbol,
            initial_capital=args.capital,
            commission=args.commission,
            provenance=provenance,
            cost_model=cost_model.description
        )

        return report_export_outcome(export_result, exporter_manager.get_exporter_names())

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())