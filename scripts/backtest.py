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
from niffler.strategies.registry import (
    create_strategy,
    get_available_strategies,
)
from niffler.exporters import ExporterManager
from niffler.utils.provenance import collect_provenance
from niffler.utils.run_identity import RUN_KIND_BACKTEST
from niffler.config.logging import setup_logging
from scripts.common import (
    EXPORTER_OPTION_FLAGS,  # noqa: F401 - re-exported for callers and tests
    PARAMS_TABLE,
    StrategyParameters,
    add_cost_model_arguments,
    add_engine_arguments,
    add_experiment_arguments,
    add_exporter_arguments,
    add_risk_manager_arguments,
    add_strategy_parameter_arguments,
    build_exporter_options,  # noqa: F401 - re-exported for callers and tests
    build_run_config,
    build_run_identity,
    configure_exporters,
    load_ohlcv_csv,
    report_cost_model,
    report_export_outcome,
    report_run_identity,
    resolve_strategy_parameters,
)
from scripts.config_file import (
    add_config_arguments,
    apply_config,
    report_config,
    typed_on_command_line,
)


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


# Convenience flags that map onto strategy constructor parameters. The flag name
# is only needed to phrase errors in the terms the user actually typed.
STRATEGY_PARAMETER_FLAGS = {
    'short_window': '--short-window',
    'long_window': '--long-window',
    'position_size': '--position-size',
}


def build_strategy_parameters(args, config=None) -> StrategyParameters:
    """Resolve strategy parameters, adding this script's convenience flags.

    The work is :func:`scripts.common.resolve_strategy_parameters`, shared with
    ``analyze.py``; only the convenience flags are this script's own. A parameter
    the chosen strategy does not accept raises rather than being dropped, so
    ``--strategy rsi --short-window 5`` fails loudly instead of silently running
    an RSI backtest with default settings.

    Args:
        args: Parsed command line arguments.
        config: The value ``apply_config`` returned, or None.

    Returns:
        The resolved parameters and the run that produced them, if any.

    Raises:
        ValueError: If a source is malformed, or a supplied parameter is not
            accepted by the chosen strategy.
    """
    return resolve_strategy_parameters(
        args, args.strategy, config=config, flags=STRATEGY_PARAMETER_FLAGS
    )


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
    add_strategy_parameter_arguments(parser)
    add_experiment_arguments(parser)
    parser.add_argument('--short-window', type=int, default=None,
                       help='Short MA window (simple_ma; default: strategy default)')
    parser.add_argument('--long-window', type=int, default=None,
                       help='Long MA window (simple_ma; default: strategy default)')
    parser.add_argument('--position-size', type=float, default=None,
                       help='Position size as fraction of portfolio (default: 1.0)')
    
    # Backtest parameters. --capital keeps its established spelling; the dest
    # is what build_run_config reads, and every script spells it the same way.
    parser.add_argument('--capital', '--initial-capital', dest='initial_capital',
                       type=float, default=10000.0,
                       help='Initial capital amount (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate per trade (default: 0.001)')

    # Transaction costs (slippage, spread, liquidity)
    add_cost_model_arguments(parser)

    # Benchmark, annualisation, order floor and statistical significance. This
    # script is the only one that prints the bootstrap Sharpe interval, so it
    # is the only one that gets the flags for it.
    add_engine_arguments(parser, bootstrap=True)


    # Output options: --exporters and friends, identical in every script.
    add_exporter_arguments(parser, default='console')
    parser.add_argument('--symbol', default=None,
                       help='Symbol identifier for the data (default: extracted from filename)')

    # Data processing options
    parser.add_argument('--clean', action='store_true',
                       help='Apply data cleaning pipeline to the CSV file before backtesting')
    
    # Risk management options, shared with optimize/analyze/compare/screen so a
    # strategy is sized and stopped the same way wherever it is measured.
    add_risk_manager_arguments(parser)


    # Logging options
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')

    # Persisted defaults, folded in after every flag is declared and before
    # parsing, so a flag typed on the command line still wins.
    add_config_arguments(parser)
    config = apply_config(parser, 'backtest', tables=(PARAMS_TABLE,))

    args = parser.parse_args()
    experiment_typed = typed_on_command_line(parser, 'experiment')

    # Configure logging
    setup_logging(level=args.log_level)
    report_config(config)
    
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

        # Engine settings, built once and before anything else, so an unusable
        # combination of flags fails before any work is done. The risk manager
        # is one of those settings now, so it is taken from here rather than
        # constructed a second time - the engine rejects two managers.
        run_config = build_run_config(args)

        risk_manager = run_config.risk_manager
        if risk_manager is not None:
            print(f"Risk Manager: {risk_manager.get_risk_metrics()['risk_management_type']}")
        
        # Initialize strategy. Construction is generic: a strategy registered in
        # niffler.strategies.registry is usable here with no change to this file.
        strategy_parameters = build_strategy_parameters(args, config)
        strategy = create_strategy(
            args.strategy,
            strategy_parameters.values,
            risk_manager=risk_manager
        )

        # Minted once, before the backtest runs, so an experiment mismatch
        # stops the run instead of surfacing after the work is done.
        identity, identity_note = build_run_identity(
            args, RUN_KIND_BACKTEST, config=config,
            parent=strategy_parameters.parent, experiment_typed=experiment_typed
        )
        report_run_identity(identity, identity_note)

        # Exporters are created before the run for the same reason: an option
        # none of them accepts, or one that cannot export a backtest, is a
        # mistake in the command and must not cost a finished backtest.
        # Construction is generic - a registered exporter needs no change here.
        exporter_manager = ExporterManager()
        configure_exporters(exporter_manager, args, RUN_KIND_BACKTEST)

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
        
        report_cost_model(run_config.cost_model)

        engine = BacktestEngine.from_config(run_config)

        print(f"Benchmark: {run_config.benchmark}")

        print("Running backtest...")

        # Run backtest
        result = engine.run_backtest(strategy, data, symbol)

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
            initial_capital=run_config.initial_capital,
            commission=run_config.commission,
            provenance=provenance,
            cost_model=engine.cost_model.description,
            risk_manager=run_config.to_metadata()['risk_manager'],
            identity=identity,
            strategy_key=args.strategy
        )

        return report_export_outcome(export_result, exporter_manager.get_exporter_names())

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())