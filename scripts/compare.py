"""Evaluate strategies across several datasets and compare them side by side.

Every other CLI in this repository takes exactly one ``--data`` file, which makes a
result an anecdote: a strategy that survives on one asset over one window has not been
shown to generalise. This script runs the same walk-forward evaluation over a set of
datasets and reports the cross-sectional distribution.

Two conventions make the comparison mean something:

**Excess over buy-and-hold, not raw return.** Absolute returns are not comparable across
assets - a long-only strategy on an asset that quadrupled will beat the same strategy on
an asset that went sideways, without carrying any information about the strategy. Every
row is therefore reported against a buy-and-hold benchmark computed over the *same*
out-of-sample span and charged the *same* commission and cost model.

**Non-overlapping folds by default.** ``analyze.py`` defaults to ``step_months=3`` with
``test_window_months=6``, so consecutive out-of-sample windows share half their bars and
N folds are not N independent observations. Here ``--step`` defaults to ``--test_window``,
because the whole point of this script is counting independent evidence.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from niffler.analysis import WalkForwardAnalyzer
from niffler.backtesting.run_config import RunConfig
from niffler.config.logging import setup_logging
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
    add_risk_manager_arguments,
    build_run_config,
    load_ohlcv_csv,
    report_run_config,
)
from scripts.config_file import add_config_arguments, apply_config, report_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldSchedule:
    """How the walk-forward folds are cut and fitted.

    Separated from the parsed command line so ``evaluate`` can be called by
    another script (``screen.py``) without fabricating an argparse namespace.
    The engine settings are *not* here: they live in ``RunConfig``, and having
    two objects that both carry commission is exactly the split this change
    exists to remove.

    Attributes:
        train_window_months: In-sample window each fold fits on.
        test_window_months: Out-of-sample window each fold is graded on.
        step_months: Months between folds. Defaulting it to the test window is
            what keeps folds non-overlapping, so ``None`` here means "match the
            test window" rather than "3".
        optimization_method: Per-fold optimizer.
        optimization_metric: Metric the per-fold optimizer selects by.
        anchored: Anchor every training window at the first bar.
        n_jobs: Parallel jobs for folds.
        clean: Run the preprocessing pipeline on each dataset first.
    """

    train_window_months: int = 12
    test_window_months: int = 6
    step_months: Optional[int] = None
    optimization_method: str = 'grid'
    optimization_metric: str = 'total_return'
    anchored: bool = False
    n_jobs: Optional[int] = None
    clean: bool = False

    @property
    def effective_step_months(self) -> int:
        """Months between folds, defaulting to non-overlapping windows."""
        return (self.step_months if self.step_months is not None
                else self.test_window_months)


def symbol_from_path(path: str) -> str:
    """Derive a display symbol from a data file name.

    Args:
        path: Path to the CSV file.

    Returns:
        The leading underscore-delimited token of the file stem, upper-cased -
        ``data/BTCUSDT_research.csv`` gives ``BTCUSDT``.
    """
    return Path(path).stem.split('_')[0].upper()


def paired_folds(result) -> List[tuple]:
    """Pair every out-of-sample fold with buy-and-hold over the *same* bars.

    This pairing is the whole point of the script: comparing a six-month fold return
    against a buy-and-hold figure computed over the entire multi-year span would flatter
    or damn a strategy purely through the length of the window.

    The benchmark is **not** recomputed here. ``BacktestEngine`` defaults to
    ``benchmark='buy_and_hold'`` and ``WalkForwardAnalyzer`` runs every fold through it,
    so each fold already carries a benchmark priced over its own bars, entered through
    ``_execute_buy_trade`` and charged the same commission and cost model as the strategy.
    Computing a second one here would be a parallel implementation that can drift from the
    one the rest of the platform reports - the same reason there is only one FIFO pairing
    routine and one equity-metrics module. (Verified identical to six decimal places
    across seven SPY folds before this was collapsed into the built-in field.)

    Args:
        result: The ``AnalysisResult`` from a walk-forward run.

    Returns:
        ``(strategy_return_pct, benchmark_return_pct)`` per fold that has a benchmark. A
        fold whose benchmark is None is dropped rather than compared against a fabricated
        baseline.
    """
    return [
        (fold.total_return_pct, fold.benchmark_return_pct)
        for fold in result.individual_results
        if fold.benchmark_return_pct is not None
    ]


def evaluate(data_path: str, strategy: str, run_config: RunConfig,
             schedule: FoldSchedule) -> Dict[str, Any]:
    """Walk-forward one strategy on one dataset and summarise it as a row.

    Args:
        data_path: Path to the OHLCV CSV.
        strategy: Registered strategy name.
        run_config: Engine settings every fold runs under.
        schedule: How the folds are cut and fitted.

    Returns:
        A row dict. On failure the row carries ``error`` and every metric is None,
        so a broken pair is visibly missing rather than silently absent.
    """
    symbol = symbol_from_path(data_path)
    row: Dict[str, Any] = {'symbol': symbol, 'strategy': strategy, 'error': None}

    try:
        data = load_ohlcv_csv(data_path, clean=schedule.clean)
        analyzer = WalkForwardAnalyzer(
            strategy_class=get_strategy_class(strategy),
            parameter_space=get_parameter_space(strategy),
            train_window_months=schedule.train_window_months,
            test_window_months=schedule.test_window_months,
            step_months=schedule.effective_step_months,
            optimization_method=schedule.optimization_method,
            optimization_metric=schedule.optimization_metric,
            anchored=schedule.anchored,
            n_jobs=schedule.n_jobs,
            run_config=run_config,
        )
        result = analyzer.analyze(data, symbol)
    except Exception as e:
        # One unusable pair must not abort a batch of twenty, but it is reported and
        # makes the run exit non-zero rather than quietly shrinking the table.
        logger.error("%s / %s failed: %s", symbol, strategy, e)
        row['error'] = str(e)
        return row

    stats = {**result.combined_metrics, **result.stability_metrics}
    pairs = paired_folds(result)

    row.update({
        'folds': int(stats.get('total_periods', 0)),
        'compared_folds': len(pairs),
        'failed_folds': result.failed_runs,
        'oos_sharpe': stats.get('combined_sharpe_ratio'),
        'median_efficiency': stats.get('median_efficiency_ratio'),
        'positive_fold_pct': stats.get('positive_return_pct'),
    })

    if pairs:
        strategy_returns = pd.Series([p[0] for p in pairs])
        benchmark_returns = pd.Series([p[1] for p in pairs])
        excess = strategy_returns - benchmark_returns
        row.update({
            'median_fold_pct': float(strategy_returns.median()),
            'median_bh_pct': float(benchmark_returns.median()),
            'median_excess_pct': float(excess.median()),
            # The headline robustness number: on how many independent out-of-sample
            # windows did the strategy actually beat simply holding the asset?
            'beat_bh_pct': float((excess > 0).mean() * 100.0),
        })
    else:
        row.update({'median_fold_pct': None, 'median_bh_pct': None,
                    'median_excess_pct': None, 'beat_bh_pct': None})
    return row


def render(rows: List[Dict[str, Any]]) -> None:
    """Print the comparison table, sorted by excess over buy-and-hold.

    Args:
        rows: Result rows from :func:`evaluate`.
    """
    ok = [r for r in rows if r['error'] is None]
    bad = [r for r in rows if r['error'] is not None]

    print("\n" + "=" * 100)
    print("CROSS-ASSET COMPARISON - median out-of-sample fold vs buy-and-hold")
    print("=" * 100)

    if ok:
        header = (f"{'ASSET':<10}{'STRATEGY':<12}{'FOLD %':>9}{'B&H %':>9}"
                  f"{'EXCESS':>9}{'BEAT%':>8}{'SHARPE':>8}{'MED EFF':>9}{'N':>4}")
        print(header)
        print("-" * len(header))

        def key(r):
            v = r['median_excess_pct']
            return v if v is not None else float('-inf')

        for r in sorted(ok, key=key, reverse=True):
            def fmt(v, nd=2):
                return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"
            print(f"{r['symbol']:<10}{r['strategy']:<12}"
                  f"{fmt(r['median_fold_pct']):>9}{fmt(r['median_bh_pct']):>9}"
                  f"{fmt(r['median_excess_pct']):>9}{fmt(r['beat_bh_pct'],1):>8}"
                  f"{fmt(r['oos_sharpe']):>8}{fmt(r['median_efficiency']):>9}"
                  f"{r['compared_folds']:>4}")

    if bad:
        print(f"\nFAILED ({len(bad)}):")
        for r in bad:
            print(f"  {r['symbol']} / {r['strategy']}: {r['error']}")

    print("""
COLUMNS: FOLD % and B&H % are medians over that pair's out-of-sample folds, each fold
compared against buy-and-hold on the SAME bars with the SAME costs. EXCESS is the median
of the per-fold differences. BEAT% is the share of folds where the strategy beat holding
- the robustness number, since it does not let one exceptional fold carry a verdict.

READ THIS: a positive excess on one asset is one observation. What matters is the pattern
down the column - a strategy that clears the benchmark on a single asset has not been
shown to generalise. No correction is applied for the parameter search behind each fold,
so these figures still flatter the strategy.""")
    print("=" * 100)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Compare strategies across several datasets (walk-forward)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Every registered strategy on every research dataset
  python scripts/compare.py --data data/*_research.csv

  # One strategy across assets, with costs
  python scripts/compare.py --data data/BTCUSDT_research.csv data/SPY_research.csv \\
    --strategy breakout --cost-model fixed --slippage-bps 5 --half-spread-bps 5
        """
    )
    parser.add_argument('--data', nargs='+', required=True,
                        help='One or more OHLCV CSV files')
    parser.add_argument('--strategy', nargs='+', default=None,
                        choices=get_available_strategies(),
                        help='Strategies to compare (default: all registered)')
    parser.add_argument('--output', default=None,
                        help='Write the rows to this JSON file')
    parser.add_argument('--clean', action='store_true',
                        help='Run the preprocessing pipeline on each dataset first')

    parser.add_argument('--capital', '--initial-capital', dest='initial_capital',
                        type=float, default=10000.0,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate (default: 0.001)')

    parser.add_argument('--train_window', type=int, default=12,
                        help='Training window in months (default: 12)')
    parser.add_argument('--test_window', type=int, default=6,
                        help='Test window in months (default: 6)')
    parser.add_argument('--step', type=int, default=None,
                        help='Months between folds (default: --test_window, '
                             'which keeps out-of-sample windows non-overlapping)')
    parser.add_argument('--anchored', action='store_true',
                        help='Anchor the training window at the start of the data')
    parser.add_argument('--optimization_method', default='grid',
                        choices=get_available_optimizers(),
                        help='Per-fold optimizer (default: grid)')
    parser.add_argument('--optimization_metric', default='total_return',
                        help='Per-fold selection metric (default: total_return)')
    parser.add_argument('--n_jobs', '--jobs', dest='n_jobs', type=int, default=None,
                        help='Parallel jobs for folds (default: auto)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    add_cost_model_arguments(parser)
    # No --benchmark here: every row of this table is an excess over
    # buy-and-hold on the same bars, so 'none' would empty the table rather
    # than configure it.
    add_engine_arguments(parser, benchmark=False)
    # Every asset in the table is measured under the same risk configuration,
    # which is the only way the column is a comparison.
    add_risk_manager_arguments(parser)

    # Persisted defaults, folded in before parsing so a flag still wins. Last,
    # so [risk] keys resolve against dests the parser has already declared.
    add_config_arguments(parser)
    config = apply_config(parser, 'compare')

    args = parser.parse_args()

    setup_logging(level=args.log_level)
    report_config(config)

    schedule = FoldSchedule(
        train_window_months=args.train_window,
        test_window_months=args.test_window,
        step_months=args.step,
        optimization_method=args.optimization_method,
        optimization_metric=args.optimization_metric,
        anchored=args.anchored,
        n_jobs=args.n_jobs,
        clean=args.clean,
    )
    step_months = schedule.effective_step_months

    strategies = args.strategy or get_available_strategies()

    missing = [p for p in args.data if not os.path.exists(p)]
    if missing:
        print(f"Error: data file(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    run_config = build_run_config(args)
    report_run_config(run_config)

    print(f"Comparing {len(strategies)} strategy(ies) across {len(args.data)} dataset(s) "
          f"= {len(strategies) * len(args.data)} walk-forward runs")
    print(f"Folds: train {args.train_window}m / test {args.test_window}m / "
          f"step {step_months}m"
          f"{' (OVERLAPPING)' if step_months < args.test_window else ''}")

    rows: List[Dict[str, Any]] = []
    for path in args.data:
        for strategy in strategies:
            print(f"  {symbol_from_path(path)} / {strategy} ...", flush=True)
            rows.append(evaluate(path, strategy, run_config, schedule))

    render(rows)

    if args.output:
        payload = {
            # One record per dataset: a single provenance block would hash one
            # file and imply it covered every row in the table.
            'provenance': {p: collect_provenance(p) for p in args.data},
            'settings': {
                'train_window_months': args.train_window,
                'test_window_months': args.test_window,
                'step_months': step_months,
                **run_config.to_metadata(),
            },
            'rows': rows,
        }
        with open(args.output, 'w') as f:
            safe_json_dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.output}")

    return 1 if any(r['error'] for r in rows) else 0


if __name__ == '__main__':
    sys.exit(main())
