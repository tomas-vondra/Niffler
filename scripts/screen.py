#!/usr/bin/env python3
"""Run the research pipeline as a funnel and stop at the first gate that fails.

The four scripts in this repository are meant to be run in an order -
backtest, optimize, walk-forward, cross-asset compare - and each step is only
worth taking if the previous one cleared a bar. In practice nothing enforces
that: it is entirely possible to spend an afternoon optimising a strategy that
made eight trades in five years, or to admire a walk-forward efficiency ratio
computed from a parameter surface that was an isolated spike. This script makes
the sequence and its gates explicit, and stops with a stated reason.

The stages and the question each one answers
--------------------------------------------
1. **Backtest** on the primary dataset - *did it trade enough to say anything?*
   The gate is the round-trip count against the same
   ``min_trades_for_significance`` the engine already refuses to render a
   verdict below. There is one such number and this is it.
2. **Optimize** - *is the winner a plateau or a spike, and did the rest of the
   grid beat doing nothing?* Both numbers come from
   :mod:`niffler.optimization.plateau`, which reads scores the optimisation
   already produced.
3. **Walk-forward** - *does the fitted edge survive out-of-sample?* The gate is
   the median walk-forward efficiency ratio.
4. **Compare across assets** - *does it generalise?* The gate is BEAT%, the
   share of out-of-sample folds that beat buy-and-hold on the same bars, pooled
   over every asset screened.
5. **Holdout** (only with ``--holdout-data``) - *does it hold on bars no
   decision has seen?* One backtest of the stage-2 winner on a file that starts
   after the primary dataset ends. Nothing is fitted on it. The gates are a
   minimum number of completed round trips and excess return over buy-and-hold.

The holdout is spent by looking at it
-------------------------------------
Stages 1-4 can be rerun as often as the research takes; every rerun is another
decision made with the research data in view. The holdout is the one number no
such decision has touched, and that is true exactly once: a strategy adjusted
after a holdout run and screened again has been fitted to the holdout. So the
path cannot come from ``niffler.toml`` - it has to be typed - and every holdout
run exports the file's hash, so the number of looks can be counted afterwards.
For the same reason ``--force`` does not reach it: a strategy that already
failed a gate has its verdict, and a look spent confirming it is a look lost.

Thresholds
----------
Every threshold is a flag, and the chosen value is printed whether or not the
gate fires - a gate you cannot see is not a gate. Three of the four defaults are
**judgment calls, not results**: there is no theory that says a median
efficiency ratio of 0.30 is the line between a real edge and a fitted one. They
are set where a reasonable person would want to look again, and they are meant
to be argued with. The exception is the trade-count gate, which reuses the
framework's existing ``DEFAULT_MIN_TRADES``. The holdout gates default to one
completed round trip and to zero excess over buy-and-hold: the least that counts
as having traded, and a break-even line, rather than judgments.

Exit codes
----------
``0`` every gate that ran passed. ``3`` a gate stopped the run - a normal,
expected outcome and emphatically not an error, which is why it is not ``1``.
``1`` is reserved for a genuine failure (unreadable data, a broken run) and
argparse owns ``2`` for a usage error, so a stop needs a code of its own.
``--force`` runs every stage regardless (except the holdout, which a failed
gate leaves unspent), but a run whose gates failed still
exits ``3``: the exit code reports the verdict, ``--force`` only controls how
much work is done before the verdict is printed.

This script implements no analysis of its own. Every number it gates on is
computed by the library or by ``compare.py``; it only decides whether to
continue.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.run_config import RunConfig
from niffler.config.logging import setup_logging
from niffler.optimization import plateau as plateau_analysis
from niffler.optimization.optimizer_factory import (
    create_optimizer,
    get_available_optimizers,
    get_parameter_space,
)
from niffler.strategies.registry import (
    create_strategy,
    get_available_strategies,
    get_strategy_class,
)
from niffler.exporters import ExporterManager
from niffler.utils.provenance import collect_provenance, provenance_fingerprint
from niffler.utils.run_identity import RUN_KIND_SCREEN
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
)
from scripts.compare import (
    FoldSchedule,
    comparison_details,
    evaluate,
    render,
    symbol_from_path,
)
from scripts.config_file import (
    add_config_arguments,
    apply_config,
    report_config,
    typed_on_command_line,
)
from scripts.optimize import CLI_MAX_RESULTS_IN_MEMORY

logger = logging.getLogger(__name__)

#: Every gate that ran cleared its threshold.
EXIT_OK = 0
#: The run could not be completed (unreadable data, a failed analysis).
EXIT_ERROR = 1
#: A gate stopped the run. Not an error: it is the script working.
EXIT_STOPPED = 3

#: Judgment call. Retention below this is what :mod:`plateau` already calls an
#: "isolated spike", so the gate is set exactly where the existing vocabulary
#: stops describing a plateau. Above it the surface is at least partly flat.
DEFAULT_MIN_RETENTION = plateau_analysis.ISOLATED_SPIKE_RETENTION

#: Judgment call. If fewer than a tenth of the searched grid beats buy-and-hold,
#: the winner is the right tail of a distribution that mostly loses to doing
#: nothing, and the best cell of such a grid is the most likely to be noise.
DEFAULT_MIN_GRID_BEAT = 0.10

#: Judgment call. The efficiency ratio is out-of-sample performance per bar over
#: in-sample performance per bar; 1.0 means the fitted edge survived intact and
#: 0.0 means none of it did. 0.30 asks for roughly a third of it to survive,
#: which is a low bar deliberately - the intent is to catch curve-fitting, not
#: to select a strategy.
DEFAULT_MIN_EFFICIENCY = 0.30

#: Judgment call, and the least arbitrary of the four: a long-only strategy that
#: beats simply holding the asset on fewer than half of its independent
#: out-of-sample windows has not beaten a coin toss against the alternative of
#: doing nothing.
DEFAULT_MIN_BEAT_PCT = 50.0

#: Stage names, used in the STOPPED line so a reader knows where the funnel
#: ended without reading the whole log.
STAGE_BACKTEST = 'backtest'
STAGE_OPTIMIZE = 'optimize'
STAGE_WALK_FORWARD = 'walk-forward'
STAGE_COMPARE = 'compare'
STAGE_HOLDOUT = 'holdout'

#: Not a judgment call in the way the other four are: zero is the line between
#: beating the passive alternative and losing to it on bars nothing was fitted
#: on. It is still a flag, because how much excess is enough is arguable.
DEFAULT_MIN_HOLDOUT_EXCESS = 0.0

#: A strategy that stays flat while the asset falls has positive excess over
#: buy-and-hold without having done anything. One completed round trip is the
#: least that separates "beat holding" from "did not trade".
DEFAULT_MIN_HOLDOUT_TRADES = 1

_SEPARATOR = '=' * 78


@dataclass(frozen=True)
class Gate:
    """One threshold, its measured value, and the verdict between them.

    Deliberately pure: it holds no data and runs nothing, so every gate's
    behaviour - including the awkward cases - is testable without a market.

    A ``None`` value is **not** a pass. ``retention``,
    ``fraction_beating_baseline`` and ``median_efficiency_ratio`` are all
    legitimately ``None`` in cases the library is careful to distinguish from
    zero (no scored neighbour, a score-biased grid, no fold with a defined
    ratio). Treating that as a pass would wave through exactly the runs there is
    no evidence about, and treating it as 0.0 would invent a measurement.

    Attributes:
        stage: Which stage produced the value.
        quantity: What was measured, phrased for the STOPPED line.
        value: The measurement, or None when it could not be computed.
        threshold: The minimum the value must reach.
        flag: The CLI flag that sets that threshold.
        precision: Decimal places used when rendering the value; 0 renders it
            as an integer count.
        unknown_reason: Why the value is None, when the producer said.
    """

    stage: str
    quantity: str
    value: Optional[float]
    threshold: float
    flag: str
    precision: int = 2
    unknown_reason: Optional[str] = None

    @property
    def passed(self) -> bool:
        """True only when a value exists and reaches the threshold."""
        return self.value is not None and self.value >= self.threshold

    def _format(self, value: float) -> str:
        return f"{value:.0f}" if self.precision == 0 else f"{value:.{self.precision}f}"

    def describe(self) -> str:
        """Render the gate's verdict as one line.

        Returns:
            The ``STOPPED at ...`` line when the gate fires, or the equivalent
            ``passed`` line when it does not. The threshold and the flag that
            set it appear either way.
        """
        threshold = self._format(self.threshold)

        if self.value is None:
            reason = f" ({self.unknown_reason})" if self.unknown_reason else ""
            return (f"STOPPED at {self.stage}: {self.quantity} is None{reason} - "
                    f"cannot be compared against {threshold} ({self.flag})")

        value = self._format(self.value)
        if self.passed:
            return (f"passed {self.stage}: {self.quantity} {value} >= {threshold} "
                    f"({self.flag})")
        return (f"STOPPED at {self.stage}: {self.quantity} {value} < {threshold} "
                f"({self.flag})")


@dataclass
class StageResult:
    """What one stage measured, and whether its gates cleared.

    Attributes:
        name: Stage name.
        gates: Gates evaluated by this stage, in the order they are reported.
        detail: Context lines printed above the gate verdicts.
        payload: JSON-safe record of the stage for ``--output``.
        skipped_reason: Set when the stage did not run at all, which is neither
            a pass nor a stop and is reported as itself.
    """

    name: str
    gates: List[Gate] = field(default_factory=list)
    detail: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    skipped_reason: Optional[str] = None

    @property
    def failed_gates(self) -> List[Gate]:
        """Gates that did not clear, in report order."""
        return [gate for gate in self.gates if not gate.passed]


def run_backtest_stage(data, symbol: str, strategy_name: str,
                       run_config: RunConfig) -> StageResult:
    """Stage 1: does the strategy trade often enough to say anything at all?

    The strategy runs on its registered defaults. That is the point of the
    stage: a strategy whose out-of-the-box behaviour on this asset is eight
    round trips in five years has nothing to optimise, and the optimiser would
    happily spend an hour finding the best of several hundred equally
    meaningless results.

    Args:
        data: The primary OHLCV dataset.
        symbol: Symbol identifier for reporting.
        strategy_name: Registered strategy name.
        run_config: Engine settings, whose ``min_trades_for_significance`` is
            the gate.

    Returns:
        The stage result, gated on the realised round-trip count.
    """
    engine = BacktestEngine.from_config(run_config)
    result = engine.run_backtest(create_strategy(strategy_name, {}), data, symbol)

    benchmark = result.benchmark_return_pct
    benchmark_text = 'n/a' if benchmark is None else f"{benchmark:.2f}%"

    stage = StageResult(name=STAGE_BACKTEST)
    stage.detail = [
        f"return {result.total_return_pct:.2f}%  vs buy-and-hold {benchmark_text}  "
        f"fills {result.total_trades}  round trips {result.round_trip_count}",
    ]
    stage.gates = [Gate(
        stage=STAGE_BACKTEST,
        quantity='round trips',
        value=float(result.round_trip_count),
        threshold=float(run_config.min_trades_for_significance),
        flag='--min-trades-for-significance',
        precision=0,
    )]
    stage.payload = {
        'symbol': symbol,
        'total_return_pct': result.total_return_pct,
        'benchmark_return_pct': benchmark,
        'total_trades': result.total_trades,
        'round_trips': result.round_trip_count,
        'is_sample_sufficient': result.is_sample_sufficient,
    }
    return stage


def run_optimize_stage(data, strategy_name: str, run_config: RunConfig,
                       method: str, metric: str, trials: int, seed: Optional[int],
                       n_jobs: Optional[int], min_retention: float,
                       min_grid_beat: float) -> StageResult:
    """Stage 2: is the winner a plateau, on a grid that beat doing nothing?

    Nothing here is recomputed. The plateau analysis reads the scores the
    optimisation already produced, exactly as ``optimize.py`` reports them, and
    the in-memory cap is raised to ``optimize.py``'s own ceiling - a
    score-truncated result set reports **no** distribution rather than one
    computed from its best-scoring survivors, and would gate on ``None``.

    Args:
        data: The primary OHLCV dataset.
        strategy_name: Registered strategy name.
        run_config: Engine settings every candidate backtest runs under.
        method: Optimizer name.
        metric: Metric the optimizer selects by, and the surface is built from.
        trials: Trials for random search.
        seed: Random-search seed. A screening verdict that cannot be reproduced
            is not a verdict, so it is passed through rather than left to
            whatever entropy the process happened to have.
        n_jobs: Parallel jobs.
        min_retention: Plateau-retention gate.
        min_grid_beat: Gate on the fraction of the grid beating the baseline.

    Returns:
        The stage result, gated on plateau retention and on the share of the
        grid that beat buy-and-hold.

    Raises:
        ValueError: If the optimisation produced no usable result.
    """
    optimizer = create_optimizer(
        method=method,
        strategy_class=get_strategy_class(strategy_name),
        parameter_space=get_parameter_space(strategy_name),
        data=data,
        sort_by=metric,
        n_jobs=n_jobs,
        run_config=run_config,
        max_results_in_memory=CLI_MAX_RESULTS_IN_MEMORY,
    )

    results = (optimizer.optimize(n_trials=trials, seed=seed) if method == 'random'
               else optimizer.optimize())
    if not results:
        raise ValueError(f"{method} optimisation produced no valid results")

    if optimizer.results_truncated:
        selection = plateau_analysis.SELECTION_TRUNCATED
    elif method == 'random':
        selection = plateau_analysis.SELECTION_SAMPLED
    else:
        selection = plateau_analysis.SELECTION_EXHAUSTIVE

    report = plateau_analysis.analyse_results(results, metric=metric,
                                              selection=selection)
    plateau = report.plateau
    distribution = report.distribution

    retention = plateau.retention if plateau is not None else None
    retention_reason = (plateau.retention_reason if plateau is not None
                        else 'no cell scored')
    verdict = plateau.verdict if plateau is not None else 'no verdict'

    beat_fraction = distribution.fraction_beating_baseline
    beat_reason = (distribution.unreliable_reason if not distribution.reliable
                   else 'no do-nothing baseline for this metric')

    stage = StageResult(name=STAGE_OPTIMIZE)
    stage.detail = [
        f"{len(results)} combination(s) evaluated by {method} on {metric} "
        f"({selection})",
        f"winner {results[0].parameters}  plateau verdict: {verdict}",
    ]
    stage.gates = [
        Gate(stage=STAGE_OPTIMIZE, quantity='plateau retention', value=retention,
             threshold=min_retention, flag='--min-retention',
             unknown_reason=retention_reason),
        Gate(stage=STAGE_OPTIMIZE, quantity='grid fraction beating buy-and-hold',
             value=beat_fraction, threshold=min_grid_beat, flag='--min-grid-beat',
             unknown_reason=beat_reason),
    ]
    stage.payload = {
        'method': method,
        'metric': metric,
        'selection': selection,
        'combinations': len(results),
        'winner_parameters': results[0].parameters,
        'plateau_verdict': verdict,
        'plateau_retention': retention,
        'grid_fraction_beating_baseline': beat_fraction,
        'baseline_label': distribution.baseline_label,
    }
    return stage


def run_walk_forward_stage(row: Dict[str, Any], min_efficiency: float) -> StageResult:
    """Stage 3: does the fitted edge survive on bars the optimiser never saw?

    The fold row is produced by ``compare.evaluate``, the same function stage 4
    uses, so the primary asset's walk-forward is run once and read twice.

    Args:
        row: A ``compare.evaluate`` row for the primary dataset.
        min_efficiency: Median efficiency-ratio gate.

    Returns:
        The stage result, gated on the median walk-forward efficiency ratio.

    Raises:
        ValueError: If the walk-forward run itself failed.
    """
    if row['error'] is not None:
        raise ValueError(f"walk-forward failed: {row['error']}")

    efficiency = row['median_efficiency']

    stage = StageResult(name=STAGE_WALK_FORWARD)
    stage.detail = [
        f"{row['folds']} fold(s), {row['failed_folds']} failed  "
        f"out-of-sample Sharpe {_number(row['oos_sharpe'])}  "
        f"positive folds {_number(row['positive_fold_pct'], 1)}%",
    ]
    stage.gates = [Gate(
        stage=STAGE_WALK_FORWARD,
        quantity='median efficiency',
        value=efficiency,
        threshold=min_efficiency,
        flag='--min-efficiency',
        unknown_reason='no fold had a defined efficiency ratio',
    )]
    stage.payload = {k: row[k] for k in
                     ('symbol', 'folds', 'compared_folds', 'failed_folds',
                      'oos_sharpe', 'median_efficiency', 'positive_fold_pct')}
    return stage


def pooled_beat_pct(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Share of every compared out-of-sample fold that beat buy-and-hold.

    Pooled over folds rather than averaged over assets, so an asset that only
    produced two comparable folds does not carry the same weight as one that
    produced twelve.

    Args:
        rows: ``compare.evaluate`` rows.

    Returns:
        The percentage, or None when no fold anywhere could be compared - which
        is an absence of evidence, not a score of zero.
    """
    usable = [r for r in rows
              if r['error'] is None and r.get('beat_bh_pct') is not None
              and r.get('compared_folds')]
    compared = sum(r['compared_folds'] for r in usable)
    if compared == 0:
        return None

    beats = sum(r['beat_bh_pct'] / 100.0 * r['compared_folds'] for r in usable)
    return beats / compared * 100.0


def run_compare_stage(rows: List[Dict[str, Any]], min_beat_pct: float) -> StageResult:
    """Stage 4: does the edge hold up on assets other than the one it was found on?

    Args:
        rows: ``compare.evaluate`` rows for every screened dataset.
        min_beat_pct: BEAT% gate.

    Returns:
        The stage result, gated on pooled BEAT%.
    """
    beat = pooled_beat_pct(rows)
    compared = sum(r.get('compared_folds') or 0 for r in rows if r['error'] is None)
    failures = [r for r in rows if r['error'] is not None]

    stage = StageResult(name=STAGE_COMPARE)
    stage.detail = [
        f"{len(rows)} dataset(s), {compared} comparable out-of-sample fold(s)"
        + (f", {len(failures)} dataset(s) failed" if failures else ""),
    ]
    stage.gates = [Gate(
        stage=STAGE_COMPARE,
        quantity='BEAT%',
        value=beat,
        threshold=min_beat_pct,
        flag='--min-beat-pct',
        precision=1,
        unknown_reason='no fold on any asset carried a benchmark to compare against',
    )]
    stage.payload = {'pooled_beat_pct': beat, 'compared_folds': compared, 'rows': rows}
    return stage


def check_holdout_follows_research(research, holdout) -> None:
    """Refuse a holdout that overlaps the data the strategy was fitted on.

    Args:
        research: The primary OHLCV dataset.
        holdout: The holdout OHLCV dataset.

    Raises:
        ValueError: If either frame is empty, or the holdout's first bar is not
            strictly after the research data's last bar.
    """
    if research.empty or holdout.empty:
        raise ValueError("holdout check needs at least one bar in both the "
                         "research and the holdout dataset")

    research_end = research.index[-1]
    holdout_start = holdout.index[0]
    if holdout_start <= research_end:
        raise ValueError(
            f"holdout data starts {holdout_start} but the research data runs to "
            f"{research_end}: the holdout must start strictly after it, or the "
            f"parameters were fitted on bars the holdout then scores them on")


def run_holdout_stage(holdout, symbol: str, strategy_name: str,
                      parameters: Dict[str, Any], run_config: RunConfig,
                      min_excess: float,
                      min_trades: int = DEFAULT_MIN_HOLDOUT_TRADES) -> StageResult:
    """Stage 5: does the fitted strategy beat holding on bars nothing has seen?

    One backtest, with the parameters stage 2 chose on the research data.
    Nothing is optimised here: a search over holdout bars would turn them into
    research bars.

    The significance assessment is the engine's own and is reported as it
    stands. Below ``min_trades_for_significance`` it refuses a verdict, and that
    refusal is printed rather than gated on or worked around: a short holdout
    often cannot support a p-value, which is a fact about the holdout.

    Args:
        holdout: The holdout OHLCV dataset.
        symbol: Symbol identifier for reporting.
        strategy_name: Registered strategy name.
        parameters: The winning parameters from the optimize stage.
        run_config: Engine settings, the same ones every earlier stage ran under.
        min_excess: Gate on excess return over buy-and-hold, in percentage points.
        min_trades: Gate on completed round trips. Reported first, because
            below it the excess is not a measurement of the strategy.

    Returns:
        The stage result, gated on round trips and on excess return over
        buy-and-hold.
    """
    engine = BacktestEngine.from_config(run_config)
    result = engine.run_backtest(create_strategy(strategy_name, parameters),
                                 holdout, symbol)

    stage = StageResult(name=STAGE_HOLDOUT)
    stage.detail = [
        f"{len(holdout)} bar(s) from {holdout.index[0]} to {holdout.index[-1]}, "
        f"parameters {parameters}",
        f"return {result.total_return_pct:.2f}%  vs buy-and-hold "
        f"{_number(result.benchmark_return_pct)}%  fills {result.total_trades}  "
        f"round trips {result.round_trip_count}",
    ]
    if result.significance_verdict:
        stage.detail.append(f"significance: {result.significance_verdict}")
    if result.is_sample_sufficient and result.p_value is not None:
        stage.detail.append(
            f"t-statistic {_number(result.t_statistic, 3)}  "
            f"p-value {result.p_value:.4f} (two-sided)")

    if result.round_trip_count < min_trades:
        stage.detail.append(
            f"the strategy completed {result.round_trip_count} round trip(s) on "
            f"the holdout: it did not trade enough for its excess over "
            f"buy-and-hold to show anything about it")

    stage.gates = [
        Gate(
            stage=STAGE_HOLDOUT,
            quantity='round trips',
            value=float(result.round_trip_count),
            threshold=float(min_trades),
            flag='--min-holdout-trades',
            precision=0,
        ),
        Gate(
            stage=STAGE_HOLDOUT,
            quantity='excess over buy-and-hold (pp)',
            value=result.excess_return_pct,
            threshold=min_excess,
            flag='--min-holdout-excess',
            unknown_reason='no benchmark could be established on the holdout',
        ),
    ]
    stage.payload = {
        'symbol': symbol,
        'parameters': parameters,
        'first_bar': str(holdout.index[0]),
        'last_bar': str(holdout.index[-1]),
        'bars': len(holdout),
        'total_return_pct': result.total_return_pct,
        'benchmark_return_pct': result.benchmark_return_pct,
        'excess_return_pct': result.excess_return_pct,
        'total_trades': result.total_trades,
        'round_trips': result.round_trip_count,
        'is_sample_sufficient': result.is_sample_sufficient,
        'p_value': result.p_value,
        'significance_verdict': result.significance_verdict,
    }
    return stage


def _number(value: Optional[float], precision: int = 2) -> str:
    """Render a metric that may legitimately be absent."""
    return 'n/a' if value is None else f"{value:.{precision}f}"


def report_stage(stage: StageResult) -> None:
    """Print one stage's context and the verdict of each of its gates."""
    print()
    print(f"--- {stage.name} ---")
    if stage.skipped_reason:
        print(f"SKIPPED: {stage.skipped_reason}")
        return
    for line in stage.detail:
        print(f"  {line}")
    for gate in stage.gates:
        print(f"  {gate.describe()}")


def build_parser() -> argparse.ArgumentParser:
    """Build the screening CLI.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description='Run the research pipeline as a funnel, stopping at the first '
                    'gate a strategy fails',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Screen one strategy on SPY, comparing across four assets at the last stage
  python scripts/screen.py --data data/SPY_research.csv --strategy breakout \\
    --compare-data data/QQQ_research.csv data/GLD_research.csv data/BTCUSDT_research.csv

  # Same, with realistic fills and a stricter out-of-sample bar
  python scripts/screen.py --data data/SPY_research.csv --strategy simple_ma \\
    --cost-model fixed --slippage-bps 5 --min-efficiency 0.5

  # Report every stage even after one fails
  python scripts/screen.py --data data/SPY_research.csv --strategy rsi --force

  # Finish with one look at bars no decision has seen. Run it once.
  python scripts/screen.py --data data/SPY_research.csv --strategy breakout \\
    --compare-data data/QQQ_research.csv --holdout-data data/SPY_holdout.csv

Exit codes: 0 = every gate passed, 3 = a gate stopped the run (a normal
outcome), 1 = the run failed.
        """
    )

    parser.add_argument('--data', required=True,
                        help='Primary OHLCV CSV: stages 1-3 run on this file')
    parser.add_argument('--strategy', required=True,
                        choices=get_available_strategies(),
                        help='Strategy to screen')
    parser.add_argument('--compare-data', nargs='+', default=None,
                        help='Further datasets for the cross-asset stage. Without '
                             'them that stage is skipped and said to be skipped: '
                             'one asset is not a cross-asset comparison')
    parser.add_argument('--holdout-data', default=None,
                        help='Holdout OHLCV CSV for a final stage: one backtest of '
                             'the stage-2 winner, nothing fitted. It must start '
                             'after --data ends. Looking at it spends it, so it '
                             'must be typed here and cannot come from niffler.toml, '
                             'and it only runs once every earlier gate has passed: '
                             '--force does not spend it on a strategy that failed')
    parser.add_argument('--clean', action='store_true',
                        help='Run the preprocessing pipeline on each dataset first')
    add_exporter_arguments(parser, default='console')
    parser.add_argument('--output', default=None,
                        help='Write the full screening record to this JSON file; '
                             'implies the json exporter')
    parser.add_argument('--force', action='store_true',
                        help='Run every stage even after a gate fails. The failure '
                             'is still reported and the run still exits 3')

    parser.add_argument('--capital', '--initial-capital', dest='initial_capital',
                        type=float, default=10000.0,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate (default: 0.001)')

    gates = parser.add_argument_group(
        'gate thresholds',
        'Judgment calls, not results. --min-trades-for-significance reuses the '
        'framework constant, --min-holdout-excess defaults to break-even and '
        '--min-holdout-trades to one round trip; '
        'the other three are set where a reasonable person would want to look '
        'again, and are meant to be argued with.')
    gates.add_argument('--min-retention', type=float, default=DEFAULT_MIN_RETENTION,
                       help=f"Stage 2: plateau retention the winner's neighbourhood "
                            f"must keep, where 1.0 is a flat plateau and 0.0 an "
                            f"isolated spike (default: {DEFAULT_MIN_RETENTION:g}, the "
                            f"value below which plateau.py already calls a surface an "
                            f"isolated spike)")
    gates.add_argument('--min-grid-beat', type=float, default=DEFAULT_MIN_GRID_BEAT,
                       help=f"Stage 2: fraction of the searched grid that must beat "
                            f"buy-and-hold (default: {DEFAULT_MIN_GRID_BEAT:g} - a "
                            f"judgment call)")
    gates.add_argument('--min-efficiency', type=float, default=DEFAULT_MIN_EFFICIENCY,
                       help=f"Stage 3: median walk-forward efficiency ratio, i.e. how "
                            f"much of the fitted edge survived out-of-sample "
                            f"(default: {DEFAULT_MIN_EFFICIENCY:g} - a judgment call)")
    gates.add_argument('--min-beat-pct', type=float, default=DEFAULT_MIN_BEAT_PCT,
                       help=f"Stage 4: percentage of out-of-sample folds, pooled over "
                            f"assets, that must beat buy-and-hold on the same bars "
                            f"(default: {DEFAULT_MIN_BEAT_PCT:g} - a judgment call, "
                            f"and the coin-toss line against doing nothing)")
    gates.add_argument('--min-holdout-excess', type=float,
                       default=DEFAULT_MIN_HOLDOUT_EXCESS,
                       help=f"Stage 5: return over buy-and-hold on the holdout, in "
                            f"percentage points (default: "
                            f"{DEFAULT_MIN_HOLDOUT_EXCESS:g} - the line between "
                            f"beating the passive alternative and losing to it)")
    gates.add_argument('--min-holdout-trades', type=int,
                       default=DEFAULT_MIN_HOLDOUT_TRADES,
                       help=f"Stage 5: completed round trips on the holdout "
                            f"(default: {DEFAULT_MIN_HOLDOUT_TRADES} - a strategy "
                            f"that never traded has an excess over buy-and-hold "
                            f"that shows nothing)")

    search = parser.add_argument_group('search and folds')
    search.add_argument('--optimization-method', '--optimization_method', default='grid',
                        choices=get_available_optimizers(),
                        help='Optimizer used in stage 2 and per fold (default: grid)')
    search.add_argument('--optimization-metric', '--optimization_metric', default='total_return',
                        help='Metric the optimizer selects by, and the metric the '
                             'plateau surface is built from (default: total_return)')
    search.add_argument('--trials', type=int, default=100,
                        help='Trials for random search (default: 100)')
    search.add_argument('--seed', type=int, default=None,
                        help='Random-search seed, so a screening verdict can be '
                             'reproduced (default: none)')
    search.add_argument('--train-window', '--train_window', type=int, default=12,
                        help='Training window in months (default: 12)')
    search.add_argument('--test-window', '--test_window', type=int, default=6,
                        help='Test window in months (default: 6)')
    search.add_argument('--step', type=int, default=None,
                        help='Months between folds (default: --test-window, which '
                             'keeps out-of-sample windows non-overlapping)')
    search.add_argument('--anchored', action='store_true',
                        help='Anchor every training window at the first bar')
    search.add_argument('--jobs', '--n-jobs', '--n_jobs', dest='n_jobs', type=int, default=None,
                        help='Parallel jobs (default: auto)')

    add_cost_model_arguments(parser)
    # No --benchmark: stages 2 and 4 both gate on beating buy-and-hold, so
    # 'none' would leave two of the four gates with nothing to measure.
    add_engine_arguments(parser, benchmark=False)
    # All four stages run under one risk configuration, so a funnel cannot pass
    # a strategy on numbers the configured risk layer would never have produced.
    add_risk_manager_arguments(parser)
    add_experiment_arguments(parser)

    parser.add_argument('--log-level', default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: WARNING, so the funnel is '
                             'readable)')

    add_config_arguments(parser)
    return parser


def main() -> int:
    """Screen one strategy through the funnel.

    Returns:
        ``EXIT_OK`` when every gate that ran passed, ``EXIT_STOPPED`` when a
        gate fired, ``EXIT_ERROR`` when the run could not be completed.
    """
    parser = build_parser()

    # Persisted defaults, folded in before parsing so a flag still wins.
    config = apply_config(parser, 'screen')

    args = parser.parse_args()
    experiment_typed = typed_on_command_line(parser, 'experiment')
    setup_logging(level=args.log_level)
    report_config(config)

    datasets = [args.data] + list(args.compare_data or [])
    required = datasets + ([args.holdout_data] if args.holdout_data else [])
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        print(f"Error: data file(s) not found: {', '.join(missing)}", file=sys.stderr)
        return EXIT_ERROR

    # A path in niffler.toml would be read by every run of the funnel, and a
    # holdout scored on every iteration is research data under another name.
    if args.holdout_data and not typed_on_command_line(parser, 'holdout_data'):
        print("Error: holdout_data is set in the configuration file. It must be "
              "typed as --holdout-data: read from a file it would be spent on "
              "every run.", file=sys.stderr)
        return EXIT_ERROR

    try:
        run_config = build_run_config(args)
        # The funnel is one run in one process; its stages are rows of it.
        identity, identity_note = build_run_identity(
            args, RUN_KIND_SCREEN, config=config, experiment_typed=experiment_typed
        )
        # Created before the funnel runs, so an unusable exporter stops the
        # screen now rather than after every stage has been computed.
        exporter_manager = ExporterManager()
        configure_exporters(exporter_manager, args, RUN_KIND_SCREEN)
    except (ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR

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

    symbol = symbol_from_path(args.data)
    print(_SEPARATOR)
    print(f"SCREENING {args.strategy} on {symbol} ({len(datasets)} dataset(s))")
    print(_SEPARATOR)
    report_run_config(run_config)
    report_run_identity(identity, identity_note)

    stages: List[StageResult] = []
    stopped_at: Optional[Gate] = None
    # Only the datasets the run actually opened. A provenance block for a file
    # a gate stopped the funnel before reaching would claim it was part of the
    # evidence.
    screened: List[str] = []
    # The walk-forward rows, one per dataset the funnel reached. Empty when a
    # gate stopped it before stage 3.
    walk_forward_rows: List[Dict[str, Any]] = []
    # Set only when the holdout backtest actually ran.
    holdout_stage: Optional[StageResult] = None

    def record(stage: StageResult) -> bool:
        """Report a stage and say whether the funnel should continue."""
        nonlocal stopped_at
        stages.append(stage)
        report_stage(stage)
        failed = stage.failed_gates
        if failed and stopped_at is None:
            stopped_at = failed[0]
        return not failed or args.force

    try:
        data = load_ohlcv_csv(args.data, clean=args.clean)
        screened.append(args.data)

        holdout = None
        if args.holdout_data:
            # Checked before any stage runs: reading the dates is not a look at
            # the result, and a mis-cut file should not cost a whole funnel.
            holdout = load_ohlcv_csv(args.holdout_data, clean=args.clean)
            check_holdout_follows_research(data, holdout)

        if record(run_backtest_stage(data, symbol, args.strategy, run_config)):
            optimize_stage = run_optimize_stage(
                data, args.strategy, run_config,
                method=args.optimization_method,
                metric=args.optimization_metric,
                trials=args.trials,
                seed=args.seed,
                n_jobs=args.n_jobs,
                min_retention=args.min_retention,
                min_grid_beat=args.min_grid_beat)
            if record(optimize_stage):

                rows = [evaluate(args.data, args.strategy, run_config, schedule)]
                walk_forward_rows = rows
                if record(run_walk_forward_stage(rows[0], args.min_efficiency)):
                    if args.compare_data:
                        print(f"\nwalking forward {len(args.compare_data)} further "
                              f"dataset(s)...")
                        for path in args.compare_data:
                            print(f"  {symbol_from_path(path)} ...", flush=True)
                            rows.append(
                                evaluate(path, args.strategy, run_config, schedule))
                            screened.append(path)
                        render(rows)
                        reached_holdout = record(
                            run_compare_stage(rows, args.min_beat_pct))
                    else:
                        skipped = StageResult(
                            name=STAGE_COMPARE,
                            skipped_reason='no --compare-data given; a single asset '
                                           'is one observation, not a cross-asset '
                                           'comparison')
                        stages.append(skipped)
                        report_stage(skipped)
                        reached_holdout = True

                    if reached_holdout and holdout is not None \
                            and stopped_at is not None:
                        # Only --force gets here with a failed gate behind it.
                        skipped = StageResult(
                            name=STAGE_HOLDOUT,
                            skipped_reason='an earlier gate failed; the holdout '
                                           'was not spent on a strategy that '
                                           'already has its verdict (--force '
                                           'does not override this)')
                        stages.append(skipped)
                        report_stage(skipped)
                    elif reached_holdout and holdout is not None:
                        holdout_stage = run_holdout_stage(
                            holdout, symbol_from_path(args.holdout_data),
                            args.strategy,
                            optimize_stage.payload['winner_parameters'],
                            run_config, args.min_holdout_excess,
                            args.min_holdout_trades)
                        screened.append(args.holdout_data)
                        record(holdout_stage)
                    elif reached_holdout:
                        skipped = StageResult(
                            name=STAGE_HOLDOUT,
                            skipped_reason='no --holdout-data given; every number '
                                           'above comes from data the search has '
                                           'seen')
                        stages.append(skipped)
                        report_stage(skipped)
    except Exception as e:
        # A CLI boundary: report and exit non-zero rather than raising a
        # traceback at a user who asked a research question.
        logger.error("Screening failed: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR

    print()
    print(_SEPARATOR)
    if stopped_at is None:
        print(f"PASSED all {len(stages)} stage(s). That is a reason to look harder, "
              f"not a result: no correction is applied for the parameter search "
              f"behind any of these numbers.")
        if holdout_stage is not None:
            print("The holdout is the one number here that search never saw, and "
                  "it is now spent: adjust the strategy and screen again, and the "
                  "same file is research data.")
    else:
        print(stopped_at.describe())
        if args.force:
            # Every later gate ran, so report all of them rather than letting
            # the first failure hide the ones behind it.
            later = [gate for stage in stages for gate in stage.failed_gates
                     if gate is not stopped_at]
            for gate in later:
                print(gate.describe())
            print("(--force ran the remaining stages anyway; the verdict stands.)")
    print(_SEPARATOR)

    # One record per dataset the run actually read: a single block would hash one
    # file and imply it covered the whole screen, and a block for a dataset the
    # funnel stopped short of would claim evidence that was never gathered.
    provenance = {path: collect_provenance(path) for path in screened}

    # The holdout is exported as a row of its own, carrying the hash of the file
    # it read, so "how many runs looked at this holdout" is a count over rows.
    detail_rows = list(walk_forward_rows)
    holdout_sha256 = None
    if holdout_stage is not None:
        holdout_sha256 = provenance_fingerprint(
            provenance.get(args.holdout_data))['data_sha256']
        holdout_stage.payload.update({
            'data_path': args.holdout_data,
            'data_sha256': holdout_sha256,
        })
        detail_rows.append({
            **holdout_stage.payload,
            'stage': STAGE_HOLDOUT,
            'strategy': args.strategy,
            'error': None,
        })
    record = exporter_manager.create_run_record(
        identity, args.strategy,
        {
            'requested_datasets': datasets,
            'holdout_data': args.holdout_data,
            'strategy': args.strategy,
            'settings': {
                'train_window_months': schedule.train_window_months,
                'test_window_months': schedule.test_window_months,
                'step_months': schedule.effective_step_months,
                'optimization_method': schedule.optimization_method,
                'optimization_metric': schedule.optimization_metric,
                **run_config.to_metadata(),
            },
            'thresholds': {
                'min_trades_for_significance': run_config.min_trades_for_significance,
                'min_retention': args.min_retention,
                'min_grid_beat': args.min_grid_beat,
                'min_efficiency': args.min_efficiency,
                'min_beat_pct': args.min_beat_pct,
                'min_holdout_excess': args.min_holdout_excess,
                'min_holdout_trades': args.min_holdout_trades,
            },
            'stages': [
                {
                    'name': stage.name,
                    'skipped_reason': stage.skipped_reason,
                    'gates': [
                        {
                            'quantity': gate.quantity,
                            'value': gate.value,
                            'threshold': gate.threshold,
                            'flag': gate.flag,
                            'passed': gate.passed,
                        }
                        for gate in stage.gates
                    ],
                    **stage.payload,
                }
                for stage in stages
            ],
            'stopped_at': stopped_at.describe() if stopped_at is not None else None,
        },
        provenance=provenance,
        settings=run_config.to_metadata(),
        symbol=symbol,
        summary={
            'passed': stopped_at is None,
            'stopped_at': stopped_at.describe() if stopped_at is not None else None,
            'stopped_at_quantity': stopped_at.quantity if stopped_at is not None else None,
            'forced': bool(args.force),
            'n_stages': len(stages),
            # None unless the holdout backtest ran: a requested holdout a gate
            # stopped short of was not looked at.
            'holdout_data_sha256': holdout_sha256,
            'holdout_excess_pct': (holdout_stage.payload['excess_return_pct']
                                   if holdout_stage is not None else None),
            # One entry per gate: which stage, what was measured, against what.
            'stages': [
                {'stage': stage.name, 'quantity': gate.quantity, 'value': gate.value,
                 'threshold': gate.threshold, 'passed': gate.passed}
                for stage in stages for gate in stage.gates
            ],
        },
        details=comparison_details(detail_rows, provenance),
    )
    # A screen whose record could not be written has failed, whatever the gates said.
    if report_export_outcome(exporter_manager.export_run(record), what='Screen'):
        return EXIT_ERROR

    return EXIT_OK if stopped_at is None else EXIT_STOPPED


if __name__ == '__main__':
    sys.exit(main())
