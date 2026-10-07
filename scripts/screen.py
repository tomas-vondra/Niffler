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
2. **Optimize** - *is the winner a plateau or a spike, did the rest of the
   grid beat doing nothing, and is the winner above what a search this size
   finds by luck?* The first two numbers come from
   :mod:`niffler.optimization.plateau` and the third from
   :mod:`niffler.optimization.deflated_sharpe`; all three read results the
   optimisation already produced.
3. **Walk-forward** - *does the fitted edge survive out-of-sample?* The gate is
   the median walk-forward efficiency ratio.
4. **Compare across assets** - *does it generalise?* The gate is BEAT%, the
   share of out-of-sample folds that beat buy-and-hold on the same bars, pooled
   over every asset screened.
5. **Holdout** (only with ``--holdout-data``) - *does it hold on bars no
   decision has seen?* One backtest of the stage-2 winner per holdout file,
   each starting after every research dataset in the run ends. Nothing is
   fitted on any of them. The gates are a minimum number of completed round
   trips and excess return over buy-and-hold; with several files both are
   pooled, and each file is still reported and exported on its own.

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
gate fires - a gate you cannot see is not a gate. Four of the five research-stage
defaults are **judgment calls, not results**: there is no theory that says a
median efficiency ratio of 0.30 is the line between a real edge and a fitted
one. They are set where a reasonable person would want to look again, and they
are meant to be argued with. The exception is the trade-count gate, which reuses
the framework's existing ``DEFAULT_MIN_TRADES``. The holdout gates default to one
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
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.run_config import RunConfig
from niffler.config.logging import setup_logging
from niffler.optimization import deflated_sharpe as search_luck_analysis
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
    warn_if_holdout_data,
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

#: Judgment call. The grid-relative probability is the chance that the winner's
#: true Sharpe is above the best a search this size finds among equally good
#: combinations. Below a half the winner is more likely the luckiest draw than a
#: better parameter set; the conventional 95% would stop nearly every grid,
#: because counting every combination as independent over-corrects.
DEFAULT_MIN_GRID_RELATIVE_PROBABILITY = 0.5

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
#: Rule fencing off lines a reader must not skim past.
_BANNER = '!' * 66


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
                       min_grid_beat: float,
                       min_grid_relative_probability: float =
                       DEFAULT_MIN_GRID_RELATIVE_PROBABILITY) -> StageResult:
    """Stage 2: is the winner a plateau, on a grid that beat doing nothing?

    Nothing here is recomputed. The plateau analysis reads the scores the
    optimisation already produced, exactly as ``optimize.py`` reports them, and
    the in-memory cap is raised to ``optimize.py``'s own ceiling - a
    score-truncated result set reports **no** distribution rather than one
    computed from its best-scoring survivors, and would gate on ``None``.

    The search-luck gate reads the same results. When the luck could not be
    assessed (a truncated result set, too few trials, no spread between them)
    its value is ``None`` and, like every other gate, that stops the funnel: a
    winner whose luck could not be judged has not cleared the gate.

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
        min_grid_relative_probability: Gate on the probability that the winner
            is above the best of that many equally good combinations.

    Returns:
        The stage result, gated on plateau retention, on the share of the grid
        that beat buy-and-hold, and on the grid-relative probability.

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

    luck = assess_search_luck(results, selection, run_config)

    stage = StageResult(name=STAGE_OPTIMIZE)
    stage.detail = [
        f"{len(results)} combination(s) evaluated by {method} on {metric} "
        f"({selection})",
        f"winner {results[0].parameters}  plateau verdict: {verdict}",
    ] + search_luck_detail(luck)
    stage.gates = [
        Gate(stage=STAGE_OPTIMIZE, quantity='plateau retention', value=retention,
             threshold=min_retention, flag='--min-retention',
             unknown_reason=retention_reason),
        Gate(stage=STAGE_OPTIMIZE, quantity='grid fraction beating buy-and-hold',
             value=beat_fraction, threshold=min_grid_beat, flag='--min-grid-beat',
             unknown_reason=beat_reason),
        Gate(stage=STAGE_OPTIMIZE, quantity='grid-relative probability',
             value=luck.grid_relative_probability,
             threshold=min_grid_relative_probability,
             flag='--min-grid-relative-probability',
             unknown_reason=f"search luck not assessable: {luck.status}"),
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
        'search_luck': search_luck_analysis.summary_fields(luck),
    }
    return stage


def assess_search_luck(results, selection: str,
                       run_config: RunConfig) -> search_luck_analysis.DeflatedSharpe:
    """Judge the winner against the best a search this size finds by luck.

    Args:
        results: The optimisation results, best first.
        selection: One of the ``plateau.SELECTION_*`` constants.
        run_config: Engine settings, which decide the annualisation the figures
            are shown in.

    Returns:
        The analysis. A failure to run it comes back as a not-computed result
        carrying the reason, so it reaches the gate as "not assessable" instead
        of ending the funnel as an error or passing unseen.
    """
    try:
        engine = BacktestEngine.from_config(run_config)
        periods_per_year = engine.resolve_periods_per_year(
            results[0].backtest_result.portfolio_values.index)
        return search_luck_analysis.analyse_results(
            results, selection, periods_per_year=periods_per_year)
    except Exception as e:
        return search_luck_analysis.DeflatedSharpe(
            status=search_luck_analysis.STATUS_UNDEFINED, reason=str(e),
            trials_evaluated=len(results))


def search_luck_detail(luck: search_luck_analysis.DeflatedSharpe) -> List[str]:
    """Render the search-luck figures, or say loudly that there are none.

    Args:
        luck: The analysis from :func:`assess_search_luck`.

    Returns:
        Context lines for the optimize stage.
    """
    if not luck.is_computed:
        return [
            _BANNER,
            f"SEARCH LUCK NOT ASSESSABLE ({luck.status}): {luck.reason}",
            "This is not a pass. It stops the funnel like a failed gate: a winner",
            "whose luck could not be judged has not been shown to be above it.",
            _BANNER,
        ]

    return [
        f"search luck over {luck.trials:g} trial(s): winner Sharpe "
        f"{_number(luck.annualised(luck.winner.sharpe), 3)} vs grid-relative luck "
        f"line {_number(luck.annualised(luck.grid_relative_luck_line), 3)} "
        f"(annualised)",
    ]


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


def check_holdout_follows_research(research: Dict[str, Any],
                                   holdouts: Dict[str, Any]) -> None:
    """Refuse a holdout that overlaps any data the strategy was judged on.

    The line is the latest last bar across every research file, not the last bar
    of the file a holdout happens to share a symbol with: a walk-forward on one
    asset over a period is a decision made with that period in view, whichever
    asset the holdout then scores.

    Args:
        research: Every research dataset in the run, keyed by path.
        holdouts: Every holdout dataset, keyed by path.

    Raises:
        ValueError: If any frame is empty, or any holdout's first bar is not
            strictly after the latest last research bar.
    """
    empty = [path for path, frame in {**research, **holdouts}.items() if frame.empty]
    if empty or not research or not holdouts:
        raise ValueError("holdout check needs at least one bar in every research "
                         "and holdout dataset" +
                         (f"; empty: {', '.join(empty)}" if empty else ""))

    latest_path, latest = max(research.items(), key=lambda item: item[1].index[-1])
    research_end = latest.index[-1]
    for path, holdout in holdouts.items():
        holdout_start = holdout.index[0]
        if holdout_start <= research_end:
            raise ValueError(
                f"holdout data {path} starts {holdout_start} but the research data "
                f"runs to {research_end} ({latest_path}): every holdout must start "
                f"strictly after the last research bar, or the parameters were "
                f"fitted on bars the holdout then scores them on")


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


def run_pooled_holdout_stage(holdouts: List[Tuple[str, Any]], strategy_name: str,
                             parameters: Dict[str, Any], run_config: RunConfig,
                             min_excess: float,
                             min_trades: int = DEFAULT_MIN_HOLDOUT_TRADES) -> StageResult:
    """Stage 5 over several holdout files: one verdict from one backtest each.

    One holdout file is one asset, and often too few round trips to say much.
    Several are pooled the way ``compare.py`` pools folds: the unit of
    observation is the file, the excess is the **median** of the per-file
    figures so one exceptional file cannot carry the verdict, and the number of
    files that beat buy-and-hold is reported beside it.

    A file the strategy never traded on is left out of the excess pool and said
    to be: flat while the asset fell is positive excess for nothing, and a
    median would count it as a win.

    Args:
        holdouts: ``(symbol, frame)`` per holdout file, in the order given.
        strategy_name: Registered strategy name.
        parameters: The winning parameters from the optimize stage, used
            unchanged on every file.
        run_config: Engine settings, the same ones every earlier stage ran under.
        min_excess: Gate on the pooled excess, in percentage points.
        min_trades: Gate on completed round trips summed over the files.

    Returns:
        The stage result, gated on pooled round trips and pooled excess. Its
        payload carries one entry per file under ``files``.
    """
    per_file = [run_holdout_stage(frame, symbol, strategy_name, parameters,
                                  run_config, min_excess)
                for symbol, frame in holdouts]
    files = [single.payload for single in per_file]

    round_trips = sum(payload['round_trips'] for payload in files)
    traded = [payload for payload in files if payload['round_trips'] > 0]
    idle = [payload['symbol'] for payload in files if payload['round_trips'] == 0]
    unmeasured = [payload['symbol'] for payload in traded
                  if payload['excess_return_pct'] is None]

    pooled_excess: Optional[float] = None
    beating: Optional[int] = None
    if not traded:
        excess_reason = 'no holdout file completed a round trip'
    elif unmeasured:
        excess_reason = (f"no benchmark could be established on "
                         f"{', '.join(unmeasured)}")
    else:
        excess_reason = None
        excesses = [payload['excess_return_pct'] for payload in traded]
        pooled_excess = float(statistics.median(excesses))
        beating = sum(1 for excess in excesses if excess > 0)

    stage = StageResult(name=STAGE_HOLDOUT)
    stage.detail = [
        f"{len(files)} holdout file(s), one backtest each with parameters "
        f"{parameters}; nothing fitted on any of them",
    ]
    for single in per_file:
        stage.detail.extend(f"[{single.payload['symbol']}] {line}"
                            for line in single.detail)
    stage.detail.append(
        f"pooled round trips {round_trips} = the sum over the {len(files)} file(s)")
    if pooled_excess is not None:
        stage.detail.append(
            f"pooled excess {pooled_excess:.2f} pp = the median of the per-file "
            f"excess over buy-and-hold across the {len(traded)} file(s) that traded "
            f"(compare.py's convention for folds: one exceptional file cannot "
            f"carry it)")
        stage.detail.append(
            f"{beating} of {len(traded)} file(s) that traded beat buy-and-hold")
    if idle:
        stage.detail.append(
            f"left out of the excess pool, no completed round trip: "
            f"{', '.join(idle)}")

    stage.gates = [
        Gate(
            stage=STAGE_HOLDOUT,
            quantity='pooled round trips',
            value=float(round_trips),
            threshold=float(min_trades),
            flag='--min-holdout-trades',
            precision=0,
        ),
        Gate(
            stage=STAGE_HOLDOUT,
            quantity='pooled excess over buy-and-hold (pp)',
            value=pooled_excess,
            threshold=min_excess,
            flag='--min-holdout-excess',
            unknown_reason=excess_reason,
        ),
    ]
    stage.payload = {
        'parameters': parameters,
        'n_files': len(files),
        'files_traded': len(traded),
        'files_beating_buy_and_hold': beating,
        'pooled_round_trips': round_trips,
        'pooled_excess_pct': pooled_excess,
        'files': files,
    }
    return stage


def holdout_files(stage: StageResult) -> List[Dict[str, Any]]:
    """The per-file records of a holdout stage, whether it pooled or not."""
    return stage.payload.get('files', [stage.payload])


def holdout_summary(stage: Optional[StageResult]) -> Dict[str, Any]:
    """The holdout figures a run summary carries, for one file or several.

    Args:
        stage: The holdout stage, or None when no holdout backtest ran.

    Returns:
        ``holdout_files``, ``holdout_files_beating``, ``holdout_round_trips``,
        ``holdout_excess_pct`` and ``holdout_data_sha256``. All None when the
        holdout was not looked at. The hash is set for exactly one file: a run
        over several has no single fingerprint, and each exported row names its
        own.
    """
    if stage is None:
        return {'holdout_files': None, 'holdout_files_beating': None,
                'holdout_round_trips': None, 'holdout_excess_pct': None,
                'holdout_data_sha256': None}

    files = holdout_files(stage)
    if 'files' in stage.payload:
        beating = stage.payload['files_beating_buy_and_hold']
        round_trips = stage.payload['pooled_round_trips']
        excess = stage.payload['pooled_excess_pct']
    else:
        excess = stage.payload.get('excess_return_pct')
        round_trips = stage.payload.get('round_trips')
        beating = None if excess is None or not round_trips else int(excess > 0)
    return {
        'holdout_files': len(files),
        'holdout_files_beating': beating,
        'holdout_round_trips': round_trips,
        'holdout_excess_pct': excess,
        'holdout_data_sha256': (files[0].get('data_sha256') if len(files) == 1
                                else None),
    }


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

  # The same look pooled over several instruments: one backtest per file
  python scripts/screen.py --data data/SPY_research.csv --strategy breakout \\
    --compare-data data/QQQ_research.csv \\
    --holdout-data data/SPY_holdout.csv data/QQQ_holdout.csv

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
    parser.add_argument('--holdout-data', nargs='+', default=None,
                        help='Holdout OHLCV CSV(s) for a final stage: one backtest of '
                             'the stage-2 winner per file, nothing fitted. Each must '
                             'start after --data and every --compare-data file ends. '
                             'Several files are pooled: round trips are summed and '
                             'the excess over buy-and-hold is the median of the '
                             'per-file figures. Looking at it spends it, so it '
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
        'the other four are set where a reasonable person would want to look '
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
    gates.add_argument('--min-grid-relative-probability', type=float,
                       default=DEFAULT_MIN_GRID_RELATIVE_PROBABILITY,
                       help=f"Stage 2: probability that the winner is truly above "
                            f"the best of that many equally good combinations "
                            f"(default: {DEFAULT_MIN_GRID_RELATIVE_PROBABILITY:g} - a "
                            f"judgment call: below it the winner is more likely the "
                            f"luckiest draw than a better parameter set). When it "
                            f"cannot be assessed the gate stops the run")
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
    holdout_paths = ([args.holdout_data] if isinstance(args.holdout_data, str)
                     else list(args.holdout_data or []))
    required = datasets + holdout_paths
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

    # Only the research roles: --holdout-data is where such a file belongs.
    warn_if_holdout_data(datasets)

    # Twice in the list is two rows and two votes in the pool for one look.
    if len(set(holdout_paths)) != len(holdout_paths):
        print("Error: a holdout file is listed more than once in --holdout-data.",
              file=sys.stderr)
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

        holdouts: Dict[str, Any] = {}
        if holdout_paths:
            # Checked before any stage runs: reading the dates is not a look at
            # the result, and a mis-cut file should not cost a whole funnel.
            holdouts = {path: load_ohlcv_csv(path, clean=args.clean)
                        for path in holdout_paths}
            research = {args.data: data}
            for path in args.compare_data or []:
                research[path] = load_ohlcv_csv(path, clean=args.clean)
            check_holdout_follows_research(research, holdouts)

        if record(run_backtest_stage(data, symbol, args.strategy, run_config)):
            optimize_stage = run_optimize_stage(
                data, args.strategy, run_config,
                method=args.optimization_method,
                metric=args.optimization_metric,
                trials=args.trials,
                seed=args.seed,
                n_jobs=args.n_jobs,
                min_retention=args.min_retention,
                min_grid_beat=args.min_grid_beat,
                min_grid_relative_probability=args.min_grid_relative_probability)
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

                    if reached_holdout and holdouts and stopped_at is not None:
                        # Only --force gets here with a failed gate behind it.
                        skipped = StageResult(
                            name=STAGE_HOLDOUT,
                            skipped_reason='an earlier gate failed; the holdout '
                                           'was not spent on a strategy that '
                                           'already has its verdict (--force '
                                           'does not override this)')
                        stages.append(skipped)
                        report_stage(skipped)
                    elif reached_holdout and len(holdouts) == 1:
                        holdout_stage = run_holdout_stage(
                            holdouts[holdout_paths[0]],
                            symbol_from_path(holdout_paths[0]),
                            args.strategy,
                            optimize_stage.payload['winner_parameters'],
                            run_config, args.min_holdout_excess,
                            args.min_holdout_trades)
                        screened.extend(holdout_paths)
                        record(holdout_stage)
                    elif reached_holdout and holdouts:
                        holdout_stage = run_pooled_holdout_stage(
                            [(symbol_from_path(path), holdouts[path])
                             for path in holdout_paths],
                            args.strategy,
                            optimize_stage.payload['winner_parameters'],
                            run_config, args.min_holdout_excess,
                            args.min_holdout_trades)
                        screened.extend(holdout_paths)
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
              f"not a result: the search-luck gate corrects for this one "
              f"parameter search only, counting every combination as independent, "
              f"and for none of the strategies or grids tried before it.")
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

    # Each holdout file is exported as a row of its own, carrying the hash of the
    # file it read, so "how many runs looked at this holdout" is a count over rows.
    detail_rows = list(walk_forward_rows)
    if holdout_stage is not None:
        for path, payload in zip(holdout_paths, holdout_files(holdout_stage)):
            payload.update({
                'data_path': path,
                'data_sha256': provenance_fingerprint(
                    provenance.get(path))['data_sha256'],
            })
            detail_rows.append({
                **payload,
                'stage': STAGE_HOLDOUT,
                'strategy': args.strategy,
                'error': None,
            })
    record = exporter_manager.create_run_record(
        identity, args.strategy,
        {
            'requested_datasets': datasets,
            # One path stays a string, as it always was; several are a list.
            'holdout_data': (holdout_paths[0] if len(holdout_paths) == 1
                             else holdout_paths or None),
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
                'min_grid_relative_probability': args.min_grid_relative_probability,
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
            **holdout_summary(holdout_stage),
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
