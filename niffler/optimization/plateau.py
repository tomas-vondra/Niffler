"""
Parameter plateau analysis.

An optimizer returns the maximum of N noisy estimates. On pure noise, the best
of 375 parameter combinations still looks impressive, which is why the winner's
own score says almost nothing about whether the strategy has an edge. The
*shape of the score surface around the winner* says considerably more: there is
no economic mechanism by which a 10/30 crossover works while 11/30 and 10/31
fail, so an isolated spike is noise, whereas a broad contiguous region of
similar scores is at least consistent with a real effect that does not care
whether it was measured over 10 days or 12.

This module is pure analysis. It re-runs nothing and recomputes no metric: it
reads the scores that :class:`~niffler.optimization.optimization_result.OptimizationResult`
objects already carry, which the optimizer computed and then threw away.

What is computed
----------------

**The surface.** Every parameter that appears in the results becomes an axis,
whose ticks are that parameter's distinct observed values in ascending order.
The surface is the full k-dimensional cartesian product of those axes; a cell
is one parameter combination. Adjacency is defined on *tick indices*, not on
parameter values, so it is only meaningful for parameters whose ordering is
meaningful (numeric ranges). A ``choice`` parameter over unordered labels is
sorted deterministically but its neighbours are an artefact of that sort - do
not read a plateau along such an axis as evidence.

**Cell states.** A cell is one of four things, and they are never conflated:

``ok``
    A result exists, it traded at least once, and its metric is finite.
    Only these cells enter any statistic.
``no_trades``
    A result exists but the strategy never traded. Its 0% return is not a
    mediocre result, it is the absence of one.
``non_finite``
    A result exists but the metric is ``NaN``/``inf`` (or ``None``, which the
    metric accessors map to ``-inf``). A degenerate estimate, not a bad score.
``missing``
    No result for this combination: it errored during evaluation, was never
    sampled (random search), or was discarded by the optimizer's in-memory
    result cap.

**Direction.** Metric direction is taken from
:data:`~niffler.optimization.base_optimizer.BaseOptimizer.METRICS_CONFIG`, the
same table the optimizer sorts by, so the two can never disagree. Every
comparison in this module is done on the *oriented* score
(``score`` if higher-is-better else ``-score``), so ``max_drawdown`` - a
negative percentage where -5 is better than -40 - ranks shallowest-first
everywhere, and no code path takes ``max()`` for "worst".

**The plateau score (neighbourhood retention).** Let ``m`` be the median
oriented score over ``ok`` cells, ``w`` the winner's oriented score, and ``E =
w - m`` the winner's *edge* over a typical grid cell. Let ``n`` be the mean
oriented score of the winner's ``ok`` immediate neighbours - all cells at
Chebyshev distance 1 in tick-index space, i.e. up to ``3**k - 1`` of them,
excluding the winner itself. Then::

    retention = (n - m) / E

It is 1.0 when the neighbourhood scores as well as the winner (a plateau), 0.0
when the neighbourhood is indistinguishable from a typical grid cell (an
isolated spike), and negative when the winner's neighbours are *worse* than
typical. Anchoring on the median rather than on zero makes it scale-free and
sign-safe: it is a ratio of two differences, so it behaves identically for a
metric measured in percent, in Sharpe units, or in negative drawdown percent.
It is ``None``, never a number, when ``E <= 0`` (no dispersion to retain) or
when the winner has no ``ok`` neighbour at all; the count of neighbours it was
actually computed from is always reported next to it, because a "robust"
verdict from one of eight neighbours is not one.

**The plateau region.** Cells whose oriented score is at least
``w - tolerance * E`` (default tolerance 0.25, i.e. cells retaining >=75% of the
winner's edge over the median) that are connected to the winner by von Neumann
adjacency - single steps along one axis at a time. Reported as a cell count, as
a fraction of ``ok`` cells, and as a per-parameter extent.

**The plateau centre** is the region member closest to the region's centroid in
tick-index space, ties broken by higher oriented score and then by tick index.
The middle of a plateau generalises better than its highest point, but this is
reported *beside* the optimizer's winner and never replaces it.

**Distribution statistics** cover the whole surface, not the top row: min,
quartiles, median, max and mean over ``ok`` cells, plus the fraction of them
beating a do-nothing baseline. The baseline is the run's own buy-and-hold
benchmark where the metric has one (all results share it: it is a function of
the data, capital, commission and cost model, all constant across a grid), and
is labelled exactly for what it is when it is not.

Multi-parameter surfaces
------------------------
The analysis above is k-dimensional: neighbourhoods, regions and extents use
every parameter. Only the ASCII heatmap has to collapse to two axes, and it
does so by **slicing, not marginalising**: two axes are displayed (by default
the two with the most ticks) and the remaining parameters are pinned at the
winner's values. Every rendered cell is therefore a real backtest rather than
an average over regimes, at the cost of showing one slab of the surface; the
header states which slab and how many cells are not shown.
"""

import csv
import logging
import math
import statistics
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from niffler.utils.json_utils import sanitize_numeric_values

from .base_optimizer import BaseOptimizer
from .optimization_result import OptimizationResult

logger = logging.getLogger(__name__)

#: Cell states. Kept as plain strings so they survive into CSV unchanged.
CELL_OK = 'ok'
CELL_NO_TRADES = 'no_trades'
CELL_NON_FINITE = 'non_finite'
CELL_MISSING = 'missing'

#: How the evaluated combinations were selected out of the parameter space.
#: This is the caller's knowledge, not the module's, and it decides whether
#: distribution statistics may be reported at all.
SELECTION_EXHAUSTIVE = 'exhaustive'   # every combination was evaluated (grid search)
SELECTION_SAMPLED = 'sampled'         # an unbiased random subset (random search)
SELECTION_TRUNCATED = 'truncated'     # a score-biased subset (optimizer memory cap)

SELECTION_CHOICES = (SELECTION_EXHAUSTIVE, SELECTION_SAMPLED, SELECTION_TRUNCATED)

#: Default width of the plateau band, as a fraction of the winner's edge over
#: the grid median. 0.25 means "cells retaining at least 75% of the edge".
DEFAULT_TOLERANCE = 0.25

#: Retention at or above this is described as a broad plateau.
BROAD_PLATEAU_RETENTION = 0.75

#: Retention below this is described as an isolated spike.
ISOLATED_SPIKE_RETENTION = 0.40

#: Characters the heatmap uses for the four cell states. The scale itself is
#: digits, so a level can never be mistaken for a missing cell.
HEATMAP_LEVELS = '0123456789'
HEATMAP_NO_TRADES = '-'
HEATMAP_NON_FINITE = '?'
HEATMAP_MISSING = '.'
HEATMAP_WINNER = '*'
HEATMAP_CENTRE = 'C'

_SEPARATOR = '=' * 78

#: Baseline per metric: which BacktestResult field carries the buy-and-hold
#: equivalent, how to describe it, and what to fall back to when no benchmark
#: ran. A fallback of None means "there is no honest substitute": a zero
#: baseline for max_drawdown would be a drawdown no strategy can beat, and
#: reporting "0% of the grid beat it" would be an artefact of the fabrication.
_BASELINES: Dict[str, Tuple[Optional[str], str, Optional[float], Optional[str]]] = {
    'total_return': (
        'benchmark_return_pct', 'buy-and-hold total return',
        0.0, 'zero return (no benchmark available)',
    ),
    'sharpe_ratio': (
        'benchmark_sharpe_ratio', 'buy-and-hold Sharpe ratio',
        0.0, 'zero Sharpe ratio (no benchmark available)',
    ),
    'max_drawdown': (
        'benchmark_max_drawdown', 'buy-and-hold max drawdown',
        None, None,
    ),
    'excess_return_pct': (
        None, 'zero excess return over buy-and-hold',
        0.0, 'zero excess return over buy-and-hold',
    ),
}

#: Benchmark values differing by more than this across results mean the results
#: did not all come from one dataset, and the baseline is not a constant.
_BASELINE_TOLERANCE = 1e-6


class PlateauError(ValueError):
    """A surface could not be built from the given results."""


@dataclass(frozen=True)
class SurfaceCell:
    """
    One parameter combination on the surface.

    Attributes:
        index: Tick indices of the cell, one per axis, in axis order
        parameters: The parameter combination itself
        status: One of CELL_OK / CELL_NO_TRADES / CELL_NON_FINITE / CELL_MISSING
        score: The analysed metric's raw value, None unless the status is
            CELL_OK or CELL_NO_TRADES
        oriented_score: ``score`` re-signed so that larger is always better,
            None unless the status is CELL_OK
        total_trades: Trades the combination executed, None when missing
        result: The underlying optimization result, None when missing
    """

    index: Tuple[int, ...]
    parameters: Dict[str, Any]
    status: str
    score: Optional[float] = None
    oriented_score: Optional[float] = None
    total_trades: Optional[int] = None
    result: Optional[OptimizationResult] = None

    @property
    def is_ok(self) -> bool:
        """Whether this cell may enter a statistic."""
        return self.status == CELL_OK


@dataclass
class ParameterSurface:
    """
    The full score surface of an optimisation run.

    Attributes:
        metric: Name of the analysed metric (a key of ``METRICS_CONFIG``)
        higher_is_better: Direction of that metric, taken from the optimizer
        axes: Ordered mapping of parameter name to its ascending tick values
        cells: Every cell of the cartesian product, keyed by tick index
        selection: How the evaluated subset was chosen (see SELECTION_*)
        evaluated: Number of results the surface was built from
        duplicates: Results dropped because their parameters were already seen
    """

    metric: str
    higher_is_better: bool
    axes: Dict[str, List[Any]]
    cells: Dict[Tuple[int, ...], SurfaceCell]
    selection: str = SELECTION_EXHAUSTIVE
    evaluated: int = 0
    duplicates: int = 0

    @property
    def parameter_names(self) -> List[str]:
        """Axis names, in axis order."""
        return list(self.axes.keys())

    @property
    def shape(self) -> Tuple[int, ...]:
        """Number of ticks per axis, in axis order."""
        return tuple(len(values) for values in self.axes.values())

    @property
    def size(self) -> int:
        """Total number of cells in the cartesian product."""
        total = 1
        for length in self.shape:
            total *= length
        return total

    @property
    def ok_cells(self) -> List[SurfaceCell]:
        """Cells that carry a usable score, in tick-index order."""
        return [cell for _, cell in sorted(self.cells.items()) if cell.is_ok]

    @property
    def status_counts(self) -> Dict[str, int]:
        """Cell count per state, always carrying all four keys."""
        counts = {CELL_OK: 0, CELL_NO_TRADES: 0, CELL_NON_FINITE: 0, CELL_MISSING: 0}
        for cell in self.cells.values():
            counts[cell.status] += 1
        return counts

    @property
    def coverage(self) -> float:
        """Fraction of the cartesian product that has a result at all."""
        if not self.cells:
            return 0.0
        missing = self.status_counts[CELL_MISSING]
        return (len(self.cells) - missing) / len(self.cells)

    def cell(self, index: Tuple[int, ...]) -> Optional[SurfaceCell]:
        """
        Look a cell up by tick index.

        Args:
            index: Tick indices, one per axis

        Returns:
            The cell, or None when the index is out of bounds
        """
        return self.cells.get(index)

    def best_cell(self) -> Optional[SurfaceCell]:
        """
        The highest-scoring ``ok`` cell.

        Returns:
            The winner, or None when no cell carries a score. Ties are broken
            by tick index so the answer is deterministic.
        """
        ok = self.ok_cells
        if not ok:
            return None
        return max(ok, key=lambda cell: (cell.oriented_score, tuple(-i for i in cell.index)))


@dataclass
class DistributionStats:
    """
    Distribution of the metric across the whole surface.

    Attributes:
        metric: The analysed metric
        count: Number of ``ok`` cells the statistics describe
        minimum/q1/median/q3/maximum/mean: Order statistics over those cells
        baseline: The do-nothing comparison value, None when there is no honest
            one for this metric
        baseline_label: Exactly what that baseline is. A zero-return baseline is
            never described as buy-and-hold
        beat_count: ``ok`` cells beating the baseline, None without a baseline
        fraction_beating_baseline: ``beat_count / count``
        reliable: False when the evaluated subset was selected by score, in
            which case the order statistics describe survivors rather than the
            grid and are not reported at all
        unreliable_reason: Why, when ``reliable`` is False
        status_counts: Cell count per state, so a surface where most cells never
            traded cannot read as healthy on the remainder
    """

    metric: str
    count: int
    minimum: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    baseline: Optional[float] = None
    baseline_label: Optional[str] = None
    beat_count: Optional[int] = None
    fraction_beating_baseline: Optional[float] = None
    reliable: bool = True
    unreliable_reason: Optional[str] = None
    status_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class PlateauScore:
    """
    Robustness of the winner's neighbourhood, and the region around it.

    Attributes:
        metric: The analysed metric
        winner_index: Tick index of the highest-scoring cell
        winner_parameters: Its parameter combination
        winner_score: Its raw metric value
        grid_median: Median oriented score over ``ok`` cells, the anchor
        edge: ``winner_oriented - grid_median``
        retention: The plateau score, None when it is not computable
        retention_reason: Why it is None, when it is
        neighbours_ok: ``ok`` neighbours the score was computed from
        neighbours_total: In-bounds neighbours the winner has at all
        neighbour_mean: Mean oriented score of those ``ok`` neighbours
        tolerance: Band width used for the region, as a fraction of the edge
        threshold: Oriented score a cell must reach to join the region
        region_size: Cells in the connected region containing the winner
        region_fraction: ``region_size`` over the number of ``ok`` cells
        region_extent: Per-parameter (min, max) value across region members
        centre_index/centre_parameters: The region's centre cell
        centre_score: That cell's raw metric value
        boundary_parameters: Parameters along which the region reaches the edge
            of the searched space. A plateau that runs off the grid has not been
            shown to end there; the optimum may simply lie outside the range
            that was searched
    """

    metric: str
    winner_index: Tuple[int, ...]
    winner_parameters: Dict[str, Any]
    winner_score: float
    grid_median: float
    edge: float
    retention: Optional[float] = None
    retention_reason: Optional[str] = None
    neighbours_ok: int = 0
    neighbours_total: int = 0
    neighbour_mean: Optional[float] = None
    tolerance: float = DEFAULT_TOLERANCE
    threshold: float = 0.0
    region_size: int = 1
    region_fraction: float = 0.0
    region_extent: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    centre_index: Optional[Tuple[int, ...]] = None
    centre_parameters: Optional[Dict[str, Any]] = None
    centre_score: Optional[float] = None
    boundary_parameters: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """
        One-line reading of the retention score.

        Returns:
            A description of the surface around the winner. "no verdict" when
            retention could not be computed - which is a statement about the
            data, not a negative result.
        """
        if self.retention is None:
            return 'no verdict'
        if self.retention >= BROAD_PLATEAU_RETENTION:
            return 'broad plateau'
        if self.retention >= ISOLATED_SPIKE_RETENTION:
            return 'partial plateau'
        return 'isolated spike'


@dataclass
class PlateauReport:
    """
    Everything the analysis produced: surface, distribution and plateau score.

    Attributes:
        surface: The score surface itself
        distribution: Whole-grid distribution statistics
        plateau: Winner neighbourhood analysis, None when no cell scored
    """

    surface: ParameterSurface
    distribution: DistributionStats
    plateau: Optional[PlateauScore] = None


# ---------------------------------------------------------------------------
# Building the surface
# ---------------------------------------------------------------------------


def _sort_key(value: Any) -> Tuple[int, Any]:
    """
    Order axis ticks, keeping mixed and non-numeric types sortable.

    Args:
        value: A parameter value

    Returns:
        A key placing numbers before other values, both groups internally
        ordered, so the axis order is deterministic for any parameter space.
    """
    if isinstance(value, bool):
        return (1, str(value))
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, str(value))


def _metric_direction(metric: str) -> Tuple[bool, Any]:
    """
    Look a metric's direction and accessor up in the optimizer's own table.

    Args:
        metric: Metric name

    Returns:
        ``(higher_is_better, accessor)``

    Raises:
        PlateauError: If the metric is not one the optimizer knows
    """
    if metric not in BaseOptimizer.METRICS_CONFIG:
        available = ', '.join(BaseOptimizer.METRICS_CONFIG)
        raise PlateauError(f"Unknown metric '{metric}'. Available: {available}")
    return BaseOptimizer.METRICS_CONFIG[metric]


def _classify(result: OptimizationResult, accessor: Any,
              higher_is_better: bool) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Decide a result's cell state and score.

    A combination that never traded is not a badly-scoring combination, and a
    metric that came back ``NaN``/``inf``/``None`` is not a low one, so both get
    their own state instead of a number that would average in as mediocre.

    Args:
        result: The optimization result
        accessor: Metric accessor from ``METRICS_CONFIG``
        higher_is_better: Direction of the metric

    Returns:
        ``(status, score, oriented_score)``
    """
    trades = getattr(result.backtest_result, 'total_trades', 0) or 0

    try:
        raw = accessor(result)
    except (AttributeError, TypeError):
        return CELL_NON_FINITE, None, None

    if raw is None:
        return CELL_NON_FINITE, None, None

    try:
        score = float(raw)
    except (TypeError, ValueError):
        return CELL_NON_FINITE, None, None

    if not math.isfinite(score):
        return CELL_NON_FINITE, None, None

    if trades <= 0:
        return CELL_NO_TRADES, score, None

    return CELL_OK, score, (score if higher_is_better else -score)


def _cell_rank(status: str, oriented: Optional[float]) -> Tuple[int, float]:
    """
    Rank two candidate cells for the same parameters, best first.

    Random search can sample the same combination twice; keeping whichever
    arrived first would make the surface depend on result ordering.

    Args:
        status: Candidate cell state
        oriented: Candidate oriented score, if any

    Returns:
        A sort key where larger is preferred.
    """
    return (1 if status == CELL_OK else 0,
            oriented if oriented is not None else float('-inf'))


def build_surface(results: Sequence[OptimizationResult],
                  metric: str = 'total_return',
                  selection: str = SELECTION_EXHAUSTIVE) -> ParameterSurface:
    """
    Build the score surface from optimisation results.

    Nothing is re-run and no metric is recomputed: the scores already on each
    result are read through the optimizer's own accessor table.

    Args:
        results: Optimisation results, in any order
        metric: Metric to build the surface of; must be a key of
            ``BaseOptimizer.METRICS_CONFIG``
        selection: How the evaluated combinations were chosen - one of
            SELECTION_EXHAUSTIVE, SELECTION_SAMPLED or SELECTION_TRUNCATED.
            Only the caller knows this, and it decides whether distribution
            statistics may be trusted.

    Returns:
        The surface, with every cell of the cartesian product present and
        classified.

    Raises:
        PlateauError: If ``results`` is empty, the metric is unknown, the
            selection is unknown, the results do not all carry the same
            parameter names, or the cartesian product would be unusably large.
    """
    if selection not in SELECTION_CHOICES:
        raise PlateauError(
            f"Unknown selection '{selection}'. Available: {', '.join(SELECTION_CHOICES)}"
        )
    if not results:
        raise PlateauError('Cannot build a parameter surface from zero results')

    higher_is_better, accessor = _metric_direction(metric)

    names = list(results[0].parameters.keys())
    if not names:
        raise PlateauError('Cannot build a parameter surface from parameterless results')
    name_set = set(names)
    for result in results:
        if set(result.parameters.keys()) != name_set:
            raise PlateauError(
                'All results must share the same parameter names; got '
                f'{sorted(name_set)} and {sorted(result.parameters.keys())}'
            )

    axes: Dict[str, List[Any]] = {}
    for name in names:
        values = {result.parameters[name] for result in results}
        axes[name] = sorted(values, key=_sort_key)

    tick_index = {name: {value: i for i, value in enumerate(values)}
                  for name, values in axes.items()}

    # Best cell per parameter combination, so a duplicated sample cannot make
    # the surface depend on the order results arrived in.
    best: Dict[Tuple[int, ...], Tuple[Tuple[int, float], SurfaceCell]] = {}
    duplicates = 0
    for result in results:
        index = tuple(tick_index[name][result.parameters[name]] for name in names)
        status, score, oriented = _classify(result, accessor, higher_is_better)
        candidate = SurfaceCell(
            index=index,
            parameters=dict(result.parameters),
            status=status,
            score=score,
            oriented_score=oriented,
            total_trades=int(getattr(result.backtest_result, 'total_trades', 0) or 0),
            result=result,
        )
        rank = _cell_rank(status, oriented)
        if index in best:
            duplicates += 1
            if rank <= best[index][0]:
                continue
        best[index] = (rank, candidate)

    cells: Dict[Tuple[int, ...], SurfaceCell] = {}
    for index in product(*[range(len(values)) for values in axes.values()]):
        if index in best:
            cells[index] = best[index][1]
        else:
            cells[index] = SurfaceCell(
                index=index,
                parameters={name: axes[name][index[position]]
                            for position, name in enumerate(names)},
                status=CELL_MISSING,
            )

    return ParameterSurface(
        metric=metric,
        higher_is_better=higher_is_better,
        axes=axes,
        cells=cells,
        selection=selection,
        evaluated=len(results),
        duplicates=duplicates,
    )


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------


def resolve_baseline(results: Sequence[OptimizationResult],
                     metric: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Find the do-nothing baseline the metric should be measured against.

    The results already carry a real buy-and-hold benchmark, charged the same
    commission and cost model as the strategy and entered no earlier than the
    execution lag allows (see :mod:`niffler.backtesting.benchmark`). It is a
    function of the data and the engine configuration, both constant across an
    optimisation run, so any result's copy of it is the run's baseline; the
    others are checked for agreement and a disagreement is reported rather than
    averaged away.

    Args:
        results: Optimisation results
        metric: The analysed metric

    Returns:
        ``(baseline, label)``. Both are None when the metric has no do-nothing
        equivalent (``win_rate``, ``total_trades``) or when a fabricated one
        would be misleading (``max_drawdown`` with no benchmark). The label
        describes exactly what the number is: a zero-return fallback is never
        called buy-and-hold.
    """
    spec = _BASELINES.get(metric)
    if spec is None:
        return None, None

    field_name, label, fallback, fallback_label = spec

    if field_name is None:
        return fallback, label

    values = [
        value for value in (
            getattr(result.backtest_result, field_name, None) for result in results
        )
        if value is not None and math.isfinite(float(value))
    ]

    if not values:
        return fallback, fallback_label

    baseline = float(values[0])
    spread = max(abs(float(value) - baseline) for value in values)
    if spread > _BASELINE_TOLERANCE:
        logger.warning(
            f"Benchmark {field_name} is not constant across results (spread "
            f"{spread:.6g}); the results did not all come from one dataset. "
            f"Using the median."
        )
        baseline = float(statistics.median(float(value) for value in values))

    return baseline, label


def _quantiles(values: List[float]) -> Tuple[float, float, float]:
    """
    First quartile, median and third quartile of a sample.

    ``statistics.quantiles`` needs at least two points, so short samples fall
    back to the median rather than raising.

    Args:
        values: Sample, unsorted

    Returns:
        ``(q1, median, q3)``
    """
    median = float(statistics.median(values))
    if len(values) < 2:
        return median, median, median
    q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    return float(q1), median, float(q3)


def summarize_distribution(surface: ParameterSurface,
                           results: Sequence[OptimizationResult]) -> DistributionStats:
    """
    Describe the metric across the whole surface, not just its top row.

    Args:
        surface: The surface to summarise
        results: The results it was built from, used only to resolve the
            baseline

    Returns:
        The distribution statistics. When the evaluated subset was selected by
        score (``SELECTION_TRUNCATED``) the order statistics and the
        beat-the-baseline fraction are withheld entirely rather than reported
        with a caveat: survivors of a "keep the best half" purge beat the
        baseline far more often than the grid does, which is precisely the rosy
        picture these statistics exist to counter.
    """
    counts = surface.status_counts
    ok = surface.ok_cells
    baseline, label = resolve_baseline(results, surface.metric)

    stats = DistributionStats(
        metric=surface.metric,
        count=len(ok),
        baseline=baseline,
        baseline_label=label,
        status_counts=counts,
    )

    if surface.selection == SELECTION_TRUNCATED:
        stats.reliable = False
        stats.unreliable_reason = (
            'the optimizer discarded the worst-scoring results to cap memory, so '
            'the surviving combinations are a score-biased subset of the grid'
        )
        return stats

    if not ok:
        stats.reliable = False
        stats.unreliable_reason = 'no combination produced a usable score'
        return stats

    scores = [float(cell.score) for cell in ok]
    q1, median, q3 = _quantiles(scores)
    stats.minimum = min(scores)
    stats.q1 = q1
    stats.median = median
    stats.q3 = q3
    stats.maximum = max(scores)
    stats.mean = float(statistics.fmean(scores))

    if baseline is not None:
        sign = 1.0 if surface.higher_is_better else -1.0
        beat = sum(1 for score in scores if sign * score > sign * baseline)
        stats.beat_count = beat
        stats.fraction_beating_baseline = beat / len(scores)

    return stats


# ---------------------------------------------------------------------------
# Plateau score
# ---------------------------------------------------------------------------


def _neighbour_indices(index: Tuple[int, ...], shape: Tuple[int, ...],
                       moore: bool) -> List[Tuple[int, ...]]:
    """
    In-bounds neighbours of a cell.

    Args:
        index: Tick index of the cell
        shape: Ticks per axis
        moore: True for Chebyshev-distance-1 neighbours (the full surrounding
            block, used for the retention score), False for von Neumann
            neighbours - one step along one axis (used for region connectivity,
            where diagonal-only contact is not contiguity)

    Returns:
        Every in-bounds neighbour, excluding the cell itself.
    """
    neighbours: List[Tuple[int, ...]] = []

    if moore:
        offsets: Iterable[Tuple[int, ...]] = product((-1, 0, 1), repeat=len(index))
    else:
        offsets = []
        for axis in range(len(index)):
            for step in (-1, 1):
                offset = [0] * len(index)
                offset[axis] = step
                offsets.append(tuple(offset))

    for offset in offsets:
        if not any(offset):
            continue
        candidate = tuple(i + o for i, o in zip(index, offset))
        if all(0 <= value < length for value, length in zip(candidate, shape)):
            neighbours.append(candidate)

    return neighbours


def _region_centre(members: List[SurfaceCell]) -> SurfaceCell:
    """
    The member closest to the region's centroid in tick-index space.

    Args:
        members: Cells of the region, at least one

    Returns:
        The centre cell. Ties are broken by higher oriented score and then by
        tick index, so the answer never depends on iteration order.
    """
    dimensions = len(members[0].index)
    centroid = [
        sum(cell.index[axis] for cell in members) / len(members)
        for axis in range(dimensions)
    ]

    def key(cell: SurfaceCell) -> Tuple[float, float, Tuple[int, ...]]:
        distance = sum((cell.index[axis] - centroid[axis]) ** 2
                       for axis in range(dimensions))
        oriented = cell.oriented_score if cell.oriented_score is not None else float('-inf')
        return (distance, -oriented, cell.index)

    return min(members, key=key)


def analyse_plateau(surface: ParameterSurface,
                    tolerance: float = DEFAULT_TOLERANCE) -> Optional[PlateauScore]:
    """
    Score the surface around its winner.

    See the module docstring for the exact definitions of the retention score
    and of the plateau region.

    Args:
        surface: The surface to analyse
        tolerance: Width of the plateau band as a fraction of the winner's edge
            over the grid median. Must lie in [0, 1]; 0.25 (the default) admits
            cells retaining at least 75% of that edge.

    Returns:
        The plateau score, or None when no cell on the surface carries a usable
        score.

    Raises:
        PlateauError: If the tolerance is outside [0, 1]
    """
    if not 0.0 <= tolerance <= 1.0:
        raise PlateauError(f"tolerance must lie in [0, 1], got {tolerance}")

    winner = surface.best_cell()
    if winner is None:
        return None

    ok_by_index = {cell.index: cell for cell in surface.ok_cells}
    oriented = [float(cell.oriented_score) for cell in ok_by_index.values()]
    grid_median = float(statistics.median(oriented))
    winner_oriented = float(winner.oriented_score)
    edge = winner_oriented - grid_median

    shape = surface.shape
    neighbour_positions = _neighbour_indices(winner.index, shape, moore=True)
    neighbour_scores = [
        float(ok_by_index[position].oriented_score)
        for position in neighbour_positions if position in ok_by_index
    ]

    score = PlateauScore(
        metric=surface.metric,
        winner_index=winner.index,
        winner_parameters=dict(winner.parameters),
        winner_score=float(winner.score),
        grid_median=grid_median,
        edge=edge,
        neighbours_ok=len(neighbour_scores),
        neighbours_total=len(neighbour_positions),
        tolerance=tolerance,
    )

    if neighbour_scores:
        score.neighbour_mean = float(statistics.fmean(neighbour_scores))

    if not neighbour_scores:
        score.retention_reason = (
            'the winner has no scored neighbour: every adjacent combination is '
            'missing, non-finite or never traded'
        )
    elif edge <= 0.0:
        score.retention_reason = (
            'the winner does not score above the grid median, so there is no '
            'edge for a neighbourhood to retain'
        )
    else:
        score.retention = (score.neighbour_mean - grid_median) / edge

    # Region: connected cells retaining at least (1 - tolerance) of the edge.
    threshold = winner_oriented - tolerance * edge if edge > 0 else winner_oriented
    score.threshold = threshold

    members: List[SurfaceCell] = []
    seen = {winner.index}
    queue = [winner.index]
    while queue:
        position = queue.pop()
        cell = ok_by_index[position]
        members.append(cell)
        for candidate in _neighbour_indices(position, shape, moore=False):
            if candidate in seen or candidate not in ok_by_index:
                continue
            if float(ok_by_index[candidate].oriented_score) >= threshold:
                seen.add(candidate)
                queue.append(candidate)

    score.region_size = len(members)
    score.region_fraction = len(members) / len(ok_by_index)
    score.region_extent = {
        name: (min((cell.parameters[name] for cell in members), key=_sort_key),
               max((cell.parameters[name] for cell in members), key=_sort_key))
        for name in surface.parameter_names
    }

    # A region that reaches the first or last tick of an axis has not been shown
    # to end there: the search simply stopped.
    for position, name in enumerate(surface.parameter_names):
        indices = [cell.index[position] for cell in members]
        if min(indices) == 0 or max(indices) == shape[position] - 1:
            score.boundary_parameters.append(name)

    centre = _region_centre(members)
    score.centre_index = centre.index
    score.centre_parameters = dict(centre.parameters)
    score.centre_score = float(centre.score)

    return score


def analyse_results(results: Sequence[OptimizationResult],
                    metric: str = 'total_return',
                    selection: str = SELECTION_EXHAUSTIVE,
                    tolerance: float = DEFAULT_TOLERANCE) -> PlateauReport:
    """
    Run the whole analysis over a set of optimisation results.

    Args:
        results: Optimisation results
        metric: Metric to analyse (a key of ``BaseOptimizer.METRICS_CONFIG``)
        selection: How the evaluated combinations were chosen (SELECTION_*)
        tolerance: Plateau band width, as a fraction of the winner's edge

    Returns:
        Surface, distribution statistics and plateau score together.

    Raises:
        PlateauError: If the surface cannot be built or the tolerance is invalid
    """
    surface = build_surface(results, metric=metric, selection=selection)
    return PlateauReport(
        surface=surface,
        distribution=summarize_distribution(surface, results),
        plateau=analyse_plateau(surface, tolerance=tolerance),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """
    Format a parameter value compactly for a label.

    Args:
        value: Parameter value

    Returns:
        A short string: floats lose their trailing zeros, everything else is
        rendered with str().
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def _format_metric(value: Optional[float], metric: str) -> str:
    """
    Format a metric value with the unit the CLI prints it in.

    Args:
        value: The value, possibly None
        metric: Metric name

    Returns:
        A formatted string, or 'n/a' when the value is None.
    """
    if value is None:
        return 'n/a'
    if metric in ('total_return', 'max_drawdown', 'win_rate', 'excess_return_pct'):
        return f'{value:.2f}%'
    if metric == 'sharpe_ratio':
        return f'{value:.3f}'
    if metric == 'total_trades':
        return f'{value:.0f}'
    return f'{value:.4f}'


def choose_display_axes(surface: ParameterSurface) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick the two parameters the heatmap puts on its axes.

    The two axes with the most ticks are chosen, because they are the ones a
    surface can actually show a shape along; ties keep axis order. A
    one-parameter surface returns ``(name, None)`` and is drawn as a single row.

    Args:
        surface: The surface to display

    Returns:
        ``(x_parameter, y_parameter)``; the second is None for a
        one-dimensional surface.
    """
    names = surface.parameter_names
    if not names:
        return None, None
    if len(names) == 1:
        return names[0], None

    ranked = sorted(
        names, key=lambda name: (-len(surface.axes[name]), names.index(name))
    )
    first, second = ranked[0], ranked[1]
    # Keep the original parameter order between the two chosen axes so the
    # picture does not flip when two axes have the same number of ticks.
    if names.index(second) < names.index(first):
        first, second = second, first
    return first, second


def _level_char(cell: SurfaceCell, low: float, high: float) -> str:
    """
    Map a cell to its heatmap character.

    Args:
        cell: The cell
        low: Lowest oriented score on the surface
        high: Highest oriented score on the surface

    Returns:
        A digit for a scored cell (linear in the oriented score, so a flat
        surface looks flat instead of being stretched into artificial
        contrast), or the state character for anything else.
    """
    if cell.status == CELL_MISSING:
        return HEATMAP_MISSING
    if cell.status == CELL_NO_TRADES:
        return HEATMAP_NO_TRADES
    if cell.status == CELL_NON_FINITE:
        return HEATMAP_NON_FINITE

    if high <= low:
        return HEATMAP_LEVELS[-1]

    position = (float(cell.oriented_score) - low) / (high - low)
    level = int(position * (len(HEATMAP_LEVELS) - 1) + 1e-12)
    level = max(0, min(len(HEATMAP_LEVELS) - 1, level))
    return HEATMAP_LEVELS[level]


def render_heatmap(surface: ParameterSurface,
                   plateau: Optional[PlateauScore] = None,
                   x_parameter: Optional[str] = None,
                   y_parameter: Optional[str] = None) -> str:
    """
    Render the surface as an ASCII heatmap.

    Two parameters are displayed; any others are **pinned at the winner's
    values** rather than averaged over, so every character is one real backtest.
    The header names the slice and says how many cells it leaves out.

    Args:
        surface: The surface to render
        plateau: Plateau score, used to mark the winner and the plateau centre
        x_parameter: Parameter on the columns (default: chosen automatically)
        y_parameter: Parameter on the rows (default: chosen automatically)

    Returns:
        The rendered block, ending without a trailing newline.

    Raises:
        PlateauError: If a named parameter is not on the surface
    """
    names = surface.parameter_names
    for name in (x_parameter, y_parameter):
        if name is not None and name not in names:
            raise PlateauError(
                f"Parameter '{name}' is not on this surface. Available: {', '.join(names)}"
            )

    if x_parameter is None and y_parameter is None:
        x_parameter, y_parameter = choose_display_axes(surface)
    elif x_parameter is None:
        x_parameter = next((name for name in names if name != y_parameter), None)
    elif y_parameter is None and len(names) > 1:
        y_parameter = next((name for name in names if name != x_parameter), None)

    if x_parameter is None:
        return 'PARAMETER SURFACE: nothing to display (no parameters)'

    ok = surface.ok_cells
    lows = [float(cell.oriented_score) for cell in ok]
    low, high = (min(lows), max(lows)) if lows else (0.0, 0.0)

    x_values = surface.axes[x_parameter]
    y_values = surface.axes[y_parameter] if y_parameter else [None]

    pinned = {
        name: (plateau.winner_parameters[name] if plateau is not None
               else (surface.best_cell().parameters[name] if surface.best_cell() else
                     surface.axes[name][0]))
        for name in names if name not in (x_parameter, y_parameter)
    }

    lines: List[str] = []
    direction = 'higher is better' if surface.higher_is_better else 'lower is better'
    lines.append(f'PARAMETER SURFACE - {surface.metric} ({direction})')
    lines.append(f'  columns: {x_parameter}    rows: {y_parameter or "(single row)"}')

    if pinned:
        slice_text = ', '.join(f'{name}={_format_value(value)}'
                               for name, value in pinned.items())
        shown = len(x_values) * len(y_values)
        lines.append(f'  slice: {slice_text} (the winner\'s values); '
                     f'{surface.size - shown} of {surface.size} cells not shown')

    if surface.coverage < 1.0:
        lines.append(f'  coverage: {surface.coverage * 100:.1f}% of the grid has a result; '
                     'the rest was never evaluated,')
        lines.append('  so most of what this picture shows is the sampling, not the strategy')

    gutter = max(len(_format_value(value)) for value in y_values) if y_parameter else 0
    gutter = max(gutter, 1)
    labels = [_format_value(value) for value in x_values]
    label_height = max(len(label) for label in labels)

    for row in range(label_height):
        padded = ''.join(label.rjust(label_height)[row] for label in labels)
        lines.append(f'  {"".ljust(gutter)} | {padded}')
    lines.append(f'  {"".ljust(gutter)} +-{"-" * len(labels)}')

    for y_index, y_value in enumerate(y_values):
        characters: List[str] = []
        for x_index in range(len(x_values)):
            index = []
            for name in names:
                if name == x_parameter:
                    index.append(x_index)
                elif name == y_parameter:
                    index.append(y_index)
                else:
                    index.append(surface.axes[name].index(pinned[name]))
            key = tuple(index)
            cell = surface.cells[key]
            if plateau is not None and key == plateau.winner_index:
                characters.append(HEATMAP_WINNER)
            elif plateau is not None and key == plateau.centre_index:
                characters.append(HEATMAP_CENTRE)
            else:
                characters.append(_level_char(cell, low, high))
        row_label = _format_value(y_value).rjust(gutter) if y_parameter else ''.ljust(gutter)
        lines.append(f'  {row_label} | {"".join(characters)}')

    lines.append('')
    lines.append(f'  legend: {HEATMAP_LEVELS[0]}..{HEATMAP_LEVELS[-1]} = linear scale '
                 f'from {_format_metric(low if surface.higher_is_better else -low, surface.metric)} '
                 f'to {_format_metric(high if surface.higher_is_better else -high, surface.metric)}'
                 f'   {HEATMAP_WINNER} = winner   {HEATMAP_CENTRE} = plateau centre')
    lines.append(f'          {HEATMAP_NO_TRADES} = never traded   '
                 f'{HEATMAP_NON_FINITE} = non-finite metric   '
                 f'{HEATMAP_MISSING} = no result (errored, unsampled or discarded)')

    return '\n'.join(lines)


def render_distribution(stats: DistributionStats, surface: ParameterSurface) -> str:
    """
    Render the whole-grid distribution block.

    This is the counterweight to printing only the winner, so it always states
    how many combinations were scored, how the rest fell out, and how many beat
    doing nothing - in plain language when that number is small.

    Args:
        stats: The distribution statistics
        surface: The surface they describe

    Returns:
        The rendered block.
    """
    counts = stats.status_counts or surface.status_counts
    lines: List[str] = []
    lines.append(f'GRID DISTRIBUTION - {stats.metric}')

    if surface.selection == SELECTION_SAMPLED:
        scope = (f'{stats.count} scored of {surface.evaluated} sampled combinations '
                 f'({surface.size} in the full grid)')
    else:
        scope = f'{stats.count} scored of {surface.size} combinations in the grid'
    lines.append(f'  {scope}')
    lines.append(f'  cells: {counts[CELL_OK]} scored | {counts[CELL_NO_TRADES]} never traded '
                 f'| {counts[CELL_NON_FINITE]} non-finite | {counts[CELL_MISSING]} no result')

    if not stats.reliable:
        lines.append(f'  NO DISTRIBUTION REPORTED: {stats.unreliable_reason}.')
        lines.append('  Quartiles and the beat-the-baseline fraction are withheld rather')
        lines.append('  than printed from a biased sample, which would flatter the grid.')
        return '\n'.join(lines)

    lines.append(
        f'  min {_format_metric(stats.minimum, stats.metric)} | '
        f'q1 {_format_metric(stats.q1, stats.metric)} | '
        f'median {_format_metric(stats.median, stats.metric)} | '
        f'q3 {_format_metric(stats.q3, stats.metric)} | '
        f'max {_format_metric(stats.maximum, stats.metric)}'
    )

    if stats.baseline is None:
        lines.append('  baseline: none exists for this metric, so no '
                     '"beats doing nothing" count is reported')
        return '\n'.join(lines)

    lines.append(f'  baseline: {stats.baseline_label} = '
                 f'{_format_metric(stats.baseline, stats.metric)}')

    fraction = stats.fraction_beating_baseline or 0.0
    lines.append(f'  beating the baseline: {stats.beat_count} of {stats.count} '
                 f'({fraction * 100:.1f}%)')

    if stats.median is not None:
        gap = stats.median - stats.baseline
        better = gap > 0 if surface.higher_is_better else gap < 0
        lines.append(
            f'  the median combination {"beat" if better else "trailed"} the baseline by '
            f'{abs(gap):.2f} ({stats.metric} units)'
        )

    lines.append(f'  READ THIS: the winner is the maximum of {stats.count} estimates on one')
    lines.append('  dataset, not an effect. There is no multiple-testing correction here.')
    if fraction <= 0.10:
        lines.append(f'  Only {fraction * 100:.1f}% of the grid beat the baseline at all - a '
                     'single winner')
        lines.append('  that beats it is the kind of thing a grid this size produces by chance.')

    return '\n'.join(lines)


def render_plateau(plateau: Optional[PlateauScore], surface: ParameterSurface,
                   show_centre: bool = False) -> str:
    """
    Render the plateau / robustness block.

    Args:
        plateau: The plateau score, or None when nothing scored
        surface: The surface it came from
        show_centre: Whether to print the plateau-centre recommendation. It is
            opt-in and clearly labelled, because it is *not* what the optimizer
            returned as its winner

    Returns:
        The rendered block.
    """
    if plateau is None:
        return ('PLATEAU ANALYSIS\n'
                '  no combination produced a usable score, so there is no surface to read')

    lines: List[str] = []
    lines.append('PLATEAU ANALYSIS')
    lines.append(f'  winner: {plateau.winner_parameters}')
    lines.append(f'    {plateau.metric} = {_format_metric(plateau.winner_score, plateau.metric)}'
                 f'   grid median = {_format_metric(plateau.grid_median if surface.higher_is_better else -plateau.grid_median, plateau.metric)}')

    if plateau.retention is None:
        lines.append(f'  plateau score: no verdict - {plateau.retention_reason}')
    else:
        lines.append(f'  plateau score: {plateau.retention:.2f} ({plateau.verdict})')
        lines.append(f'    = mean edge of the {plateau.neighbours_ok} scored neighbours over the '
                     'grid median,')
        lines.append("      divided by the winner's own edge over it. 1.00 = the neighbourhood is")
        lines.append('      as good as the peak; 0.00 = the peak is an isolated spike.')
    lines.append(f'    computed from {plateau.neighbours_ok} of {plateau.neighbours_total} '
                 'adjacent combinations')

    lines.append(f'  plateau region: {plateau.region_size} contiguous combinations '
                 f'({plateau.region_fraction * 100:.1f}% of the scored grid) within '
                 f'{(1 - plateau.tolerance) * 100:.0f}% of the peak edge')
    for name, (low, high) in plateau.region_extent.items():
        edge = ' (reaches the edge of the searched range)' \
            if name in plateau.boundary_parameters else ''
        lines.append(f'    {name}: {_format_value(low)} .. {_format_value(high)}{edge}')

    if plateau.boundary_parameters:
        lines.append('  CAVEAT: the region runs into the edge of the parameter space along '
                     f'{", ".join(plateau.boundary_parameters)}.')
        lines.append('  It has not been shown to end there - the search stopped there. Widen')
        lines.append('  the range before reading this plateau as a bounded region.')

    if show_centre and plateau.centre_parameters is not None:
        lines.append('  plateau centre (NOT the optimizer\'s winner - reported for comparison):')
        lines.append(f'    {plateau.centre_parameters}')
        lines.append(f'    {plateau.metric} = {_format_metric(plateau.centre_score, plateau.metric)}')
        lines.append('    The middle of a region tends to survive out-of-sample better than its')
        lines.append('    highest point, but this run has not tested that claim on held-out data.')

    return '\n'.join(lines)


def render_report(report: PlateauReport, show_heatmap: bool = False,
                  show_centre: bool = False) -> str:
    """
    Render the full console report.

    Args:
        report: The analysis to render
        show_heatmap: Whether to include the ASCII heatmap
        show_centre: Whether to include the plateau-centre recommendation

    Returns:
        The rendered report.
    """
    blocks = [_SEPARATOR,
              render_distribution(report.distribution, report.surface),
              '',
              render_plateau(report.plateau, report.surface, show_centre=show_centre)]

    if show_heatmap:
        blocks.append('')
        blocks.append(render_heatmap(report.surface, report.plateau))

    blocks.append(_SEPARATOR)
    return '\n'.join(blocks)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

#: Metrics written beside the analysed one, so an external plot can colour the
#: surface by something else without re-running the optimisation.
_CSV_METRICS = ('total_return_pct', 'sharpe_ratio', 'max_drawdown', 'win_rate',
                'total_trades', 'excess_return_pct')


def write_surface_csv(surface: ParameterSurface, path: str,
                      plateau: Optional[PlateauScore] = None) -> int:
    """
    Write the full surface to CSV, one row per cell of the cartesian product.

    Missing cells are written with empty metric fields and ``status=missing``,
    never as zeros: a combination that errored is not a combination that
    returned nothing. Non-finite values become empty fields through the shared
    :func:`~niffler.utils.json_utils.sanitize_numeric_values`, so the file has
    no ``inf``/``nan`` literals for a plotting library to choke on.

    Args:
        surface: The surface to write
        path: Destination file path
        plateau: Plateau score, used to flag the winner, the region members and
            the region centre

    Returns:
        The number of data rows written.
    """
    members: set = set()
    if plateau is not None:
        # Recomputing membership here would risk drifting from analyse_plateau,
        # so the region is re-derived from the same threshold instead.
        members = {
            cell.index for cell in surface.ok_cells
            if float(cell.oriented_score) >= plateau.threshold
        }

    names = surface.parameter_names
    header = (list(names) + ['status', 'metric', 'metric_value']
              + list(_CSV_METRICS) + ['is_winner', 'within_plateau_band', 'is_plateau_centre'])

    rows = 0
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for _, cell in sorted(surface.cells.items()):
            values: List[Any] = [cell.parameters[name] for name in names]
            values.append(cell.status)
            values.append(surface.metric)
            values.append(cell.score)

            backtest = cell.result.backtest_result if cell.result is not None else None
            for metric_name in _CSV_METRICS:
                values.append(getattr(backtest, metric_name, None) if backtest else None)

            values.append(plateau is not None and cell.index == plateau.winner_index)
            values.append(cell.index in members)
            values.append(plateau is not None and cell.index == plateau.centre_index)

            writer.writerow(
                ['' if value is None else value
                 for value in sanitize_numeric_values(values)]
            )
            rows += 1

    logger.info(f"Parameter surface written to {path} ({rows} cells)")
    return rows
