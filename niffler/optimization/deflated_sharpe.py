"""
Deflated Sharpe ratio: is the optimizer's winner above what the search finds by luck?

A parameter search evaluates N combinations and keeps the best one. The best of
N estimates is biased upwards even when no combination has any edge at all: flip
ten coins 396 times and somebody gets nine heads. A significance test run on the
winner alone judges it as if it were the only thing tried, which is exactly the
overstatement :mod:`niffler.backtesting.significance` warns about and does not
correct.

This module corrects it for the search, following Bailey & Lopez de Prado,
"The Deflated Sharpe Ratio" (2014):

1. **The luck line.** From the number of trials and the spread of their Sharpe
   ratios, the Sharpe the best of N *zero-edge* trials is expected to show::

       E[max SR] = sqrt(V[SR]) * ((1 - g) * Z^-1[1 - 1/N] + g * Z^-1[1 - 1/(N*e)])

   with ``g`` the Euler-Mascheroni constant and ``Z^-1`` the inverse normal CDF.
2. **The deflated Sharpe.** The probability that the winner's *true* Sharpe is
   above that line, given how noisy its own estimate is - which depends on the
   number of bars and on the skewness and kurtosis of its returns::

       DSR = Z[(SR - E[max SR]) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2)]

Two luck lines
--------------
"By luck" needs a null, and the published one - no trial has any edge - is easy
to clear for the wrong reason. Every combination of a long-only strategy on an
asset that rose carries the same market exposure, so the whole grid sits well
above zero and its winner clears a zero-centred luck line without the search
having found anything. A second line is therefore reported beside the first:
the grid's own mean Sharpe plus the same allowance. Its null is that every
combination is as good as the average one and the differences between them are
noise, which is the question a parameter search actually raises: did tuning
find anything?

Everything is computed in **per-bar** Sharpe, so no annualisation factor enters
the probability. The annualised figures printed beside it exist only to be read
next to the engine's own Sharpe, and the factor for them is passed in from
:meth:`BacktestEngine.resolve_periods_per_year`.

What it does not do
-------------------
* **N is the raw trial count by default.** Neighbouring parameter sets are close
  to the same strategy, so the number of *independent* trials is smaller than N.
  A smaller N lowers the luck line, so counting every trial over-corrects rather
  than under-corrects. ``effective_trials`` overrides it; estimating it from the
  trials themselves is not implemented.
* **One search.** Ten strategies with one grid each are ten searches; nothing
  here counts the other nine.
* **Neither null is "no edge over the market".** The first luck line is cleared
  by exposure alone; the second asks only whether the winner stands out from
  its own grid. Buy-and-hold is a separate comparison and stays one.
* **A truncated result set is refused.** When the optimizer discarded its
  worst-scoring results to cap memory, the survivors' Sharpe ratios are a
  score-biased sample and their spread is not the grid's.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from niffler.backtesting import metrics

from .optimization_result import OptimizationResult
from .plateau import SELECTION_TRUNCATED

EULER_MASCHERONI = 0.5772156649015329

#: Conventional bar for calling the winner distinguishable from a lucky one.
DEFAULT_CONFIDENCE = 0.95

STATUS_OK = 'ok'
STATUS_TRUNCATED = 'truncated'
STATUS_TOO_FEW_TRIALS = 'too_few_trials'
STATUS_NO_DISPERSION = 'no_dispersion'
STATUS_NO_RETURNS = 'no_returns'
STATUS_UNDEFINED = 'undefined'

#: Where the N in the formula came from.
TRIALS_EVALUATED = 'evaluated'
TRIALS_OVERRIDE = 'override'

#: What an exported optimization summary carries. ``deflated_sharpe`` and
#: ``deflated_sharpe_vs_grid`` are probabilities; the Sharpe figures are annualised.
SUMMARY_FIELDS = (
    'deflated_sharpe',
    'deflated_sharpe_vs_grid',
    'deflated_sharpe_status',
    'deflated_sharpe_trials',
    'deflated_sharpe_trials_source',
    'expected_max_sharpe',
    'expected_max_sharpe_vs_grid',
    'trial_sharpe_mean',
    'trial_sharpe_std',
)

_NORMAL = NormalDist()


@dataclass(frozen=True)
class ReturnMoments:
    """Per-bar Sharpe and shape of one return series."""
    sharpe: float
    skewness: float
    #: Non-excess kurtosis: 3.0 for a normal distribution.
    kurtosis: float
    observations: int


@dataclass(frozen=True)
class DeflatedSharpe:
    """
    The winner of a search, judged against the best a search that size finds by luck.

    Every numeric field is ``None`` unless ``status`` is :data:`STATUS_OK`, and
    ``reason`` then says why. None means "not computed" and must never be
    rendered or exported as zero.

    Sharpe figures are per-bar. ``periods_per_year`` is carried only so they can
    be shown annualised; it plays no part in ``probability``.
    """
    status: str
    reason: str = ''
    #: N used in the formula.
    trials: Optional[float] = None
    trials_source: str = TRIALS_EVALUATED
    #: Combinations the search evaluated.
    trials_evaluated: int = 0
    #: Of those, how many had returns with any dispersion to compute a Sharpe from.
    trials_with_sharpe: int = 0
    trial_sharpe_mean: Optional[float] = None
    trial_sharpe_std: Optional[float] = None
    #: The luck line: expected best Sharpe of ``trials`` zero-edge trials.
    expected_max_sharpe: Optional[float] = None
    #: The stricter luck line: the same allowance above the trials' own mean.
    expected_max_sharpe_vs_grid: Optional[float] = None
    winner: Optional[ReturnMoments] = None
    #: Probability that the winner's true Sharpe is above the luck line. This
    #: is the published deflated Sharpe ratio.
    probability: Optional[float] = None
    #: The same probability against the stricter line.
    probability_vs_grid: Optional[float] = None
    periods_per_year: Optional[float] = None

    @property
    def is_computed(self) -> bool:
        """Whether a probability was computed at all."""
        return self.status == STATUS_OK

    def annualised(self, per_bar: Optional[float]) -> Optional[float]:
        """
        Scale a per-bar Sharpe figure to a year.

        Args:
            per_bar: A per-bar Sharpe ratio or a standard deviation of them

        Returns:
            The annualised figure, or None when either the figure or the
            annualisation factor is unknown.
        """
        if per_bar is None or self.periods_per_year is None:
            return None
        return per_bar * math.sqrt(self.periods_per_year)


def expected_max_sharpe(trials: float, trial_sharpe_std: float) -> float:
    """
    Sharpe the best of ``trials`` zero-edge trials is expected to show.

    Args:
        trials: Number of independent trials, at least 1
        trial_sharpe_std: Standard deviation of the Sharpe ratio across trials

    Returns:
        The expected maximum, in the units of ``trial_sharpe_std``. Zero for a
        single trial: nothing was selected, so there is nothing to deflate.

    Raises:
        ValueError: If ``trials`` is below 1 or the spread is negative or not finite.
    """
    if not math.isfinite(trials) or trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials!r}")
    if not math.isfinite(trial_sharpe_std) or trial_sharpe_std < 0:
        raise ValueError(
            f"trial_sharpe_std must be finite and non-negative, got {trial_sharpe_std!r}")
    if trials == 1:
        return 0.0

    upper = _NORMAL.inv_cdf(1.0 - 1.0 / trials)
    lower = _NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return trial_sharpe_std * (
        (1.0 - EULER_MASCHERONI) * upper + EULER_MASCHERONI * lower)


def probabilistic_sharpe(sharpe: float, benchmark_sharpe: float, observations: int,
                         skewness: float, kurtosis: float) -> Optional[float]:
    """
    Probability that a true Sharpe ratio exceeds ``benchmark_sharpe``.

    Args:
        sharpe: The estimated Sharpe, per bar
        benchmark_sharpe: The Sharpe to clear, per bar
        observations: Number of returns the estimate was computed from
        skewness: Skewness of those returns
        kurtosis: Their non-excess kurtosis (3.0 for a normal distribution)

    Returns:
        A probability in [0, 1], or None when it is undefined: fewer than two
        observations, or moments so extreme that the estimator's variance is
        not positive.
    """
    if observations < 2:
        return None

    variance_term = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    if not math.isfinite(variance_term) or variance_term <= 0:
        return None

    z = (sharpe - benchmark_sharpe) * math.sqrt(observations - 1) / math.sqrt(variance_term)
    return _NORMAL.cdf(z)


def return_moments(returns: Sequence[float]) -> Optional[ReturnMoments]:
    """
    Per-bar Sharpe, skewness and kurtosis of a return series.

    Args:
        returns: Per-bar returns

    Returns:
        The moments, or None when the series has fewer than two observations or
        no dispersion - a flat equity curve has no Sharpe ratio to deflate.
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None

    std = float(values.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None

    mean = float(values.mean())
    centred = values - mean
    second = float(np.mean(centred ** 2))
    if second <= 0:
        return None

    return ReturnMoments(
        sharpe=mean / std,
        skewness=float(np.mean(centred ** 3)) / second ** 1.5,
        kurtosis=float(np.mean(centred ** 4)) / second ** 2,
        observations=len(values),
    )


def _moments_of_result(result: OptimizationResult) -> Optional[ReturnMoments]:
    """Moments of one trial's equity curve, or None when it has none to speak of."""
    portfolio_values = getattr(result.backtest_result, 'portfolio_values', None)
    if not isinstance(portfolio_values, pd.Series) or portfolio_values.empty:
        return None
    return return_moments(metrics.periodic_returns(portfolio_values))


def deflated_sharpe(winner: ReturnMoments, trial_sharpes: Sequence[float],
                    trials_evaluated: int,
                    effective_trials: Optional[float] = None,
                    periods_per_year: Optional[float] = None) -> DeflatedSharpe:
    """
    Deflate a winner's Sharpe by the search that selected it.

    Args:
        winner: Moments of the winner's per-bar returns
        trial_sharpes: Per-bar Sharpe of every trial that has one
        trials_evaluated: Combinations the search evaluated, including those
            with no Sharpe
        effective_trials: N to use instead of ``trials_evaluated``
        periods_per_year: Annualisation factor, for display only

    Returns:
        The result. Its status is not OK when fewer than two trials have a
        Sharpe, when they all share one, or when the winner's moments leave the
        probability undefined.

    Raises:
        ValueError: If ``effective_trials`` is below 1.
    """
    if effective_trials is not None and (
            not math.isfinite(effective_trials) or effective_trials < 1):
        raise ValueError(f"effective_trials must be at least 1, got {effective_trials!r}")

    sharpes = [float(value) for value in trial_sharpes if math.isfinite(value)]
    common: Dict[str, Any] = {
        'trials_evaluated': trials_evaluated,
        'trials_with_sharpe': len(sharpes),
        'periods_per_year': periods_per_year,
    }

    if len(sharpes) < 2:
        return DeflatedSharpe(
            status=STATUS_TOO_FEW_TRIALS,
            reason=(f'only {len(sharpes)} trial(s) produced a Sharpe ratio, and the '
                    f'spread across trials needs at least two'),
            **common)

    spread = float(np.std(sharpes, ddof=1))
    if not math.isfinite(spread) or spread <= 0:
        return DeflatedSharpe(
            status=STATUS_NO_DISPERSION,
            reason='every trial has the same Sharpe ratio, so there is no spread to deflate by',
            **common)

    trials = float(effective_trials) if effective_trials is not None else float(trials_evaluated)
    source = TRIALS_OVERRIDE if effective_trials is not None else TRIALS_EVALUATED
    luck_line = expected_max_sharpe(trials, spread)
    mean = float(np.mean(sharpes))
    measured: Dict[str, Any] = {
        'trials': trials,
        'trials_source': source,
        'trial_sharpe_mean': mean,
        'trial_sharpe_std': spread,
        'expected_max_sharpe': luck_line,
        'expected_max_sharpe_vs_grid': mean + luck_line,
        'winner': winner,
        **common,
    }

    probability = probabilistic_sharpe(
        winner.sharpe, luck_line, winner.observations, winner.skewness, winner.kurtosis)
    probability_vs_grid = probabilistic_sharpe(
        winner.sharpe, mean + luck_line, winner.observations, winner.skewness,
        winner.kurtosis)
    if probability is None or probability_vs_grid is None:
        return DeflatedSharpe(
            status=STATUS_UNDEFINED,
            reason=("the winner's skewness and kurtosis leave the variance of its "
                    "Sharpe estimate non-positive"),
            **measured)

    return DeflatedSharpe(status=STATUS_OK, probability=probability,
                          probability_vs_grid=probability_vs_grid, **measured)


def analyse_results(results: Sequence[OptimizationResult], selection: str,
                    effective_trials: Optional[float] = None,
                    periods_per_year: Optional[float] = None) -> DeflatedSharpe:
    """
    Deflate the Sharpe of the first result by the search that produced the list.

    The winner is ``results[0]``: whatever the optimizer ranked first, by
    whatever metric it was asked to sort on. When that metric is not the Sharpe
    ratio the winner's Sharpe is at most the grid's best, so judging it against
    the expected *best* Sharpe is the stricter reading.

    Args:
        results: The optimisation results, best first, exactly as returned
        selection: One of the ``plateau.SELECTION_*`` constants
        effective_trials: N to use instead of the number of results
        periods_per_year: Annualisation factor, for display only

    Returns:
        The result. A truncated result set is refused outright: its survivors
        were selected by score, so the spread of their Sharpe ratios is not the
        spread of the grid's.

    Raises:
        ValueError: If ``effective_trials`` is below 1.
    """
    if effective_trials is not None and (
            not math.isfinite(effective_trials) or effective_trials < 1):
        raise ValueError(f"effective_trials must be at least 1, got {effective_trials!r}")

    evaluated = len(results)
    if selection == SELECTION_TRUNCATED:
        return DeflatedSharpe(
            status=STATUS_TRUNCATED,
            reason=('the optimizer discarded its worst-scoring results to cap memory, so '
                    'the surviving trials are a score-biased sample'),
            trials_evaluated=evaluated,
            periods_per_year=periods_per_year)

    moments: List[Optional[ReturnMoments]] = [_moments_of_result(result) for result in results]
    sharpes = [moment.sharpe for moment in moments if moment is not None]

    winner = moments[0] if moments else None
    if winner is None:
        return DeflatedSharpe(
            status=STATUS_NO_RETURNS,
            reason=("the winner's equity curve has no dispersion, so it has no Sharpe "
                    "ratio to deflate"),
            trials_evaluated=evaluated,
            trials_with_sharpe=len(sharpes),
            periods_per_year=periods_per_year)

    return deflated_sharpe(
        winner, sharpes, evaluated,
        effective_trials=effective_trials,
        periods_per_year=periods_per_year)


def summary_fields(result: Optional[DeflatedSharpe]) -> Dict[str, Any]:
    """
    The figures an exported optimization summary carries.

    Sharpe figures are annualised here, because the summary's own
    ``sharpe_ratio`` is, and a luck line in different units beside it would be
    read as a comparison it is not.

    Args:
        result: The analysis, or None when it could not be run

    Returns:
        The fields named in :data:`SUMMARY_FIELDS`. Anything not computed is
        None.
    """
    if result is None:
        return dict.fromkeys(SUMMARY_FIELDS)

    return {
        'deflated_sharpe': result.probability,
        'deflated_sharpe_vs_grid': result.probability_vs_grid,
        'deflated_sharpe_status': result.status,
        'deflated_sharpe_trials': result.trials,
        'deflated_sharpe_trials_source': (
            result.trials_source if result.trials is not None else None),
        'expected_max_sharpe': result.annualised(result.expected_max_sharpe),
        'expected_max_sharpe_vs_grid': result.annualised(result.expected_max_sharpe_vs_grid),
        'trial_sharpe_mean': result.annualised(result.trial_sharpe_mean),
        'trial_sharpe_std': result.annualised(result.trial_sharpe_std),
    }


def _sharpe_text(result: DeflatedSharpe, per_bar: float) -> Tuple[str, str]:
    """A Sharpe figure and the unit it is shown in."""
    annual = result.annualised(per_bar)
    if annual is None:
        return f'{per_bar:.4f}', 'per bar'
    return f'{annual:.3f}', 'annualised'


def render_report(result: DeflatedSharpe, confidence: float = DEFAULT_CONFIDENCE) -> str:
    """
    Render the deflated-Sharpe block in plain language.

    Args:
        result: The analysis
        confidence: Probability the winner has to reach to be called above the
            luck line

    Returns:
        The rendered block. When nothing was computed it says so and why,
        rather than printing a zero.
    """
    lines: List[str] = ['DEFLATED SHARPE - is the winner above what a search this size finds by luck?']

    if result.trials is None:
        lines.append(f'  NOT COMPUTED: {result.reason}.')
        lines.append('  No luck line and no probability are reported rather than')
        lines.append('  printed from numbers that do not describe the search.')
        return '\n'.join(lines)

    if result.trials_source == TRIALS_OVERRIDE:
        lines.append(f'  trials counted: {result.trials:g} (--effective-trials; '
                     f'{result.trials_evaluated} combinations were evaluated)')
    else:
        lines.append(f'  trials counted: {result.trials:g} - every combination evaluated.')
        lines.append('    Neighbouring parameter sets are near-duplicates, so fewer of them are')
        lines.append('    independent; counting them all over-corrects rather than')
        lines.append('    under-corrects. --effective-trials overrides the count.')

    spread, unit = _sharpe_text(result, result.trial_sharpe_std)
    mean, _ = _sharpe_text(result, result.trial_sharpe_mean)
    lines.append(f'  trial Sharpe ratios: mean {mean}, spread {spread} {unit} '
                 f'(over {result.trials_with_sharpe} trials with a Sharpe)')

    winner_sharpe, _ = _sharpe_text(result, result.winner.sharpe)
    lines.append(f'  winner: {winner_sharpe} {unit} over {result.winner.observations} bars '
                 f'(skewness {result.winner.skewness:+.2f}, '
                 f'kurtosis {result.winner.kurtosis:.2f})')

    if result.probability is None or result.probability_vs_grid is None:
        lines.append(f'  NO PROBABILITY: {result.reason}.')
        return '\n'.join(lines)

    luck, _ = _sharpe_text(result, result.expected_max_sharpe)
    luck_vs_grid, _ = _sharpe_text(result, result.expected_max_sharpe_vs_grid)
    lines.append('  "By luck" needs a null, so there are two luck lines - the best Sharpe')
    lines.append('  that many trials with that spread are expected to show if:')
    lines.append('  1. no combination has any edge at all (the published deflated Sharpe)')
    lines.append(f'       luck line {luck} {unit} | probability the winner is truly '
                 f'above it: {result.probability * 100:.1f}%')
    lines.append('  2. every combination is as good as the average one, and tuning found nothing')
    lines.append(f'       luck line {luck_vs_grid} {unit} | probability the winner is truly '
                 f'above it: {result.probability_vs_grid * 100:.1f}%')

    clears_zero = result.probability >= confidence
    clears_grid = result.probability_vs_grid >= confidence
    if clears_zero and clears_grid:
        lines.append(f'  Both clear the conventional {confidence:.0%} bar: the winner is more '
                     f'than the')
        lines.append('  best of that many lucky tries, under either reading.')
    elif clears_zero:
        lines.append(f'  Only the first clears the conventional {confidence:.0%} bar. The '
                     f'strategy family')
        lines.append('  is above zero, but the winner is NOT distinguishable from the best of')
        lines.append('  that many equally good combinations: the search is no evidence for')
        lines.append('  these particular parameters.')
    else:
        lines.append(f'  The winner does NOT clear the conventional {confidence:.0%} bar: it is not')
        lines.append('  distinguishable from the best of that many lucky tries.')

    lines.append('  Neither line is "no edge over the market": a long-only strategy on an')
    lines.append('  asset that rose clears the first from exposure alone. This corrects')
    lines.append('  for this one search only. Still one asset, one window.')
    return '\n'.join(lines)
