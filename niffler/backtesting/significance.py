"""
Statistical significance of a backtest.

A backtest tells you what happened. It does not tell you whether what happened
is distinguishable from luck. This module supplies the two cheapest honest
answers to that question and, just as importantly, a gate that refuses to answer
it when the sample is too small.

What is tested
--------------
The t-test is on the **mean round-trip return**, not on the mean per-bar return.
Round trips are the closest thing this framework has to independent draws: a
per-bar series is dominated by bars where the strategy is flat (a return of
exactly zero) and by strong serial dependence while it is in a position, so
testing it would inflate the sample size, shrink the standard error and
manufacture significance out of nothing. Round trips are pulled from
:func:`niffler.backtesting.round_trip.pair_trades` - the single FIFO pairing
routine - so the sample the test runs on is the same sample the win rate and
profit factor come from.

What this does NOT prove
------------------------
A small p-value here is weak evidence, and the docs and the console say so:

* It is **one asset over one window**. Nothing here is a claim about any other
  market or any other period.
* If the parameters were fitted on this same data, the p-value is **not**
  corrected for that. Searching a grid of 200 parameter sets and reporting the
  best one's p-value at the 5% level finds "significance" by construction.
  Multiple-testing correction and the deflated Sharpe ratio are deliberately out
  of scope here.
* Round trips are treated as i.i.d. Overlapping positions, regime persistence
  and volatility clustering all violate that to some degree.
* The t-test assumes an approximately normal *mean*. Trade returns are skewed
  and fat-tailed; the central limit theorem rescues the mean slowly, which is
  another reason for the minimum-sample gate.

The gate
--------
Below ``min_trades`` round trips the framework reports the numbers and
explicitly **refuses a verdict**. A strategy with 12 round trips tells you
nothing regardless of its win rate, and a confident-looking p-value printed next
to it is worse than no p-value at all.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics
from .round_trip import RoundTrip

#: Round trips below which no verdict is rendered. Thirty is the conventional
#: rule-of-thumb point where the t-distribution has settled down enough for a
#: skewed sample's mean to be roughly normal. It is a convention, not a
#: theorem - hence ``min_trades`` is a parameter.
DEFAULT_MIN_TRADES = 30

#: Significance level used to phrase the verdict.
DEFAULT_ALPHA = 0.05

#: Two-sided confidence level for the bootstrap Sharpe interval.
DEFAULT_CONFIDENCE_LEVEL = 0.95

#: Bootstrap resamples used when a confidence interval is requested. The
#: bootstrap is the only expensive part of this module, so it is opt-in: the
#: engine defaults to 0 (skip it) and the backtest CLI turns it on.
DEFAULT_BOOTSTRAP_SAMPLES = 1000

#: Fixed default seed. The audit found workers silently drawing fresh entropy;
#: every random draw here takes its seed explicitly rather than touching global
#: state, so two runs of the same backtest produce the same interval.
DEFAULT_BOOTSTRAP_SEED = 42

#: Smallest denominator treated as usable in the continued fraction below.
_FP_MIN = 1e-300


@dataclass
class SignificanceResult:
    """
    Outcome of the significance assessment of one backtest.

    Attributes:
        round_trips: Number of realised round trips the test ran on
        min_trades: Threshold below which no verdict is rendered
        is_sample_sufficient: True when ``round_trips >= min_trades``
        mean_trade_return_pct: Mean round-trip return as a percentage of the
            entry notional
        t_statistic: t-statistic of that mean against a null of zero, or None
            when it is undefined (fewer than two trades, or zero dispersion)
        p_value: Two-sided p-value of the t-statistic, or None when undefined.
            Halve it for a one-sided "the edge is positive" reading.
        sharpe_ci_low: Lower bound of the bootstrap Sharpe interval, or None
            when the bootstrap was not run
        sharpe_ci_high: Upper bound of the bootstrap Sharpe interval, or None
        confidence_level: Confidence level of that interval
        bootstrap_samples: Number of resamples drawn (0 = bootstrap skipped)
        bootstrap_seed: Seed the resamples were drawn from
        verdict: One-line, deliberately unexcited summary
    """

    round_trips: int = 0
    min_trades: int = DEFAULT_MIN_TRADES
    is_sample_sufficient: bool = False
    mean_trade_return_pct: Optional[float] = None
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    sharpe_ci_low: Optional[float] = None
    sharpe_ci_high: Optional[float] = None
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    bootstrap_samples: int = 0
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    verdict: str = ''

    @property
    def is_significant(self) -> Optional[bool]:
        """
        Whether the mean trade return is distinguishable from zero.

        Returns:
            True/False at the 5% level, or **None** when the sample is below the
            gate or the test is undefined. None means "no verdict", and callers
            must render it as such rather than collapsing it to False.
        """
        if not self.is_sample_sufficient or self.p_value is None:
            return None
        return self.p_value < DEFAULT_ALPHA


def assess_significance(round_trips: Sequence[RoundTrip],
                        portfolio_values: Optional[pd.Series] = None,
                        periods_per_year: float = 252.0,
                        min_trades: int = DEFAULT_MIN_TRADES,
                        bootstrap_samples: int = 0,
                        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
                        seed: int = DEFAULT_BOOTSTRAP_SEED) -> SignificanceResult:
    """
    Test whether a backtest's edge is distinguishable from noise.

    Args:
        round_trips: Realised round trips from
            :func:`niffler.backtesting.round_trip.pair_trades`
        portfolio_values: Strategy equity curve, needed for the bootstrap Sharpe
            interval. Omit it (or pass ``bootstrap_samples=0``) to skip it.
        periods_per_year: Annualisation factor, inferred from the data by the
            engine and passed in so the bootstrap Sharpe matches the reported one
        min_trades: Round trips below which no verdict is rendered
        bootstrap_samples: Resamples for the Sharpe confidence interval; 0 skips
            the bootstrap entirely
        confidence_level: Two-sided confidence level for that interval
        seed: Seed for the bootstrap, passed explicitly rather than relying on
            global numpy state

    Returns:
        The assessment, with a verdict that says "too small to tell" whenever it
        is too small to tell

    Raises:
        ValueError: If ``min_trades`` is negative, ``bootstrap_samples`` is
            negative, or ``confidence_level`` is outside (0, 1)
    """
    if min_trades < 0:
        raise ValueError(f"min_trades cannot be negative, got {min_trades}")
    if bootstrap_samples < 0:
        raise ValueError(f"bootstrap_samples cannot be negative, got {bootstrap_samples}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1, got {confidence_level}"
        )

    returns_pct = trade_return_percentages(round_trips)
    count = len(returns_pct)

    result = SignificanceResult(
        round_trips=count,
        min_trades=min_trades,
        is_sample_sufficient=count >= min_trades,
        confidence_level=confidence_level,
        bootstrap_seed=seed,
    )

    if count:
        result.mean_trade_return_pct = float(np.mean(returns_pct))
        result.t_statistic, result.p_value = one_sample_t_test(returns_pct)

    if bootstrap_samples > 0 and portfolio_values is not None:
        low, high = bootstrap_sharpe_interval(
            portfolio_values, periods_per_year,
            samples=bootstrap_samples, confidence_level=confidence_level, seed=seed
        )
        result.sharpe_ci_low = low
        result.sharpe_ci_high = high
        result.bootstrap_samples = bootstrap_samples

    result.verdict = _build_verdict(result)
    return result


def trade_return_percentages(round_trips: Sequence[RoundTrip]) -> List[float]:
    """
    Round-trip P&L as a percentage of the capital each round trip tied up.

    Percentages rather than currency amounts, because the position size changes
    with the account balance: a $200 win on $1,000 and a $2,000 win on $10,000
    are the same trade, and averaging their dollar values would weight late
    trades far more heavily than early ones.

    Args:
        round_trips: Realised round trips

    Returns:
        One percentage per round trip, net of both commissions. Round trips with
        a non-positive entry notional are skipped, since their return is not
        defined.
    """
    returns: List[float] = []
    for rt in round_trips:
        entry_value = rt.entry_price * rt.quantity
        if entry_value <= 0:
            continue
        returns.append(rt.pnl / entry_value * 100)
    return returns


def one_sample_t_test(sample: Sequence[float]) -> tuple:
    """
    Two-sided one-sample t-test of a sample's mean against zero.

    Args:
        sample: Observations, e.g. per-round-trip returns

    Returns:
        ``(t_statistic, p_value)``, or ``(None, None)`` when the test is
        undefined: fewer than two observations, or a sample with zero
        dispersion (every trade identical), where the t-statistic is infinite
        and the p-value meaningless.
    """
    values = np.asarray(sample, dtype=float)
    n = values.size
    if n < 2:
        return None, None

    # ddof=1: the sample standard deviation, which is what a t-test uses.
    std = float(values.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None, None

    t_statistic = float(values.mean() / (std / math.sqrt(n)))
    return t_statistic, student_t_two_sided_p(t_statistic, n - 1)


def student_t_two_sided_p(t_statistic: float, degrees_of_freedom: int) -> float:
    """
    Two-sided p-value of Student's t distribution.

    Implemented here rather than pulled in from scipy. scipy is a large
    dependency to add for one special function, and the alternative shortcut -
    approximating the t distribution with a normal - is wrong in exactly the
    regime that matters: at 30 observations the normal approximation understates
    a two-sided p-value by roughly 15%, which is the difference between "0.048,
    significant" and "0.056, not". The exact identity is used instead::

        P(|T| >= |t|) = I_x(df/2, 1/2),  x = df / (df + t^2)

    where ``I_x`` is the regularised incomplete beta function.

    Args:
        t_statistic: Observed t-statistic
        degrees_of_freedom: Degrees of freedom (n - 1 for a one-sample test)

    Returns:
        The two-sided p-value, in [0, 1]

    Raises:
        ValueError: If the degrees of freedom are not positive
    """
    if degrees_of_freedom <= 0:
        raise ValueError(
            f"degrees_of_freedom must be positive, got {degrees_of_freedom}"
        )
    if not math.isfinite(t_statistic):
        return 0.0

    df = float(degrees_of_freedom)
    x = df / (df + t_statistic * t_statistic)
    return regularized_incomplete_beta(df / 2.0, 0.5, x)


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """
    Regularised incomplete beta function ``I_x(a, b)``.

    Evaluated with the standard modified-Lentz continued fraction, using the
    symmetry ``I_x(a, b) = 1 - I_{1-x}(b, a)`` to stay on the rapidly converging
    side of the expansion.

    Args:
        a: First shape parameter, positive
        b: Second shape parameter, positive
        x: Point of evaluation, in [0, 1]

    Returns:
        ``I_x(a, b)`` in [0, 1]

    Raises:
        ValueError: If a or b is not positive, or x is outside [0, 1]
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"a and b must be positive, got a={a}, b={b}")
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")

    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)

    # The continued fraction converges quickly only for x below the
    # distribution's mode; reflect otherwise.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float,
                             max_iterations: int = 300,
                             tolerance: float = 3e-16) -> float:
    """
    Continued fraction for the incomplete beta function (modified Lentz).

    Args:
        a: First shape parameter
        b: Second shape parameter
        x: Point of evaluation, on the fast-converging side of the mode
        max_iterations: Iteration cap; convergence is typically reached in well
            under twenty
        tolerance: Relative change below which the expansion is considered
            converged

    Returns:
        The value of the continued fraction
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FP_MIN:
        d = _FP_MIN
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m

        # Even step.
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < _FP_MIN:
            d = _FP_MIN
        c = 1.0 + numerator / c
        if abs(c) < _FP_MIN:
            c = _FP_MIN
        d = 1.0 / d
        h *= d * c

        # Odd step.
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < _FP_MIN:
            d = _FP_MIN
        c = 1.0 + numerator / c
        if abs(c) < _FP_MIN:
            c = _FP_MIN
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < tolerance:
            break

    return h


def bootstrap_sharpe_interval(portfolio_values: pd.Series, periods_per_year: float,
                              samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
                              confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
                              seed: int = DEFAULT_BOOTSTRAP_SEED) -> tuple:
    """
    Percentile bootstrap confidence interval for the annualised Sharpe ratio.

    Per-bar returns are resampled **with replacement** and the Sharpe ratio is
    recomputed on each resample; the interval is the empirical percentile range
    of those values. A wide interval straddling zero is the useful case: it says
    the point estimate is a coin toss dressed up as a number.

    The resampling is i.i.d., which discards serial dependence: volatility
    clustering and momentum make the true interval wider than this one. A
    stationary or moving-block bootstrap would preserve that structure and is a
    reasonable follow-up; the block-bootstrap machinery already exists in
    :mod:`niffler.analysis.monte_carlo_analyzer` for the market-path question.

    The seed is an explicit argument and the generator is created locally.
    Nothing here reads or writes global numpy random state, so the interval is
    reproducible whatever else the process has been doing.

    Args:
        portfolio_values: Strategy equity curve
        periods_per_year: Annualisation factor
        samples: Number of bootstrap resamples
        confidence_level: Two-sided confidence level, e.g. 0.95
        seed: Seed for the resampling

    Returns:
        ``(low, high)`` bounds, or ``(None, None)`` when there are fewer than
        two returns to resample

    Raises:
        ValueError: If ``samples`` is not positive or ``confidence_level`` is
            outside (0, 1)
    """
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1, got {confidence_level}"
        )

    returns = metrics.periodic_returns(portfolio_values).to_numpy(dtype=float)
    n = returns.size
    if n < 2:
        return None, None

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(samples, n))
    resampled = returns[draws]

    means = resampled.mean(axis=1)
    stds = resampled.std(axis=1, ddof=1)

    scale = math.sqrt(periods_per_year)
    # A resample with no dispersion has no defined Sharpe ratio; it contributes
    # 0.0 rather than an inf that would drag a percentile to the moon.
    sharpes = np.where(stds > 0, scale * means / np.where(stds > 0, stds, 1.0), 0.0)

    tail = (1.0 - confidence_level) / 2.0 * 100
    low, high = np.percentile(sharpes, [tail, 100.0 - tail])
    return float(low), float(high)


def _build_verdict(result: SignificanceResult) -> str:
    """
    Phrase the assessment in one line, without overclaiming.

    Args:
        result: The populated assessment

    Returns:
        A sentence that says "no verdict" whenever there is no verdict
    """
    if result.round_trips == 0:
        return "No completed round trips: there is nothing to test."

    if not result.is_sample_sufficient:
        return (
            f"SAMPLE TOO SMALL: {result.round_trips} round trip(s), "
            f"{result.min_trades} required. No verdict - at this size the "
            f"results are indistinguishable from noise whatever they look like."
        )

    if result.p_value is None:
        return (
            f"Undefined test on {result.round_trips} round trips: the trade "
            f"returns have no dispersion, so no t-statistic exists."
        )

    mean = result.mean_trade_return_pct or 0.0
    if result.p_value < DEFAULT_ALPHA:
        headline = (
            f"Mean round-trip return {mean:+.3f}% differs from zero at the "
            f"{DEFAULT_ALPHA:.0%} level (p={result.p_value:.4f}, "
            f"n={result.round_trips})."
        )
    else:
        headline = (
            f"Mean round-trip return {mean:+.3f}% is NOT distinguishable from "
            f"zero (p={result.p_value:.4f}, n={result.round_trips})."
        )

    return (
        f"{headline} One asset, one window, no correction for parameters "
        f"fitted on this same data."
    )
