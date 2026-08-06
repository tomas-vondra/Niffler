"""
Equity-curve metrics shared by the engine and the benchmark.

The engine used to compute the max drawdown and the Sharpe ratio inline. The
moment a second equity curve exists - the buy-and-hold benchmark - that inline
code becomes a fork waiting to happen: one copy annualises from the index, the
other quietly hardcodes ``sqrt(252)``, and the two numbers stop being
comparable. Both curves go through the functions here instead, and the
annualisation factor is passed in from
:meth:`BacktestEngine.resolve_periods_per_year` so strategy and benchmark are
always annualised identically.

Sign convention: ``max_drawdown_pct`` is **negative** (-40.0 means the curve
lost 40% from its peak). The worst drawdown is the *smallest* number, so
ranking on it uses ``min``/``max`` accordingly - never ``max()`` for "worst".
"""

import numpy as np
import pandas as pd


def periodic_returns(portfolio_values: pd.Series) -> pd.Series:
    """
    Simple per-bar returns of an equity curve.

    Args:
        portfolio_values: Mark-to-market equity per bar

    Returns:
        Percentage change per bar with the undefined first observation dropped
    """
    return portfolio_values.pct_change().dropna()


def max_drawdown_pct(portfolio_values: pd.Series) -> float:
    """
    Worst peak-to-trough decline of an equity curve, in percent.

    Args:
        portfolio_values: Mark-to-market equity per bar

    Returns:
        A **negative** percentage: -40.0 means the curve fell 40% below its
        running peak. 0.0 for a curve that never falls below its peak, and 0.0
        for an empty curve.
    """
    if portfolio_values.empty:
        return 0.0

    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max
    worst = drawdown.min()
    if pd.isna(worst):
        return 0.0
    return float(worst) * 100


def sharpe_ratio(portfolio_values: pd.Series, periods_per_year: float) -> float:
    """
    Annualised Sharpe ratio of an equity curve, with a zero risk-free rate.

    The annualisation factor is a parameter rather than a constant on purpose:
    it is inferred from the data's own bar spacing by
    :meth:`BacktestEngine.resolve_periods_per_year`. A hardcoded ``sqrt(252)``
    overstates the Sharpe of daily crypto by ~20% and of hourly data by an order
    of magnitude.

    Args:
        portfolio_values: Mark-to-market equity per bar
        periods_per_year: Number of bars that make up one year

    Returns:
        The annualised Sharpe ratio, or 0.0 when there are fewer than two
        returns or the returns have no dispersion (a flat curve has no
        risk-adjusted performance to speak of).
    """
    returns = periodic_returns(portfolio_values)
    return sharpe_ratio_of_returns(returns, periods_per_year)


def sharpe_ratio_of_returns(returns: pd.Series, periods_per_year: float) -> float:
    """
    Annualised Sharpe ratio of a return series, with a zero risk-free rate.

    Args:
        returns: Per-bar returns
        periods_per_year: Number of bars that make up one year

    Returns:
        The annualised Sharpe ratio, or 0.0 when the series is too short or has
        zero standard deviation
    """
    if len(returns) <= 1:
        return 0.0

    std = returns.std()
    if not np.isfinite(std) or std <= 0:
        return 0.0

    return float(np.sqrt(periods_per_year) * returns.mean() / std)
