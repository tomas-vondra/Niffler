"""
Buy-and-hold benchmark.

A backtest that reports "+40%" and nothing else is not a result, it is a
number. If the asset itself did +120% over the same bars, +40% is a failure
wearing a success's clothes. This module produces the thing that number has to
be measured against: what would have happened by buying the asset once, at the
first bar the strategy itself could have traded, and doing nothing else.

Two properties make the comparison fair rather than decorative:

* **The benchmark pays the same costs.** Its entry goes through the engine's own
  :meth:`~niffler.backtesting.backtest_engine.BacktestEngine._execute_buy_trade`,
  so it is charged the same commission, priced by the same
  :class:`~niffler.backtesting.cost_model.CostModel`, capped by the same
  participation limit and sized by the same budget solver. A cost-free benchmark
  against a cost-charged strategy is a rigged comparison that always flatters
  buy-and-hold.
* **The benchmark obeys the same execution timing.** It buys at
  ``data.index[engine.execution_lag]``, i.e. the earliest bar a signal observed
  on the first bar could possibly have filled on. Buying at the very first bar's
  open under ``next_bar_open`` would hand the benchmark a bar of look-ahead the
  strategy never had.

Exit convention
---------------
The benchmark **holds to the end** and is marked to market at the final bar's
close. It is never liquidated, so it never pays an exit cost. That is not a
favour: it is exactly what the engine does with a strategy position that is
still open on the last bar - ``final_capital`` is a mark-to-market, not a
liquidation. Charging the benchmark an exit the strategy is not charged would
tilt the comparison the other way. The consequence, stated plainly, is that
neither side's terminal position pays to get out; if that ever changes it has to
change for both at once.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import metrics
from .trade import Trade

#: Buy the asset once and hold it to the end of the data.
BENCHMARK_BUY_AND_HOLD = 'buy_and_hold'

#: Run no benchmark at all; the comparison fields stay unset.
BENCHMARK_NONE = 'none'

#: Benchmarks selectable from the engine and the command line.
BENCHMARK_CHOICES = (BENCHMARK_BUY_AND_HOLD, BENCHMARK_NONE)

logger = logging.getLogger(__name__)


class BenchmarkError(RuntimeError):
    """
    A requested benchmark could not be established.

    Raised rather than returning ``None`` so a run never reports strategy
    numbers next to a silently missing comparison. Callers who genuinely do not
    want a benchmark ask for ``benchmark='none'`` instead.
    """


@dataclass
class BenchmarkResult:
    """
    Equity curve and metrics of a passive benchmark over the strategy's bars.

    Attributes:
        name: Benchmark identifier, e.g. ``'buy_and_hold'``
        portfolio_values: Mark-to-market equity per bar, on the **same index**
            as the strategy's curve
        total_return_pct: Total return over the whole window, in percent
        sharpe_ratio: Annualised Sharpe ratio, annualised with the same factor
            the strategy used
        max_drawdown: Worst peak-to-trough decline, a **negative** percentage
        entry_trade: The single buy the benchmark executed, carrying the
            commission and slippage it paid
    """

    name: str
    portfolio_values: pd.Series
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    entry_trade: Optional[Trade] = field(default=None)

    @property
    def total_commission(self) -> float:
        """Commission the benchmark paid to establish its position."""
        return float(self.entry_trade.commission) if self.entry_trade else 0.0

    @property
    def total_slippage(self) -> float:
        """Cash the benchmark gave up to slippage and spread on entry."""
        return float(self.entry_trade.slippage_cost) if self.entry_trade else 0.0

    @property
    def total_cost(self) -> float:
        """Everything the benchmark paid to trade: commission plus slippage."""
        return self.total_commission + self.total_slippage

    @property
    def returns(self) -> pd.Series:
        """Per-bar returns of the benchmark's equity curve."""
        return metrics.periodic_returns(self.portfolio_values)


def compute_benchmark(engine, data: pd.DataFrame, symbol: str,
                      benchmark: str = BENCHMARK_BUY_AND_HOLD
                      ) -> Optional[BenchmarkResult]:
    """
    Build the requested benchmark over the same bars as a strategy run.

    Args:
        engine: The :class:`BacktestEngine` whose costs, commission, capital and
            execution timing the benchmark must inherit
        data: The exact price data the strategy was run on
        symbol: Symbol identifier, recorded on the benchmark's entry trade
        benchmark: ``'buy_and_hold'`` or ``'none'``

    Returns:
        The benchmark result, or None when ``benchmark='none'``

    Raises:
        ValueError: If the benchmark name is not one of BENCHMARK_CHOICES
        BenchmarkError: If the benchmark's entry could not be filled
    """
    if benchmark is None or benchmark == BENCHMARK_NONE:
        return None
    if benchmark != BENCHMARK_BUY_AND_HOLD:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. Available: {', '.join(BENCHMARK_CHOICES)}"
        )
    return compute_buy_and_hold(engine, data, symbol)


def compute_buy_and_hold(engine, data: pd.DataFrame, symbol: str) -> BenchmarkResult:
    """
    Buy the asset at the first executable bar and hold it to the last one.

    The entry is executed through the engine, not re-implemented here, so the
    benchmark pays the same commission, is priced by the same cost model and is
    truncated by the same participation cap as any strategy order. The cash left
    over after the buy stays as cash and is carried in the equity curve, which
    matters when a liquidity cap prevents the whole capital from being deployed.

    Args:
        engine: The BacktestEngine supplying capital, commission, cost model and
            execution timing
        data: Price data with OHLCV columns and a DatetimeIndex
        symbol: Symbol identifier for the entry trade

    Returns:
        The benchmark's equity curve and metrics

    Raises:
        BenchmarkError: If the data has no bar the benchmark could enter on, or
            the entry order does not execute (below the minimum order value, or
            a bar the cost model says cannot absorb any of it)
    """
    first_tradeable = engine.execution_lag
    if len(data) <= first_tradeable:
        raise BenchmarkError(
            f"Buy-and-hold needs at least {first_tradeable + 1} bars to enter under "
            f"execution timing '{engine.execution_timing}', got {len(data)}"
        )

    entry_index, entry_trade = _first_fillable_entry(engine, data, symbol, first_tradeable)
    entry_timestamp = data.index[entry_index]

    cash = engine.initial_capital - entry_trade.value - entry_trade.commission
    quantity = entry_trade.quantity

    # Flat before the entry bar, invested from it onwards.
    closes = data['close'].to_numpy(dtype=float)
    values = pd.Series(engine.initial_capital, index=data.index, dtype=float)
    values.iloc[entry_index:] = cash + quantity * closes[entry_index:]

    periods_per_year = engine.resolve_periods_per_year(data.index)
    total_return_pct = (
        (float(values.iloc[-1]) - engine.initial_capital) / engine.initial_capital * 100
    )

    logger.info(
        f"BENCHMARK buy_and_hold: bought {quantity:.6f} units of {symbol} at "
        f"${entry_trade.price:,.2f} on {entry_timestamp} "
        f"(commission ${entry_trade.commission:,.2f}, slippage "
        f"${entry_trade.slippage_cost:,.2f}); held to the last bar for "
        f"{total_return_pct:.2f}%"
    )

    return BenchmarkResult(
        name=BENCHMARK_BUY_AND_HOLD,
        portfolio_values=values,
        total_return_pct=total_return_pct,
        sharpe_ratio=metrics.sharpe_ratio(values, periods_per_year),
        max_drawdown=metrics.max_drawdown_pct(values),
        entry_trade=entry_trade,
    )


def _first_fillable_entry(engine, data: pd.DataFrame, symbol: str,
                          first_tradeable: int) -> tuple:
    """
    Find the first bar buy-and-hold can actually get filled on, and fill it.

    Normally that is the first bar the engine's execution timing allows. It can
    be later: under a liquidity-aware cost model a bar that traded nothing
    cannot absorb an order at any price, and a passive investor facing a halted
    market simply buys on the next day they can. Waiting is not look-ahead -
    the bar's own volume is information the engine already uses to fill the
    strategy's orders on that same bar - but it does mean the benchmark misses
    whatever the price did in the meantime, which is exactly what would have
    happened.

    Args:
        engine: The BacktestEngine supplying capital, commission and cost model
        data: Price data with OHLCV columns
        symbol: Symbol identifier for the entry trade
        first_tradeable: Index of the earliest bar the execution timing allows

    Returns:
        ``(entry_index, entry_trade)``

    Raises:
        BenchmarkError: If no bar in the whole window can absorb the order
    """
    prices = data[engine.execution_price_column].to_numpy(dtype=float)
    volumes = data['volume'].to_numpy(dtype=float)

    for index in range(first_tradeable, len(data)):
        trade = engine._execute_buy_trade(
            timestamp=data.index[index],
            symbol=symbol,
            price=float(prices[index]),
            position_size=1.0,
            available_cash=engine.initial_capital,
            bar_volume=float(volumes[index]),
        )
        if trade is not None:
            if index > first_tradeable:
                logger.warning(
                    f"BENCHMARK buy_and_hold could not enter {symbol} until "
                    f"{data.index[index]}: bars {data.index[first_tradeable]} onwards "
                    f"could not absorb the order. The benchmark misses the price move "
                    f"over those {index - first_tradeable} bar(s), as a real buyer "
                    f"would have."
                )
            return index, trade

    raise BenchmarkError(
        f"Buy-and-hold could not enter {symbol} on any bar between "
        f"{data.index[first_tradeable]} and {data.index[-1]}: an order for "
        f"${engine.initial_capital:,.2f} was everywhere either below the minimum order "
        f"value of ${engine.min_order_value:,.2f} or more than the bar's volume can "
        f"absorb. There is no passive alternative to compare against on this data."
    )


def information_ratio(strategy_values: pd.Series, benchmark_values: pd.Series,
                      periods_per_year: float) -> float:
    """
    Annualised information ratio of a strategy against a benchmark.

    The information ratio is the mean of the per-bar **active** return
    (strategy minus benchmark) divided by its standard deviation, annualised
    with the same factor both curves use. It answers "per unit of tracking
    error, how much did deviating from the benchmark pay?".

    Why this and not alpha/beta: a regression of the strategy on a single-asset
    buy-and-hold does produce an intercept, but calling that intercept "alpha"
    invites a CAPM reading it has not earned - the benchmark here is one
    instrument, not a market portfolio, and this framework has no risk-free
    rate. The information ratio needs neither assumption. It does inherit the
    Sharpe ratio's blind spot: it treats upside and downside tracking error
    alike, and it says nothing about whether the sample is long enough to mean
    anything (see :mod:`niffler.backtesting.significance` for that).

    Args:
        strategy_values: Strategy equity curve
        benchmark_values: Benchmark equity curve on the same index
        periods_per_year: Number of bars that make up one year

    Returns:
        The annualised information ratio, or 0.0 when the active return series
        is too short or has no dispersion (a strategy that tracks the benchmark
        exactly has no information to speak of)

    Raises:
        ValueError: If the two curves are not on the same index
    """
    if not strategy_values.index.equals(benchmark_values.index):
        raise ValueError(
            "Strategy and benchmark equity curves must share an index; "
            f"got {len(strategy_values)} and {len(benchmark_values)} bars"
        )

    active = (metrics.periodic_returns(strategy_values)
              - metrics.periodic_returns(benchmark_values))
    return metrics.sharpe_ratio_of_returns(active, periods_per_year)
