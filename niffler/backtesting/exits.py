"""
Exits a strategy sets for itself.

A strategy chooses its exits on the bar that produces the buy signal, in four
optional columns next to ``signal`` (names in
:mod:`niffler.strategies.base_strategy`):

``stop_price``
    An absolute stop below the entry. On any later bar while the position is
    open, a value in the column *moves* the stop - upwards only, never looser -
    so a strategy can run its own trail (a Chandelier or ATR stop) one bar at a
    time.
``take_profit_price``
    An absolute target, filled like a resting limit sell.
``trailing_stop_pct``
    A stop that trails the highest high since entry by this fraction.
``max_bars_held``
    Leave at the open of the bar this many bars after the entry fill.

The engine reads every value from the *signal* bar, which is the bar the
strategy has already seen, so an exit cannot look ahead any more than a signal
can. The position-level state those values turn into lives on the portfolio as
an :class:`ExitPlan`; this module holds that state and nothing that trades.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from niffler.strategies.base_strategy import (
    MAX_BARS_HELD_COLUMN,
    STOP_PRICE_COLUMN,
    TAKE_PROFIT_COLUMN,
    TRAILING_STOP_COLUMN,
)

#: ``Trade.exit_reason`` values. A buy carries None.
EXIT_SIGNAL = 'signal'
EXIT_RISK_STOP = 'risk_stop'
EXIT_STOP = 'stop'
EXIT_TRAILING_STOP = 'trailing_stop'
EXIT_TAKE_PROFIT = 'take_profit'
EXIT_TIME = 'time'

EXIT_REASONS = (EXIT_SIGNAL, EXIT_RISK_STOP, EXIT_STOP, EXIT_TRAILING_STOP,
                EXIT_TAKE_PROFIT, EXIT_TIME)


@dataclass(frozen=True)
class ExitColumns:
    """The four exit columns of a strategy's output, NaN where unset.

    Attributes:
        stop_price: Absolute stop per bar
        take_profit_price: Absolute target per bar
        trailing_stop_pct: Trail distance per bar, as a fraction in (0, 1)
        max_bars_held: Holding limit per bar, an integer >= 1
    """
    stop_price: np.ndarray
    take_profit_price: np.ndarray
    trailing_stop_pct: np.ndarray
    max_bars_held: np.ndarray

    @classmethod
    def from_signals(cls, signals_df, expected_length: int) -> 'ExitColumns':
        """Read and validate the exit columns of a strategy's output.

        A column the strategy does not emit is all NaN, i.e. that exit is off.
        Every value is checked, not only those on buy bars: a malformed column
        is a bug in the strategy whichever bar it shows on.

        Args:
            signals_df: DataFrame returned by ``generate_signals``
            expected_length: Number of bars in the price data

        Returns:
            The validated columns

        Raises:
            ValueError: If any set value is out of range or not finite
        """
        def column(name: str) -> np.ndarray:
            if name not in signals_df.columns:
                return np.full(expected_length, np.nan)
            values = signals_df[name].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"Exit column '{name}' contains infinite values")
            return values

        stop = column(STOP_PRICE_COLUMN)
        target = column(TAKE_PROFIT_COLUMN)
        trail = column(TRAILING_STOP_COLUMN)
        bars = column(MAX_BARS_HELD_COLUMN)

        _reject(stop, stop <= 0, STOP_PRICE_COLUMN, "must be positive")
        _reject(target, target <= 0, TAKE_PROFIT_COLUMN, "must be positive")
        _reject(trail, (trail <= 0) | (trail >= 1), TRAILING_STOP_COLUMN,
                "must be a fraction strictly between 0 and 1")
        _reject(bars, (bars < 1) | (bars != np.floor(bars)), MAX_BARS_HELD_COLUMN,
                "must be a whole number of bars, at least 1")

        return cls(stop, target, trail, bars)

    @property
    def any_set(self) -> bool:
        """True when the strategy set at least one exit on at least one bar."""
        return any(
            not np.isnan(values).all()
            for values in (self.stop_price, self.take_profit_price,
                           self.trailing_stop_pct, self.max_bars_held)
        )

    def plan_for_entry(self, signal_index: int, entry_price: float,
                       entry_bar: int) -> Optional['ExitPlan']:
        """The exit plan a fresh position adopts from its signal bar.

        Args:
            signal_index: Bar that produced the buy signal
            entry_price: Price the entry filled at, which anchors the trail
            entry_bar: Bar the entry filled on

        Returns:
            The plan, or None when the signal bar sets no exit
        """
        if signal_index < 0:
            return None
        stop = _value(self.stop_price, signal_index)
        target = _value(self.take_profit_price, signal_index)
        trail = _value(self.trailing_stop_pct, signal_index)
        bars = _value(self.max_bars_held, signal_index)
        if stop is None and target is None and trail is None and bars is None:
            return None
        return ExitPlan(
            stop_price=stop,
            take_profit_price=target,
            trailing_stop_pct=trail,
            max_bars_held=None if bars is None else int(bars),
            trail_anchor=entry_price,
            entry_bar=entry_bar,
        )

    def stop_update(self, signal_index: int) -> Optional[float]:
        """The stop the strategy asked for on ``signal_index``, if any."""
        if signal_index < 0:
            return None
        return _value(self.stop_price, signal_index)


@dataclass
class ExitPlan:
    """Exit state of the one open position.

    Attributes:
        stop_price: Strategy stop, or None
        take_profit_price: Target, or None
        trailing_stop_pct: Trail distance below ``trail_anchor``, or None
        max_bars_held: Holding limit, or None
        trail_anchor: Highest price seen since entry, starting at the entry fill
        entry_bar: Bar index the position was opened on
        bars_held: Completed bars the position has been held for
    """
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_bars_held: Optional[int] = None
    trail_anchor: Optional[float] = None
    entry_bar: int = 0
    bars_held: int = 0

    @property
    def trailing_level(self) -> Optional[float]:
        """Where the trailing stop sits now, or None without a trail."""
        if self.trailing_stop_pct is None or self.trail_anchor is None:
            return None
        return self.trail_anchor * (1.0 - self.trailing_stop_pct)

    def stop_level(self) -> Optional[float]:
        """The binding stop: the higher of the strategy stop and the trail."""
        levels = [level for level in (self.stop_price, self.trailing_level)
                  if level is not None]
        return max(levels) if levels else None

    def stop_reason(self) -> str:
        """Which of the two stops is binding, for the trade log."""
        trail = self.trailing_level
        if trail is not None and (self.stop_price is None or trail > self.stop_price):
            return EXIT_TRAILING_STOP
        return EXIT_STOP

    def tighten_stop(self, stop_price: Optional[float]) -> None:
        """Adopt a stop only if it is higher than the one in force."""
        if stop_price is None:
            return
        if self.stop_price is None or stop_price > self.stop_price:
            self.stop_price = stop_price

    def time_exit_due(self) -> bool:
        """True once the position has been held for ``max_bars_held`` bars."""
        return self.max_bars_held is not None and self.bars_held >= self.max_bars_held

    def record_bar(self, bar_high: float, update_anchor: bool = True) -> None:
        """Account for one more completed bar of holding.

        Called after the bar's exits and orders, so the high a bar sets only
        moves the trail for the *next* bar: within one bar the engine cannot
        tell whether the high came before or after the low.

        Args:
            bar_high: Highest price traded on the bar just completed
            update_anchor: False when that high traded before the position
                was opened, so it must not lift the trail
        """
        self.bars_held += 1
        if not update_anchor:
            return
        if self.trail_anchor is None or bar_high > self.trail_anchor:
            self.trail_anchor = bar_high


def _value(values: np.ndarray, index: int) -> Optional[float]:
    value = float(values[index])
    return None if math.isnan(value) else value


def _reject(values: np.ndarray, invalid: np.ndarray, name: str, rule: str) -> None:
    invalid = invalid & ~np.isnan(values)
    if invalid.any():
        raise ValueError(f"Exit column '{name}' {rule}, got {values[invalid][0]}")
