"""What the engine can express, declared as data.

A strategy spec (:mod:`niffler.strategies.spec`) lists the capabilities its rules
need under ``[requires]``. This module is the one list those names are checked
against, and the one place that says which of them the engine supports today.

A spec that needs an unsupported capability is recorded as **unsupported**, with
the reasons below, and the scaffold refuses to generate code for it. That is the
point of the list: an idea the engine cannot express is written down as such
rather than approximated into something it is not. When a capability lands, its
entry flips to ``supported=True`` and every spec waiting on it becomes buildable.

Pure data with no imports from the rest of Niffler, so anything may read it.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Capability:
    """One thing a strategy's rules may need from the engine.

    Attributes:
        description: What the capability is, in a strategy author's terms.
        supported: Whether the engine can express it today.
        note: For a supported capability, how a strategy uses it. For an
            unsupported one, why not - this text is the reason recorded
            against every spec that needs it.
    """

    description: str
    supported: bool
    note: str


CAPABILITIES: Dict[str, Capability] = {
    # --- supported -----------------------------------------------------------
    'long_entry': Capability(
        description='Buy on a signal computed at the close of a bar.',
        supported=True,
        note="Emit signal 1. The order is a market order filled at the next bar's open.",
    ),
    'signal_exit': Capability(
        description='Sell an open long on a signal computed at the close of a bar.',
        supported=True,
        note="Emit signal -1. Filled at the next bar's open; a -1 while flat does nothing.",
    ),
    'fractional_position': Capability(
        description='Commit a fixed fraction of the portfolio, between 0 and 1, per entry.',
        supported=True,
        note='Every strategy takes position_size; it is never a spec parameter.',
    ),
    'scale_in': Capability(
        description='Buy again while already long.',
        supported=True,
        note='Entry price becomes the quantity-weighted average; a stop only ever tightens.',
    ),
    'fixed_pct_stop': Capability(
        description='A stop-loss at a fixed percentage below the entry price.',
        supported=True,
        note='Set by the run (--risk-manager fixed --stop-loss-pct), not by the strategy.',
    ),
    'single_instrument_bars': Capability(
        description='OHLCV bars of the one instrument being traded, at daily or coarser spacing.',
        supported=True,
        note='Columns open, high, low, close and volume; the spec names which it reads.',
    ),

    # --- not supported -------------------------------------------------------
    'strategy_stop': Capability(
        description='A stop price the strategy chooses itself, e.g. below a swing low or at 2 ATR.',
        supported=False,
        note='The only stop is a fixed percentage set by the risk manager (roadmap build order step 1).',
    ),
    'take_profit': Capability(
        description='Exit at a profit target price.',
        supported=False,
        note='There is no take-profit order (roadmap build order step 1).',
    ),
    'trailing_stop': Capability(
        description='A stop that follows the price up and never moves down.',
        supported=False,
        note='There is no trailing stop (roadmap build order step 1).',
    ),
    'time_exit': Capability(
        description='Exit after a fixed number of bars in the trade.',
        supported=False,
        note='There is no exit after N bars (roadmap build order step 1).',
    ),
    'short_entry': Capability(
        description='Sell short.',
        supported=False,
        note='Niffler is long-only by design; shorting touches the portfolio, borrow costs and the benchmark.',
    ),
    'limit_entry': Capability(
        description='Enter with a limit order at a chosen price.',
        supported=False,
        note="Every order is a market order filled at the next bar's open.",
    ),
    'stop_entry': Capability(
        description='Enter with a stop order when the price trades through a level during the bar.',
        supported=False,
        note="Every order is a market order filled at the next bar's open.",
    ),
    'same_bar_close_fill': Capability(
        description='Fill at the close of the bar whose close produced the signal.',
        supported=False,
        note='Deliberately never offered: it trades at a price the signal already saw (look-ahead).',
    ),
    'intraday_bars': Capability(
        description='Bars shorter than a day.',
        supported=False,
        note='The engine copes, but no intraday research or holdout data exists.',
    ),
    'multiple_instruments': Capability(
        description='Read or trade more than one instrument, e.g. pairs or rotation.',
        supported=False,
        note='Every script takes one data file and the portfolio is single-symbol.',
    ),
    'external_series': Capability(
        description='Read a series other than the traded OHLCV, e.g. VIX, rates or fundamentals.',
        supported=False,
        note='A strategy sees only the OHLCV frame of the instrument it trades.',
    ),
    'leverage': Capability(
        description='Hold a position worth more than the portfolio.',
        supported=False,
        note='position_size is capped at 1.',
    ),
}


def supported_capabilities() -> List[str]:
    """Return the names of the capabilities the engine supports, in declaration order."""
    return [name for name, capability in CAPABILITIES.items() if capability.supported]


def unsupported_capabilities() -> List[str]:
    """Return the names of the capabilities the engine does not support."""
    return [name for name, capability in CAPABILITIES.items() if not capability.supported]
