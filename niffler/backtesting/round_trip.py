import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import List

from .trade import Trade, TradeSide

#: Quantities below this are treated as fully matched during trade pairing.
QUANTITY_EPSILON = 1e-12


@dataclass
class RoundTrip:
    """
    A realised round-trip: a quantity bought and later sold.

    Produced by the engine's single FIFO pairing routine. Every trade statistic
    (win rate, profit factor, average/largest win/loss, win/loss counts) is
    derived from these objects so the numbers can never disagree with each other.

    Partial fills produce several round trips out of one entry or one exit; the
    commissions of the underlying executions are apportioned pro-rata by the
    matched quantity.

    Attributes:
        symbol: Traded symbol
        entry_timestamp: Timestamp of the buy execution
        exit_timestamp: Timestamp of the sell execution
        quantity: Matched quantity for this round trip
        entry_price: Buy fill price
        exit_price: Sell fill price
        entry_commission: Share of the buy commission attributable to `quantity`
        exit_commission: Share of the sell commission attributable to `quantity`
    """
    symbol: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    quantity: float
    entry_price: float
    exit_price: float
    entry_commission: float = 0.0
    exit_commission: float = 0.0

    @property
    def gross_pnl(self) -> float:
        """Profit/loss before commission, in account currency."""
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl(self) -> float:
        """Realised profit/loss net of both entry and exit commission."""
        return self.gross_pnl - self.entry_commission - self.exit_commission

    @property
    def is_win(self) -> bool:
        """True if the round trip made money net of commission."""
        return self.pnl > 0

    @property
    def is_loss(self) -> bool:
        """True if the round trip lost money net of commission."""
        return self.pnl < 0


def pair_trades(trades: List[Trade]) -> List[RoundTrip]:
    """
    Pair buy and sell executions into realised round trips, FIFO.

    This is the single source of truth for trade-level P&L: the backtest engine's
    metrics and every exporter that reports positions must go through it, or the
    same backtest ends up with two contradicting sets of numbers.

    Sells are matched against the oldest open buys; a sell larger than the oldest
    buy spills over into the next one and a sell smaller than it leaves the
    remainder open, so partial fills are handled exactly. Commission of a
    partially matched execution is apportioned pro-rata by quantity, and the
    resulting P&L is net of both the entry and the exit commission.

    Args:
        trades: Chronological list of executions

    Returns:
        List of realised round trips (unclosed buys are simply not included)
    """
    open_lots: deque = deque()
    round_trips: List[RoundTrip] = []

    for trade in trades:
        if trade.quantity <= 0:
            continue

        if trade.side == TradeSide.BUY:
            open_lots.append({
                'timestamp': trade.timestamp,
                'price': trade.price,
                'quantity': trade.quantity,
                'commission_per_unit': trade.commission / trade.quantity,
            })
            continue

        remaining = trade.quantity
        exit_commission_per_unit = trade.commission / trade.quantity

        while remaining > QUANTITY_EPSILON and open_lots:
            lot = open_lots[0]
            matched = min(lot['quantity'], remaining)

            round_trips.append(RoundTrip(
                symbol=trade.symbol,
                entry_timestamp=lot['timestamp'],
                exit_timestamp=trade.timestamp,
                quantity=matched,
                entry_price=lot['price'],
                exit_price=trade.price,
                entry_commission=lot['commission_per_unit'] * matched,
                exit_commission=exit_commission_per_unit * matched,
            ))

            lot['quantity'] -= matched
            remaining -= matched
            if lot['quantity'] <= QUANTITY_EPSILON:
                open_lots.popleft()

    return round_trips
