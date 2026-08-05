import pandas as pd
from dataclasses import dataclass
from enum import Enum


class TradeSide(Enum):
    """Enum for trade side."""
    BUY = 'buy'
    SELL = 'sell'


@dataclass
class Trade:
    """
    Represents a single trade execution.

    Attributes:
        timestamp: Bar timestamp at which the trade was filled
        symbol: Traded symbol
        side: BUY or SELL
        price: Fill price
        quantity: Number of units traded
        value: Notional value of the trade (price * quantity), excluding commission
        commission: Commission charged for this execution, in account currency.
            Optional with a 0.0 default so existing callers keep working.
    """
    timestamp: pd.Timestamp
    symbol: str
    side: TradeSide
    price: float
    quantity: float
    value: float
    commission: float = 0.0