import pandas as pd
from dataclasses import dataclass
from enum import Enum


class TradeSide(Enum):
    """Enum for trade side."""
    BUY = 'buy'
    SELL = 'sell'


@dataclass
class Trade:
    """Represents a single trade execution."""
    timestamp: pd.Timestamp
    symbol: str
    side: TradeSide
    price: float
    quantity: float
    value: float