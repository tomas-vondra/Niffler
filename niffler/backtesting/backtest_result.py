import pandas as pd
from typing import List
from dataclasses import dataclass
from .trade import Trade


@dataclass
class BacktestResult:
    """Contains the results of a backtest run."""
    strategy_name: str
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    trades: List[Trade]
    portfolio_values: pd.Series
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int