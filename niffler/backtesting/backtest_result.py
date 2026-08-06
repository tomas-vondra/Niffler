import pandas as pd
from typing import List
from dataclasses import dataclass
from .trade import Trade


@dataclass
class BacktestResult:
    """
    Contains the results of a backtest run.

    ``total_commission`` and ``total_slippage`` are the two halves of what the
    run paid to trade: commission charged on filled notional, and the cash given
    up to slippage and spread by the configured cost model. Both are appended
    last with a 0.0 default so results constructed by older callers still build.
    """
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
    profit_factor: float
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    num_winning_trades: int = 0
    num_losing_trades: int = 0
    total_commission: float = 0.0
    total_slippage: float = 0.0