import pandas as pd
from typing import List, Optional
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

    Benchmark and significance
    --------------------------
    The ``benchmark_*`` / ``excess_return_pct`` / ``information_ratio`` fields
    describe the run against a passive alternative (see
    :mod:`niffler.backtesting.benchmark`); they are ``None`` when no benchmark
    was requested. The significance fields describe whether the strategy's edge
    is distinguishable from noise (see
    :mod:`niffler.backtesting.significance`).

    ``significance_verdict`` and ``is_sample_sufficient`` are the honest pair:
    when the sample is below ``significance_min_trades`` there is **no verdict**,
    and ``is_significant`` is deliberately ``None`` rather than ``False``. Do not
    collapse it - "we cannot tell" and "we tested and it is not there" are
    different statements.

    Every field here is appended with a default, in the same style as
    ``commission`` / ``total_slippage``, so results built by older callers keep
    constructing.
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

    # --- Benchmark comparison (None when no benchmark was run) ---------------
    benchmark_name: Optional[str] = None
    benchmark_return_pct: Optional[float] = None
    benchmark_sharpe_ratio: Optional[float] = None
    #: Negative percentage, like max_drawdown.
    benchmark_max_drawdown: Optional[float] = None
    #: Commission plus slippage the benchmark itself paid. Present so a reader
    #: can confirm the comparison was not rigged by giving buy-and-hold free
    #: fills.
    benchmark_total_cost: Optional[float] = None
    #: Strategy total return minus benchmark total return, in percentage points.
    excess_return_pct: Optional[float] = None
    #: Annualised mean active return over its standard deviation.
    information_ratio: Optional[float] = None
    #: Why a requested benchmark could not be established, when that happens.
    #: The comparison fields above are then all None, and this string says why
    #: rather than leaving the reader to guess at an absent comparison.
    benchmark_error: Optional[str] = None

    # --- Statistical significance --------------------------------------------
    #: Realised round trips, which is what the t-test's sample size is. Not the
    #: same as total_trades, which counts individual executions.
    round_trip_count: int = 0
    mean_trade_return_pct: Optional[float] = None
    t_statistic: Optional[float] = None
    #: Two-sided p-value; halve it for a one-sided reading.
    p_value: Optional[float] = None
    sharpe_ci_low: Optional[float] = None
    sharpe_ci_high: Optional[float] = None
    #: Confidence level of the bootstrap interval above, so the bounds are never
    #: reported without saying what they are bounds of.
    sharpe_ci_confidence: float = 0.95
    significance_min_trades: int = 0
    is_sample_sufficient: bool = False
    significance_verdict: str = ''

    @property
    def is_significant(self) -> Optional[bool]:
        """
        Whether the mean round-trip return is distinguishable from zero.

        Returns:
            True/False at the 5% level, or **None** when the sample was below
            the minimum-trades gate or the test was undefined. None means "no
            verdict was rendered" and must not be reported as a negative result.
        """
        from .significance import DEFAULT_ALPHA

        if not self.is_sample_sufficient or self.p_value is None:
            return None
        return self.p_value < DEFAULT_ALPHA

    @property
    def beats_benchmark(self) -> Optional[bool]:
        """
        Whether the strategy out-returned its benchmark over the same bars.

        Returns:
            True/False, or None when no benchmark was run. A True here says
            nothing about whether the difference is statistically real - that is
            what the significance fields are for.
        """
        if self.excess_return_pct is None:
            return None
        return self.excess_return_pct > 0