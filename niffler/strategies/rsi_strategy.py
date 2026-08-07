import pandas as pd

from .base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """Mean-reversion strategy on Wilder's Relative Strength Index.

    Buys when RSI climbs back **out** of oversold territory and sells when it
    falls back **out** of overbought territory. Entering on the crossing rather
    than on the level means an instrument that stays oversold for thirty bars
    produces one signal, not thirty.

    Like every strategy here, signals are computed from the closing price of the
    bar they belong to. That is intentional and bias-free: ``BacktestEngine``
    defers execution to the next bar's open, so a signal is never filled at a
    price the strategy already saw.
    """

    #: Optimisation search space. Keys must be ``__init__`` keyword arguments -
    #: see :mod:`niffler.strategies.registry`. The oversold and overbought ranges
    #: are deliberately disjoint (max 35 < min 65) so no lattice point can
    #: produce the contradictory ``oversold >= overbought`` combination.
    #: 15 x 4 x 4 x 6 = 1440 combinations.
    PARAMETER_SPEC = {
        'rsi_period': {'type': 'int', 'min': 7, 'max': 21, 'step': 1},
        'oversold': {'type': 'int', 'min': 20, 'max': 35, 'step': 5},
        'overbought': {'type': 'int', 'min': 65, 'max': 80, 'step': 5},
        'position_size': {'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.1},
    }

    def __init__(self, rsi_period: int = 14, oversold: float = 30.0,
                 overbought: float = 70.0, position_size: float = 1.0,
                 risk_manager=None):
        """Initialize the strategy.

        Args:
            rsi_period: Lookback period for Wilder's RSI.
            oversold: RSI level below which the market is considered oversold.
                A buy is emitted when RSI crosses back above it.
            overbought: RSI level above which the market is considered
                overbought. A sell is emitted when RSI crosses back below it.
            position_size: Fraction of portfolio to use for each trade (0.0 to 1.0).
            risk_manager: Risk manager instance for position sizing and stop-loss.

        Raises:
            ValueError: If the period is not positive, the thresholds are outside
                0-100, or oversold is not strictly below overbought.
        """
        if rsi_period < 1:
            raise ValueError(f"rsi_period must be >= 1, got {rsi_period}")
        if not 0 <= oversold <= 100:
            raise ValueError(f"oversold must be between 0 and 100, got {oversold}")
        if not 0 <= overbought <= 100:
            raise ValueError(f"overbought must be between 0 and 100, got {overbought}")
        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) must be strictly below overbought ({overbought})"
            )

        parameters = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought,
            'position_size': position_size
        }
        super().__init__("RSI Mean Reversion", parameters, risk_manager)

        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_size = position_size

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Compute Wilder's RSI.

        Wilder's smoothing is an exponentially weighted mean with
        ``alpha = 1 / period``, which pandas expresses directly. ``min_periods``
        keeps the warm-up window NaN rather than emitting an RSI derived from a
        handful of bars, so no signal fires before the indicator is meaningful.

        Args:
            close: Closing price series.

        Returns:
            RSI series on the same index, NaN during the warm-up window.
        """
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False,
                            min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False,
                            min_periods=self.rsi_period).mean()

        rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

        # A window with no downside gives avg_loss == 0. The limit is RSI 100,
        # but 0/0 evaluates to NaN, so the two degenerate cases are pinned
        # explicitly: all-gain is maximally strong, dead flat is neutral.
        rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
        rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)

        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from RSI threshold crossings.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            DataFrame with 'rsi', 'signal' and 'position_size' columns added.

        Raises:
            ValueError: If the data does not have the required OHLCV format.
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data format")

        df = data.copy()
        df['rsi'] = self._calculate_rsi(df['close'])

        rsi_prev = df['rsi'].shift(1)

        # Crossings only. A comparison against a NaN warm-up value is False, so
        # the warm-up window cannot produce a signal.
        buy_condition = (df['rsi'] > self.oversold) & (rsi_prev <= self.oversold)
        sell_condition = (df['rsi'] < self.overbought) & (rsi_prev >= self.overbought)

        df['signal'] = 0
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        df['position_size'] = self.position_size

        return df

    def get_description(self) -> str:
        """Return strategy description."""
        risk_desc = ""
        if self.risk_manager is not None:
            risk_metrics = self.risk_manager.get_risk_metrics()
            risk_desc = f" Risk management: {risk_metrics.get('risk_management_type', 'Unknown')}"

        return (f"RSI Mean Reversion Strategy with a {self.rsi_period}-period RSI, "
                f"buying when RSI crosses back above {self.oversold} and selling "
                f"when it crosses back below {self.overbought}. "
                f"Position size: {self.position_size * 100}%.{risk_desc}")
