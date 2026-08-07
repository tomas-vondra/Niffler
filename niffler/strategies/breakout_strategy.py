import pandas as pd

from .base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Donchian channel breakout strategy.

    Buys when the close breaks above the highest high of the preceding
    ``entry_window`` bars, and sells when it breaks below the lowest low of the
    preceding ``exit_window`` bars. The asymmetric windows are the point: a
    breakout is confirmed slowly and abandoned quickly.

    Unlike the moving-average and RSI strategies this one reads the ``high`` and
    ``low`` columns, not just ``close``.

    Signals are computed from the closing price of the bar they belong to, which
    is bias-free because ``BacktestEngine`` defers execution to the next bar's
    open.
    """

    #: Optimisation search space. Keys must be ``__init__`` keyword arguments -
    #: see :mod:`niffler.strategies.registry`. 11 x 6 x 6 = 396 combinations.
    PARAMETER_SPEC = {
        'entry_window': {'type': 'int', 'min': 10, 'max': 60, 'step': 5},
        'exit_window': {'type': 'int', 'min': 5, 'max': 30, 'step': 5},
        'position_size': {'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.1},
    }

    def __init__(self, entry_window: int = 20, exit_window: int = 10,
                 position_size: float = 1.0, risk_manager=None):
        """Initialize the strategy.

        Args:
            entry_window: Number of preceding bars whose highest high the close
                must exceed to trigger a buy.
            exit_window: Number of preceding bars whose lowest low the close must
                break below to trigger a sell.
            position_size: Fraction of portfolio to use for each trade (0.0 to 1.0).
            risk_manager: Risk manager instance for position sizing and stop-loss.

        Raises:
            ValueError: If either window is not positive.
        """
        if entry_window < 1:
            raise ValueError(f"entry_window must be >= 1, got {entry_window}")
        if exit_window < 1:
            raise ValueError(f"exit_window must be >= 1, got {exit_window}")

        parameters = {
            'entry_window': entry_window,
            'exit_window': exit_window,
            'position_size': position_size
        }
        super().__init__("Donchian Breakout", parameters, risk_manager)

        self.entry_window = entry_window
        self.exit_window = exit_window
        self.position_size = position_size

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from Donchian channel breakouts.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            DataFrame with 'channel_high', 'channel_low', 'signal' and
            'position_size' columns added.

        Raises:
            ValueError: If the data does not have the required OHLCV format.
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data format")

        df = data.copy()

        # .shift(1) is what makes this bias-free, and it is not optional. A bare
        # rolling().max() window ends on the current bar, so it already contains
        # today's high: comparing today's close against it would be testing the
        # close against a band that today itself set. The channel must describe
        # only the bars that closed before this one.
        df['channel_high'] = df['high'].rolling(window=self.entry_window).max().shift(1)
        df['channel_low'] = df['low'].rolling(window=self.exit_window).min().shift(1)

        # Comparisons against the NaN warm-up window are False, so no signal can
        # fire before both channels are fully formed.
        above = df['close'] > df['channel_high']
        below = df['close'] < df['channel_low']

        # Signal on the transition into a breakout, not on every bar that happens
        # to remain outside the channel, so one breakout is one signal.
        df['signal'] = 0
        df.loc[above & ~above.shift(1, fill_value=False), 'signal'] = 1
        df.loc[below & ~below.shift(1, fill_value=False), 'signal'] = -1
        df['position_size'] = self.position_size

        return df

    def get_description(self) -> str:
        """Return strategy description."""
        risk_desc = ""
        if self.risk_manager is not None:
            risk_metrics = self.risk_manager.get_risk_metrics()
            risk_desc = f" Risk management: {risk_metrics.get('risk_management_type', 'Unknown')}"

        return (f"Donchian Breakout Strategy buying a close above the "
                f"{self.entry_window}-bar high and selling a close below the "
                f"{self.exit_window}-bar low. "
                f"Position size: {self.position_size * 100}%.{risk_desc}")
