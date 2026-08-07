import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.strategies.rsi_strategy import RSIStrategy


def frame_from_close(close_values) -> pd.DataFrame:
    """Build a minimal valid OHLCV frame from a closing price series."""
    close = pd.Series(close_values, dtype=float)
    return pd.DataFrame(
        {
            'open': close,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': [1000.0] * len(close),
        },
        index=pd.date_range('2024-01-01', periods=len(close), freq='D'),
    )


class TestRSICalculation(unittest.TestCase):
    """Properties of Wilder's RSI at its boundaries."""

    def test_monotonically_rising_series_pins_rsi_to_100(self):
        strategy = RSIStrategy(rsi_period=5)
        rsi = strategy._calculate_rsi(pd.Series(np.arange(1, 40, dtype=float)))

        self.assertTrue(np.allclose(rsi.dropna(), 100.0))

    def test_monotonically_falling_series_pins_rsi_to_zero(self):
        strategy = RSIStrategy(rsi_period=5)
        rsi = strategy._calculate_rsi(pd.Series(np.arange(40, 1, -1, dtype=float)))

        self.assertTrue(np.allclose(rsi.dropna(), 0.0))

    def test_flat_series_is_neutral_rather_than_nan(self):
        """Zero gain and zero loss is 0/0; it must resolve to 50, not NaN."""
        strategy = RSIStrategy(rsi_period=5)
        rsi = strategy._calculate_rsi(pd.Series([100.0] * 30))

        resolved = rsi.dropna()
        self.assertTrue(len(resolved) > 0)
        self.assertTrue(np.allclose(resolved, 50.0))

    def test_warmup_window_is_nan(self):
        strategy = RSIStrategy(rsi_period=14)
        rsi = strategy._calculate_rsi(pd.Series(np.linspace(100, 120, 40)))

        self.assertTrue(rsi.iloc[:14].isna().all())
        self.assertFalse(rsi.iloc[20:].isna().any())

    def test_rsi_stays_within_bounds(self):
        rng = np.random.default_rng(3)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
        rsi = RSIStrategy(rsi_period=14)._calculate_rsi(close).dropna()

        self.assertGreaterEqual(rsi.min(), 0.0)
        self.assertLessEqual(rsi.max(), 100.0)


class TestRSISignals(unittest.TestCase):
    """Signal logic, isolated from the indicator by injecting a known RSI series."""

    def _signals_for_rsi(self, rsi_values, **kwargs):
        data = frame_from_close([100.0] * len(rsi_values))
        rsi = pd.Series(rsi_values, index=data.index, dtype=float)

        strategy = RSIStrategy(**kwargs)
        with patch.object(RSIStrategy, '_calculate_rsi', return_value=rsi):
            return strategy.generate_signals(data)['signal'].tolist()

    def test_buy_fires_when_rsi_climbs_out_of_oversold(self):
        # index:            0     1   2   3   4
        signals = self._signals_for_rsi([np.nan, 25, 28, 35, 32],
                                        oversold=30, overbought=70)

        self.assertEqual([0, 0, 0, 1, 0], signals)

    def test_sell_fires_when_rsi_falls_out_of_overbought(self):
        signals = self._signals_for_rsi([np.nan, 75, 72, 68, 71],
                                        oversold=30, overbought=70)

        self.assertEqual([0, 0, 0, -1, 0], signals)

    def test_staying_oversold_produces_one_signal_not_many(self):
        """Entering on the crossing, not the level, is what keeps this to one signal."""
        signals = self._signals_for_rsi([np.nan, 20, 21, 22, 19, 45, 44, 43],
                                        oversold=30, overbought=70)

        self.assertEqual(1, signals.count(1))
        self.assertEqual(0, signals.count(-1))

    def test_nan_warmup_cannot_produce_a_signal(self):
        signals = self._signals_for_rsi([np.nan] * 5 + [45.0],
                                        oversold=30, overbought=70)

        self.assertEqual([0] * 6, signals)

    def test_position_size_column_carries_the_configured_fraction(self):
        data = frame_from_close(np.linspace(100, 130, 60))
        result = RSIStrategy(rsi_period=5, position_size=0.25).generate_signals(data)

        self.assertTrue((result['position_size'] == 0.25).all())

    def test_invalid_data_is_rejected(self):
        strategy = RSIStrategy()
        with self.assertRaises(ValueError):
            strategy.generate_signals(pd.DataFrame({'close': [1.0, 2.0]}))


class TestRSIValidation(unittest.TestCase):
    """Constructor guards. The shipped lattice cannot reach these, --params can."""

    def test_oversold_must_be_below_overbought(self):
        with self.assertRaises(ValueError):
            RSIStrategy(oversold=70, overbought=30)

    def test_equal_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            RSIStrategy(oversold=50, overbought=50)

    def test_period_must_be_positive(self):
        with self.assertRaises(ValueError):
            RSIStrategy(rsi_period=0)

    def test_thresholds_must_be_within_zero_and_one_hundred(self):
        with self.assertRaises(ValueError):
            RSIStrategy(oversold=-1)
        with self.assertRaises(ValueError):
            RSIStrategy(overbought=101)

    def test_description_mentions_the_configured_levels(self):
        description = RSIStrategy(rsi_period=9, oversold=25, overbought=75).get_description()

        self.assertIn('9', description)
        self.assertIn('25', description)
        self.assertIn('75', description)


if __name__ == '__main__':
    unittest.main()
