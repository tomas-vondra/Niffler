import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.strategies.breakout_strategy import BreakoutStrategy


def frame(highs, lows, closes) -> pd.DataFrame:
    """Build an OHLCV frame from explicit high/low/close series."""
    return pd.DataFrame(
        {
            'open': closes,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': [1000.0] * len(closes),
        },
        index=pd.date_range('2024-01-01', periods=len(closes), freq='D'),
    )


class TestDonchianChannel(unittest.TestCase):
    """The channel must describe only bars that closed before the current one."""

    def test_channel_high_excludes_the_current_bar(self):
        """Without the .shift(1) the channel contains today's own high.

        That would make the breakout test compare the close against a band the
        current bar helped set - a look-ahead, and one that also makes a genuine
        breakout impossible to detect since the high is always >= the close.
        """
        highs = [10.0, 11.0, 12.0, 13.0, 50.0, 14.0]
        data = frame(highs, [h - 2 for h in highs], [h - 1 for h in highs])

        result = BreakoutStrategy(entry_window=3, exit_window=3).generate_signals(data)

        # At index 4 the channel must be max(high[1:4]) = 13.0. The spike of 50.0
        # is index 4's own high and must not appear until the next bar.
        self.assertEqual(13.0, result['channel_high'].iloc[4])
        # At index 5 the spike at index 4 has entered the window.
        self.assertEqual(50.0, result['channel_high'].iloc[5])

    def test_channel_low_excludes_the_current_bar(self):
        lows = [10.0, 9.0, 8.0, 7.0, 1.0, 6.0]
        data = frame([low + 2 for low in lows], lows, [low + 1 for low in lows])

        result = BreakoutStrategy(entry_window=3, exit_window=3).generate_signals(data)

        # min(low[1:4]) = 7.0; the trough of 1.0 is index 4's own low.
        self.assertEqual(7.0, result['channel_low'].iloc[4])
        self.assertEqual(1.0, result['channel_low'].iloc[5])

    def test_channel_matches_an_explicit_prior_window_maximum(self):
        rng = np.random.default_rng(11)
        highs = rng.uniform(90, 110, 60)
        lows = highs - rng.uniform(1, 5, 60)
        data = frame(highs, lows, (highs + lows) / 2)

        window = 10
        result = BreakoutStrategy(entry_window=window,
                                  exit_window=window).generate_signals(data)

        for i in range(window, 60):
            with self.subTest(bar=i):
                self.assertAlmostEqual(
                    float(np.max(highs[i - window:i])),
                    float(result['channel_high'].iloc[i])
                )

    def test_warmup_channel_is_nan_and_emits_no_signal(self):
        data = frame([10.0] * 10, [8.0] * 10, [9.0] * 10)

        result = BreakoutStrategy(entry_window=5, exit_window=5).generate_signals(data)

        self.assertTrue(result['channel_high'].iloc[:5].isna().all())
        self.assertEqual(0, result['signal'].iloc[:5].abs().sum())


class TestBreakoutSignals(unittest.TestCase):

    def test_breaking_above_the_channel_buys(self):
        closes = [10.0, 10.0, 10.0, 10.0, 20.0]
        data = frame([c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)

        result = BreakoutStrategy(entry_window=3, exit_window=3).generate_signals(data)

        self.assertEqual(1, result['signal'].iloc[4])

    def test_breaking_below_the_channel_sells(self):
        closes = [10.0, 10.0, 10.0, 10.0, 2.0]
        data = frame([c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)

        result = BreakoutStrategy(entry_window=3, exit_window=3).generate_signals(data)

        self.assertEqual(-1, result['signal'].iloc[4])

    def test_a_sustained_breakout_signals_once(self):
        """A trend that stays outside the channel is one entry, not one per bar."""
        closes = [10.0, 10.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        data = frame([c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)

        signals = BreakoutStrategy(entry_window=3,
                                   exit_window=3).generate_signals(data)['signal']

        self.assertEqual(1, (signals == 1).sum())

    def test_position_size_column_carries_the_configured_fraction(self):
        closes = list(np.linspace(100, 140, 60))
        data = frame([c + 1 for c in closes], [c - 1 for c in closes], closes)

        result = BreakoutStrategy(entry_window=10, exit_window=5,
                                  position_size=0.4).generate_signals(data)

        self.assertTrue((result['position_size'] == 0.4).all())

    def test_invalid_data_is_rejected(self):
        with self.assertRaises(ValueError):
            BreakoutStrategy().generate_signals(pd.DataFrame({'close': [1.0, 2.0]}))


class TestBreakoutValidation(unittest.TestCase):

    def test_windows_must_be_positive(self):
        with self.assertRaises(ValueError):
            BreakoutStrategy(entry_window=0)
        with self.assertRaises(ValueError):
            BreakoutStrategy(exit_window=0)

    def test_description_mentions_the_configured_windows(self):
        description = BreakoutStrategy(entry_window=55, exit_window=13).get_description()

        self.assertIn('55', description)
        self.assertIn('13', description)


if __name__ == '__main__':
    unittest.main()
