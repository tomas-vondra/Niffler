"""
Unit tests for the statistical-significance assessment.

Three things are being defended here. First, the t-distribution implemented in
this repository has to agree with published tables - a normal approximation
would be wrong in exactly the small-sample regime the gate exists for. Second,
the bootstrap has to be reproducible from an explicit seed and must not touch
global numpy state. Third, and most important, the minimum-trades gate has to
actually fire: a strategy with a handful of round trips must get no verdict,
however good its numbers look.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.round_trip import RoundTrip
from niffler.backtesting.significance import (
    DEFAULT_MIN_TRADES,
    assess_significance,
    bootstrap_sharpe_interval,
    one_sample_t_test,
    regularized_incomplete_beta,
    student_t_two_sided_p,
    trade_return_percentages,
)
from niffler.strategies.base_strategy import BaseStrategy


class ScriptedStrategy(BaseStrategy):
    """Emits a fixed list of signals, one per bar."""

    def __init__(self, signals):
        super().__init__("ScriptedStrategy", {}, None)
        self.signals = signals

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = list(self.signals[:len(df)])
        df['position_size'] = 1.0
        return df

    def validate_data(self, data):
        return True

    def get_description(self):
        return "Scripted strategy for significance tests"


def make_round_trips(returns_pct, entry_price=100.0, quantity=1.0):
    """Build round trips with prescribed percentage returns, net of commission."""
    trips = []
    base = pd.Timestamp('2024-01-01')
    for i, pct in enumerate(returns_pct):
        exit_price = entry_price * (1 + pct / 100.0)
        trips.append(RoundTrip(
            symbol="TEST",
            entry_timestamp=base + pd.Timedelta(days=2 * i),
            exit_timestamp=base + pd.Timedelta(days=2 * i + 1),
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
        ))
    return trips


class TestStudentTDistribution(unittest.TestCase):
    """The hand-rolled t distribution must match published critical values."""

    def test_matches_published_critical_values(self):
        """Textbook two-sided 5% critical points land on p = 0.05."""
        for t_value, degrees in [(2.228, 10), (2.571, 5), (2.042, 30), (1.984, 99)]:
            with self.subTest(t=t_value, df=degrees):
                self.assertAlmostEqual(student_t_two_sided_p(t_value, degrees),
                                       0.05, places=3)

    def test_matches_published_one_percent_points(self):
        for t_value, degrees in [(3.169, 10), (4.032, 5), (2.750, 29)]:
            with self.subTest(t=t_value, df=degrees):
                self.assertAlmostEqual(student_t_two_sided_p(t_value, degrees),
                                       0.01, places=3)

    def test_is_symmetric_in_the_sign_of_t(self):
        self.assertAlmostEqual(student_t_two_sided_p(1.7, 12),
                               student_t_two_sided_p(-1.7, 12))

    def test_a_zero_statistic_is_certainly_not_significant(self):
        self.assertAlmostEqual(student_t_two_sided_p(0.0, 20), 1.0)

    def test_differs_materially_from_the_normal_approximation(self):
        """The reason this is not a normal: at n=30 the difference decides calls."""
        exact = student_t_two_sided_p(2.0, 29)
        # Two-sided normal tail for the same statistic.
        normal = 0.045500263896358
        self.assertGreater(exact, normal)
        self.assertGreater(exact / normal, 1.10)

    def test_rejects_non_positive_degrees_of_freedom(self):
        with self.assertRaises(ValueError):
            student_t_two_sided_p(1.0, 0)

    def test_incomplete_beta_is_symmetric_at_the_midpoint(self):
        self.assertAlmostEqual(regularized_incomplete_beta(0.5, 0.5, 0.5), 0.5)
        self.assertAlmostEqual(regularized_incomplete_beta(2.0, 3.0, 0.0), 0.0)
        self.assertAlmostEqual(regularized_incomplete_beta(2.0, 3.0, 1.0), 1.0)

    def test_incomplete_beta_validates_its_arguments(self):
        with self.assertRaises(ValueError):
            regularized_incomplete_beta(0.0, 1.0, 0.5)
        with self.assertRaises(ValueError):
            regularized_incomplete_beta(1.0, 1.0, 1.5)


class TestOneSampleTTest(unittest.TestCase):
    """The t-statistic itself, including the cases where it does not exist."""

    def test_known_sample(self):
        sample = [1.0, 2.0, 3.0, 4.0, 5.0]
        t_stat, p_value = one_sample_t_test(sample)

        # mean 3, sd 1.5811, n 5 -> t = 3 / (1.5811/sqrt(5)) = 4.2426
        self.assertAlmostEqual(t_stat, 4.242640687119285, places=9)
        self.assertAlmostEqual(p_value, 0.0132356, places=6)

    def test_a_single_observation_has_no_test(self):
        self.assertEqual(one_sample_t_test([1.0]), (None, None))

    def test_a_sample_with_no_dispersion_has_no_test(self):
        """Every trade identical means an infinite t and a meaningless p."""
        self.assertEqual(one_sample_t_test([2.0] * 40), (None, None))

    def test_a_mean_of_zero_is_maximally_insignificant(self):
        _, p_value = one_sample_t_test([-1.0, 1.0, -2.0, 2.0])
        self.assertAlmostEqual(p_value, 1.0)


class TestTradeReturnPercentages(unittest.TestCase):
    """Round trips become percentages of the capital they tied up."""

    def test_percentages_not_currency(self):
        small = RoundTrip("T", pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'),
                          quantity=1.0, entry_price=100.0, exit_price=110.0)
        large = RoundTrip("T", pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-04'),
                          quantity=100.0, entry_price=100.0, exit_price=110.0)

        percentages = trade_return_percentages([small, large])

        # Same trade at two sizes: the same return, not a hundredfold one.
        self.assertAlmostEqual(percentages[0], percentages[1])
        self.assertAlmostEqual(percentages[0], 10.0)

    def test_commission_is_included(self):
        gross = RoundTrip("T", pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'),
                          quantity=1.0, entry_price=100.0, exit_price=110.0)
        net = RoundTrip("T", pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'),
                        quantity=1.0, entry_price=100.0, exit_price=110.0,
                        entry_commission=1.0, exit_commission=1.0)

        self.assertLess(trade_return_percentages([net])[0],
                        trade_return_percentages([gross])[0])

    def test_a_zero_notional_round_trip_is_skipped(self):
        degenerate = RoundTrip("T", pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'),
                               quantity=0.0, entry_price=100.0, exit_price=110.0)

        self.assertEqual(trade_return_percentages([degenerate]), [])


class TestMinimumTradesGate(unittest.TestCase):
    """The framework must refuse to judge a sample it cannot judge."""

    def test_the_gate_fires_below_the_threshold(self):
        """Twenty-nine excellent trades still buy no verdict at a gate of 30."""
        result = assess_significance(make_round_trips([3.0, 1.0] * 14 + [3.0]),
                                     min_trades=30)

        self.assertEqual(result.round_trips, 29)
        self.assertFalse(result.is_sample_sufficient)
        self.assertIsNone(result.is_significant)
        self.assertIn('SAMPLE TOO SMALL', result.verdict)
        # The p-value is still computed, so the data is not hidden - only the
        # verdict is withheld.
        self.assertIsNotNone(result.p_value)

    def test_the_gate_opens_at_the_threshold(self):
        result = assess_significance(make_round_trips([3.0, 1.0] * 15), min_trades=30)

        self.assertEqual(result.round_trips, 30)
        self.assertTrue(result.is_sample_sufficient)
        self.assertIsNotNone(result.is_significant)
        self.assertNotIn('SAMPLE TOO SMALL', result.verdict)

    def test_the_threshold_is_configurable(self):
        trips = make_round_trips([3.0, 1.0, 2.0, 0.5, 1.5])

        self.assertFalse(assess_significance(trips, min_trades=10).is_sample_sufficient)
        self.assertTrue(assess_significance(trips, min_trades=5).is_sample_sufficient)

    def test_no_round_trips_at_all(self):
        result = assess_significance([])

        self.assertEqual(result.round_trips, 0)
        self.assertIsNone(result.is_significant)
        self.assertIn('nothing to test', result.verdict)

    def test_the_default_gate_is_thirty(self):
        self.assertEqual(DEFAULT_MIN_TRADES, 30)
        self.assertEqual(assess_significance([]).min_trades, 30)

    def test_the_engine_default_threshold_reaches_the_result(self):
        data = _price_frame([100.0, 100.0, 110.0, 110.0, 105.0])
        engine = BacktestEngine(commission=0.0)

        result = engine.run_backtest(ScriptedStrategy([1, 0, -1, 0, 0]), data, "TEST")

        self.assertEqual(result.round_trip_count, 1)
        self.assertEqual(result.significance_min_trades, DEFAULT_MIN_TRADES)
        self.assertFalse(result.is_sample_sufficient)
        self.assertIsNone(result.is_significant)
        self.assertIn('SAMPLE TOO SMALL', result.significance_verdict)


class TestVerdicts(unittest.TestCase):
    """What a sufficient sample is and is not allowed to claim."""

    def test_a_strong_consistent_edge_is_called_significant(self):
        result = assess_significance(make_round_trips([2.0, 3.0, 1.5, 2.5] * 10),
                                     min_trades=30)

        self.assertTrue(result.is_significant)
        self.assertLess(result.p_value, 0.05)

    def test_a_coin_toss_is_not_called_significant(self):
        """Symmetric wins and losses average to zero however many there are."""
        result = assess_significance(make_round_trips([5.0, -5.0] * 30), min_trades=30)

        self.assertFalse(result.is_significant)
        self.assertGreater(result.p_value, 0.05)

    def test_every_verdict_carries_the_multiple_testing_caveat(self):
        result = assess_significance(make_round_trips([2.0, 3.0, 1.5, 2.5] * 10),
                                     min_trades=30)

        self.assertIn('one window', result.verdict)
        self.assertIn('fitted on this same data', result.verdict)

    def test_identical_trades_leave_the_test_undefined(self):
        result = assess_significance(make_round_trips([2.0] * 40), min_trades=30)

        self.assertIsNone(result.p_value)
        self.assertIsNone(result.is_significant)
        self.assertIn('no dispersion', result.verdict)

    def test_invalid_configuration_raises(self):
        with self.assertRaises(ValueError):
            assess_significance([], min_trades=-1)
        with self.assertRaises(ValueError):
            assess_significance([], bootstrap_samples=-1)
        with self.assertRaises(ValueError):
            assess_significance([], confidence_level=1.5)


class TestBootstrapSharpeInterval(unittest.TestCase):
    """Reproducible from an explicit seed, and never from global state."""

    def setUp(self):
        rng = np.random.default_rng(3)
        index = pd.date_range('2024-01-01', periods=250, freq='D')
        self.curve = pd.Series(
            10000 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 250))), index=index
        )

    def test_the_same_seed_gives_the_same_interval(self):
        first = bootstrap_sharpe_interval(self.curve, 365.0, samples=200, seed=7)
        second = bootstrap_sharpe_interval(self.curve, 365.0, samples=200, seed=7)

        self.assertEqual(first, second)

    def test_different_seeds_give_different_intervals(self):
        first = bootstrap_sharpe_interval(self.curve, 365.0, samples=200, seed=7)
        second = bootstrap_sharpe_interval(self.curve, 365.0, samples=200, seed=8)

        self.assertNotEqual(first, second)

    def test_global_numpy_state_is_neither_read_nor_written(self):
        """A seeded global generator must not change the answer, or be changed."""
        np.random.seed(1234)
        before = bootstrap_sharpe_interval(self.curve, 365.0, samples=200, seed=7)
        global_draw_after_first = np.random.random()

        np.random.seed(1234)
        after = bootstrap_sharpe_interval(self.curve, 365.0, samples=200, seed=7)
        global_draw_after_second = np.random.random()

        self.assertEqual(before, after)
        self.assertEqual(global_draw_after_first, global_draw_after_second)

    def test_the_interval_brackets_the_point_estimate_roughly(self):
        low, high = bootstrap_sharpe_interval(self.curve, 365.0, samples=500, seed=7)

        self.assertLess(low, high)

    def test_a_wider_confidence_level_gives_a_wider_interval(self):
        narrow = bootstrap_sharpe_interval(self.curve, 365.0, samples=500,
                                           confidence_level=0.50, seed=7)
        wide = bootstrap_sharpe_interval(self.curve, 365.0, samples=500,
                                         confidence_level=0.99, seed=7)

        self.assertLess(wide[0], narrow[0])
        self.assertGreater(wide[1], narrow[1])

    def test_a_curve_too_short_to_resample(self):
        index = pd.date_range('2024-01-01', periods=1, freq='D')
        self.assertEqual(
            bootstrap_sharpe_interval(pd.Series([100.0], index=index), 252.0, samples=10),
            (None, None)
        )

    def test_the_bootstrap_is_skipped_by_default(self):
        """It is the only expensive part, so it is opt-in."""
        result = assess_significance(make_round_trips([1.0] * 40),
                                     portfolio_values=self.curve, min_trades=30)

        self.assertIsNone(result.sharpe_ci_low)
        self.assertEqual(result.bootstrap_samples, 0)

    def test_requesting_the_bootstrap_populates_the_interval(self):
        result = assess_significance(make_round_trips([1.0, 2.0] * 20),
                                     portfolio_values=self.curve,
                                     periods_per_year=365.0,
                                     bootstrap_samples=200, min_trades=30, seed=5)

        self.assertIsNotNone(result.sharpe_ci_low)
        self.assertLess(result.sharpe_ci_low, result.sharpe_ci_high)
        self.assertEqual(result.bootstrap_samples, 200)
        self.assertEqual(result.bootstrap_seed, 5)

    def test_the_engine_reproduces_its_interval(self):
        data = _price_frame([100.0, 101.0, 99.0, 103.0, 98.0, 104.0, 102.0])

        def run():
            engine = BacktestEngine(commission=0.0, bootstrap_samples=200,
                                    bootstrap_seed=99)
            return engine.run_backtest(ScriptedStrategy([1, -1, 1, -1, 1, -1, 0]),
                                       data, "TEST")

        first, second = run(), run()

        self.assertEqual(first.sharpe_ci_low, second.sharpe_ci_low)
        self.assertEqual(first.sharpe_ci_high, second.sharpe_ci_high)
        self.assertEqual(first.sharpe_ci_confidence, 0.95)


def _price_frame(closes):
    """Build an OHLCV frame where each bar's open equals its close."""
    index = pd.date_range('2024-01-01', periods=len(closes), freq='D')
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * len(closes),
    }, index=index)


if __name__ == '__main__':
    unittest.main()
