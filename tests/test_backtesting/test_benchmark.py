"""
Unit tests for the buy-and-hold benchmark and the comparison metrics.

The properties that matter here are the ones that keep the comparison honest:
the benchmark is charged the same costs as the strategy, it enters no earlier
than the strategy's execution timing allows, it lives on the same index, and an
absent benchmark stays visibly absent rather than collapsing to a zero excess
return.
"""

import logging
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.benchmark import (
    BENCHMARK_BUY_AND_HOLD,
    BENCHMARK_NONE,
    BenchmarkError,
    compute_benchmark,
    compute_buy_and_hold,
    information_ratio,
)
from niffler.backtesting.cost_model import FixedSlippageModel, VolumeShareSlippageModel
from niffler.strategies.base_strategy import BaseStrategy


class ScriptedStrategy(BaseStrategy):
    """Emits a fixed list of signals, one per bar."""

    def __init__(self, signals, position_sizes=None, risk_manager=None):
        super().__init__("ScriptedStrategy", {}, risk_manager)
        self.signals = signals
        self.position_sizes = position_sizes

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = list(self.signals[:len(df)])
        df['position_size'] = (self.position_sizes[:len(df)]
                               if self.position_sizes is not None else 1.0)
        return df

    def validate_data(self, data):
        return True

    def get_description(self):
        return "Scripted strategy for benchmark tests"


def make_data(closes, volume=1_000_000.0):
    """Build an OHLCV frame where each bar's open equals its close."""
    index = pd.date_range('2024-01-01', periods=len(closes), freq='D')
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [volume] * len(closes),
    }, index=index)


def enable_logging(test_case):
    """Undo any global logging.disable left behind by another test module.

    ``assertLogs`` sees nothing while logging is disabled, which makes the
    warning assertions here pass alone and fail under full discovery.
    """
    previous = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    test_case.addCleanup(logging.disable, previous)


class TestBuyAndHoldMechanics(unittest.TestCase):
    """Where the benchmark enters, what it holds, and what curve it produces."""

    def setUp(self):
        enable_logging(self)

    def test_entry_respects_next_bar_open_timing(self):
        """The benchmark may not buy on a bar the strategy could not have."""
        data = make_data([100.0, 200.0, 400.0, 800.0])
        engine = BacktestEngine(commission=0.0)

        benchmark = compute_buy_and_hold(engine, data, "TEST")

        # A signal from bar 0 fills at bar 1's open of 200, not bar 0's 100.
        self.assertEqual(benchmark.entry_trade.price, 200.0)
        self.assertEqual(benchmark.entry_trade.timestamp, data.index[1])
        # 10000 at 200 -> 50 units, held to a close of 800 -> 40000.
        self.assertAlmostEqual(benchmark.total_return_pct, 300.0)

    def test_entry_respects_same_bar_close_timing(self):
        """Under the look-ahead timing the benchmark enters on bar 0's close."""
        data = make_data([100.0, 200.0, 400.0, 800.0])
        engine = BacktestEngine(commission=0.0, execution_timing='same_bar_close')

        benchmark = compute_buy_and_hold(engine, data, "TEST")

        self.assertEqual(benchmark.entry_trade.price, 100.0)
        self.assertEqual(benchmark.entry_trade.timestamp, data.index[0])

    def test_curve_is_flat_before_the_entry_bar(self):
        """Capital sits in cash until the benchmark can actually buy."""
        data = make_data([100.0, 100.0, 200.0])
        engine = BacktestEngine(commission=0.0)

        benchmark = compute_buy_and_hold(engine, data, "TEST")

        self.assertEqual(benchmark.portfolio_values.iloc[0], engine.initial_capital)

    def test_curve_shares_the_strategy_index(self):
        """Strategy and benchmark must be comparable bar for bar."""
        data = make_data([100.0, 110.0, 120.0, 130.0])
        engine = BacktestEngine()

        result = engine.run_backtest(ScriptedStrategy([1, 0, -1, 0]), data, "TEST")
        benchmark = compute_buy_and_hold(engine, data, "TEST")

        self.assertTrue(result.portfolio_values.index.equals(benchmark.portfolio_values.index))

    def test_max_drawdown_is_negative(self):
        """The sign convention is shared with the strategy's drawdown."""
        data = make_data([100.0, 100.0, 50.0, 60.0])
        engine = BacktestEngine(commission=0.0)

        benchmark = compute_buy_and_hold(engine, data, "TEST")

        self.assertLess(benchmark.max_drawdown, 0.0)

    def test_leftover_cash_stays_in_the_curve(self):
        """A liquidity-capped entry leaves cash, and the curve must carry it."""
        data = make_data([100.0, 100.0, 100.0, 100.0], volume=50.0)
        engine = BacktestEngine(
            commission=0.0,
            cost_model=VolumeShareSlippageModel(impact_coefficient=0.0,
                                                half_spread_bps=0.0,
                                                max_participation=0.1)
        )

        with self.assertLogs(level='WARNING'):
            benchmark = compute_buy_and_hold(engine, data, "TEST")

        # 10% of a 50-unit bar is 5 units at 100 = 500 invested, 9500 in cash.
        self.assertAlmostEqual(benchmark.entry_trade.quantity, 5.0)
        self.assertAlmostEqual(benchmark.portfolio_values.iloc[-1], 10000.0)

    def test_entry_waits_for_a_bar_that_can_absorb_it(self):
        """A halted bar delays the passive buy instead of aborting the run."""
        data = make_data([100.0, 100.0, 100.0, 100.0])
        data.loc[data.index[1], 'volume'] = 0.0
        engine = BacktestEngine(commission=0.0,
                                cost_model=VolumeShareSlippageModel())

        with self.assertLogs(level='WARNING'):
            benchmark = compute_buy_and_hold(engine, data, "TEST")

        self.assertEqual(benchmark.entry_trade.timestamp, data.index[2])

    def test_a_wholly_untradeable_window_raises(self):
        """No bar can absorb the order, so there is no benchmark to report."""
        data = make_data([100.0, 100.0, 100.0], volume=0.0)
        engine = BacktestEngine(commission=0.0,
                                cost_model=VolumeShareSlippageModel())

        with self.assertLogs(level='WARNING'):
            with self.assertRaises(BenchmarkError):
                compute_buy_and_hold(engine, data, "TEST")


class TestBenchmarkPaysTheSameCosts(unittest.TestCase):
    """A cost-free benchmark would flatter every strategy that trades."""

    def setUp(self):
        self.data = make_data([100.0, 100.0, 110.0, 120.0])

    def _benchmark_return(self, cost_model, commission=0.0):
        engine = BacktestEngine(commission=commission, cost_model=cost_model)
        return compute_buy_and_hold(engine, self.data, "TEST").total_return_pct

    def test_raising_slippage_lowers_the_benchmark_return(self):
        """The headline property: costs reach the benchmark, not only the strategy."""
        frictionless = self._benchmark_return(None)
        slipped = self._benchmark_return(FixedSlippageModel(slippage_bps=50.0))
        very_slipped = self._benchmark_return(FixedSlippageModel(slippage_bps=200.0))

        self.assertLess(slipped, frictionless)
        self.assertLess(very_slipped, slipped)

    def test_raising_commission_lowers_the_benchmark_return(self):
        """Commission is charged on the benchmark's entry too."""
        free = self._benchmark_return(None, commission=0.0)
        charged = self._benchmark_return(None, commission=0.01)

        self.assertLess(charged, free)

    def test_costs_paid_are_reported(self):
        """The reader can check the benchmark was charged, not just be told so."""
        engine = BacktestEngine(commission=0.001,
                                cost_model=FixedSlippageModel(slippage_bps=50.0))

        benchmark = compute_buy_and_hold(engine, self.data, "TEST")

        self.assertGreater(benchmark.total_commission, 0.0)
        self.assertGreater(benchmark.total_slippage, 0.0)
        self.assertAlmostEqual(benchmark.total_cost,
                               benchmark.total_commission + benchmark.total_slippage)

    def test_a_costed_run_charges_both_sides(self):
        """Through the engine: the reported benchmark return falls with slippage."""
        strategy = ScriptedStrategy([1, 0, -1, 0])

        cheap = BacktestEngine(commission=0.0).run_backtest(strategy, self.data, "TEST")
        dear = BacktestEngine(
            commission=0.0, cost_model=FixedSlippageModel(slippage_bps=100.0)
        ).run_backtest(strategy, self.data, "TEST")

        self.assertLess(dear.benchmark_return_pct, cheap.benchmark_return_pct)
        self.assertGreater(dear.benchmark_total_cost, cheap.benchmark_total_cost)


class TestComparisonMetrics(unittest.TestCase):
    """Excess return and information ratio, including the degenerate cases."""

    def test_a_never_trading_strategy_gives_back_the_whole_benchmark(self):
        """Doing nothing costs exactly what the benchmark made."""
        data = make_data([100.0, 100.0, 150.0, 200.0])
        engine = BacktestEngine(commission=0.0)

        result = engine.run_backtest(ScriptedStrategy([0, 0, 0, 0]), data, "TEST")

        self.assertEqual(result.total_return_pct, 0.0)
        self.assertAlmostEqual(result.excess_return_pct, -result.benchmark_return_pct)
        self.assertFalse(result.beats_benchmark)

    def test_a_strategy_identical_to_buy_and_hold_has_zero_excess(self):
        """Buying at the first executable bar and holding IS the benchmark."""
        data = make_data([100.0, 100.0, 130.0, 90.0, 170.0])
        engine = BacktestEngine(commission=0.001,
                                cost_model=FixedSlippageModel(slippage_bps=25.0))

        # A buy signal on bar 0 fills at bar 1's open; nothing ever sells.
        result = engine.run_backtest(ScriptedStrategy([1, 0, 0, 0, 0]), data, "TEST")

        self.assertAlmostEqual(result.excess_return_pct, 0.0, places=9)
        self.assertAlmostEqual(result.total_return_pct, result.benchmark_return_pct,
                               places=9)
        self.assertAlmostEqual(result.information_ratio, 0.0, places=9)
        # No round trip ever closed, so there is nothing to test and no verdict.
        self.assertEqual(result.round_trip_count, 0)
        self.assertIsNone(result.is_significant)

    def test_flat_prices_leave_both_sides_flat(self):
        """With no price movement the only difference is what each side paid."""
        data = make_data([100.0] * 6)
        engine = BacktestEngine(commission=0.0)

        result = engine.run_backtest(ScriptedStrategy([0] * 6), data, "TEST")

        self.assertAlmostEqual(result.benchmark_return_pct, 0.0)
        self.assertAlmostEqual(result.excess_return_pct, 0.0)

    def test_excess_return_is_in_percentage_points(self):
        """+40% against +120% is -80, not a ratio."""
        data = make_data([100.0, 100.0, 200.0])
        engine = BacktestEngine(commission=0.0)

        result = engine.run_backtest(ScriptedStrategy([0, 0, 0]), data, "TEST")

        self.assertAlmostEqual(result.benchmark_return_pct, 100.0)
        self.assertAlmostEqual(result.excess_return_pct, -100.0)

    def test_information_ratio_rejects_mismatched_curves(self):
        """Comparing curves on different bars would be meaningless."""
        index = pd.date_range('2024-01-01', periods=4, freq='D')
        strategy = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
        benchmark = pd.Series([1.0, 2.0, 3.0], index=index[:3])

        with self.assertRaises(ValueError):
            information_ratio(strategy, benchmark, 252.0)

    def test_information_ratio_of_a_perfect_tracker_is_zero(self):
        """No deviation from the benchmark means no information."""
        index = pd.date_range('2024-01-01', periods=5, freq='D')
        curve = pd.Series([100.0, 110.0, 105.0, 130.0, 120.0], index=index)

        self.assertEqual(information_ratio(curve, curve.copy(), 252.0), 0.0)


class TestBenchmarkSelection(unittest.TestCase):
    """Choosing a benchmark, and what an absent one looks like."""

    def setUp(self):
        enable_logging(self)
        self.data = make_data([100.0, 100.0, 120.0, 130.0])

    def test_none_leaves_every_comparison_field_unset(self):
        """Absent must read as absent, never as a zero excess return."""
        engine = BacktestEngine(benchmark=BENCHMARK_NONE)

        result = engine.run_backtest(ScriptedStrategy([1, 0, -1, 0]), self.data, "TEST")

        self.assertIsNone(result.benchmark_name)
        self.assertIsNone(result.benchmark_return_pct)
        self.assertIsNone(result.excess_return_pct)
        self.assertIsNone(result.information_ratio)
        self.assertIsNone(result.beats_benchmark)

    def test_buy_and_hold_is_the_default(self):
        result = BacktestEngine().run_backtest(
            ScriptedStrategy([1, 0, -1, 0]), self.data, "TEST"
        )

        self.assertEqual(result.benchmark_name, BENCHMARK_BUY_AND_HOLD)

    def test_an_unknown_benchmark_is_rejected(self):
        with self.assertRaises(ValueError):
            BacktestEngine(benchmark='spx')

        with self.assertRaises(ValueError):
            compute_benchmark(BacktestEngine(), self.data, "TEST", 'spx')

    def test_compute_benchmark_returns_none_for_none(self):
        self.assertIsNone(
            compute_benchmark(BacktestEngine(), self.data, "TEST", BENCHMARK_NONE)
        )

    def test_an_unavailable_benchmark_is_reported_not_fatal(self):
        """The strategy's own numbers survive; the missing comparison is visible."""
        data = make_data([100.0, 100.0, 100.0, 100.0], volume=0.0)
        engine = BacktestEngine(commission=0.0, cost_model=VolumeShareSlippageModel())

        with self.assertLogs(level='WARNING') as logs:
            result = engine.run_backtest(ScriptedStrategy([1, 0, 0, 0]), data, "TEST")

        self.assertIsNotNone(result.benchmark_error)
        self.assertIsNone(result.benchmark_return_pct)
        self.assertIsNone(result.excess_return_pct)
        self.assertTrue(any('BENCHMARK UNAVAILABLE' in line for line in logs.output))


class TestSharedAnnualisation(unittest.TestCase):
    """Both curves must be annualised by the same inferred factor."""

    def test_benchmark_sharpe_uses_the_inferred_factor(self):
        """Daily crypto bars annualise on 365, not a hardcoded 252."""
        rng = np.random.default_rng(11)
        closes = list(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 200))))
        data = make_data(closes)
        engine = BacktestEngine(commission=0.0)

        benchmark = compute_buy_and_hold(engine, data, "TEST")

        returns = benchmark.portfolio_values.pct_change().dropna()
        expected = np.sqrt(365.0) * returns.mean() / returns.std()
        self.assertAlmostEqual(benchmark.sharpe_ratio, expected, places=9)

    def test_an_explicit_override_reaches_the_benchmark(self):
        rng = np.random.default_rng(12)
        closes = list(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 120))))
        data = make_data(closes)

        inferred = compute_buy_and_hold(
            BacktestEngine(commission=0.0), data, "TEST"
        ).sharpe_ratio
        overridden = compute_buy_and_hold(
            BacktestEngine(commission=0.0, periods_per_year=12.0), data, "TEST"
        ).sharpe_ratio

        self.assertNotAlmostEqual(inferred, overridden)


if __name__ == '__main__':
    unittest.main()
