"""
Exits a strategy sets for itself: stop price, take-profit, trailing stop and a
holding limit.

Every scenario is a handful of hand-built bars, so each expected fill price can be
read straight off the bar it lands on. The rules under test are the ones a resting
order at a broker obeys: a stop fills at ``min(open, stop)``, a target at
``max(open, target)``, a time exit at the open, a bar touching both a stop and a
target is booked as the stop, a trail moves only on completed bars, and the cost
model prices every one of them adversely.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.cost_model import FixedSlippageModel
from niffler.backtesting.exits import ExitColumns, ExitPlan
from niffler.backtesting.trade import TradeSide
from niffler.risk.base_risk_manager import RiskDecision
from niffler.strategies.base_strategy import BaseStrategy
from niffler.strategies.registry import STRATEGY_CLASSES, create_strategy


def bars(rows):
    """OHLC rows -> a daily OHLCV frame."""
    index = pd.date_range('2024-01-01', periods=len(rows), freq='D')
    frame = pd.DataFrame(rows, columns=['open', 'high', 'low', 'close'], index=index)
    frame['volume'] = 1_000_000.0
    return frame


def flat_bar(price=100.0):
    return (price, price + 1.0, price - 1.0, price)


class ScriptedExitStrategy(BaseStrategy):
    """Emits fixed signals and, optionally, fixed exit columns."""

    def __init__(self, signals, exits=None, risk_manager=None):
        super().__init__("ScriptedExitStrategy", {}, risk_manager)
        self.signals = signals
        self.exits = exits or {}

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = self.signals[:len(df)]
        df['position_size'] = 1.0
        for column, values in self.exits.items():
            df[column] = values[:len(df)]
        return df

    def get_description(self):
        return "Scripted exits"


def column(n, values):
    """A NaN column of length n with {bar: value} set."""
    out = [np.nan] * n
    for index, value in values.items():
        out[index] = value
    return out


def run(rows, signals, exits=None, engine=None, risk_manager=None):
    engine = engine or BacktestEngine(commission=0.0, benchmark=None)
    strategy = ScriptedExitStrategy(signals, exits, risk_manager=risk_manager)
    return engine.run_backtest(strategy, bars(rows), symbol='TEST')


def sells(result):
    return [t for t in result.trades if t.side == TradeSide.SELL]


class TestStopPrice(unittest.TestCase):
    """A stop the strategy chose, filled like the risk manager's stop."""

    def test_a_stop_fills_at_the_stop_when_the_low_trades_through_it(self):
        rows = [flat_bar(), flat_bar(), flat_bar(), (99.0, 100.0, 94.0, 96.0), flat_bar()]
        signals = [1, 0, 0, 0, 0]
        result = run(rows, signals, {'stop_price': column(5, {0: 95.0})})

        (sell,) = sells(result)
        self.assertEqual(sell.timestamp, bars(rows).index[3])
        self.assertAlmostEqual(sell.price, 95.0)
        self.assertEqual(sell.exit_reason, 'stop')

    def test_a_gap_through_the_stop_fills_at_the_open(self):
        rows = [flat_bar(), flat_bar(), (90.0, 91.0, 89.0, 90.0), flat_bar()]
        result = run(rows, [1, 0, 0, 0], {'stop_price': column(4, {0: 95.0})})

        (sell,) = sells(result)
        self.assertAlmostEqual(sell.price, 90.0)

    def test_the_entry_bar_is_checked_after_the_entry_fills(self):
        # Fills at bar 1's open (100), then trades down to 94 on the same bar.
        rows = [flat_bar(), (100.0, 101.0, 94.0, 97.0), flat_bar()]
        result = run(rows, [1, 0, 0], {'stop_price': column(3, {0: 95.0})})

        buy, sell = result.trades
        self.assertEqual(buy.timestamp, sell.timestamp)
        self.assertAlmostEqual(sell.price, 95.0)

    def test_a_stop_not_reached_leaves_the_position_open(self):
        rows = [flat_bar()] * 5
        result = run(rows, [1, 0, 0, 0, 0], {'stop_price': column(5, {0: 95.0})})

        self.assertEqual(sells(result), [])

    def test_a_later_value_moves_the_stop_up(self):
        # Entry at bar 1 with a stop of 90; bar 2 asks for 98, in force from bar 3.
        rows = [flat_bar(), flat_bar(), flat_bar(), (99.5, 100.0, 97.0, 98.0), flat_bar()]
        exits = {'stop_price': column(5, {0: 90.0, 2: 98.0})}
        result = run(rows, [1, 0, 0, 0, 0], exits)

        (sell,) = sells(result)
        self.assertEqual(sell.timestamp, bars(rows).index[3])
        self.assertAlmostEqual(sell.price, 98.0)

    def test_a_later_value_never_loosens_the_stop(self):
        rows = [flat_bar(), flat_bar(), flat_bar(), (99.0, 100.0, 94.0, 96.0), flat_bar()]
        exits = {'stop_price': column(5, {0: 95.0, 2: 80.0})}
        result = run(rows, [1, 0, 0, 0, 0], exits)

        (sell,) = sells(result)
        self.assertAlmostEqual(sell.price, 95.0)

    def test_a_stop_is_only_adopted_from_the_signal_bar_not_the_fill_bar(self):
        # Bar 1 is the fill bar; its own stop value is not known before it opens.
        rows = [flat_bar(), (100.0, 101.0, 96.0, 97.0), flat_bar()]
        exits = {'stop_price': column(3, {1: 99.0})}
        result = run(rows, [1, 0, 0], exits)

        self.assertEqual(sells(result), [])


class TestTakeProfit(unittest.TestCase):

    def test_a_target_fills_at_the_target(self):
        rows = [flat_bar(), flat_bar(), (104.0, 111.0, 103.0, 109.0), flat_bar(109.0)]
        result = run(rows, [1, 0, 0, 0], {'take_profit_price': column(4, {0: 110.0})})

        (sell,) = sells(result)
        self.assertAlmostEqual(sell.price, 110.0)
        self.assertEqual(sell.exit_reason, 'take_profit')

    def test_a_gap_above_the_target_fills_at_the_open(self):
        rows = [flat_bar(), flat_bar(), (115.0, 116.0, 114.0, 115.0), flat_bar(115.0)]
        result = run(rows, [1, 0, 0, 0], {'take_profit_price': column(4, {0: 110.0})})

        (sell,) = sells(result)
        self.assertAlmostEqual(sell.price, 115.0)

    def test_a_bar_touching_stop_and_target_is_booked_as_the_stop(self):
        rows = [flat_bar(), flat_bar(), (100.0, 112.0, 94.0, 100.0), flat_bar()]
        exits = {'stop_price': column(4, {0: 95.0}),
                 'take_profit_price': column(4, {0: 110.0})}
        result = run(rows, [1, 0, 0, 0], exits)

        (sell,) = sells(result)
        self.assertAlmostEqual(sell.price, 95.0)
        self.assertEqual(sell.exit_reason, 'stop')


class TestTrailingStop(unittest.TestCase):

    def test_the_trail_follows_the_highest_high_since_entry(self):
        # Entry at 100; highs run to 120, so a 10% trail sits at 108.
        rows = [flat_bar(),
                (100.0, 110.0, 99.5, 109.0),
                (109.0, 120.0, 108.5, 118.0),
                (118.0, 119.0, 107.0, 107.5),
                flat_bar(107.0)]
        result = run(rows, [1, 0, 0, 0, 0], {'trailing_stop_pct': column(5, {0: 0.10})})

        (sell,) = sells(result)
        self.assertEqual(sell.timestamp, bars(rows).index[3])
        self.assertAlmostEqual(sell.price, 108.0)
        self.assertEqual(sell.exit_reason, 'trailing_stop')

    def test_a_bars_own_high_does_not_lift_the_trail_within_that_bar(self):
        # Bar 2 makes a high of 130 and a low of 112. Against the previous high
        # (110 -> trail 99) nothing happens; reading its own high first (-> 117)
        # would invent an exit nobody could have placed.
        rows = [flat_bar(),
                (100.0, 110.0, 99.5, 109.0),
                (113.0, 130.0, 112.0, 125.0),
                flat_bar(125.0)]
        result = run(rows, [1, 0, 0, 0], {'trailing_stop_pct': column(4, {0: 0.10})})

        self.assertEqual(sells(result), [])

    def test_the_higher_of_stop_and_trail_binds(self):
        rows = [flat_bar(),
                (100.0, 101.0, 99.5, 100.0),
                (100.0, 100.5, 96.0, 97.0),
                flat_bar()]
        exits = {'stop_price': column(4, {0: 97.0}),
                 'trailing_stop_pct': column(4, {0: 0.20})}
        result = run(rows, [1, 0, 0, 0], exits)

        (sell,) = sells(result)
        self.assertAlmostEqual(sell.price, 97.0)
        self.assertEqual(sell.exit_reason, 'stop')


class TestTimeExit(unittest.TestCase):

    def test_leaves_at_the_open_n_bars_after_the_entry(self):
        rows = [flat_bar(100.0 + i) for i in range(8)]
        result = run(rows, [1, 0, 0, 0, 0, 0, 0, 0], {'max_bars_held': column(8, {0: 3})})

        buy, sell = result.trades
        index = bars(rows).index
        self.assertEqual(buy.timestamp, index[1])
        self.assertEqual(sell.timestamp, index[4])
        self.assertAlmostEqual(sell.price, rows[4][0])
        self.assertEqual(sell.exit_reason, 'time')

    def test_counts_the_same_bars_under_same_bar_close(self):
        rows = [flat_bar(100.0 + i) for i in range(8)]
        engine = BacktestEngine(commission=0.0, benchmark=None,
                                execution_timing='same_bar_close')
        result = run(rows, [1, 0, 0, 0, 0, 0, 0, 0], {'max_bars_held': column(8, {0: 3})},
                     engine=engine)

        buy, sell = result.trades
        index = bars(rows).index
        self.assertEqual(buy.timestamp, index[0])
        self.assertEqual(sell.timestamp, index[3])

    def test_a_signal_sell_before_the_limit_is_a_signal_exit(self):
        rows = [flat_bar()] * 8
        result = run(rows, [1, 0, -1, 0, 0, 0, 0, 0], {'max_bars_held': column(8, {0: 5})})

        (sell,) = sells(result)
        self.assertEqual(sell.exit_reason, 'signal')


class TestExitsAndTheRestOfTheEngine(unittest.TestCase):

    def test_costs_make_a_stop_exit_worse_than_the_stop(self):
        rows = [flat_bar(), flat_bar(), flat_bar(), (99.0, 100.0, 94.0, 96.0), flat_bar()]
        engine = BacktestEngine(commission=0.001, benchmark=None,
                                cost_model=FixedSlippageModel(slippage_bps=10.0))
        result = run(rows, [1, 0, 0, 0, 0], {'stop_price': column(5, {0: 95.0})},
                     engine=engine)

        (sell,) = sells(result)
        self.assertLess(sell.price, 95.0)
        self.assertGreater(sell.slippage_cost, 0.0)
        self.assertGreater(sell.commission, 0.0)

    def test_costs_make_a_target_exit_worse_than_the_target(self):
        rows = [flat_bar(), flat_bar(), (104.0, 111.0, 103.0, 109.0), flat_bar(109.0)]
        engine = BacktestEngine(commission=0.0, benchmark=None,
                                cost_model=FixedSlippageModel(slippage_bps=10.0))
        result = run(rows, [1, 0, 0, 0], {'take_profit_price': column(4, {0: 110.0})},
                     engine=engine)

        (sell,) = sells(result)
        self.assertLess(sell.price, 110.0)

    def test_the_bar_of_an_exit_does_not_act_on_its_signal(self):
        rows = [flat_bar(), flat_bar(), (99.0, 100.0, 94.0, 96.0), flat_bar(), flat_bar()]
        # The buy signal on bar 1 would fill on bar 2, the bar the stop is hit.
        result = run(rows, [1, 1, 0, 0, 0], {'stop_price': column(5, {0: 95.0})})

        self.assertEqual([t.side for t in result.trades], [TradeSide.BUY, TradeSide.SELL])

    def test_a_new_position_starts_with_a_fresh_plan(self):
        rows = [flat_bar(), flat_bar(), (99.0, 100.0, 94.0, 96.0),
                flat_bar(), flat_bar(), flat_bar(), flat_bar()]
        exits = {'max_bars_held': column(7, {3: 2}), 'stop_price': column(7, {0: 95.0})}
        result = run(rows, [1, 0, 0, 1, 0, 0, 0], exits)

        reasons = [t.exit_reason for t in sells(result)]
        self.assertEqual(reasons, ['stop', 'time'])

    def test_the_risk_managers_higher_stop_goes_first(self):
        class StopAt:
            def evaluate_trade(self, signal, current_price, portfolio_value,
                               historical_data, portfolio):
                return RiskDecision(allow_trade=True, position_size=1.0,
                                    stop_loss_price=97.0, max_risk_per_trade=0.02)

            def should_close_position(self, current_price, entry_price,
                                      stop_loss_price, signal, unrealized_pnl):
                return current_price <= stop_loss_price, "risk stop"

        rows = [flat_bar(), flat_bar(), flat_bar(), (99.0, 100.0, 94.0, 96.0), flat_bar()]
        result = run(rows, [1, 0, 0, 0, 0], {'stop_price': column(5, {0: 95.0})},
                     risk_manager=StopAt())

        (sell,) = sells(result)
        self.assertAlmostEqual(sell.price, 97.0)
        self.assertEqual(sell.exit_reason, 'risk_stop')

    def test_buys_carry_no_exit_reason(self):
        rows = [flat_bar()] * 4
        result = run(rows, [1, 0, -1, 0])

        self.assertIsNone(result.trades[0].exit_reason)
        self.assertEqual(result.trades[1].exit_reason, 'signal')

    def test_all_nan_exit_columns_change_nothing(self):
        index = pd.date_range('2023-01-01', periods=300, freq='D')
        closes = [100.0 + 20.0 * math.sin(i / 9.0) for i in range(300)]
        data = pd.DataFrame({'open': closes, 'high': [c * 1.01 for c in closes],
                             'low': [c * 0.99 for c in closes], 'close': closes,
                             'volume': 1_000_000.0}, index=index)

        for name in STRATEGY_CLASSES:
            with self.subTest(strategy=name):
                strategy = create_strategy(name)
                plain = BacktestEngine().run_backtest(strategy, data)

                original = strategy.generate_signals

                def with_nan_exits(frame, original=original):
                    out = original(frame)
                    for col in ('stop_price', 'take_profit_price',
                                'trailing_stop_pct', 'max_bars_held'):
                        out[col] = np.nan
                    return out

                strategy.generate_signals = with_nan_exits
                padded = BacktestEngine().run_backtest(strategy, data)

                self.assertEqual(plain.final_capital, padded.final_capital)
                self.assertEqual(len(plain.trades), len(padded.trades))


class TestExitColumnValidation(unittest.TestCase):

    def frame(self, **columns):
        df = pd.DataFrame({'signal': [0, 0, 0]})
        for name, values in columns.items():
            df[name] = values
        return df

    def test_invalid_values_raise(self):
        cases = {
            'stop_price': [np.nan, -1.0, np.nan],
            'take_profit_price': [0.0, np.nan, np.nan],
            'trailing_stop_pct': [np.nan, 1.0, np.nan],
            'max_bars_held': [np.nan, 2.5, np.nan],
        }
        for name, values in cases.items():
            with self.subTest(column=name):
                with self.assertRaises(ValueError) as caught:
                    ExitColumns.from_signals(self.frame(**{name: values}), 3)
                self.assertIn(name, str(caught.exception))

    def test_zero_bars_and_infinite_values_raise(self):
        with self.assertRaises(ValueError):
            ExitColumns.from_signals(self.frame(max_bars_held=[0, np.nan, np.nan]), 3)
        with self.assertRaises(ValueError):
            ExitColumns.from_signals(self.frame(stop_price=[np.inf, np.nan, np.nan]), 3)

    def test_a_missing_column_means_that_exit_is_off(self):
        columns = ExitColumns.from_signals(self.frame(), 3)

        self.assertFalse(columns.any_set)
        self.assertIsNone(columns.plan_for_entry(0, entry_price=100.0, entry_bar=1))

    def test_a_backtest_with_an_invalid_column_raises(self):
        rows = [flat_bar()] * 3
        with self.assertRaises(ValueError):
            run(rows, [1, 0, 0], {'trailing_stop_pct': [0.0, np.nan, np.nan]})


class TestExitPlan(unittest.TestCase):

    def test_trailing_level_and_reason(self):
        plan = ExitPlan(stop_price=90.0, trailing_stop_pct=0.05, trail_anchor=100.0)

        self.assertAlmostEqual(plan.stop_level(), 95.0)
        self.assertEqual(plan.stop_reason(), 'trailing_stop')

        plan.record_bar(80.0)
        self.assertAlmostEqual(plan.trail_anchor, 100.0, msg="the anchor never falls")
        self.assertEqual(plan.bars_held, 1)

    def test_time_exit_due(self):
        plan = ExitPlan(max_bars_held=2)
        self.assertFalse(plan.time_exit_due())
        plan.record_bar(1.0)
        plan.record_bar(1.0)
        self.assertTrue(plan.time_exit_due())


if __name__ == '__main__':
    unittest.main()
