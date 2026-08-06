"""
Unit tests for transaction costs inside the backtest engine.

These cover the wiring rather than the models themselves (see
``test_cost_model.py``): that fills are priced adversely on entries, exits and
stops, that the cash budget is solved against the price actually paid, that a
liquidity cap produces a recorded partial fill rather than a vanished order, and
that the frictionless default leaves every previously produced number alone.
"""

import logging
import sys
import unittest
from pathlib import Path

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.cost_model import (
    FixedSlippageModel,
    VolumeShareSlippageModel,
    ZeroCostModel,
)
from niffler.backtesting.portfolio import Portfolio
from niffler.backtesting.trade import TradeSide
from niffler.risk.base_risk_manager import RiskDecision
from niffler.strategies.base_strategy import BaseStrategy


class ScriptedStrategy(BaseStrategy):
    """Emits a fixed list of signals, one per bar."""

    def __init__(self, signals, position_sizes=None, risk_manager=None):
        super().__init__("ScriptedStrategy", {}, risk_manager)
        self.signals = signals
        self.position_sizes = position_sizes

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = self.signals[:len(df)]
        df['position_size'] = (self.position_sizes[:len(df)]
                               if self.position_sizes is not None else 1.0)
        return df

    def validate_data(self, data):
        return True

    def get_description(self):
        return "Scripted strategy for cost tests"


class StopRiskManager:
    """Arms a stop at a fixed price and closes the position once it is breached."""

    def __init__(self, stop_loss_price):
        self.stop_loss_price = stop_loss_price
        self.positions = {}

    def evaluate_trade(self, signal, current_price, portfolio_value, historical_data,
                       current_position):
        return RiskDecision(
            position_size=1.0,
            stop_loss_price=self.stop_loss_price,
            max_risk_per_trade=1.0,
            allow_trade=True,
            reason="fixed stop"
        )

    def should_close_position(self, current_price, entry_price, stop_loss_price, signal,
                              unrealized_pnl):
        if stop_loss_price is not None and current_price <= stop_loss_price:
            return True, "stop breached"
        return False, "stop intact"

    def update_position_state(self, symbol, position_size, entry_price, stop_loss_price,
                              entry_timestamp):
        self.positions[symbol] = entry_price

    def clear_position(self, symbol):
        self.positions.pop(symbol, None)


def make_data(closes, volume=1_000_000.0, lows=None):
    """Build a flat-ish OHLCV frame where open == close of each bar."""
    index = pd.date_range('2024-01-01', periods=len(closes), freq='D')
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': lows if lows is not None else [c * 0.99 for c in closes],
        'close': closes,
        'volume': [volume] * len(closes),
    }, index=index)


class TestFrictionlessDefault(unittest.TestCase):
    """The default must leave existing results untouched."""

    def setUp(self):
        self.data = make_data([100.0, 100.0, 110.0, 110.0])
        self.strategy = ScriptedStrategy([1, 0, -1, 0])

    def test_default_cost_model_is_zero_cost(self):
        self.assertIsInstance(BacktestEngine().cost_model, ZeroCostModel)

    def test_default_matches_an_explicit_zero_cost_model(self):
        default = BacktestEngine().run_backtest(self.strategy, self.data, "TEST")
        explicit = BacktestEngine(cost_model=ZeroCostModel()).run_backtest(
            ScriptedStrategy([1, 0, -1, 0]), self.data, "TEST"
        )

        self.assertEqual(default.final_capital, explicit.final_capital)
        self.assertEqual([t.price for t in default.trades],
                         [t.price for t in explicit.trades])

    def test_fills_land_on_the_untouched_next_bar_open(self):
        result = BacktestEngine().run_backtest(self.strategy, self.data, "TEST")

        self.assertEqual(result.trades[0].price, 100.0)
        self.assertEqual(result.trades[0].slippage_cost, 0.0)
        self.assertEqual(result.total_slippage, 0.0)

    def test_rejects_something_that_is_not_a_cost_model(self):
        with self.assertRaises(TypeError):
            BacktestEngine(cost_model="fixed")


class TestCostsAreAdverseInTheEngine(unittest.TestCase):
    """Entries pay up, exits give up - in the engine, not just the model."""

    def setUp(self):
        self.data = make_data([100.0, 100.0, 100.0, 100.0])
        self.engine = BacktestEngine(
            commission=0.0,
            cost_model=FixedSlippageModel(slippage_bps=50.0, half_spread_bps=10.0)
        )

    def test_buy_fills_above_and_sell_below_the_reference_price(self):
        result = self.engine.run_backtest(ScriptedStrategy([1, 0, -1, 0]),
                                          self.data, "TEST")

        buy_trade = next(t for t in result.trades if t.side == TradeSide.BUY)
        sell_trade = next(t for t in result.trades if t.side == TradeSide.SELL)

        self.assertGreater(buy_trade.price, 100.0)
        self.assertLess(sell_trade.price, 100.0)

    def test_a_round_trip_at_a_flat_price_loses_money(self):
        """Frictionless, this trade is a wash; with costs it must be a loss."""
        result = self.engine.run_backtest(ScriptedStrategy([1, 0, -1, 0]),
                                          self.data, "TEST")

        self.assertLess(result.final_capital, 10000.0)
        self.assertGreater(result.total_slippage, 0.0)

    def test_slippage_cost_is_recorded_per_trade_and_totalled(self):
        result = self.engine.run_backtest(ScriptedStrategy([1, 0, -1, 0]),
                                          self.data, "TEST")

        for trade in result.trades:
            self.assertGreater(trade.slippage_cost, 0.0)
            self.assertAlmostEqual(
                trade.slippage_cost, abs(trade.price - 100.0) * trade.quantity
            )

        self.assertAlmostEqual(result.total_slippage,
                               sum(t.slippage_cost for t in result.trades))

    def test_commission_is_totalled_too(self):
        engine = BacktestEngine(
            commission=0.002,
            cost_model=FixedSlippageModel(slippage_bps=10.0)
        )

        result = engine.run_backtest(ScriptedStrategy([1, 0, -1, 0]), self.data, "TEST")

        self.assertAlmostEqual(result.total_commission,
                               sum(t.commission for t in result.trades))
        self.assertGreater(result.total_commission, 0.0)

    def test_entry_price_is_the_price_actually_paid(self):
        """The cost basis must be the fill, or every stop distance is wrong."""
        portfolio = Portfolio(10000.0)
        strategy = ScriptedStrategy([1])
        trades = []

        self.engine._process_buy(strategy, portfolio, trades,
                                 pd.Timestamp('2024-01-02'), "TEST",
                                 100.0, 1.0, None, bar_volume=1e9)

        self.assertEqual(portfolio.entry_price, trades[0].price)
        self.assertGreater(portfolio.entry_price, 100.0)


class TestCashCannotGoNegative(unittest.TestCase):
    """
    The buy budget has to be solved against the slipped price.

    Sizing on the reference price and paying slippage on top spends cash the
    portfolio does not have; with a heavy enough cost model it goes negative.
    """

    def _buy_everything(self, cost_model, commission=0.001):
        engine = BacktestEngine(initial_capital=10000.0, commission=commission,
                                cost_model=cost_model)
        portfolio = Portfolio(10000.0)
        trades = []

        engine._process_buy(ScriptedStrategy([1]), portfolio, trades,
                            pd.Timestamp('2024-01-02'), "TEST",
                            100.0, 1.0, None, bar_volume=1e9)

        return portfolio, trades

    def test_full_capital_buy_with_heavy_slippage_leaves_cash_non_negative(self):
        for bps in (10.0, 100.0, 1_000.0, 5_000.0):
            with self.subTest(slippage_bps=bps):
                portfolio, trades = self._buy_everything(
                    FixedSlippageModel(slippage_bps=bps)
                )

                self.assertEqual(len(trades), 1)
                self.assertGreaterEqual(portfolio.cash, 0.0)

    def test_full_capital_buy_spends_essentially_everything(self):
        portfolio, trades = self._buy_everything(FixedSlippageModel(slippage_bps=500.0))

        self.assertAlmostEqual(portfolio.cash, 0.0, places=6)
        self.assertAlmostEqual(trades[0].quantity * trades[0].price,
                               trades[0].value, places=6)

    def test_a_volume_capped_full_capital_buy_also_leaves_cash_non_negative(self):
        portfolio, trades = self._buy_everything(
            VolumeShareSlippageModel(impact_coefficient=0.5, max_participation=0.2)
        )

        self.assertGreaterEqual(portfolio.cash, 0.0)
        self.assertEqual(len(trades), 1)


class TestStopLossPaysCosts(unittest.TestCase):
    """A stop is an exit like any other: it pays, it never earns."""

    def setUp(self):
        # Other test modules disable logging globally at import time, which would
        # make every assertLogs assertion here vacuously fail. Re-enable for the
        # duration of each test and restore whatever level was in force afterwards.
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, previous_disable_level)

        # Bar 1 opens at 100 and trades down to 90, breaching a stop at 95.
        self.data = make_data([100.0, 100.0, 100.0, 100.0],
                              lows=[99.0, 90.0, 99.0, 99.0])
        self.risk_manager = StopRiskManager(stop_loss_price=95.0)

    def test_long_stop_fills_at_or_below_the_stop_price(self):
        engine = BacktestEngine(
            commission=0.0,
            cost_model=FixedSlippageModel(slippage_bps=100.0)
        )
        portfolio = Portfolio(10000.0)
        portfolio.position = 10.0
        portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)
        strategy = ScriptedStrategy([0], risk_manager=self.risk_manager)
        trades = []

        triggered = engine._process_stop_loss(
            strategy, portfolio, trades, pd.Timestamp('2024-01-02'), "TEST",
            price=100.0, bar_low=90.0, bar_high=101.0, bar_volume=1e9
        )

        self.assertTrue(triggered)
        self.assertEqual(len(trades), 1)
        # Reference fill is min(open, stop) = 95; costs may only worsen it.
        self.assertLess(trades[0].price, 95.0)
        self.assertGreater(trades[0].slippage_cost, 0.0)

    def test_costs_never_improve_a_gapped_stop_fill(self):
        """Opening below the stop fills at the open, then costs make it worse."""
        engine = BacktestEngine(
            commission=0.0,
            cost_model=FixedSlippageModel(slippage_bps=100.0)
        )
        portfolio = Portfolio(10000.0)
        portfolio.position = 10.0
        portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)
        trades = []

        engine._process_stop_loss(
            ScriptedStrategy([0], risk_manager=self.risk_manager), portfolio, trades,
            pd.Timestamp('2024-01-02'), "TEST",
            price=90.0, bar_low=88.0, bar_high=91.0, bar_volume=1e9
        )

        self.assertLess(trades[0].price, 90.0)

    def test_a_partially_filled_stop_leaves_the_rest_open_with_its_stop(self):
        engine = BacktestEngine(
            commission=0.0,
            cost_model=VolumeShareSlippageModel(impact_coefficient=0.0,
                                                max_participation=0.5)
        )
        portfolio = Portfolio(10000.0)
        portfolio.position = 10.0
        portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)
        trades = []

        with self.assertLogs(level='WARNING') as logs:
            triggered = engine._process_stop_loss(
                ScriptedStrategy([0], risk_manager=self.risk_manager), portfolio,
                trades, pd.Timestamp('2024-01-02'), "TEST",
                price=100.0, bar_low=90.0, bar_high=101.0, bar_volume=10.0
            )

        self.assertTrue(triggered)
        self.assertAlmostEqual(trades[0].quantity, 5.0)
        self.assertAlmostEqual(portfolio.position, 5.0)
        self.assertEqual(portfolio.stop_loss, 95.0)
        self.assertTrue(any('PARTIALLY FILLED' in line for line in logs.output))
        # The risk manager must still believe a position is open.
        self.assertIn("TEST", self.risk_manager.positions)

    def test_an_unfillable_bar_leaves_the_stop_unexecuted_and_says_so(self):
        engine = BacktestEngine(commission=0.0, cost_model=VolumeShareSlippageModel())
        portfolio = Portfolio(10000.0)
        portfolio.position = 10.0
        portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)
        trades = []

        with self.assertLogs(level='WARNING') as logs:
            triggered = engine._process_stop_loss(
                ScriptedStrategy([0], risk_manager=self.risk_manager), portfolio,
                trades, pd.Timestamp('2024-01-02'), "TEST",
                price=100.0, bar_low=90.0, bar_high=101.0, bar_volume=0.0
            )

        self.assertFalse(triggered)
        self.assertEqual(trades, [])
        self.assertAlmostEqual(portfolio.position, 10.0)
        self.assertTrue(any('STOP LOSS NOT EXECUTED' in line for line in logs.output))


class TestLiquidityLimits(unittest.TestCase):
    """A capped order is truncated and recorded, never silently dropped."""

    def setUp(self):
        # Other test modules disable logging globally at import time, which would
        # make every assertLogs assertion here vacuously fail. Re-enable for the
        # duration of each test and restore whatever level was in force afterwards.
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, previous_disable_level)

        self.data = make_data([100.0, 100.0, 100.0, 100.0], volume=50.0)

    def test_a_capped_buy_is_recorded_at_the_reduced_quantity(self):
        engine = BacktestEngine(
            commission=0.0,
            cost_model=VolumeShareSlippageModel(impact_coefficient=0.0,
                                                max_participation=0.1)
        )

        with self.assertLogs(level='WARNING') as logs:
            result = engine.run_backtest(ScriptedStrategy([1, 0, 0, 0]),
                                         self.data, "TEST")

        self.assertEqual(len(result.trades), 1)
        # 10% of a 50-unit bar.
        self.assertAlmostEqual(result.trades[0].quantity, 5.0)
        self.assertTrue(any('PARTIAL FILL' in line for line in logs.output))

    def test_a_bar_that_traded_nothing_fills_nothing(self):
        data = make_data([100.0, 100.0, 100.0, 100.0], volume=0.0)
        engine = BacktestEngine(commission=0.0, cost_model=VolumeShareSlippageModel())

        with self.assertLogs(level='WARNING') as logs:
            result = engine.run_backtest(ScriptedStrategy([1, 0, 0, 0]), data, "TEST")

        self.assertEqual(result.trades, [])
        self.assertTrue(any('ORDER NOT FILLED' in line for line in logs.output))

    def test_a_zero_volume_bar_still_fills_under_the_frictionless_default(self):
        """ZeroCostModel ignores volume, so old backtests on such data still run."""
        data = make_data([100.0, 100.0, 100.0, 100.0], volume=0.0)

        result = BacktestEngine().run_backtest(ScriptedStrategy([1, 0, 0, 0]),
                                               data, "TEST")

        self.assertEqual(len(result.trades), 1)


class TestCostsShowUpInPerformance(unittest.TestCase):
    """Costs must reduce the reported edge, not sit in a field nobody reads."""

    def test_a_costed_run_underperforms_the_frictionless_one(self):
        data = make_data([100.0, 100.0, 110.0, 110.0, 120.0, 120.0])
        signals = [1, 0, -1, 1, -1, 0]

        free = BacktestEngine().run_backtest(ScriptedStrategy(signals), data, "TEST")
        costed = BacktestEngine(
            cost_model=FixedSlippageModel(slippage_bps=25.0, half_spread_bps=5.0)
        ).run_backtest(ScriptedStrategy(signals), data, "TEST")

        self.assertLess(costed.final_capital, free.final_capital)
        self.assertLess(costed.total_return_pct, free.total_return_pct)
        self.assertGreater(costed.total_slippage, 0.0)
        self.assertEqual(free.total_slippage, 0.0)


if __name__ == '__main__':
    unittest.main()
