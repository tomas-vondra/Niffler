"""One risk manager, several runs: no run may inherit another's positions.

This is the regression suite for the defect that kept a risk manager out of
validation entirely. ``BaseRiskManager`` used to hold a ``Dict[str, PositionInfo]``
that the engine mutated through ``update_position_state`` / ``clear_position``,
so a manager instance accumulated a *history*. Walk-forward runs its folds in
parallel and each fold is an independent hypothetical history; one shared manager
would carry the first fold's open position into the second, where ``max_positions``
would veto its first entry.

The tests here drive the public engine API, so they keep failing if the state ever
comes back under a different name.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.portfolio import Portfolio
from niffler.backtesting.trade import TradeSide
from niffler.risk.contract import PortfolioSnapshot
from niffler.risk.fixed_risk_manager import FixedRiskManager
from niffler.strategies.base_strategy import BaseStrategy


class ScriptedStrategy(BaseStrategy):
    """Emits the signals it was handed, one per bar."""

    def __init__(self, signals, risk_manager=None):
        super().__init__("ScriptedStrategy", {}, risk_manager)
        self.signals = signals

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = self.signals[:len(df)]
        df['position_size'] = 1.0
        return df

    def validate_data(self, data):
        return True

    def get_description(self):
        return "Scripted strategy for risk-state tests"


def rising_data(start='2024-01-01', periods=6, base=100.0):
    """Prices that only rise, so no stop can fire and the position stays open."""
    closes = [base + i * 5.0 for i in range(periods)]
    return pd.DataFrame(
        {
            'open': closes,
            'high': [c + 1.0 for c in closes],
            'low': [c - 0.5 for c in closes],
            'close': closes,
            'volume': [1_000_000.0] * periods,
        },
        index=pd.date_range(start, periods=periods, freq='D'),
    )


def make_manager():
    """A manager whose position limit is one - the boundary the leak bit at."""
    return FixedRiskManager(position_size_pct=0.2, stop_loss_pct=0.05,
                            max_positions=1, max_risk_per_trade=0.02)


class TestPositionStateDoesNotLeakBetweenRuns(unittest.TestCase):

    def setUp(self):
        self.engine = BacktestEngine(initial_capital=10000.0, commission=0.001)
        # A buy on bar 0 that is never sold: both runs end holding the position.
        self.first_data = rising_data(start='2024-01-01', base=100.0)
        self.second_data = rising_data(start='2024-06-01', base=200.0)
        self.signals = [1, 0, 0, 0, 0, 0]

    def _run(self, data, risk_manager):
        strategy = ScriptedStrategy(self.signals, risk_manager)
        return self.engine.run_backtest(strategy, data, "TEST")

    def test_the_first_run_really_does_end_holding_a_position(self):
        """The precondition: without it the leak could not be observed at all."""
        result = self._run(self.first_data, make_manager())

        buys = [t for t in result.trades if t.side == TradeSide.BUY]
        sells = [t for t in result.trades if t.side == TradeSide.SELL]
        self.assertEqual(len(buys), 1)
        self.assertEqual(len(sells), 0)

    def test_a_reused_manager_trades_exactly_like_a_fresh_one(self):
        """The regression: run A's open position must not veto run B's entry."""
        shared = make_manager()
        self._run(self.first_data, shared)
        reused = self._run(self.second_data, shared)

        fresh = self._run(self.second_data, make_manager())

        self.assertEqual(len(reused.trades), len(fresh.trades))
        self.assertGreater(len(reused.trades), 0)
        self.assertAlmostEqual(reused.final_capital, fresh.final_capital, places=9)

    def test_the_same_manager_serves_many_runs_identically(self):
        """Ten sequential runs, the shape a walk-forward fold sweep would take."""
        shared = make_manager()
        capitals = [self._run(self.second_data, shared).final_capital
                    for _ in range(10)]

        self.assertEqual(len({round(c, 9) for c in capitals}), 1)


class TestPortfolioLevelControlsStillApply(unittest.TestCase):
    """max_positions has to keep working now that it reads from the snapshot."""

    def test_max_positions_vetoes_a_scale_in_while_already_positioned(self):
        engine = BacktestEngine(initial_capital=10000.0, commission=0.001)
        risk_manager = make_manager()  # max_positions=1
        strategy = ScriptedStrategy([1, 0, 1, 0, 0, 0], risk_manager)

        result = engine.run_backtest(strategy, rising_data(), "TEST")

        buys = [t for t in result.trades if t.side == TradeSide.BUY]
        self.assertEqual(len(buys), 1, "the second buy should be vetoed at the limit")

    def test_a_higher_limit_lets_the_scale_in_through(self):
        engine = BacktestEngine(initial_capital=10000.0, commission=0.001)
        risk_manager = FixedRiskManager(position_size_pct=0.2, stop_loss_pct=0.05,
                                        max_positions=3, max_risk_per_trade=0.02)
        strategy = ScriptedStrategy([1, 0, 1, 0, 0, 0], risk_manager)

        result = engine.run_backtest(strategy, rising_data(), "TEST")

        buys = [t for t in result.trades if t.side == TradeSide.BUY]
        self.assertEqual(len(buys), 2)


class TestPortfolioOwnsTheSnapshot(unittest.TestCase):

    def test_a_flat_portfolio_reports_nothing_open(self):
        snapshot = Portfolio(10000.0).risk_snapshot(100.0)

        self.assertEqual(snapshot.open_positions, 0)
        self.assertEqual(snapshot.total_exposure, 0.0)
        self.assertEqual(snapshot.current_position, 0.0)

    def test_exposure_is_valued_at_the_price_passed_in(self):
        """Exposure means "now", not "at the price of the last fill".

        The old manager recorded the fraction at each fill and never revalued it,
        so a position that doubled in price still reported its entry-day weight
        and the exposure cap silently stopped binding.
        """
        portfolio = Portfolio(10000.0)
        portfolio.cash = 5000.0
        portfolio.position = 50.0
        portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)

        at_entry = portfolio.risk_snapshot(100.0)
        after_a_doubling = portfolio.risk_snapshot(200.0)

        self.assertAlmostEqual(at_entry.total_exposure, 0.5, places=9)
        self.assertAlmostEqual(after_a_doubling.total_exposure, 2 / 3, places=9)
        self.assertEqual(after_a_doubling.open_positions, 1)

    def test_dust_is_not_an_open_position(self):
        portfolio = Portfolio(10000.0)
        portfolio.position = Portfolio.POSITION_TOLERANCE / 2

        self.assertEqual(portfolio.risk_snapshot(100.0).open_positions, 0)


class BrokenRiskManager:
    """Has evaluate_trade under the old name only - a rename, in other words."""

    def evaluate_trade(self, signal, current_price, portfolio_value, historical_data,
                       portfolio):
        raise AssertionError("should never be reached")


class TestTheContractIsChecked(unittest.TestCase):
    """The engine used to reach its manager by pure duck typing."""

    def test_a_manager_missing_a_method_fails_before_the_first_bar(self):
        engine = BacktestEngine(initial_capital=10000.0)
        strategy = ScriptedStrategy([1, 0, 0, 0, 0, 0], BrokenRiskManager())

        with self.assertRaises(TypeError) as ctx:
            engine.run_backtest(strategy, rising_data(), "TEST")

        message = str(ctx.exception)
        self.assertIn('BrokenRiskManager', message)
        self.assertIn('should_close_position', message)

    def test_a_conforming_manager_passes_the_check(self):
        engine = BacktestEngine(initial_capital=10000.0)
        strategy = ScriptedStrategy([1, 0, 0, 0, 0, 0], make_manager())

        result = engine.run_backtest(strategy, rising_data(), "TEST")

        self.assertGreater(len(result.trades), 0)

    def test_no_risk_manager_is_still_allowed(self):
        engine = BacktestEngine(initial_capital=10000.0)
        strategy = ScriptedStrategy([1, 0, 0, 0, 0, 0])

        result = engine.run_backtest(strategy, rising_data(), "TEST")

        self.assertGreater(len(result.trades), 0)


class TestSnapshotHandedToTheManager(unittest.TestCase):

    def test_the_engine_hands_over_the_state_it_owns(self):
        seen = []

        class RecordingManager(FixedRiskManager):
            def evaluate_trade(self, signal, current_price, portfolio_value,
                               historical_data, portfolio):
                seen.append(portfolio)
                return super().evaluate_trade(signal, current_price, portfolio_value,
                                              historical_data, portfolio)

        engine = BacktestEngine(initial_capital=10000.0, commission=0.001)
        risk_manager = RecordingManager(position_size_pct=0.2, stop_loss_pct=0.05,
                                        max_positions=3, max_risk_per_trade=0.02)
        strategy = ScriptedStrategy([1, 0, 0, -1, 0, 0], risk_manager)

        engine.run_backtest(strategy, rising_data(), "TEST")

        self.assertEqual(len(seen), 2)
        self.assertTrue(all(isinstance(s, PortfolioSnapshot) for s in seen))
        self.assertEqual(seen[0].open_positions, 0)
        self.assertEqual(seen[1].open_positions, 1)
        self.assertGreater(seen[1].current_position, 0.0)


if __name__ == '__main__':
    unittest.main()
