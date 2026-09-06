"""
The engine owns its risk manager, and still honours the strategy's.

``backtest.py`` has always attached the manager to the strategy object, which is
why the optimizer and the analyzers - which build their own strategies - could
never be given one. The engine now carries it, and the strategy-attached route
stays live so ``create_strategy(..., risk_manager=)`` keeps working. Two
different managers are an error rather than a silent preference.
"""

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.run_config import RunConfig
from niffler.risk import FixedRiskManager
from niffler.strategies.registry import create_strategy
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy


def make_data(n_bars=300):
    """Build an oscillating OHLCV frame, so a moving-average cross actually fires."""
    index = pd.date_range('2023-01-01', periods=n_bars, freq='D')
    closes = [100.0 + 20.0 * math.sin(i / 9.0) for i in range(n_bars)]
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


class TestTheEngineOwnsIt(unittest.TestCase):
    """The new route: a manager on the RunConfig."""

    def setUp(self):
        self.data = make_data()
        self.parameters = {'short_window': 5, 'long_window': 20}

    def test_from_config_forwards_it(self):
        manager = FixedRiskManager()

        engine = BacktestEngine.from_config(RunConfig(risk_manager=manager))

        self.assertIs(engine.risk_manager, manager)

    def test_the_default_engine_has_none(self):
        self.assertIsNone(BacktestEngine().risk_manager)

    def test_it_changes_the_result_of_a_strategy_that_carries_none(self):
        strategy = SimpleMAStrategy(**self.parameters)

        unmanaged = BacktestEngine.from_config(RunConfig()).run_backtest(
            strategy, self.data, 'TEST')
        managed = BacktestEngine.from_config(
            RunConfig(risk_manager=FixedRiskManager(position_size_pct=0.05))
        ).run_backtest(strategy, self.data, 'TEST')

        self.assertIsNone(strategy.risk_manager)
        self.assertNotEqual(unmanaged.total_return, managed.total_return)

    def test_a_manager_missing_the_contract_is_rejected_when_configured(self):
        class NotAManager:
            pass

        with self.assertRaises(TypeError) as context:
            RunConfig(risk_manager=NotAManager())

        self.assertIn('evaluate_trade', str(context.exception))


class TestTheStrategyRouteStillWorks(unittest.TestCase):
    """The old route: a manager on the strategy, engine unaware."""

    def setUp(self):
        self.data = make_data()
        self.parameters = {'short_window': 5, 'long_window': 20}

    def test_an_unconfigured_engine_falls_back_to_the_strategy(self):
        manager = FixedRiskManager(position_size_pct=0.05)
        strategy = create_strategy('simple_ma', self.parameters, risk_manager=manager)
        engine = BacktestEngine.from_config(RunConfig())

        self.assertIs(engine._effective_risk_manager(strategy), manager)

    def test_the_fallback_still_changes_the_result(self):
        engine = BacktestEngine.from_config(RunConfig())

        unmanaged = engine.run_backtest(
            create_strategy('simple_ma', self.parameters), self.data, 'TEST')
        managed = engine.run_backtest(
            create_strategy('simple_ma', self.parameters,
                            risk_manager=FixedRiskManager(position_size_pct=0.05)),
            self.data, 'TEST')

        self.assertNotEqual(unmanaged.total_return, managed.total_return)

    def test_the_engines_manager_wins_when_it_is_the_same_object(self):
        """backtest.py hands one manager to both, which must not be an error."""
        manager = FixedRiskManager()
        strategy = create_strategy('simple_ma', self.parameters, risk_manager=manager)
        engine = BacktestEngine.from_config(RunConfig(risk_manager=manager))

        result = engine.run_backtest(strategy, self.data, 'TEST')

        self.assertIsNotNone(result)

    def test_two_different_managers_are_an_error(self):
        strategy = create_strategy('simple_ma', self.parameters,
                                   risk_manager=FixedRiskManager())
        engine = BacktestEngine.from_config(RunConfig(risk_manager=FixedRiskManager()))

        with self.assertRaises(ValueError) as context:
            engine.run_backtest(strategy, self.data, 'TEST')

        self.assertIn('risk managers', str(context.exception))


if __name__ == '__main__':
    unittest.main()
