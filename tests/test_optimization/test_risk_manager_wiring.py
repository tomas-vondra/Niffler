"""
The risk manager must reach every backtest an optimizer runs.

The optimizer constructs its own strategy objects, so a manager attached to a
strategy never reached it: every grid cell was scored with risk management off,
and the winning parameters were the ones that suited an unmanaged system.
"""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.run_config import RunConfig
from niffler.optimization.base_optimizer import BaseOptimizer
from niffler.optimization.optimizer_factory import create_optimizer
from niffler.optimization.parameter_space import ParameterSpace
from niffler.risk import FixedRiskManager, RiskDecision
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy

PARAMETER_SPACE = ParameterSpace({
    'short_window': {'type': 'int', 'min': 5, 'max': 6},
    'long_window': {'type': 'int', 'min': 20, 'max': 21},
})


class VetoRiskManager(FixedRiskManager):
    """Refuses every trade, so its presence shows up as zero trades.

    Module level, because a spawned worker unpickles it by import path.
    """

    def evaluate_trade(self, signal, current_price, portfolio_value,
                       historical_data, portfolio):
        return RiskDecision(position_size=0.0, stop_loss_price=None,
                            max_risk_per_trade=0.0, allow_trade=False,
                            reason='vetoed by test')


def make_data(n_bars=200):
    """Build an oscillating OHLCV frame, so a moving-average cross actually fires."""
    index = pd.date_range('2024-01-01', periods=n_bars, freq='D')
    closes = [100.0 + 20.0 * math.sin(i / 9.0) for i in range(n_bars)]
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


class TestOptimizerCarriesTheRiskManager(unittest.TestCase):
    """Every path an optimizer can take must run the same risk rules."""

    def setUp(self):
        self.data = make_data()
        self.risk_manager = VetoRiskManager()

    def _optimizer(self, risk_manager, n_jobs=1):
        return create_optimizer(
            method='grid',
            strategy_class=SimpleMAStrategy,
            parameter_space=PARAMETER_SPACE,
            data=self.data,
            n_jobs=n_jobs,
            run_config=RunConfig(risk_manager=risk_manager),
        )

    def test_the_reusable_engine_gets_it(self):
        optimizer = self._optimizer(self.risk_manager)

        self.assertIs(optimizer._backtest_engine.risk_manager, self.risk_manager)

    def test_the_default_is_still_no_risk_management(self):
        self.assertIsNone(self._optimizer(None)._backtest_engine.risk_manager)

    def test_the_parallel_worker_gets_it_too(self):
        """The static worker builds its own engine, so it needs it as an argument."""
        optimizer = self._optimizer(self.risk_manager, n_jobs=2)

        executor = Mock()
        executor.submit.return_value = Mock(result=Mock(return_value=None))

        with patch('niffler.optimization.base_optimizer.ProcessPoolExecutor') as pool:
            pool.return_value.__enter__.return_value = executor
            with patch('niffler.optimization.base_optimizer.as_completed',
                       return_value=[]):
                optimizer._evaluate_parallel([{'short_window': 5, 'long_window': 20}])

        self.assertEqual(executor.submit.call_count, 1)
        self.assertIs(executor.submit.call_args[0][-1].risk_manager, self.risk_manager)

    def test_the_static_worker_applies_it(self):
        unmanaged = evaluate_one(None)
        vetoed = evaluate_one(self.risk_manager)

        self.assertGreater(unmanaged.backtest_result.total_trades, 0)
        self.assertEqual(vetoed.backtest_result.total_trades, 0)

    def test_it_reaches_workers_across_the_spawn_boundary(self):
        """Real processes, a vetoing manager, and therefore no trades anywhere."""
        traded = self._optimizer(None, n_jobs=2).optimize()
        vetoed = self._optimizer(self.risk_manager, n_jobs=2).optimize()

        self.assertTrue(traded)
        self.assertTrue(vetoed)
        self.assertGreater(sum(r.backtest_result.total_trades for r in traded), 0)
        self.assertEqual(sum(r.backtest_result.total_trades for r in vetoed), 0)

    def test_it_is_recorded_in_the_saved_metadata(self):
        optimizer = self._optimizer(self.risk_manager)

        metadata = optimizer.run_config.to_metadata()['risk_manager']

        self.assertEqual(metadata['class'], 'VetoRiskManager')
        # Not a registered class, so it cannot be rebuilt by name and the record
        # says so rather than naming one that would build something else.
        self.assertIsNone(metadata['name'])


def evaluate_one(risk_manager):
    """Run one static-worker evaluation under the given risk manager."""
    return BaseOptimizer._evaluate_single_combination_static(
        {'short_window': 5, 'long_window': 20},
        SimpleMAStrategy,
        make_data(),
        RunConfig(risk_manager=risk_manager),
    )


if __name__ == '__main__':
    unittest.main()
