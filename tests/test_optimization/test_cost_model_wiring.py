"""
The cost model must reach every backtest an optimizer runs.

Optimising frictionlessly and then trading the winning parameters with real
costs picks the parameter set most sensitive to costs, which is the opposite of
what the search was asked for.
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

from niffler.backtesting.cost_model import FixedSlippageModel, ZeroCostModel
from niffler.backtesting.run_config import RunConfig
from niffler.optimization.optimizer_factory import create_optimizer
from niffler.optimization.parameter_space import ParameterSpace
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy

PARAMETER_SPACE = ParameterSpace({
    'short_window': {'type': 'int', 'min': 5, 'max': 6},
    'long_window': {'type': 'int', 'min': 20, 'max': 21},
})


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


class TestOptimizerCarriesTheCostModel(unittest.TestCase):
    """Every path an optimizer can take must see the same market."""

    def setUp(self):
        self.data = make_data()
        self.cost_model = FixedSlippageModel(slippage_bps=25.0)

    def _optimizer(self, cost_model, n_jobs=1):
        return create_optimizer(
            method='grid',
            strategy_class=SimpleMAStrategy,
            parameter_space=PARAMETER_SPACE,
            data=self.data,
            n_jobs=n_jobs,
            run_config=RunConfig(cost_model=cost_model),
        )

    def test_the_reusable_engine_gets_it(self):
        optimizer = self._optimizer(self.cost_model)

        self.assertIs(optimizer._backtest_engine.cost_model, self.cost_model)

    def test_the_default_stays_frictionless(self):
        optimizer = self._optimizer(None)

        self.assertIsInstance(optimizer._backtest_engine.cost_model, ZeroCostModel)

    def test_the_parallel_worker_gets_it_too(self):
        """The static worker builds its own engine, so it needs it as an argument."""
        optimizer = self._optimizer(self.cost_model, n_jobs=2)

        executor = Mock()
        executor.submit.return_value = Mock(result=Mock(return_value=None))

        with patch('niffler.optimization.base_optimizer.ProcessPoolExecutor') as pool:
            pool.return_value.__enter__.return_value = executor
            with patch('niffler.optimization.base_optimizer.as_completed',
                       return_value=[]):
                optimizer._evaluate_parallel([{'short_window': 5, 'long_window': 20}])

        self.assertEqual(executor.submit.call_count, 1)
        self.assertIs(executor.submit.call_args[0][-1].cost_model, self.cost_model)

    def test_the_static_worker_applies_it(self):
        free = optimizer_result(None)
        costed = optimizer_result(self.cost_model)

        self.assertLess(costed.backtest_result.final_capital,
                        free.backtest_result.final_capital)

    def test_costs_are_recorded_in_the_saved_metadata(self):
        optimizer = self._optimizer(self.cost_model)
        results = optimizer.optimize()

        self.assertTrue(results)
        self.assertGreater(results[0].backtest_result.total_slippage, 0.0)


def optimizer_result(cost_model):
    """Run one static-worker evaluation with the given cost model."""
    from niffler.optimization.base_optimizer import BaseOptimizer

    return BaseOptimizer._evaluate_single_combination_static(
        {'short_window': 5, 'long_window': 20},
        SimpleMAStrategy,
        make_data(),
        RunConfig(cost_model=cost_model),
    )


if __name__ == '__main__':
    unittest.main()
