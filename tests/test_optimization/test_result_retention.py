"""
Unit tests for the optimizer's in-memory result cap.

The cap discards the *worst-scoring* half of the results. That never changes the
winner - the running best survives every purge - but it does leave a sample
biased towards high scores, which is fatal for anything describing the whole
grid (see :mod:`niffler.optimization.plateau`). These tests pin down that the
cap is configurable, that hitting it is reported rather than silent, and that
the default library cap is smaller than the default simple_ma grid, which is how
the bias got in unnoticed in the first place.
"""

import sys
import unittest
from pathlib import Path
from typing import List
from unittest.mock import Mock

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.optimization.base_optimizer import BaseOptimizer
from niffler.optimization.grid_search_optimizer import GridSearchOptimizer
from niffler.optimization.optimization_result import OptimizationResult
from niffler.optimization.parameter_space import ParameterSpace
from niffler.optimization.optimizer_factory import get_parameter_space
from niffler.strategies.base_strategy import BaseStrategy

SIMPLE_MA_PARAMETER_SPACE = get_parameter_space('simple_ma')


class RetentionTestStrategy(BaseStrategy):
    """Minimal strategy so the optimizer will construct."""

    def __init__(self, param1=10):
        super().__init__("RetentionTestStrategy", {"param1": param1})
        self.param1 = param1

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['position_size'] = 0.0
        return signals

    def get_description(self) -> str:
        return "Strategy used by the result-retention tests"


class RetentionTestOptimizer(BaseOptimizer):
    """Concrete optimizer that evaluates nothing by itself."""

    def optimize(self) -> List[OptimizationResult]:
        return []


def make_result(value: float) -> OptimizationResult:
    """An optimization result whose total return is ``value``."""
    backtest = Mock()
    backtest.total_return_pct = value
    backtest.total_return = value
    backtest.total_trades = 1
    return OptimizationResult(parameters={'param1': value}, backtest_result=backtest)


class TestResultRetention(unittest.TestCase):
    """The cap, the flag, and the default that is too small for the default grid."""

    def setUp(self):
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        self.data = pd.DataFrame({
            'open': [100.0] * 20,
            'high': [101.0] * 20,
            'low': [99.0] * 20,
            'close': [100.5] * 20,
            'volume': [1000.0] * 20,
        }, index=dates)
        self.parameter_space = ParameterSpace(
            {'param1': {'type': 'int', 'min': 1, 'max': 10, 'step': 1}})

    def _optimizer(self, **kwargs) -> RetentionTestOptimizer:
        return RetentionTestOptimizer(
            strategy_class=RetentionTestStrategy,
            parameter_space=self.parameter_space,
            data=self.data,
            **kwargs
        )

    def test_default_cap_is_the_class_constant(self):
        optimizer = self._optimizer()

        self.assertEqual(optimizer.max_results_in_memory,
                         BaseOptimizer.MAX_RESULTS_IN_MEMORY)

    def test_cap_is_configurable(self):
        optimizer = self._optimizer(max_results_in_memory=25000)

        self.assertEqual(optimizer.max_results_in_memory, 25000)

    def test_cap_below_two_is_rejected(self):
        with self.assertRaises(ValueError):
            self._optimizer(max_results_in_memory=1)

    def test_nothing_is_truncated_before_the_cap_is_hit(self):
        optimizer = self._optimizer(max_results_in_memory=10)

        results = []
        for value in range(10):
            results = optimizer._manage_memory_efficient_results(results, make_result(value))

        self.assertEqual(len(results), 10)
        self.assertFalse(optimizer.results_truncated)

    def test_hitting_the_cap_is_reported(self):
        optimizer = self._optimizer(max_results_in_memory=10)

        results = []
        for value in range(11):
            results = optimizer._manage_memory_efficient_results(results, make_result(value))

        self.assertTrue(optimizer.results_truncated)
        self.assertEqual(len(results), 5)

    def test_truncation_discards_the_losers_not_a_random_half(self):
        # This is the property that makes the surviving sample unusable for
        # whole-grid statistics, and the reason results_truncated exists.
        optimizer = self._optimizer(max_results_in_memory=10)

        results = []
        for value in range(11):
            results = optimizer._manage_memory_efficient_results(results, make_result(value))

        survivors = sorted(result.backtest_result.total_return_pct for result in results)
        self.assertEqual(survivors, [6.0, 7.0, 8.0, 9.0, 10.0])

    def test_truncation_never_drops_the_winner(self):
        optimizer = self._optimizer(max_results_in_memory=10)

        results = []
        for value in list(range(11)) + [999.0]:
            results = optimizer._manage_memory_efficient_results(results, make_result(value))

        self.assertEqual(max(result.backtest_result.total_return_pct
                             for result in results), 999.0)

    def test_raising_the_cap_keeps_every_result(self):
        optimizer = self._optimizer(max_results_in_memory=2000)

        results = []
        for value in range(1500):
            results = optimizer._manage_memory_efficient_results(results, make_result(value))

        self.assertEqual(len(results), 1500)
        self.assertFalse(optimizer.results_truncated)

    def test_default_simple_ma_grid_is_larger_than_the_default_cap(self):
        # 16 x 17 x 6 = 1632 combinations against a default cap of 1000: an
        # unmodified grid search over the shipped parameter space discards the
        # worst results. The CLI raises the cap for exactly this reason.
        optimizer = GridSearchOptimizer(
            strategy_class=RetentionTestStrategy,
            parameter_space=SIMPLE_MA_PARAMETER_SPACE,
            data=self.data,
        )

        combinations = optimizer._estimate_combinations_count()

        self.assertGreater(combinations, BaseOptimizer.MAX_RESULTS_IN_MEMORY)


if __name__ == '__main__':
    unittest.main()
