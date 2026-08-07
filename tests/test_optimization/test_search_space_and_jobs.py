"""
Unit tests for three optimizer defects found while building plateau analysis.

1. Random search ignored ``step`` for integer parameters, so ``--method grid``
   and ``--method random`` searched different spaces (17 vs 81 values for a
   ``long_window`` declared 20..100 step 5). The float branch had the same
   fault in subtler form: it sampled ``k * step`` rather than
   ``min + k * step``, which can fall below ``min``.
2. ``GridSearchOptimizer.optimize`` ignored ``n_jobs`` entirely, so ``--jobs 8``
   silently did nothing. Parallel evaluation must also give byte-identical
   results whatever the worker count.
3. ``analyze_best_metrics`` announced a "best" ``total_trades``, which is a
   count and has no best.

The parallel tests run real backtests: mocks do not cross a process boundary,
so a stubbed engine would prove nothing about the pool.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.optimization.base_optimizer import BaseOptimizer
from niffler.optimization.grid_search_optimizer import GridSearchOptimizer
from niffler.optimization.parameter_space import ParameterSpace
from niffler.optimization.random_search_optimizer import RandomSearchOptimizer
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy


def make_price_data(periods=90):
    """A deterministic price series with enough shape to produce trades."""
    index = pd.date_range('2024-01-01', periods=periods, freq='D')
    closes = [100.0 + (i % 17) * 1.5 - (i % 7) * 2.0 + i * 0.3 for i in range(periods)]
    return pd.DataFrame({
        'open': closes,
        'high': [close * 1.01 for close in closes],
        'low': [close * 0.99 for close in closes],
        'close': closes,
        'volume': [10_000.0] * periods,
    }, index=index)


class TestSteppedSampling(unittest.TestCase):
    """Random search samples the lattice grid search enumerates."""

    def setUp(self):
        self.data = make_price_data(40)
        self.space = ParameterSpace({
            'short_window': {'type': 'int', 'min': 5, 'max': 20, 'step': 1},
        })

    def _optimizer(self, space=None):
        return RandomSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=space or self.space,
            data=self.data,
            n_jobs=1,
        )

    def test_int_values_respect_step(self):
        optimizer = self._optimizer()
        config = {'type': 'int', 'min': 20, 'max': 100, 'step': 5}

        values = {optimizer._generate_random_parameter_value('long_window', config)
                  for _ in range(500)}

        self.assertTrue(values)
        for value in values:
            self.assertEqual((value - 20) % 5, 0)
            self.assertGreaterEqual(value, 20)
            self.assertLessEqual(value, 100)

    def test_int_sampling_covers_exactly_the_grid_values(self):
        optimizer = self._optimizer()
        config = {'type': 'int', 'min': 20, 'max': 100, 'step': 5}

        sampled = {optimizer._generate_random_parameter_value('long_window', config)
                   for _ in range(2000)}
        enumerated = set(optimizer._generate_parameter_values('long_window', config))

        self.assertEqual(sampled, enumerated)

    def test_int_without_a_step_still_covers_every_value(self):
        optimizer = self._optimizer()
        config = {'type': 'int', 'min': 1, 'max': 5}

        sampled = {optimizer._generate_random_parameter_value('window', config)
                   for _ in range(500)}

        self.assertEqual(sampled, {1, 2, 3, 4, 5})

    def test_float_values_stay_on_the_grid_lattice(self):
        optimizer = self._optimizer()
        config = {'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.1}

        sampled = {optimizer._generate_random_parameter_value('position_size', config)
                   for _ in range(1000)}
        enumerated = set(optimizer._generate_parameter_values('position_size', config))

        self.assertEqual(sampled, enumerated)

    def test_float_minimum_that_is_not_a_multiple_of_step(self):
        # The old implementation sampled k*step, so it could return 0.5 here -
        # below the declared minimum - and never returned the grid's values.
        optimizer = self._optimizer()
        config = {'type': 'float', 'min': 0.55, 'max': 1.05, 'step': 0.1}

        sampled = {optimizer._generate_random_parameter_value('position_size', config)
                   for _ in range(1000)}

        for value in sampled:
            self.assertGreaterEqual(value, 0.55)
        self.assertEqual(sampled,
                         set(optimizer._generate_parameter_values('position_size', config)))

    def test_continuous_float_is_still_continuous(self):
        optimizer = self._optimizer()
        config = {'type': 'float', 'min': 0.0, 'max': 1.0}

        sampled = [optimizer._generate_random_parameter_value('x', config)
                   for _ in range(50)]

        self.assertGreater(len(set(sampled)), 40)
        for value in sampled:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_estimated_space_size_matches_the_sampled_lattice(self):
        space = ParameterSpace({
            'short_window': {'type': 'int', 'min': 5, 'max': 20, 'step': 1},
            'long_window': {'type': 'int', 'min': 20, 'max': 100, 'step': 5},
            'position_size': {'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.1},
        })
        optimizer = self._optimizer(space)

        self.assertEqual(optimizer._estimate_total_combinations(), 16 * 17 * 6)

    def test_random_and_grid_search_the_same_space(self):
        space = ParameterSpace({
            'short_window': {'type': 'int', 'min': 5, 'max': 9, 'step': 2},
            'long_window': {'type': 'int', 'min': 20, 'max': 40, 'step': 10},
        })
        random_optimizer = self._optimizer(space)
        grid = GridSearchOptimizer(
            strategy_class=SimpleMAStrategy, parameter_space=space,
            data=self.data, n_jobs=1)

        grid_combinations = {tuple(sorted(combination.items()))
                             for combination in grid._generate_grid_combinations_lazy()}
        sampled = {tuple(sorted(random_optimizer._generate_single_combination().items()))
                   for _ in range(500)}

        self.assertEqual(sampled, grid_combinations)


class TestGridSearchJobs(unittest.TestCase):
    """--jobs actually does something for a grid search, and changes nothing."""

    def setUp(self):
        self.data = make_price_data()
        self.space = ParameterSpace({
            'short_window': {'type': 'int', 'min': 3, 'max': 5, 'step': 1},
            'long_window': {'type': 'int', 'min': 10, 'max': 20, 'step': 5},
        })

    def _optimizer(self, n_jobs):
        return GridSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=self.space,
            data=self.data,
            n_jobs=n_jobs,
        )

    @staticmethod
    def _fingerprint(results):
        return [(tuple(sorted(result.parameters.items())),
                 result.backtest_result.total_return_pct,
                 result.backtest_result.sharpe_ratio,
                 result.backtest_result.max_drawdown,
                 result.backtest_result.total_trades)
                for result in results]

    def test_single_job_uses_the_lazy_in_process_path(self):
        optimizer = self._optimizer(1)

        with patch.object(optimizer, '_evaluate_combinations_lazy',
                          return_value=[]) as lazy:
            optimizer.optimize()

        lazy.assert_called_once()

    def test_multiple_jobs_use_the_shared_pool(self):
        optimizer = self._optimizer(4)

        with patch.object(optimizer, '_evaluate_combinations',
                          return_value=[]) as shared:
            optimizer.optimize()

        shared.assert_called_once()
        # ...and it receives the whole grid, not a generator it cannot count.
        self.assertEqual(len(shared.call_args.args[0]), 9)

    def test_results_are_identical_across_job_counts(self):
        sequential = self._optimizer(1).optimize()

        parallel_optimizer = self._optimizer(2)
        with patch.object(parallel_optimizer, '_evaluate_parallel',
                          wraps=parallel_optimizer._evaluate_parallel) as pool:
            parallel = parallel_optimizer.optimize()

        # The comparison is worthless unless the pool really ran.
        pool.assert_called_once()
        self.assertEqual(len(sequential), 9)
        self.assertEqual(len(parallel), 9)
        self.assertEqual(self._fingerprint(sequential), self._fingerprint(parallel))


class TestParallelRetentionOrder(unittest.TestCase):
    """Results are retained in submission order whatever the completion order."""

    def setUp(self):
        self.data = make_price_data(40)
        self.space = ParameterSpace(
            {'short_window': {'type': 'int', 'min': 3, 'max': 6, 'step': 1}})

    def _optimizer(self):
        return GridSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=self.space,
            data=self.data,
            n_jobs=2,
        )

    def test_out_of_order_completion_is_retained_in_order(self):
        optimizer = self._optimizer()
        combinations = [{'short_window': window} for window in range(3, 7)]
        retained = []

        def record(results, new_result):
            retained.append(new_result.parameters['short_window'])
            return results + [new_result]

        # A fake pool that completes the last submission first.
        with patch('niffler.optimization.base_optimizer.ProcessPoolExecutor',
                   _ReversedExecutor), \
                patch('niffler.optimization.base_optimizer.as_completed',
                      lambda futures: list(futures)[::-1]), \
                patch.object(optimizer, '_manage_memory_efficient_results',
                             side_effect=record):
            results = optimizer._evaluate_parallel(combinations)

        self.assertEqual(retained, [3, 4, 5, 6])
        self.assertEqual(len(results), 4)

    def test_a_failed_evaluation_does_not_stall_the_drain(self):
        optimizer = self._optimizer()
        combinations = [{'short_window': window} for window in range(3, 7)]
        retained = []

        def record(results, new_result):
            retained.append(new_result.parameters['short_window'])
            return results + [new_result]

        with patch('niffler.optimization.base_optimizer.ProcessPoolExecutor',
                   _FailFirstExecutor), \
                patch('niffler.optimization.base_optimizer.as_completed',
                      lambda futures: list(futures)[::-1]), \
                patch.object(optimizer, '_manage_memory_efficient_results',
                             side_effect=record):
            optimizer._evaluate_parallel(combinations)

        # The first submission raised; the rest must still be retained, in order.
        self.assertEqual(retained, [4, 5, 6])


class _FakeFuture:
    """A future whose result is already known."""

    def __init__(self, parameters, error=None):
        self.parameters = parameters
        self.error = error

    def done(self):
        return True

    def cancel(self):
        return False

    def result(self, timeout=None):
        if self.error is not None:
            raise self.error
        from niffler.optimization.optimization_result import OptimizationResult
        from unittest.mock import Mock

        backtest = Mock()
        backtest.total_return_pct = float(self.parameters['short_window'])
        backtest.total_trades = 1
        return OptimizationResult(parameters=self.parameters, backtest_result=backtest)


class _ReversedExecutor:
    """A stand-in pool that runs nothing and completes in reverse order."""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, parameters, *args):
        return _FakeFuture(parameters)


class _FailFirstExecutor(_ReversedExecutor):
    """Same, but the first submission raises when its result is read."""

    def __init__(self, max_workers=None):
        super().__init__(max_workers)
        self._first = True

    def submit(self, function, parameters, *args):
        if self._first:
            self._first = False
            return _FakeFuture(parameters, error=RuntimeError('worker died'))
        return _FakeFuture(parameters)


class TestDescriptiveMetrics(unittest.TestCase):
    """A trade count is not a quality metric."""

    def test_total_trades_is_declared_descriptive(self):
        self.assertIn('total_trades', BaseOptimizer.DESCRIPTIVE_METRICS)

    def test_descriptive_metrics_are_still_sortable(self):
        # Ordering results by trade count stays available; what is gone is the
        # claim that one of them is "best".
        self.assertIn('total_trades', BaseOptimizer.METRICS_CONFIG)


class TestWalkForwardStillSerial(unittest.TestCase):
    """Grid search going parallel must not nest pools inside a fold."""

    def test_per_fold_optimizer_is_created_with_one_job(self):
        from niffler.analysis import walk_forward_analyzer

        source = Path(walk_forward_analyzer.__file__).read_text(encoding='utf-8')

        self.assertIn('n_jobs=1', source)

    def test_fold_optimizer_call_passes_n_jobs_one(self):
        from niffler.analysis.walk_forward_analyzer import WalkForwardAnalyzer

        space = ParameterSpace(
            {'short_window': {'type': 'int', 'min': 3, 'max': 4, 'step': 1}})

        with patch('niffler.analysis.walk_forward_analyzer.create_optimizer') as create:
            create.return_value.optimize.return_value = []
            WalkForwardAnalyzer._optimize_on_training_window(
                train_data=make_price_data(60),
                strategy_class=SimpleMAStrategy,
                parameter_space=space,
                optimization_method='grid',
                optimization_metric='total_return',
                initial_capital=10000.0,
                commission=0.001,
            )

        self.assertTrue(create.called)
        self.assertEqual(create.call_args.kwargs.get('n_jobs'), 1)


if __name__ == '__main__':
    unittest.main()
