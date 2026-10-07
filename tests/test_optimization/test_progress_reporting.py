"""
Unit tests for progress reporting during a search.

A parallel search used to log nothing between "Evaluating N combinations" and
"Optimization completed": spawned workers do not inherit the logging
configuration, so a slow grid was indistinguishable from a hung one. Progress is
now logged from the parent process, throttled by time.

The parallel test runs real backtests: mocks do not cross a process boundary.
"""

import logging
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


def make_optimizer(n_jobs):
    space = ParameterSpace({
        'short_window': {'type': 'int', 'min': 3, 'max': 6, 'step': 1},
    })
    return GridSearchOptimizer(
        strategy_class=SimpleMAStrategy,
        parameter_space=space,
        data=make_price_data(),
        n_jobs=n_jobs,
    )


def enable_logging(test_case):
    """Undo any global logging.disable left behind by another test module."""
    previous = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    test_case.addCleanup(logging.disable, previous)


def progress_lines(captured):
    return [line for line in captured.output if 'Progress: ' in line]


class TestProgressLine(unittest.TestCase):
    """What one progress line says, and when it is withheld."""

    def setUp(self):
        enable_logging(self)
        self.optimizer = make_optimizer(n_jobs=1)

    def test_reports_count_share_elapsed_and_eta(self):
        with patch('niffler.optimization.base_optimizer.time.monotonic', return_value=130.0):
            with self.assertLogs(level='INFO') as captured:
                self.optimizer._report_progress(100, 400, started=100.0, last_reported=100.0)

        self.assertEqual(len(captured.output), 1)
        self.assertIn('Progress: 100/400 combinations (25%)', captured.output[0])
        self.assertIn('elapsed 0:00:30', captured.output[0])
        # 30s for 100 combinations leaves 90s for the other 300.
        self.assertIn('ETA 0:01:30', captured.output[0])

    def test_returns_the_time_of_the_line_it_logged(self):
        with patch('niffler.optimization.base_optimizer.time.monotonic', return_value=130.0):
            with self.assertLogs(level='INFO'):
                reported = self.optimizer._report_progress(
                    100, 400, started=100.0, last_reported=100.0)

        self.assertEqual(reported, 130.0)

    def test_is_withheld_inside_the_interval(self):
        inside = 100.0 + BaseOptimizer.PROGRESS_INTERVAL_SECONDS - 1.0
        with patch('niffler.optimization.base_optimizer.time.monotonic', return_value=inside):
            with self.assertNoLogs(level='INFO'):
                reported = self.optimizer._report_progress(
                    100, 400, started=100.0, last_reported=100.0)

        self.assertEqual(reported, 100.0)

    def test_the_last_combination_is_left_to_the_completion_line(self):
        with patch('niffler.optimization.base_optimizer.time.monotonic', return_value=500.0):
            with self.assertNoLogs(level='INFO'):
                self.optimizer._report_progress(400, 400, started=100.0, last_reported=100.0)

    def test_durations_over_an_hour_keep_the_hours(self):
        self.assertEqual(BaseOptimizer._format_seconds(3725.4), '1:02:05')


class TestProgressDuringSearch(unittest.TestCase):
    """Both evaluation paths report, from the parent process."""

    def setUp(self):
        enable_logging(self)

    def _run(self, n_jobs):
        optimizer = make_optimizer(n_jobs)
        with patch.object(BaseOptimizer, 'PROGRESS_INTERVAL_SECONDS', 0.0):
            with self.assertLogs(level='INFO') as captured:
                results = optimizer.optimize()
        return results, progress_lines(captured)

    def test_sequential_search_reports_every_combination_but_the_last(self):
        results, lines = self._run(n_jobs=1)

        self.assertEqual(len(results), 4)
        self.assertEqual(len(lines), 3)
        self.assertIn('Progress: 1/4 combinations (25%)', lines[0])
        self.assertIn('Progress: 3/4 combinations (75%)', lines[2])

    def test_parallel_search_reports_from_the_parent(self):
        results, lines = self._run(n_jobs=2)

        self.assertEqual(len(results), 4)
        self.assertEqual(len(lines), 3)
        self.assertIn('Progress: 3/4 combinations (75%)', lines[2])

    def test_sequential_random_search_reports(self):
        # Random search at n_jobs=1 takes a different loop from the lazy grid.
        optimizer = RandomSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=ParameterSpace({
                'short_window': {'type': 'int', 'min': 3, 'max': 6, 'step': 1},
            }),
            data=make_price_data(),
            n_jobs=1,
        )
        with patch.object(BaseOptimizer, 'PROGRESS_INTERVAL_SECONDS', 0.0):
            with self.assertLogs(level='INFO') as captured:
                results = optimizer.optimize(n_trials=3, seed=7)

        lines = progress_lines(captured)
        self.assertEqual(len(lines), len(results) - 1)
        self.assertIn(f'Progress: 1/{len(results)} combinations', lines[0])

    def test_a_fast_search_stays_quiet_at_the_default_interval(self):
        optimizer = make_optimizer(n_jobs=1)
        with self.assertLogs(level='INFO') as captured:
            optimizer.optimize()

        self.assertEqual(progress_lines(captured), [])


if __name__ == '__main__':
    unittest.main()
