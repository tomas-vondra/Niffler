"""
Unit tests for holding back per-backtest log lines while backtests run in bulk.

A single backtest logs each fill at INFO, which is a trade log. A sequential
396-combination grid doing the same wrote about 41,700 lines, burying the
search's own output. ``quiet_backtests`` silences the backtesting package below
WARNING for the duration of a bulk run and leaves a single backtest alone.
"""

import logging
import sys
import unittest
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.analysis.monte_carlo_analyzer import MonteCarloAnalyzer
from niffler.analysis.walk_forward_analyzer import WalkForwardAnalyzer
from niffler.backtesting.backtest_engine import BacktestEngine, quiet_backtests
from niffler.optimization.grid_search_optimizer import GridSearchOptimizer
from niffler.optimization.parameter_space import ParameterSpace
from niffler.optimization.random_search_optimizer import RandomSearchOptimizer
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy

PACKAGE_LOGGER = logging.getLogger('niffler.backtesting')
PARAMETERS = {'short_window': 3, 'long_window': 8}


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


def make_space():
    return ParameterSpace({
        'short_window': {'type': 'int', 'min': 3, 'max': 5, 'step': 1},
    })


class LoggingTestCase(unittest.TestCase):

    def setUp(self):
        # Another test module may have left logging disabled globally.
        previous = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, previous)

    def fill_lines(self, captured):
        return [line for line in captured.output if 'BUY: ' in line or 'SELL: ' in line]

    def engine_lines(self, captured):
        return [line for line in captured.output if ':niffler.backtesting' in line]


class TestQuietBacktests(LoggingTestCase):

    def run_backtest(self):
        return BacktestEngine().run_backtest(
            SimpleMAStrategy(**PARAMETERS), make_price_data(), 'TEST')

    def test_a_single_backtest_still_logs_its_fills(self):
        with self.assertLogs(level='INFO') as captured:
            result = self.run_backtest()

        self.assertTrue(result.trades)
        self.assertEqual(len(self.fill_lines(captured)), len(result.trades))

    def test_inside_the_block_the_engine_logs_nothing_below_warning(self):
        with self.assertLogs(level='INFO') as captured:
            logging.info('marker')
            with quiet_backtests():
                result = self.run_backtest()

        self.assertTrue(result.trades)
        self.assertEqual(self.engine_lines(captured), [])

    def test_warnings_still_get_through(self):
        with self.assertLogs(level='INFO') as captured:
            with quiet_backtests():
                logging.getLogger('niffler.backtesting.backtest_engine').warning('stop missed')

        self.assertEqual(len(captured.output), 1)
        self.assertIn('stop missed', captured.output[0])

    def test_the_level_is_restored_afterwards(self):
        before = PACKAGE_LOGGER.level
        with quiet_backtests():
            self.assertEqual(PACKAGE_LOGGER.level, logging.WARNING)

        self.assertEqual(PACKAGE_LOGGER.level, before)

    def test_the_level_is_restored_when_the_backtest_raises(self):
        before = PACKAGE_LOGGER.level
        with self.assertRaises(RuntimeError):
            with quiet_backtests():
                raise RuntimeError('boom')

        self.assertEqual(PACKAGE_LOGGER.level, before)

    def test_debug_turns_the_full_log_back_on(self):
        with self.assertLogs(level='DEBUG') as captured:
            with quiet_backtests():
                result = self.run_backtest()

        self.assertEqual(len(self.fill_lines(captured)), len(result.trades))


class TestBulkRunsAreQuiet(LoggingTestCase):
    """Every loop that runs many backtests in this process holds the fills back."""

    def assert_no_engine_lines(self, run):
        with self.assertLogs(level='INFO') as captured:
            logging.info('marker')
            outcome = run()
        self.assertEqual(self.engine_lines(captured), [])
        return outcome

    def test_sequential_grid_search(self):
        optimizer = GridSearchOptimizer(
            strategy_class=SimpleMAStrategy, parameter_space=make_space(),
            data=make_price_data(), n_jobs=1)

        results = self.assert_no_engine_lines(optimizer.optimize)

        self.assertTrue(any(r.backtest_result.trades for r in results))

    def test_worker_entry_point(self):
        # The pool's entry point, called in-process: under fork a worker
        # inherits the parent's handlers and would log every fill.
        result = self.assert_no_engine_lines(
            lambda: RandomSearchOptimizer._evaluate_single_combination_static(
                PARAMETERS, SimpleMAStrategy, make_price_data(), None))

        self.assertTrue(result.backtest_result.trades)

    def test_monte_carlo(self):
        analyzer = MonteCarloAnalyzer(
            strategy_class=SimpleMAStrategy, optimal_parameters=PARAMETERS,
            n_simulations=3, block_size_days=10, n_jobs=1, random_seed=7)

        result = self.assert_no_engine_lines(
            lambda: analyzer.analyze(make_price_data(200), 'TEST'))

        self.assertTrue(result.individual_results)

    def test_walk_forward(self):
        analyzer = WalkForwardAnalyzer(
            strategy_class=SimpleMAStrategy, parameter_space=make_space(),
            train_window_months=3, test_window_months=1, step_months=1, n_jobs=1)

        result = self.assert_no_engine_lines(
            lambda: analyzer.analyze(make_price_data(240), 'TEST'))

        self.assertTrue(result.individual_results)


if __name__ == '__main__':
    unittest.main()
