"""
Unit tests for the deflated-Sharpe wiring in ``scripts/optimize.py``.

The analysis itself is tested in ``tests/test_optimization/test_deflated_sharpe.py``.
What is pinned here is the wiring: the block prints on every run, a truncated
search prints a refusal instead of a number, the figures reach the exported
summary, and a bug in any of it does not cost an optimisation that has already
been saved.
"""

import argparse
import io
import logging
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.run_config import RunConfig
from niffler.optimization import deflated_sharpe, plateau
from niffler.optimization.optimization_result import OptimizationResult
from scripts import optimize

BARS = 300
# The block's own header; the plateau block also names it in passing.
BLOCK = 'DEFLATED SHARPE - '


def make_result(short_window, long_window, mean_return, seed):
    """An OptimizationResult with a daily equity curve of a known drift."""
    returns = np.random.default_rng(seed).normal(mean_return, 0.01, BARS)
    backtest = Mock()
    backtest.portfolio_values = pd.Series(
        100.0 * np.cumprod(1.0 + returns),
        index=pd.date_range('2024-01-01', periods=BARS, freq='D'))
    backtest.total_return = mean_return * 1e6
    backtest.total_return_pct = mean_return * 1e4
    backtest.sharpe_ratio = 1.0
    backtest.max_drawdown = -10.0
    backtest.win_rate = 50.0
    backtest.total_trades = 5
    backtest.benchmark_return_pct = 20.0
    backtest.benchmark_sharpe_ratio = 0.5
    backtest.benchmark_max_drawdown = -25.0
    backtest.excess_return_pct = 1.0
    backtest.round_trip_count = 2
    backtest.p_value = 0.4
    return OptimizationResult(
        parameters={'short_window': short_window, 'long_window': long_window},
        backtest_result=backtest)


def sample_results():
    """A small grid, best first, the way the optimizer hands it back."""
    grid = [(short, long_) for short in range(5, 9) for long_ in range(20, 41, 10)]
    return [make_result(short, long_, mean_return=0.002 - 0.0001 * rank, seed=rank)
            for rank, (short, long_) in enumerate(grid)]


def optimize_args(**overrides):
    values = {'method': 'grid', 'sort_by': 'total_return', 'effective_trials': None}
    values.update(overrides)
    return argparse.Namespace(**values)


class TestDeflatedSharpeArguments(unittest.TestCase):

    def parse(self, *argv):
        parser = argparse.ArgumentParser()
        optimize.add_deflated_sharpe_arguments(parser)
        return parser.parse_args(list(argv))

    def test_the_default_is_to_count_every_combination(self):
        self.assertIsNone(self.parse().effective_trials)

    def test_both_spellings_parse(self):
        self.assertEqual(self.parse('--effective-trials', '12').effective_trials, 12.0)
        self.assertEqual(self.parse('--effective_trials', '12.5').effective_trials, 12.5)


class TestAnalyseDeflatedSharpe(unittest.TestCase):

    def analyse(self, results=None, selection=plateau.SELECTION_EXHAUSTIVE, **overrides):
        return optimize.analyse_deflated_sharpe(
            sample_results() if results is None else results,
            optimize_args(**overrides), selection, RunConfig())

    def test_annualisation_comes_from_the_engine_not_a_constant(self):
        # Seven bars a week: the engine infers 365, not 252.
        self.assertEqual(self.analyse().periods_per_year, 365.0)

    def test_an_explicit_periods_per_year_wins(self):
        analysis = optimize.analyse_deflated_sharpe(
            sample_results(), optimize_args(), plateau.SELECTION_EXHAUSTIVE,
            RunConfig(periods_per_year=8760))
        self.assertEqual(analysis.periods_per_year, 8760.0)

    def test_effective_trials_reaches_the_analysis(self):
        analysis = self.analyse(effective_trials=3.0)

        self.assertEqual(analysis.trials, 3.0)
        self.assertEqual(analysis.trials_source, deflated_sharpe.TRIALS_OVERRIDE)

    def test_a_truncated_search_is_refused(self):
        analysis = self.analyse(selection=plateau.SELECTION_TRUNCATED)
        self.assertEqual(analysis.status, deflated_sharpe.STATUS_TRUNCATED)

    def test_a_failure_is_no_analysis_not_an_exception(self):
        # Another test module may have left logging globally disabled.
        previous = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, previous)

        with patch('scripts.optimize.deflated_sharpe_analysis.analyse_results',
                   side_effect=RuntimeError('boom')):
            with self.assertLogs(level='WARNING'):
                self.assertIsNone(self.analyse())


class TestExportViews(unittest.TestCase):

    DOCUMENT = {'results': [{'parameters': {'w': 20},
                             'metrics': {'total_return': 5.0, 'sharpe_ratio': 1.2,
                                         'total_trades': 9}}]}
    PLATEAU = dict(optimize._NO_PLATEAU_SUMMARY)

    def summary(self, *deflated):
        args = argparse.Namespace(method='grid', sort_by='total_return')
        summary, _ = optimize.build_export_views(
            self.DOCUMENT, args, plateau.SELECTION_EXHAUSTIVE, False, self.PLATEAU, *deflated)
        return summary

    def test_the_summary_carries_the_deflated_figures(self):
        analysis = optimize.analyse_deflated_sharpe(
            sample_results(), optimize_args(), plateau.SELECTION_EXHAUSTIVE, RunConfig())
        summary = self.summary(deflated_sharpe.summary_fields(analysis))

        self.assertEqual(summary['deflated_sharpe'], analysis.probability)
        self.assertEqual(summary['deflated_sharpe_vs_grid'], analysis.probability_vs_grid)
        self.assertEqual(summary['deflated_sharpe_trials'], 12.0)
        self.assertAlmostEqual(summary['expected_max_sharpe'],
                               analysis.expected_max_sharpe * math.sqrt(365))
        # The winner's own metrics are untouched.
        self.assertEqual(summary['sharpe_ratio'], 1.2)

    def test_without_an_analysis_every_field_is_present_and_none(self):
        summary = self.summary()

        for field in deflated_sharpe.SUMMARY_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, summary)
                self.assertIsNone(summary[field])


class TestMainWiring(unittest.TestCase):
    """main() end to end, with the optimisation itself stubbed out."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.output = os.path.join(self.directory, 'results.json')

    def tearDown(self):
        for name in os.listdir(self.directory):
            os.remove(os.path.join(self.directory, name))
        os.rmdir(self.directory)

    def _run(self, *extra_argv, truncated=False):
        optimizer = Mock()
        optimizer.optimize.return_value = sample_results()
        optimizer.analyze_best_metrics.return_value = {}
        optimizer.results_truncated = truncated
        optimizer.results_document.return_value = {'metadata': {}, 'results': []}

        argv = ['optimize.py', '--data', 'test.csv', '--strategy', 'simple_ma',
                '--output', self.output] + list(extra_argv)

        buffer = io.StringIO()
        with patch('sys.argv', argv), \
                patch('scripts.optimize.setup_logging'), \
                patch('scripts.optimize.load_and_validate_data', return_value=Mock()), \
                patch('scripts.optimize.collect_provenance', return_value={}), \
                patch('scripts.optimize.create_optimizer', return_value=optimizer), \
                redirect_stdout(buffer):
            code = optimize.main()

        return code, buffer.getvalue()

    def test_the_block_prints_without_any_flag_and_after_the_plateau(self):
        code, output = self._run()

        self.assertEqual(code, 0)
        self.assertIn(BLOCK, output)
        self.assertIn('trials counted: 12 - every combination evaluated', output)
        self.assertGreater(output.index(BLOCK), output.index('PLATEAU ANALYSIS'))

    def test_the_grid_block_points_at_the_correction_instead_of_denying_one(self):
        _, output = self._run()

        self.assertNotIn('There is no multiple-testing correction', output)
        self.assertIn('DEFLATED SHARPE block does', output)

    def test_it_does_not_depend_on_the_plateau_block(self):
        _, output = self._run('--no-plateau')

        self.assertNotIn('GRID DISTRIBUTION', output)
        self.assertIn(BLOCK, output)

    def test_effective_trials_reaches_the_block(self):
        _, output = self._run('--effective-trials', '4')

        self.assertIn('trials counted: 4 (--effective-trials; 12 combinations were evaluated)',
                      output)

    def test_a_truncated_run_prints_a_refusal_not_a_number(self):
        code, output = self._run(truncated=True)
        block = output[output.index(BLOCK):]

        self.assertEqual(code, 0)
        self.assertIn('NOT COMPUTED', block)
        self.assertNotIn('%', block)

    def test_a_failure_does_not_fail_the_run(self):
        with patch('scripts.optimize.deflated_sharpe_analysis.analyse_results',
                   side_effect=RuntimeError('boom')):
            code, output = self._run()

        self.assertEqual(code, 0)
        self.assertIn('Full results saved to', output)
        self.assertNotIn(BLOCK, output)

    def test_fewer_than_one_effective_trial_is_a_usage_error(self):
        for value in ('0', '0.5', '-3', 'nan'):
            with self.subTest(value=value):
                with redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        self._run('--effective-trials', value)

                self.assertEqual(raised.exception.code, 2)
                self.assertIn('--effective-trials must be at least 1', stderr.getvalue())


class TestBacktestCaveat(unittest.TestCase):
    """A single backtest's p-value is still uncorrected, and now says where to look."""

    def test_the_caveat_names_the_corrected_figure(self):
        from types import SimpleNamespace

        from niffler.exporters.console_exporter import ConsoleExporter

        result = SimpleNamespace(
            significance_verdict='Significant at the 5% level.', round_trip_count=40,
            is_sample_sufficient=True, mean_trade_return_pct=1.2, t_statistic=2.4,
            p_value=0.02, sharpe_ci_low=None, sharpe_ci_high=None)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            ConsoleExporter()._print_significance(result)

        self.assertIn('this p-value is not corrected for that', buffer.getvalue())
        self.assertIn('DEFLATED', buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
