"""
Unit tests for the plateau analysis wiring in ``scripts/optimize.py``.

The wiring has to hold three lines: whole-grid distribution statistics print on
every run (they are the counterweight to printing only the winner), the noisier
outputs are opt-in, and a bug in any of it must not throw away an optimisation
that has already been saved to disk.
"""

import argparse
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.optimization import plateau
from niffler.optimization.optimization_result import OptimizationResult
from scripts import optimize


def make_result(short_window, long_window, total_return=10.0, total_trades=5):
    """An OptimizationResult with just enough of a backtest result on it."""
    backtest = Mock()
    backtest.total_return = total_return * 100
    backtest.total_return_pct = total_return
    backtest.sharpe_ratio = 1.0
    backtest.max_drawdown = -10.0
    backtest.win_rate = 50.0
    backtest.total_trades = total_trades
    backtest.benchmark_return_pct = 20.0
    backtest.benchmark_sharpe_ratio = 0.5
    backtest.benchmark_max_drawdown = -25.0
    backtest.excess_return_pct = total_return - 20.0
    backtest.round_trip_count = 2
    backtest.p_value = 0.4
    return OptimizationResult(
        parameters={'short_window': short_window, 'long_window': long_window},
        backtest_result=backtest)


def sample_results():
    """A small two-parameter grid with a single peak."""
    return [
        make_result(short, long_, total_return=30.0 - abs(short - 8) - abs(long_ - 30) / 5)
        for short in range(5, 12)
        for long_ in range(20, 41, 5)
    ]


class TestPlateauArguments(unittest.TestCase):
    """The flags themselves."""

    def parse(self, *argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--sort-by', default='total_return')
        optimize.add_plateau_arguments(parser)
        return parser.parse_args(list(argv))

    def test_defaults_are_quiet_but_not_off(self):
        args = self.parse()

        self.assertFalse(args.no_plateau)
        self.assertFalse(args.plateau_heatmap)
        self.assertFalse(args.plateau_centre)
        self.assertIsNone(args.plateau_csv)
        self.assertIsNone(args.plateau_metric)
        self.assertEqual(args.plateau_tolerance, plateau.DEFAULT_TOLERANCE)

    def test_flags_parse(self):
        args = self.parse('--plateau-heatmap', '--plateau-centre',
                          '--plateau-csv', 'surface.csv',
                          '--plateau-metric', 'sharpe_ratio',
                          '--plateau-tolerance', '0.5')

        self.assertTrue(args.plateau_heatmap)
        self.assertTrue(args.plateau_centre)
        self.assertEqual(args.plateau_csv, 'surface.csv')
        self.assertEqual(args.plateau_metric, 'sharpe_ratio')
        self.assertEqual(args.plateau_tolerance, 0.5)

    def test_unknown_plateau_metric_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse('--plateau-metric', 'calmar_ratio')


class TestReportPlateau(unittest.TestCase):
    """The reporting helper."""

    def setUp(self):
        self.args = argparse.Namespace(
            sort_by='total_return', plateau_metric=None,
            plateau_tolerance=plateau.DEFAULT_TOLERANCE,
            plateau_heatmap=False, plateau_centre=False, plateau_csv=None)
        self.results = sample_results()

    def _render(self, selection=plateau.SELECTION_EXHAUSTIVE):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            optimize.report_plateau(self.results, self.args, selection)
        return buffer.getvalue()

    def test_distribution_and_plateau_print(self):
        output = self._render()

        self.assertIn('GRID DISTRIBUTION', output)
        self.assertIn('PLATEAU ANALYSIS', output)

    def test_heatmap_is_opt_in(self):
        self.assertNotIn('PARAMETER SURFACE', self._render())

        self.args.plateau_heatmap = True
        self.assertIn('PARAMETER SURFACE', self._render())

    def test_centre_is_opt_in(self):
        self.assertNotIn('plateau centre', self._render())

        self.args.plateau_centre = True
        self.assertIn('plateau centre', self._render())

    def test_metric_defaults_to_sort_by(self):
        self.args.sort_by = 'sharpe_ratio'

        self.assertIn('GRID DISTRIBUTION - sharpe_ratio', self._render())

    def test_explicit_metric_overrides_sort_by(self):
        self.args.sort_by = 'sharpe_ratio'
        self.args.plateau_metric = 'max_drawdown'

        self.assertIn('GRID DISTRIBUTION - max_drawdown', self._render())

    def test_truncated_selection_withholds_the_distribution(self):
        output = self._render(selection=plateau.SELECTION_TRUNCATED)

        self.assertIn('NO DISTRIBUTION REPORTED', output)

    def test_sampled_selection_says_so(self):
        output = self._render(selection=plateau.SELECTION_SAMPLED)

        self.assertIn('sampled combinations', output)

    def test_csv_is_written_when_asked(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'surface.csv')
        self.args.plateau_csv = path
        try:
            output = self._render()

            self.assertTrue(os.path.exists(path))
            self.assertIn('Parameter surface', output)
        finally:
            if os.path.exists(path):
                os.remove(path)
            os.rmdir(directory)


class TestMainWiring(unittest.TestCase):
    """main() end to end, with the optimisation itself stubbed out."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.output = os.path.join(self.directory, 'results.json')

    def tearDown(self):
        for name in os.listdir(self.directory):
            os.remove(os.path.join(self.directory, name))
        os.rmdir(self.directory)

    def _run(self, *extra_argv, truncated=False, results=None):
        optimizer = Mock()
        optimizer.optimize.return_value = (sample_results() if results is None
                                           else results)
        optimizer.analyze_best_metrics.return_value = {}
        optimizer.results_truncated = truncated

        argv = ['optimize.py', '--data', 'test.csv', '--strategy', 'simple_ma',
                '--output', self.output] + list(extra_argv)

        buffer = io.StringIO()
        with patch('sys.argv', argv), \
                patch('scripts.optimize.setup_logging'), \
                patch('scripts.optimize.load_and_validate_data') as load, \
                patch('scripts.optimize.collect_provenance', return_value={}), \
                patch('scripts.optimize.create_optimizer',
                      return_value=optimizer) as create:
            load.return_value = Mock()
            with redirect_stdout(buffer):
                code = optimize.main()

        return code, buffer.getvalue(), create

    def test_distribution_prints_without_any_flag(self):
        code, output, _ = self._run()

        self.assertEqual(code, 0)
        self.assertIn('GRID DISTRIBUTION', output)
        self.assertIn('PLATEAU ANALYSIS', output)

    def test_no_plateau_skips_the_whole_block(self):
        code, output, _ = self._run('--no-plateau')

        self.assertEqual(code, 0)
        self.assertNotIn('GRID DISTRIBUTION', output)

    def test_heatmap_flag_reaches_the_renderer(self):
        _, output, _ = self._run('--plateau-heatmap')

        self.assertIn('PARAMETER SURFACE', output)

    def test_csv_flag_writes_the_surface(self):
        path = os.path.join(self.directory, 'surface.csv')

        code, _, _ = self._run('--plateau-csv', path)

        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(path))

    def test_cli_raises_the_optimizer_result_cap(self):
        # The default cap (1000) is below the default simple_ma grid (1632), and
        # the discarded results are the losing ones. Whole-grid statistics are
        # printed by default now, so the CLI must keep the whole grid.
        _, _, create = self._run()

        self.assertEqual(create.call_args.kwargs['max_results_in_memory'],
                         optimize.CLI_MAX_RESULTS_IN_MEMORY)
        self.assertGreater(optimize.CLI_MAX_RESULTS_IN_MEMORY, 1632)

    def test_truncated_run_reports_a_biased_sample(self):
        _, output, _ = self._run(truncated=True)

        self.assertIn('NO DISTRIBUTION REPORTED', output)

    def test_random_search_is_labelled_as_sampled(self):
        _, output, _ = self._run('--method', 'random', '--trials', '5')

        self.assertIn('sampled combinations', output)

    def test_plateau_failure_does_not_fail_the_run(self):
        # The results have already been saved by the time the plateau block
        # runs; a reporting bug must not turn a finished optimisation into a
        # non-zero exit.
        with patch('scripts.optimize.plateau_analysis.analyse_results',
                   side_effect=RuntimeError('boom')):
            code, output, _ = self._run()

        self.assertEqual(code, 0)
        self.assertIn('Full results saved to', output)

    def test_results_are_saved_before_the_plateau_block_runs(self):
        order = []
        optimizer = Mock()
        optimizer.optimize.return_value = sample_results()
        optimizer.analyze_best_metrics.return_value = {}
        optimizer.results_truncated = False
        optimizer.save_results.side_effect = lambda *a, **k: order.append('save')

        argv = ['optimize.py', '--data', 'test.csv', '--strategy', 'simple_ma',
                '--output', self.output]

        with patch('sys.argv', argv), \
                patch('scripts.optimize.setup_logging'), \
                patch('scripts.optimize.load_and_validate_data', return_value=Mock()), \
                patch('scripts.optimize.collect_provenance', return_value={}), \
                patch('scripts.optimize.create_optimizer', return_value=optimizer), \
                patch('scripts.optimize.report_plateau',
                      side_effect=lambda *a, **k: order.append('plateau')), \
                redirect_stdout(io.StringIO()):
            optimize.main()

        self.assertEqual(order, ['save', 'plateau'])


if __name__ == '__main__':
    unittest.main()
