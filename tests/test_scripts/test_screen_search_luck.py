"""The search-luck gate of screen.py's optimize stage.

A grid search hands back its best combination, and the best of several hundred
looks good by luck alone. ``optimize.py`` prints how good; the funnel has to
stop on it. What is pinned here: the gate reads the library's figure and
computes nothing itself, the threshold reaches it from the flag and from the
configuration file, and a search whose luck could NOT be assessed stops the
funnel loudly instead of passing as if it had cleared.

Every dataset here is synthetic.
"""

import io
import math
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.run_config import RunConfig
from niffler.exporters import ExporterManager
from niffler.optimization import deflated_sharpe
from niffler.optimization.parameter_space import ParameterSpace
from scripts.screen import (
    DEFAULT_MIN_GRID_RELATIVE_PROBABILITY,
    EXIT_OK,
    EXIT_STOPPED,
    Gate,
    StageResult,
    build_parser,
    main,
    run_optimize_stage,
)

FLAG = '--min-grid-relative-probability'


def make_data(n_bars=400):
    """An oscillating OHLCV frame, so a moving-average cross actually fires."""
    index = pd.date_range('2020-01-01', periods=n_bars, freq='D')
    closes = [100.0 + 20.0 * math.sin(i / 7.0) + 0.02 * i for i in range(n_bars)]
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


def small_space():
    return ParameterSpace({
        'short_window': {'type': 'int', 'min': 3, 'max': 8, 'step': 1},
        'long_window': {'type': 'int', 'min': 15, 'max': 25, 'step': 5},
    })


def gate_of(stage):
    return next(gate for gate in stage.gates if gate.flag == FLAG)


class TestSearchLuckGate(unittest.TestCase):

    def run_stage(self, threshold=DEFAULT_MIN_GRID_RELATIVE_PROBABILITY):
        with patch('scripts.screen.get_parameter_space', return_value=small_space()):
            return run_optimize_stage(
                make_data(), 'simple_ma', RunConfig(), method='grid',
                metric='total_return', trials=10, seed=1, n_jobs=1,
                min_retention=0.0, min_grid_beat=0.0,
                min_grid_relative_probability=threshold)

    def test_the_gate_reads_the_grid_relative_probability(self):
        stage = self.run_stage()
        gate = gate_of(stage)

        self.assertIsNotNone(gate.value)
        self.assertTrue(0.0 <= gate.value <= 1.0)
        self.assertEqual(gate.value,
                         stage.payload['search_luck']['grid_relative_probability'])
        self.assertEqual(stage.payload['search_luck']['search_luck_status'],
                         deflated_sharpe.STATUS_OK)

    def test_it_is_the_figure_the_library_computes_not_a_second_one(self):
        fixed = deflated_sharpe.DeflatedSharpe(
            status=deflated_sharpe.STATUS_OK, trials=18.0,
            winner=deflated_sharpe.ReturnMoments(0.1, 0.0, 3.0, 400),
            grid_relative_luck_line=0.05, probability=0.99,
            grid_relative_probability=0.3141, periods_per_year=365.0)
        with patch('scripts.screen.search_luck_analysis.analyse_results',
                   return_value=fixed):
            stage = self.run_stage()

        # The published figure (0.99) would pass; the gate must not read it.
        self.assertEqual(gate_of(stage).value, 0.3141)
        self.assertFalse(gate_of(stage).passed)

    def test_the_threshold_decides_the_verdict(self):
        value = gate_of(self.run_stage()).value

        self.assertTrue(gate_of(self.run_stage(threshold=0.0)).passed)
        stopped = gate_of(self.run_stage(threshold=min(1.0, value + 0.01)))
        self.assertFalse(stopped.passed)
        self.assertIn(FLAG, stopped.describe())
        self.assertIn('STOPPED at optimize', stopped.describe())

    def test_the_stage_says_what_the_winner_was_compared_against(self):
        detail = ' '.join(self.run_stage().detail)

        self.assertIn('search luck over 18 trial(s)', detail)
        self.assertIn('grid-relative luck line', detail)


class TestSearchLuckNotAssessable(unittest.TestCase):

    def run_stage(self):
        with patch('scripts.screen.get_parameter_space', return_value=small_space()):
            return run_optimize_stage(
                make_data(), 'simple_ma', RunConfig(), method='grid',
                metric='total_return', trials=10, seed=1, n_jobs=1,
                min_retention=0.0, min_grid_beat=0.0,
                min_grid_relative_probability=0.0)

    def test_a_truncated_search_does_not_pass_even_a_zero_threshold(self):
        with patch('scripts.screen.CLI_MAX_RESULTS_IN_MEMORY', 4):
            stage = self.run_stage()
        gate = gate_of(stage)

        self.assertIsNone(gate.value)
        self.assertFalse(gate.passed)
        self.assertIn('search luck not assessable: truncated', gate.describe())
        self.assertEqual(stage.payload['search_luck']['search_luck_status'],
                         deflated_sharpe.STATUS_TRUNCATED)

    def test_it_is_said_loudly_and_called_not_a_pass(self):
        with patch('scripts.screen.CLI_MAX_RESULTS_IN_MEMORY', 4):
            detail = '\n'.join(self.run_stage().detail)

        self.assertIn('SEARCH LUCK NOT ASSESSABLE (truncated)', detail)
        self.assertIn('This is not a pass', detail)
        self.assertIn('stops the funnel', detail)
        self.assertIn('!' * 20, detail)

    def test_an_analysis_that_raises_is_not_assessable_rather_than_an_error(self):
        with patch('scripts.screen.search_luck_analysis.analyse_results',
                   side_effect=RuntimeError('no equity curve')):
            stage = self.run_stage()
        gate = gate_of(stage)

        self.assertIsNone(gate.value)
        self.assertFalse(gate.passed)
        self.assertIn('no equity curve', '\n'.join(stage.detail))

    def test_the_plateau_gates_are_still_reported_beside_it(self):
        with patch('scripts.screen.search_luck_analysis.analyse_results',
                   side_effect=RuntimeError('boom')):
            stage = self.run_stage()

        self.assertEqual([gate.flag for gate in stage.gates],
                         ['--min-retention', '--min-grid-beat', FLAG])


class TestSearchLuckThresholdWiring(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.data_path = os.path.join(self.tmp, 'TEST_research.csv')
        make_data().to_csv(self.data_path, index_label='timestamp')

    def run_main(self, extra_argv=(), config_text=None, luck_value=0.9):
        optimize_arguments = {}
        records = []
        calls = []

        def passing(name):
            def _stub(*args, **kwargs):
                calls.append(name)
                return StageResult(name=name, gates=[
                    Gate(stage=name, quantity='q', value=1.0, threshold=0.5,
                         flag='--flag')])
            return _stub

        def optimize(*args, **kwargs):
            calls.append('optimize')
            optimize_arguments.update(kwargs)
            return StageResult(
                name='optimize',
                gates=[Gate(stage='optimize', quantity='grid-relative probability',
                            value=luck_value,
                            threshold=kwargs['min_grid_relative_probability'],
                            flag=FLAG, unknown_reason='search luck not assessable: x')],
                payload={'winner_parameters': {'short_window': 5, 'long_window': 20}})

        real_export = ExporterManager.export_run

        def capture(manager, record):
            records.append(record)
            return real_export(manager, record)

        argv = ['screen.py', '--data', self.data_path, '--strategy', 'simple_ma']
        if config_text is not None:
            config_path = os.path.join(self.tmp, 'niffler.toml')
            with open(config_path, 'w') as handle:
                handle.write(config_text)
            argv.extend(['--config', config_path])
        argv.extend(extra_argv)

        row = {
            'symbol': 'TEST', 'strategy': 'simple_ma', 'error': None,
            'folds': 8, 'compared_folds': 8, 'failed_folds': 0,
            'oos_sharpe': 0.9, 'median_efficiency': 0.5, 'positive_fold_pct': 60.0,
            'median_fold_pct': 3.0, 'median_bh_pct': 2.0, 'median_excess_pct': 1.0,
            'beat_bh_pct': 75.0,
        }
        out = io.StringIO()
        with patch.object(sys, 'argv', argv), \
                patch('scripts.screen.run_backtest_stage', side_effect=passing('backtest')), \
                patch('scripts.screen.run_optimize_stage', side_effect=optimize), \
                patch('scripts.screen.evaluate', return_value=row), \
                patch('scripts.screen.run_walk_forward_stage',
                      side_effect=passing('walk-forward')), \
                patch.object(ExporterManager, 'export_run', capture), \
                redirect_stdout(out), redirect_stderr(io.StringIO()):
            exit_code = main()

        return {'exit': exit_code, 'arguments': optimize_arguments, 'calls': calls,
                'stdout': out.getvalue(),
                'record': records[0] if records else None}

    def test_the_default_is_a_half(self):
        self.assertEqual(DEFAULT_MIN_GRID_RELATIVE_PROBABILITY, 0.5)
        self.assertEqual(
            build_parser().parse_args(
                ['--data', 'x.csv', '--strategy', 'simple_ma']
            ).min_grid_relative_probability, 0.5)

    def test_the_default_reaches_the_stage(self):
        run = self.run_main()

        self.assertEqual(run['arguments']['min_grid_relative_probability'], 0.5)
        self.assertEqual(run['exit'], EXIT_OK)

    def test_a_typed_threshold_reaches_the_stage(self):
        run = self.run_main([FLAG, '0.8'])

        self.assertEqual(run['arguments']['min_grid_relative_probability'], 0.8)

    def test_the_threshold_may_come_from_the_configuration_file(self):
        run = self.run_main(
            config_text='[screen]\nmin_grid_relative_probability = 0.7\n')

        self.assertEqual(run['arguments']['min_grid_relative_probability'], 0.7)

    def test_the_threshold_is_recorded_on_the_run_document(self):
        run = self.run_main([FLAG, '0.8'])

        self.assertEqual(
            run['record'].document['thresholds']['min_grid_relative_probability'], 0.8)

    def test_a_winner_below_the_line_stops_the_funnel_with_exit_3(self):
        run = self.run_main(luck_value=0.4)

        self.assertEqual(run['exit'], EXIT_STOPPED)
        self.assertEqual(run['calls'], ['backtest', 'optimize'])
        self.assertIn(f'STOPPED at optimize: grid-relative probability 0.40 < 0.50 '
                      f'({FLAG})', run['stdout'])

    def test_the_threshold_is_printed_when_the_gate_passes_too(self):
        run = self.run_main(luck_value=0.9)

        self.assertIn(f'passed optimize: grid-relative probability 0.90 >= 0.50 '
                      f'({FLAG})', run['stdout'])

    def test_luck_that_could_not_be_assessed_stops_the_funnel(self):
        run = self.run_main(luck_value=None)

        self.assertEqual(run['exit'], EXIT_STOPPED)
        self.assertEqual(run['calls'], ['backtest', 'optimize'])
        self.assertIn('search luck not assessable', run['stdout'])


if __name__ == '__main__':
    unittest.main()
