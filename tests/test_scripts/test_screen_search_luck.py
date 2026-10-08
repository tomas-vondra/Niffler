"""Search luck in screen.py's optimize stage.

A grid search hands back its best combination, and the best of several hundred
looks good by luck alone. The funnel always reports how good - the figure
counts every combination as independent, which over-corrects, and the later
stages test the winner on data the search did not see, so by default it informs
and does not stop the run. It becomes a gate only when a threshold is set.

What is pinned here: the figure is the library's and is printed and exported
either way; "not set" is said where the thresholds are and recorded as None,
never as zero; and luck that could NOT be assessed is fenced off as not-a-pass
in both modes, stopping the funnel only when a threshold was asked for.

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
    report_stage,
    run_optimize_stage,
)

FLAG = '--min-grid-relative-probability'
NOT_SET = object()


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


def luck_gates(stage):
    return [gate for gate in stage.gates if gate.flag == FLAG]


def gate_of(stage):
    return luck_gates(stage)[0]


def run_stage(threshold=NOT_SET):
    arguments = {} if threshold is NOT_SET else {
        'min_grid_relative_probability': threshold}
    with patch('scripts.screen.get_parameter_space', return_value=small_space()):
        return run_optimize_stage(
            make_data(), 'simple_ma', RunConfig(), method='grid',
            metric='total_return', trials=10, seed=1, n_jobs=1,
            min_retention=0.0, min_grid_beat=0.0, **arguments)


def printed(stage):
    buf = io.StringIO()
    with redirect_stdout(buf):
        report_stage(stage)
    return buf.getvalue()


class TestSearchLuckIsReportedByDefault(unittest.TestCase):

    def test_without_a_threshold_there_is_no_search_luck_gate(self):
        stage = run_stage()

        self.assertEqual(luck_gates(stage), [])
        self.assertEqual([gate.flag for gate in stage.gates],
                         ['--min-retention', '--min-grid-beat'])
        self.assertEqual(stage.failed_gates, [])

    def test_the_figure_and_the_luck_line_are_still_printed(self):
        stage = run_stage()
        probability = stage.payload['search_luck']['grid_relative_probability']
        text = printed(stage)

        self.assertIn('search luck over 18 trial(s)', text)
        self.assertIn('grid-relative luck line', text)
        self.assertIn(f'probability the winner is truly above it {probability:.1%}',
                      text)

    def test_not_set_is_said_where_the_thresholds_are(self):
        stage = run_stage()
        probability = stage.payload['search_luck']['grid_relative_probability']
        lines = printed(stage).rstrip().splitlines()

        # After the gate verdicts, in their shape, naming the flag.
        self.assertIn('passed optimize: grid fraction beating buy-and-hold', lines[-2])
        self.assertEqual(
            lines[-1],
            f"  not gated at optimize: grid-relative probability {probability:.2f}, "
            f"no threshold set ({FLAG})")

    def test_the_figures_are_recorded_with_the_fact_that_nothing_gated_on_them(self):
        stage = run_stage()

        self.assertEqual(stage.payload['search_luck']['search_luck_status'],
                         deflated_sharpe.STATUS_OK)
        self.assertIsNotNone(stage.payload['search_luck']['grid_relative_luck_line'])
        self.assertIs(stage.payload['search_luck_gated'], False)

    def test_a_winner_far_below_the_luck_line_does_not_stop_the_stage(self):
        hopeless = deflated_sharpe.DeflatedSharpe(
            status=deflated_sharpe.STATUS_OK, trials=18.0,
            winner=deflated_sharpe.ReturnMoments(0.1, 0.0, 3.0, 400),
            grid_relative_luck_line=0.5, probability=0.5,
            grid_relative_probability=0.001, periods_per_year=365.0)
        with patch('scripts.screen.search_luck_analysis.analyse_results',
                   return_value=hopeless):
            stage = run_stage()

        self.assertEqual(stage.failed_gates, [])
        self.assertIn('grid-relative probability 0.00, no threshold set',
                      printed(stage))


class TestSearchLuckGateWhenSet(unittest.TestCase):

    def test_the_gate_reads_the_grid_relative_probability(self):
        stage = run_stage(threshold=0.0)
        gate = gate_of(stage)

        self.assertTrue(0.0 <= gate.value <= 1.0)
        self.assertEqual(gate.value,
                         stage.payload['search_luck']['grid_relative_probability'])
        self.assertIs(stage.payload['search_luck_gated'], True)
        self.assertEqual(stage.reported, [])

    def test_it_is_the_figure_the_library_computes_not_a_second_one(self):
        fixed = deflated_sharpe.DeflatedSharpe(
            status=deflated_sharpe.STATUS_OK, trials=18.0,
            winner=deflated_sharpe.ReturnMoments(0.1, 0.0, 3.0, 400),
            grid_relative_luck_line=0.05, probability=0.99,
            grid_relative_probability=0.3141, periods_per_year=365.0)
        with patch('scripts.screen.search_luck_analysis.analyse_results',
                   return_value=fixed):
            stage = run_stage(threshold=0.5)

        # The published figure (0.99) would pass; the gate must not read it.
        self.assertEqual(gate_of(stage).value, 0.3141)
        self.assertFalse(gate_of(stage).passed)

    def test_the_threshold_decides_the_verdict(self):
        value = gate_of(run_stage(threshold=0.0)).value

        self.assertTrue(gate_of(run_stage(threshold=0.0)).passed)
        stopped = gate_of(run_stage(threshold=min(1.0, value + 0.01)))
        self.assertFalse(stopped.passed)
        self.assertIn(FLAG, stopped.describe())
        self.assertIn('STOPPED at optimize', stopped.describe())

    def test_a_threshold_of_zero_is_a_gate_not_the_absence_of_one(self):
        stage = run_stage(threshold=0.0)

        self.assertEqual(len(luck_gates(stage)), 1)
        self.assertIn(f'>= 0.00 ({FLAG})', gate_of(stage).describe())


class TestSearchLuckNotAssessable(unittest.TestCase):

    def truncated(self, threshold=NOT_SET):
        with patch('scripts.screen.CLI_MAX_RESULTS_IN_MEMORY', 4):
            return run_stage(threshold)

    def test_unset_it_is_fenced_off_as_not_a_pass_and_the_stage_continues(self):
        stage = self.truncated()
        text = printed(stage)

        # The plateau gates stop a truncated search on their own account;
        # search luck adds no gate to them.
        self.assertEqual(luck_gates(stage), [])
        self.assertEqual({gate.flag for gate in stage.failed_gates},
                         {'--min-retention', '--min-grid-beat'})
        self.assertIn('SEARCH LUCK NOT ASSESSABLE (truncated)', text)
        self.assertIn('This is not a pass.', text)
        self.assertIn(f'No threshold is set ({FLAG}), so the funnel', text)
        self.assertIn('continues - with a winner whose luck was NOT judged', text)
        self.assertIn('!' * 20, text)
        self.assertNotIn('stops the funnel', text)

    def test_unset_the_threshold_line_says_there_is_no_figure(self):
        text = printed(self.truncated())

        self.assertIn(
            f"not gated at optimize: grid-relative probability None (search luck "
            f"not assessable: truncated), no threshold set ({FLAG})", text)

    def test_unset_the_missing_figure_is_exported_as_missing(self):
        stage = self.truncated()

        self.assertIsNone(stage.payload['search_luck']['grid_relative_probability'])
        self.assertEqual(stage.payload['search_luck']['search_luck_status'],
                         deflated_sharpe.STATUS_TRUNCATED)

    def test_set_it_does_not_pass_even_a_zero_threshold(self):
        stage = self.truncated(threshold=0.0)
        gate = gate_of(stage)
        text = printed(stage)

        self.assertIsNone(gate.value)
        self.assertFalse(gate.passed)
        self.assertIn('search luck not assessable: truncated', gate.describe())
        self.assertIn('This is not a pass.', text)
        self.assertIn('It stops the funnel like a failed gate', text)
        self.assertNotIn('the funnel\n  continues', text)

    def test_an_analysis_that_raises_is_not_assessable_rather_than_an_error(self):
        with patch('scripts.screen.search_luck_analysis.analyse_results',
                   side_effect=RuntimeError('no equity curve')):
            unset = run_stage()
            gated = run_stage(threshold=0.0)

        self.assertEqual(unset.failed_gates, [])
        self.assertIn('no equity curve', '\n'.join(unset.detail))
        self.assertIsNone(gate_of(gated).value)
        self.assertFalse(gate_of(gated).passed)


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
            threshold = kwargs['min_grid_relative_probability']
            stage = StageResult(
                name='optimize',
                payload={
                    'winner_parameters': {'short_window': 5, 'long_window': 20},
                    'search_luck': {'grid_relative_probability': luck_value,
                                    'grid_relative_luck_line': 1.25,
                                    'search_luck_status': 'ok'},
                })
            if threshold is not None:
                stage.gates = [Gate(
                    stage='optimize', quantity='grid-relative probability',
                    value=luck_value, threshold=threshold, flag=FLAG,
                    unknown_reason='search luck not assessable: x')]
            return stage

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

    def test_the_default_is_not_set_rather_than_zero(self):
        self.assertIsNone(DEFAULT_MIN_GRID_RELATIVE_PROBABILITY)
        self.assertIsNone(
            build_parser().parse_args(
                ['--data', 'x.csv', '--strategy', 'simple_ma']
            ).min_grid_relative_probability)

    def test_unset_reaches_the_stage_as_none(self):
        run = self.run_main()

        self.assertIsNone(run['arguments']['min_grid_relative_probability'])

    def test_a_typed_threshold_reaches_the_stage(self):
        run = self.run_main([FLAG, '0.8'])

        self.assertEqual(run['arguments']['min_grid_relative_probability'], 0.8)

    def test_a_typed_zero_reaches_the_stage_as_zero(self):
        run = self.run_main([FLAG, '0'])

        self.assertEqual(run['arguments']['min_grid_relative_probability'], 0.0)
        self.assertIsNotNone(run['arguments']['min_grid_relative_probability'])

    def test_the_threshold_may_come_from_the_configuration_file(self):
        run = self.run_main(
            config_text='[screen]\nmin_grid_relative_probability = 0.7\n')

        self.assertEqual(run['arguments']['min_grid_relative_probability'], 0.7)

    def test_a_set_threshold_is_recorded_on_the_run_document(self):
        run = self.run_main([FLAG, '0.8'])

        self.assertEqual(
            run['record'].document['thresholds']['min_grid_relative_probability'], 0.8)

    def test_not_set_is_recorded_on_the_run_document_as_none(self):
        run = self.run_main()
        thresholds = run['record'].document['thresholds']

        self.assertIn('min_grid_relative_probability', thresholds)
        self.assertIsNone(thresholds['min_grid_relative_probability'])

    def test_unset_a_winner_below_any_line_does_not_stop_the_funnel(self):
        run = self.run_main(luck_value=0.01)

        self.assertEqual(run['exit'], EXIT_OK)
        self.assertEqual(run['calls'], ['backtest', 'optimize', 'walk-forward'])

    def test_unset_luck_that_could_not_be_assessed_does_not_stop_the_funnel(self):
        run = self.run_main(luck_value=None)

        self.assertEqual(run['exit'], EXIT_OK)
        self.assertEqual(run['calls'], ['backtest', 'optimize', 'walk-forward'])

    def test_the_figures_are_exported_on_the_summary_either_way(self):
        for argv in ([], [FLAG, '0.5']):
            summary = self.run_main(argv)['record'].summary

            self.assertEqual(summary['grid_relative_probability'], 0.9)
            self.assertEqual(summary['grid_relative_luck_line'], 1.25)
            self.assertEqual(summary['search_luck_status'], 'ok')

    def test_set_a_winner_below_the_line_stops_the_funnel_with_exit_3(self):
        run = self.run_main([FLAG, '0.5'], luck_value=0.4)

        self.assertEqual(run['exit'], EXIT_STOPPED)
        self.assertEqual(run['calls'], ['backtest', 'optimize'])
        self.assertIn(f'STOPPED at optimize: grid-relative probability 0.40 < 0.50 '
                      f'({FLAG})', run['stdout'])

    def test_set_the_threshold_is_printed_when_the_gate_passes_too(self):
        run = self.run_main([FLAG, '0.5'], luck_value=0.9)

        self.assertIn(f'passed optimize: grid-relative probability 0.90 >= 0.50 '
                      f'({FLAG})', run['stdout'])

    def test_set_luck_that_could_not_be_assessed_stops_the_funnel(self):
        run = self.run_main([FLAG, '0.5'], luck_value=None)

        self.assertEqual(run['exit'], EXIT_STOPPED)
        self.assertEqual(run['calls'], ['backtest', 'optimize'])
        self.assertIn('search luck not assessable', run['stdout'])

    def test_the_closing_line_does_not_claim_a_gate_that_was_not_set(self):
        text = ' '.join(self.run_main()['stdout'].split())

        self.assertIn('no gate corrected for the parameter search', text)
        self.assertIn(f'not gated on ({FLAG} is not set)', text)
        self.assertNotIn('the search-luck gate corrects', text)

    def test_the_closing_line_names_the_gate_when_it_ran(self):
        text = ' '.join(self.run_main([FLAG, '0.5'])['stdout'].split())

        self.assertIn('the search-luck gate corrects for this one parameter search',
                      text)
        self.assertNotIn('no gate corrected', text)

    def test_help_says_it_is_off_by_default_and_why(self):
        buf = io.StringIO()
        with patch.object(sys, 'argv', ['screen.py', '--help']), redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                build_parser().parse_args()
        help_text = ' '.join(buf.getvalue().split())

        self.assertIn('default: not set', help_text)
        self.assertIn('over-corrects', help_text)
        self.assertIn('data the search did not see', help_text)


if __name__ == '__main__':
    unittest.main()
