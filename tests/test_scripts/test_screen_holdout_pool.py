"""The holdout stage of screen.py over several files.

One holdout file is one asset and often a handful of round trips. Several are
pooled into one verdict, and what is pinned here is how: one backtest per file
with the same parameters, round trips summed, excess taken as the median of the
per-file figures, a file the strategy never traded on left out of that median,
and every file still exported as its own row with its own hash. The date rule
is the other half: a holdout must start after EVERY research file in the run.

Every dataset here is synthetic. No test may read a real ``*_holdout.csv``.
"""

import hashlib
import io
import json
import math
import os
import shutil
import statistics
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
from niffler.exporters.run_record import DETAIL_COMPARISON
from scripts.screen import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_STOPPED,
    STAGE_HOLDOUT,
    Gate,
    StageResult,
    check_holdout_follows_research,
    holdout_summary,
    main,
    run_holdout_stage,
    run_pooled_holdout_stage,
)

WINNER = {'short_window': 5, 'long_window': 20}
TRADES = '--min-holdout-trades'
EXCESS = '--min-holdout-excess'


def make_data(start='2020-01-01', n_bars=600, period=7.0):
    """An oscillating OHLCV frame, so a moving-average cross actually fires."""
    index = pd.date_range(start, periods=n_bars, freq='D')
    closes = [100.0 + 20.0 * math.sin(i / period) for i in range(n_bars)]
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


def make_flat_data(start='2022-01-01', n_bars=200):
    """A constant price: no moving-average cross, so no trade at all."""
    index = pd.date_range(start, periods=n_bars, freq='D')
    return pd.DataFrame({
        'open': [100.0] * n_bars,
        'high': [100.0] * n_bars,
        'low': [100.0] * n_bars,
        'close': [100.0] * n_bars,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


def gate_for(stage, flag):
    return next(gate for gate in stage.gates if gate.flag == flag)


def file_stage(symbol, round_trips, excess):
    """What run_holdout_stage hands back for one file, with chosen figures."""
    return StageResult(
        name=STAGE_HOLDOUT,
        detail=[f"round trips {round_trips}"],
        payload={'symbol': symbol, 'parameters': WINNER, 'round_trips': round_trips,
                 'excess_return_pct': excess})


class TestHoldoutsMustFollowAllResearch(unittest.TestCase):

    def test_a_holdout_before_a_later_compare_file_ends_is_refused(self):
        research = {
            'PRIMARY_research.csv': make_data('2020-01-01', 100),
            'OTHER_research.csv': make_data('2020-01-01', 200),
        }
        # After the primary ends (2020-04-09), before the other does (2020-07-18).
        holdouts = {'PRIMARY_later.csv': make_data('2020-05-01', 50)}

        with self.assertRaises(ValueError) as raised:
            check_holdout_follows_research(research, holdouts)

        message = str(raised.exception)
        self.assertIn('2020-05-01', message)
        self.assertIn('2020-07-18', message)
        self.assertIn('OTHER_research.csv', message)
        self.assertIn('PRIMARY_later.csv', message)

    def test_every_holdout_is_checked_not_only_the_first(self):
        research = {'r.csv': make_data('2020-01-01', 100)}
        holdouts = {
            'good.csv': make_data('2020-06-01', 50),
            'bad.csv': make_data('2020-04-09', 50),
        }

        with self.assertRaises(ValueError) as raised:
            check_holdout_follows_research(research, holdouts)

        self.assertIn('bad.csv', str(raised.exception))

    def test_holdouts_after_every_research_file_are_accepted(self):
        research = {
            'a.csv': make_data('2020-01-01', 100),
            'b.csv': make_data('2020-01-01', 200),
        }
        holdouts = {
            'x.csv': make_data('2020-07-19', 50),
            'y.csv': make_data('2021-01-01', 50),
        }

        check_holdout_follows_research(research, holdouts)


class TestPooledArithmetic(unittest.TestCase):
    """The pool, with the per-file figures chosen so the answer is known."""

    def pooled(self, files, min_excess=0.0, min_trades=1):
        holdouts = [(stage.payload['symbol'], None) for stage in files]
        with patch('scripts.screen.run_holdout_stage', side_effect=files):
            return run_pooled_holdout_stage(
                holdouts, 'simple_ma', WINNER, RunConfig(), min_excess, min_trades)

    def test_round_trips_are_summed_over_the_files(self):
        stage = self.pooled([file_stage('A', 3, 1.0), file_stage('B', 4, 2.0)])

        self.assertEqual(gate_for(stage, TRADES).value, 7.0)
        self.assertEqual(stage.payload['pooled_round_trips'], 7)

    def test_the_trade_gate_reads_the_sum_not_each_file(self):
        files = [file_stage('A', 3, 1.0), file_stage('B', 4, 2.0)]

        self.assertTrue(gate_for(self.pooled(list(files), min_trades=7), TRADES).passed)
        self.assertFalse(gate_for(self.pooled(list(files), min_trades=8), TRADES).passed)

    def test_excess_is_the_median_of_the_per_file_figures(self):
        stage = self.pooled([file_stage('A', 3, -4.0), file_stage('B', 3, 1.0),
                             file_stage('C', 3, 90.0)])

        # The mean would be 29.0: one exceptional file must not carry the verdict.
        self.assertEqual(gate_for(stage, EXCESS).value, 1.0)
        self.assertEqual(stage.payload['pooled_excess_pct'], 1.0)

    def test_the_excess_gate_reads_the_pooled_figure(self):
        files = [file_stage('A', 3, -4.0), file_stage('B', 3, 1.0),
                 file_stage('C', 3, 90.0)]

        self.assertTrue(gate_for(self.pooled(list(files), min_excess=1.0), EXCESS).passed)
        stopped = gate_for(self.pooled(list(files), min_excess=1.5), EXCESS)
        self.assertFalse(stopped.passed)
        self.assertIn('pooled excess over buy-and-hold (pp) 1.00 < 1.50',
                      stopped.describe())

    def test_it_counts_the_files_that_beat_buy_and_hold(self):
        stage = self.pooled([file_stage('A', 3, -4.0), file_stage('B', 3, 1.0),
                             file_stage('C', 3, 90.0)])

        self.assertEqual(stage.payload['files_beating_buy_and_hold'], 2)
        self.assertIn('2 of 3 file(s) that traded beat buy-and-hold',
                      '\n'.join(stage.detail))

    def test_a_file_never_traded_on_is_left_out_of_the_excess_pool(self):
        """Flat while the asset fell is positive excess for nothing."""
        stage = self.pooled([file_stage('A', 3, -4.0), file_stage('B', 3, -2.0),
                             file_stage('IDLE', 0, 50.0)])
        detail = '\n'.join(stage.detail)

        self.assertEqual(gate_for(stage, EXCESS).value, -3.0)
        self.assertEqual(stage.payload['files_traded'], 2)
        self.assertEqual(stage.payload['files_beating_buy_and_hold'], 0)
        self.assertIn('left out of the excess pool, no completed round trip: IDLE',
                      detail)

    def test_no_file_traded_means_no_excess_and_no_pass(self):
        stage = self.pooled([file_stage('A', 0, 5.0), file_stage('B', 0, 5.0)],
                            min_excess=-1e9, min_trades=0)
        gate = gate_for(stage, EXCESS)

        self.assertIsNone(gate.value)
        self.assertFalse(gate.passed)
        self.assertIn('no holdout file completed a round trip', gate.describe())
        self.assertIsNone(stage.payload['files_beating_buy_and_hold'])

    def test_a_traded_file_without_a_benchmark_leaves_the_pool_unmeasured(self):
        stage = self.pooled([file_stage('A', 3, 5.0), file_stage('NOBENCH', 3, None)],
                            min_excess=-1e9)
        gate = gate_for(stage, EXCESS)

        self.assertIsNone(gate.value)
        self.assertFalse(gate.passed)
        self.assertIn('NOBENCH', gate.describe())

    def test_the_output_says_what_was_pooled_and_how(self):
        stage = self.pooled([file_stage('A', 3, -4.0), file_stage('B', 4, 1.0)])
        detail = '\n'.join(stage.detail)

        self.assertIn('2 holdout file(s), one backtest each', detail)
        self.assertIn('nothing fitted on any of them', detail)
        self.assertIn('pooled round trips 7 = the sum over the 2 file(s)', detail)
        self.assertIn('pooled excess -1.50 pp = the median of the per-file excess',
                      detail)
        self.assertIn("compare.py's convention", detail)
        self.assertIn('[A] round trips 3', detail)
        self.assertIn('[B] round trips 4', detail)

    def test_each_file_keeps_its_own_record(self):
        stage = self.pooled([file_stage('A', 3, -4.0), file_stage('B', 4, 1.0)])

        self.assertEqual([entry['symbol'] for entry in stage.payload['files']],
                         ['A', 'B'])
        self.assertEqual(stage.payload['n_files'], 2)


class TestPooledStageRunsRealBacktests(unittest.TestCase):

    def setUp(self):
        self.holdouts = [
            ('ONE', make_data('2022-01-01', 400, period=7.0)),
            ('TWO', make_data('2022-01-01', 400, period=9.0)),
            ('FLAT', make_flat_data('2022-01-01', 200)),
        ]

    def stage(self):
        return run_pooled_holdout_stage(self.holdouts, 'simple_ma', WINNER,
                                        RunConfig(), 0.0, 1)

    def test_one_backtest_per_file_with_the_parameters_it_is_given(self):
        with patch('scripts.screen.create_strategy',
                   wraps=sys.modules['scripts.screen'].create_strategy) as create:
            self.stage()

        self.assertEqual(create.call_count, 3)
        for call in create.call_args_list:
            self.assertEqual(call.args, ('simple_ma', WINNER))

    def test_nothing_is_fitted_on_any_file(self):
        with patch('scripts.screen.create_optimizer') as create_optimizer, \
                patch('scripts.screen.evaluate') as evaluate:
            self.stage()

        create_optimizer.assert_not_called()
        evaluate.assert_not_called()

    def test_it_agrees_with_the_single_file_stage_run_on_each(self):
        singles = [run_holdout_stage(frame, symbol, 'simple_ma', WINNER, RunConfig(), 0.0)
                   for symbol, frame in self.holdouts]
        traded = [s.payload['excess_return_pct'] for s in singles
                  if s.payload['round_trips'] > 0]

        stage = self.stage()

        self.assertEqual(len(traded), 2)
        self.assertEqual(stage.payload['pooled_round_trips'],
                         sum(s.payload['round_trips'] for s in singles))
        self.assertAlmostEqual(stage.payload['pooled_excess_pct'],
                               statistics.median(traded))
        self.assertIn('FLAT', '\n'.join(stage.detail))


class TestHoldoutSummary(unittest.TestCase):

    def test_no_holdout_run_is_all_none(self):
        self.assertEqual(set(holdout_summary(None).values()), {None})

    def test_one_file_keeps_its_hash_and_its_own_excess(self):
        stage = StageResult(name=STAGE_HOLDOUT, payload={
            'symbol': 'A', 'round_trips': 4, 'excess_return_pct': 2.5,
            'data_sha256': 'abc'})

        self.assertEqual(holdout_summary(stage), {
            'holdout_files': 1, 'holdout_files_beating': 1,
            'holdout_round_trips': 4, 'holdout_excess_pct': 2.5,
            'holdout_data_sha256': 'abc'})

    def test_several_files_have_no_single_hash(self):
        stage = StageResult(name=STAGE_HOLDOUT, payload={
            'n_files': 2, 'files_beating_buy_and_hold': 1, 'pooled_round_trips': 9,
            'pooled_excess_pct': -0.5,
            'files': [{'data_sha256': 'abc'}, {'data_sha256': 'def'}]})

        self.assertEqual(holdout_summary(stage), {
            'holdout_files': 2, 'holdout_files_beating': 1,
            'holdout_round_trips': 9, 'holdout_excess_pct': -0.5,
            'holdout_data_sha256': None})

    def test_the_summary_fields_are_mapped(self):
        with open(project_root / 'config' / 'elasticsearch' / 'mappings' / 'runs.json') as handle:
            properties = json.load(handle)['mappings']['properties']

        for name, kind in (('holdout_files', 'integer'),
                           ('holdout_files_beating', 'integer'),
                           ('holdout_round_trips', 'integer'),
                           ('holdout_excess_pct', 'double'),
                           ('holdout_data_sha256', 'keyword')):
            self.assertEqual(properties[name]['type'], kind, name)


class TestSeveralHoldoutsInTheFunnel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.research_path = self._write('TEST_research.csv', make_data('2020-01-01', 300))
        self.other_research = self._write('OTHER_research.csv',
                                          make_data('2020-01-01', 320))
        self.first = self._write('TEST_later.csv', make_data('2021-01-01', 200))
        self.second = self._write('OTHER_later.csv',
                                  make_data('2021-01-01', 220, period=9.0))
        self.out = os.path.join(self.tmp, 'screen.json')

    def _write(self, name, frame):
        path = os.path.join(self.tmp, name)
        frame.to_csv(path, index_label='timestamp')
        return path

    def _run(self, extra_argv, compare_passes=True, stub_holdout=True):
        calls = []
        holdout_calls = []
        records = []

        def stub(name, passed, payload=None):
            def _stub(*args, **kwargs):
                calls.append(name)
                return StageResult(
                    name=name,
                    gates=[Gate(stage=name, quantity='q', value=1.0 if passed else 0.0,
                                threshold=0.5, flag='--flag')],
                    payload=dict(payload or {}))
            return _stub

        def holdout_stub(*args, **kwargs):
            calls.append(STAGE_HOLDOUT)
            holdout_calls.append(args)
            return file_stage(args[1], 3, 2.0 * len(holdout_calls))

        real_export = ExporterManager.export_run

        def capture(manager, record):
            records.append(record)
            return real_export(manager, record)

        argv = ['screen.py', '--data', self.research_path, '--strategy', 'simple_ma',
                '--output', self.out, *extra_argv]
        row = {
            'symbol': 'TEST', 'strategy': 'simple_ma', 'error': None,
            'folds': 8, 'compared_folds': 8, 'failed_folds': 0,
            'oos_sharpe': 0.9, 'median_efficiency': 0.5, 'positive_fold_pct': 60.0,
            'median_fold_pct': 3.0, 'median_bh_pct': 2.0, 'median_excess_pct': 1.0,
            'beat_bh_pct': 75.0,
        }
        patches = [
            patch.object(sys, 'argv', argv),
            patch('scripts.screen.run_backtest_stage', side_effect=stub('backtest', True)),
            patch('scripts.screen.run_optimize_stage',
                  side_effect=stub('optimize', True, {'winner_parameters': WINNER})),
            patch('scripts.screen.evaluate', return_value=row),
            patch('scripts.screen.run_walk_forward_stage',
                  side_effect=stub('walk-forward', True)),
            patch('scripts.screen.run_compare_stage',
                  side_effect=stub('compare', compare_passes)),
            patch('scripts.screen.render'),
            patch.object(ExporterManager, 'export_run', capture),
        ]
        if stub_holdout:
            patches.append(patch('scripts.screen.run_holdout_stage',
                                 side_effect=holdout_stub))

        out, err = io.StringIO(), io.StringIO()
        for active in patches:
            active.start()
            self.addCleanup(active.stop)
        with redirect_stdout(out), redirect_stderr(err):
            exit_code = main()

        return {'exit': exit_code, 'calls': calls, 'holdout_calls': holdout_calls,
                'stdout': out.getvalue(), 'stderr': err.getvalue(),
                'record': records[0] if records else None}

    def _document(self):
        with open(self.out) as handle:
            return json.load(handle)

    def _both(self, *extra):
        return ['--compare-data', self.other_research,
                '--holdout-data', self.first, self.second, *extra]

    def _sha(self, path):
        with open(path, 'rb') as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def test_each_file_is_backtested_once_with_the_winner(self):
        run = self._run(self._both())

        self.assertEqual(run['exit'], EXIT_OK)
        self.assertEqual(run['calls'][-2:], [STAGE_HOLDOUT, STAGE_HOLDOUT])
        self.assertEqual([call[1] for call in run['holdout_calls']], ['TEST', 'OTHER'])
        for call in run['holdout_calls']:
            self.assertEqual(call[3], WINNER)

    def test_each_file_is_exported_as_its_own_row_with_its_own_hash(self):
        run = self._run(self._both())

        rows = [row for row in run['record'].details[DETAIL_COMPARISON]
                if row.get('stage') == STAGE_HOLDOUT]

        self.assertEqual([row['data_path'] for row in rows], [self.first, self.second])
        self.assertEqual([row['data_sha256'] for row in rows],
                         [self._sha(self.first), self._sha(self.second)])
        self.assertEqual([row['symbol'] for row in rows], ['TEST', 'OTHER'])
        self.assertEqual([row['excess_return_pct'] for row in rows], [2.0, 4.0])

    def test_the_summary_describes_the_pool_and_claims_no_single_hash(self):
        run = self._run(self._both())
        summary = run['record'].summary

        self.assertEqual(summary['holdout_files'], 2)
        self.assertEqual(summary['holdout_files_beating'], 2)
        self.assertEqual(summary['holdout_round_trips'], 6)
        self.assertEqual(summary['holdout_excess_pct'], 3.0)
        self.assertIsNone(summary['holdout_data_sha256'])

    def test_the_json_record_names_every_holdout_file(self):
        self._run(self._both())
        document = self._document()

        self.assertEqual(document['holdout_data'], [self.first, self.second])
        self.assertIn(self.first, document['provenance'])
        self.assertIn(self.second, document['provenance'])
        self.assertEqual(len(document['stages'][-1]['files']), 2)

    def test_one_file_still_reports_as_one_file(self):
        run = self._run(['--holdout-data', self.first])
        summary = run['record'].summary

        self.assertEqual(self._document()['holdout_data'], self.first)
        self.assertEqual(summary['holdout_files'], 1)
        self.assertEqual(summary['holdout_data_sha256'], self._sha(self.first))
        self.assertEqual(summary['holdout_excess_pct'], 2.0)
        self.assertNotIn('pooled', run['stdout'])

    def test_no_holdout_leaves_the_summary_empty(self):
        run = self._run(['--compare-data', self.other_research])

        self.assertIsNone(run['record'].summary['holdout_files'])
        self.assertIsNone(run['record'].summary['holdout_round_trips'])

    def test_a_holdout_inside_a_compare_file_is_an_error_before_any_stage(self):
        late_research = self._write('LATE_research.csv', make_data('2020-01-01', 500))

        run = self._run(['--compare-data', late_research,
                         '--holdout-data', self.first, self.second])

        self.assertEqual(run['exit'], EXIT_ERROR)
        self.assertEqual(run['calls'], [])
        self.assertIn('LATE_research.csv', run['stderr'])
        self.assertIn('2021-05-14', run['stderr'])
        self.assertIn('2021-01-01', run['stderr'])

    def test_the_same_file_twice_is_an_error(self):
        run = self._run(['--holdout-data', self.first, self.first])

        self.assertEqual(run['exit'], EXIT_ERROR)
        self.assertEqual(run['calls'], [])
        self.assertIn('more than once', run['stderr'])

    def test_force_spends_none_of_them_after_a_failed_gate(self):
        run = self._run(self._both('--force'), compare_passes=False)

        self.assertNotIn(STAGE_HOLDOUT, run['calls'])
        self.assertEqual(run['exit'], EXIT_STOPPED)
        self.assertIn('SKIPPED: an earlier gate failed', run['stdout'])
        self.assertNotIn(self.first, self._document()['provenance'])
        self.assertNotIn(self.second, self._document()['provenance'])
        self.assertIsNone(run['record'].summary['holdout_files'])
        self.assertEqual(
            [row for row in run['record'].details[DETAIL_COMPARISON]
             if row.get('stage') == STAGE_HOLDOUT], [])

    def test_the_real_stage_runs_end_to_end_on_synthetic_bars(self):
        run = self._run(self._both(EXCESS, '-1000000'), stub_holdout=False)

        self.assertEqual(run['exit'], EXIT_OK)
        self.assertIn('passed holdout: pooled round trips', run['stdout'])
        self.assertIn('passed holdout: pooled excess over buy-and-hold (pp)',
                      run['stdout'])
        self.assertIn('the median of the per-file excess', run['stdout'])
        self.assertIn('[TEST]', run['stdout'])
        self.assertIn('[OTHER]', run['stdout'])
        rows = [row for row in run['record'].details[DETAIL_COMPARISON]
                if row.get('stage') == STAGE_HOLDOUT]
        self.assertEqual([row['bars'] for row in rows], [200, 220])
        self.assertEqual([row['parameters'] for row in rows], [WINNER, WINNER])


if __name__ == '__main__':
    unittest.main()
