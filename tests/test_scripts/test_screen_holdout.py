"""The holdout stage of screen.py.

Stages 1-4 can be rerun at will, and every rerun is a decision made with the
research data in view. The holdout is the one file no such decision has seen,
so what is pinned here is mostly what the stage must NOT do: fit anything, run
on bars that overlap the research data, run when an earlier gate stopped the
funnel, or be read from a configuration file where every run would spend it.

Every dataset here is synthetic. No test may read a real ``*_holdout.csv``.
"""

import hashlib
import io
import json
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
from niffler.exporters.run_record import DETAIL_COMPARISON
from scripts.screen import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_STOPPED,
    STAGE_HOLDOUT,
    Gate,
    StageResult,
    build_parser,
    check_holdout_follows_research,
    main,
    run_holdout_stage,
)

WINNER = {'short_window': 5, 'long_window': 20}


def make_data(start='2020-01-01', n_bars=600):
    """An oscillating OHLCV frame, so a moving-average cross actually fires."""
    index = pd.date_range(start, periods=n_bars, freq='D')
    closes = [100.0 + 20.0 * math.sin(i / 7.0) for i in range(n_bars)]
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


def fold_row(**overrides):
    row = {
        'symbol': 'TEST', 'strategy': 'simple_ma', 'error': None,
        'folds': 8, 'compared_folds': 8, 'failed_folds': 0,
        'oos_sharpe': 0.9, 'median_efficiency': 0.5, 'positive_fold_pct': 60.0,
        'median_fold_pct': 3.0, 'median_bh_pct': 2.0, 'median_excess_pct': 1.0,
        'beat_bh_pct': 75.0,
    }
    row.update(overrides)
    return row


class TestHoldoutMustFollowResearch(unittest.TestCase):

    def test_a_holdout_starting_after_the_research_data_is_accepted(self):
        research = make_data('2020-01-01', 100)
        holdout = make_data('2020-04-10', 50)

        check_holdout_follows_research(research, holdout)

    def test_an_overlapping_holdout_is_refused_naming_both_dates(self):
        research = make_data('2020-01-01', 100)
        holdout = make_data('2020-03-01', 50)

        with self.assertRaises(ValueError) as raised:
            check_holdout_follows_research(research, holdout)

        self.assertIn('2020-03-01', str(raised.exception))
        self.assertIn('2020-04-09', str(raised.exception))

    def test_sharing_the_last_research_bar_is_an_overlap(self):
        research = make_data('2020-01-01', 100)
        holdout = make_data('2020-04-09', 50)

        with self.assertRaises(ValueError):
            check_holdout_follows_research(research, holdout)

    def test_an_empty_frame_is_refused(self):
        research = make_data('2020-01-01', 100)

        with self.assertRaises(ValueError):
            check_holdout_follows_research(research, research.iloc[0:0])


class TestHoldoutStage(unittest.TestCase):
    """One backtest with the parameters it is handed, gated on trades and excess."""

    def setUp(self):
        self.holdout = make_data('2022-01-01', 400)
        self.config = RunConfig()

    def _stage(self, min_excess=0.0, config=None, min_trades=1, holdout=None):
        return run_holdout_stage(self.holdout if holdout is None else holdout,
                                 'TEST', 'simple_ma', WINNER,
                                 config or self.config, min_excess, min_trades)

    def test_it_backtests_the_parameters_it_is_given(self):
        with patch('scripts.screen.create_strategy',
                   wraps=sys.modules['scripts.screen'].create_strategy) as create:
            stage = self._stage()

        create.assert_called_once_with('simple_ma', WINNER)
        self.assertEqual(stage.payload['parameters'], WINNER)

    def test_nothing_is_optimised_on_the_holdout(self):
        with patch('scripts.screen.create_optimizer') as create_optimizer, \
                patch('scripts.screen.evaluate') as evaluate:
            self._stage()

        create_optimizer.assert_not_called()
        evaluate.assert_not_called()

    def test_the_gate_is_excess_over_buy_and_hold(self):
        stage = self._stage()
        gate = gate_for(stage, '--min-holdout-excess')

        self.assertEqual(stage.name, STAGE_HOLDOUT)
        self.assertIsNotNone(gate.value)
        self.assertAlmostEqual(
            gate.value,
            stage.payload['total_return_pct'] - stage.payload['benchmark_return_pct'])

    def test_it_fires_below_the_threshold_and_passes_at_it(self):
        flag = '--min-holdout-excess'
        excess = gate_for(self._stage(), flag).value

        self.assertTrue(gate_for(self._stage(min_excess=excess), flag).passed)
        failed = gate_for(self._stage(min_excess=excess + 1.0), flag)
        self.assertFalse(failed.passed)
        self.assertIn('STOPPED at holdout', failed.describe())
        self.assertIn('--min-holdout-excess', failed.describe())

    def test_a_refused_significance_verdict_is_printed_not_gated_on(self):
        """A short holdout cannot support a p-value; that is reported as itself."""
        demanding = RunConfig(min_trades_for_significance=10_000)

        stage = self._stage(min_excess=-1e9, config=demanding)

        self.assertFalse(stage.payload['is_sample_sufficient'])
        self.assertTrue(stage.payload['significance_verdict'])
        self.assertTrue(any(stage.payload['significance_verdict'] in line
                            for line in stage.detail))
        self.assertFalse(any('p-value' in line for line in stage.detail))
        self.assertEqual([gate.flag for gate in stage.gates],
                         ['--min-holdout-trades', '--min-holdout-excess'])
        self.assertEqual(stage.failed_gates, [])

    def test_a_holdout_the_strategy_never_traded_on_cannot_pass(self):
        """Flat while buy-and-hold pays commission is positive excess for nothing."""
        stage = self._stage(holdout=make_flat_data())
        trades = gate_for(stage, '--min-holdout-trades')

        self.assertEqual(stage.payload['round_trips'], 0)
        self.assertTrue(gate_for(stage, '--min-holdout-excess').passed)
        self.assertFalse(trades.passed)
        self.assertEqual(stage.failed_gates[0], trades)
        self.assertEqual(trades.describe(),
                         'STOPPED at holdout: round trips 0 < 1 (--min-holdout-trades)')
        self.assertTrue(any('did not trade enough' in line for line in stage.detail))

    def test_the_trade_threshold_is_printed_when_it_passes_too(self):
        stage = self._stage()
        trades = gate_for(stage, '--min-holdout-trades')

        self.assertTrue(trades.passed)
        self.assertIn('>= 1 (--min-holdout-trades)', trades.describe())
        self.assertFalse(any('did not trade enough' in line for line in stage.detail))

    def test_the_trade_threshold_is_the_one_it_is_given(self):
        traded = self._stage().payload['round_trips']

        self.assertFalse(
            gate_for(self._stage(min_trades=traded + 1), '--min-holdout-trades').passed)
        self.assertTrue(gate_for(self._stage(holdout=make_flat_data(), min_trades=0),
                                 '--min-holdout-trades').passed)

    def test_the_bars_it_ran_on_are_recorded(self):
        stage = self._stage()

        self.assertEqual(stage.payload['bars'], 400)
        self.assertIn('2022-01-01', stage.payload['first_bar'])


class TestHoldoutInTheFunnel(unittest.TestCase):
    """Where the stage sits, when it is skipped, and what the run records."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.research_path = self._write('TEST_research.csv', make_data('2020-01-01', 300))
        self.holdout_path = self._write('TEST_later.csv', make_data('2021-01-01', 200))
        self.out = os.path.join(self.tmp, 'screen.json')

    def _write(self, name, frame):
        path = os.path.join(self.tmp, name)
        frame.to_csv(path, index_label='timestamp')
        return path

    def _run(self, extra_argv, compare_passes=True, holdout_passes=True,
             stub_holdout=True, config_text=None):
        """Run main() with every stage stubbed; return what it did."""
        calls = []
        holdout_arguments = {}
        records = []

        def stub(name, passed, payload=None):
            def _stub(*args, **kwargs):
                calls.append(name)
                if name == STAGE_HOLDOUT:
                    holdout_arguments['args'] = args
                return StageResult(
                    name=name,
                    gates=[Gate(stage=name, quantity='q', value=1.0 if passed else 0.0,
                                threshold=0.5, flag='--flag')],
                    payload=dict(payload or {}),
                )
            return _stub

        def capture(manager, record):
            records.append(record)
            return real_export(manager, record)

        real_export = ExporterManager.export_run

        argv = ['screen.py', '--data', self.research_path, '--strategy', 'simple_ma',
                '--output', self.out]
        if config_text is not None:
            config_path = os.path.join(self.tmp, 'niffler.toml')
            with open(config_path, 'w') as handle:
                handle.write(config_text)
            argv.extend(['--config', config_path])
        argv.extend(extra_argv)

        patches = [
            patch.object(sys, 'argv', argv),
            patch('scripts.screen.run_backtest_stage', side_effect=stub('backtest', True)),
            patch('scripts.screen.run_optimize_stage',
                  side_effect=stub('optimize', True, {'winner_parameters': WINNER})),
            patch('scripts.screen.evaluate', return_value=fold_row()),
            patch('scripts.screen.run_walk_forward_stage',
                  side_effect=stub('walk-forward', True)),
            patch('scripts.screen.run_compare_stage',
                  side_effect=stub('compare', compare_passes)),
            patch('scripts.screen.render'),
            patch.object(ExporterManager, 'export_run', capture),
        ]
        if stub_holdout:
            patches.append(patch(
                'scripts.screen.run_holdout_stage',
                side_effect=stub(STAGE_HOLDOUT, holdout_passes,
                                 {'excess_return_pct': 4.2})))

        out, err = io.StringIO(), io.StringIO()
        for active in patches:
            active.start()
            self.addCleanup(active.stop)
        with redirect_stdout(out), redirect_stderr(err):
            exit_code = main()

        return {
            'exit': exit_code, 'calls': calls, 'stdout': out.getvalue(),
            'stderr': err.getvalue(), 'holdout_arguments': holdout_arguments,
            'record': records[0] if records else None,
        }

    def _document(self):
        with open(self.out) as handle:
            return json.load(handle)

    def _with_holdout(self, *extra):
        return ['--compare-data', self.research_path,
                '--holdout-data', self.holdout_path, *extra]

    def test_it_runs_last_after_every_other_gate_passed(self):
        run = self._run(self._with_holdout())

        self.assertEqual(run['calls'],
                         ['backtest', 'optimize', 'walk-forward', 'compare', 'holdout'])
        self.assertEqual(run['exit'], EXIT_OK)

    def test_it_is_handed_the_optimize_stage_winner(self):
        run = self._run(self._with_holdout())

        self.assertEqual(run['holdout_arguments']['args'][3], WINNER)

    def test_it_is_handed_the_typed_threshold(self):
        run = self._run(self._with_holdout('--min-holdout-excess', '2.5'))

        self.assertEqual(run['holdout_arguments']['args'][5], 2.5)

    def test_it_is_handed_the_typed_trade_threshold(self):
        run = self._run(self._with_holdout('--min-holdout-trades', '7'))

        self.assertEqual(run['holdout_arguments']['args'][6], 7)
        self.assertEqual(self._document()['thresholds']['min_holdout_trades'], 7)

    def test_the_trade_threshold_defaults_to_one_round_trip(self):
        run = self._run(self._with_holdout())

        self.assertEqual(run['holdout_arguments']['args'][6], 1)
        self.assertEqual(self._document()['thresholds']['min_holdout_trades'], 1)

    def test_the_trade_threshold_may_come_from_the_config_file(self):
        run = self._run(self._with_holdout(),
                        config_text='[screen]\nmin_holdout_trades = 4\n')

        self.assertEqual(run['holdout_arguments']['args'][6], 4)

    def test_a_holdout_without_a_trade_stops_the_run(self):
        flat = self._write('TEST_flat.csv', make_flat_data('2021-01-01', 200))

        run = self._run(['--holdout-data', flat], stub_holdout=False)

        self.assertEqual(run['exit'], EXIT_STOPPED)
        self.assertIn('STOPPED at holdout: round trips 0 < 1 (--min-holdout-trades)',
                      run['stdout'])
        self.assertIn('did not trade enough', run['stdout'])

    def test_a_failed_earlier_gate_leaves_the_holdout_unread(self):
        run = self._run(self._with_holdout(), compare_passes=False)

        self.assertNotIn('holdout', run['calls'])
        self.assertEqual(run['exit'], EXIT_STOPPED)
        self.assertNotIn(self.holdout_path, self._document()['provenance'])
        self.assertIsNone(run['record'].summary['holdout_data_sha256'])
        self.assertEqual(len(run['record'].details[DETAIL_COMPARISON]), 2)

    def test_force_does_not_spend_the_holdout_after_a_failed_gate(self):
        """A strategy that already failed has its verdict; a look would be lost."""
        run = self._run(self._with_holdout('--force'), compare_passes=False)

        self.assertNotIn('holdout', run['calls'])
        self.assertEqual(run['exit'], EXIT_STOPPED)
        self.assertIn('--- holdout ---', run['stdout'])
        self.assertIn('SKIPPED: an earlier gate failed', run['stdout'])
        self.assertIn('not spent', run['stdout'])
        last = self._document()['stages'][-1]
        self.assertEqual(last['name'], STAGE_HOLDOUT)
        self.assertIn('earlier gate failed', last['skipped_reason'])
        self.assertNotIn(self.holdout_path, self._document()['provenance'])
        self.assertIsNone(run['record'].summary['holdout_data_sha256'])

    def test_force_does_not_block_the_holdout_when_every_gate_passed(self):
        run = self._run(self._with_holdout('--force'))

        self.assertEqual(run['calls'][-1], 'holdout')
        self.assertEqual(run['exit'], EXIT_OK)

    def test_a_failed_holdout_gate_stops_the_run(self):
        run = self._run(self._with_holdout(), holdout_passes=False)

        self.assertEqual(run['exit'], EXIT_STOPPED)
        self.assertIn('STOPPED at holdout', run['stdout'])

    def test_it_runs_when_the_cross_asset_stage_was_skipped(self):
        run = self._run(['--holdout-data', self.holdout_path])

        self.assertEqual(run['calls'], ['backtest', 'optimize', 'walk-forward', 'holdout'])
        self.assertEqual(run['exit'], EXIT_OK)

    def test_passing_it_says_the_holdout_is_now_spent(self):
        run = self._run(self._with_holdout())

        self.assertIn('now spent', run['stdout'])

    def test_without_a_holdout_the_run_says_none_was_run(self):
        run = self._run(['--compare-data', self.research_path])

        self.assertEqual(run['exit'], EXIT_OK)
        self.assertIn('--- holdout ---', run['stdout'])
        self.assertIn('SKIPPED: no --holdout-data given', run['stdout'])
        self.assertNotIn('now spent', run['stdout'])
        last = self._document()['stages'][-1]
        self.assertEqual(last['name'], STAGE_HOLDOUT)
        self.assertTrue(last['skipped_reason'])
        self.assertIsNone(self._document()['holdout_data'])
        self.assertIsNone(run['record'].summary['holdout_data_sha256'])

    def test_the_exported_row_carries_the_stage_and_the_file_hash(self):
        run = self._run(self._with_holdout())
        with open(self.holdout_path, 'rb') as handle:
            expected = hashlib.sha256(handle.read()).hexdigest()

        rows = run['record'].details[DETAIL_COMPARISON]
        holdout_rows = [row for row in rows if row.get('stage') == STAGE_HOLDOUT]

        self.assertEqual(len(holdout_rows), 1)
        self.assertEqual(holdout_rows[0]['data_sha256'], expected)
        self.assertEqual(holdout_rows[0]['data_path'], self.holdout_path)
        self.assertEqual(holdout_rows[0]['strategy_key'], 'simple_ma')
        self.assertEqual(run['record'].summary['holdout_data_sha256'], expected)
        self.assertEqual(run['record'].summary['holdout_excess_pct'], 4.2)

    def test_the_json_record_names_the_holdout_and_its_hash(self):
        self._run(self._with_holdout())
        document = self._document()
        with open(self.holdout_path, 'rb') as handle:
            expected = hashlib.sha256(handle.read()).hexdigest()

        self.assertEqual(document['holdout_data'], self.holdout_path)
        self.assertIn(self.holdout_path, document['provenance'])
        self.assertEqual(document['stages'][-1]['name'], STAGE_HOLDOUT)
        self.assertEqual(document['stages'][-1]['data_sha256'], expected)
        self.assertEqual(document['thresholds']['min_holdout_excess'], 0.0)

    def test_an_overlapping_holdout_is_an_error_before_any_stage_runs(self):
        overlapping = self._write('TEST_overlap.csv', make_data('2020-06-01', 200))

        run = self._run(['--holdout-data', overlapping])

        self.assertEqual(run['exit'], EXIT_ERROR)
        self.assertEqual(run['calls'], [])
        self.assertIn('2020-06-01', run['stderr'])
        self.assertIn('2020-10-26', run['stderr'])

    def test_a_missing_holdout_file_is_an_error(self):
        run = self._run(['--holdout-data', os.path.join(self.tmp, 'absent.csv')])

        self.assertEqual(run['exit'], EXIT_ERROR)
        self.assertEqual(run['calls'], [])

    def test_a_holdout_path_in_the_config_file_is_refused(self):
        """Read from a file, every run of the funnel would spend it."""
        path = self.holdout_path.replace('\\', '/')

        run = self._run([], config_text=f'[screen]\nholdout_data = "{path}"\n')

        self.assertEqual(run['exit'], EXIT_ERROR)
        self.assertEqual(run['calls'], [])
        self.assertIn('--holdout-data', run['stderr'])

    def test_the_threshold_may_come_from_the_config_file(self):
        self._run(self._with_holdout(),
                  config_text='[screen]\nmin_holdout_excess = 2.5\n')

        self.assertEqual(self._document()['thresholds']['min_holdout_excess'], 2.5)

    def test_the_real_stage_runs_end_to_end_on_synthetic_bars(self):
        run = self._run(self._with_holdout('--min-holdout-excess', '-1000000'),
                        stub_holdout=False)

        self.assertEqual(run['exit'], EXIT_OK)
        self.assertIn('passed holdout: excess over buy-and-hold (pp)', run['stdout'])
        self.assertIn('passed holdout: round trips', run['stdout'])
        self.assertIn('(--min-holdout-trades)', run['stdout'])
        self.assertIn('(--min-holdout-excess)', run['stdout'])
        row = run['record'].details[DETAIL_COMPARISON][-1]
        self.assertEqual(row['parameters'], WINNER)
        self.assertEqual(row['bars'], 200)


class TestHoldoutFlagsAreDocumented(unittest.TestCase):

    def test_help_names_both_flags_and_the_cost_of_looking(self):
        buf = io.StringIO()
        with patch.object(sys, 'argv', ['screen.py', '--help']), redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                build_parser().parse_args()
        help_text = ' '.join(buf.getvalue().split())

        self.assertIn('--holdout-data', help_text)
        self.assertIn('--min-holdout-excess', help_text)
        self.assertIn('--min-holdout-trades', help_text)
        self.assertIn('spends it', help_text)
        self.assertIn('--force does not spend it', help_text)


if __name__ == '__main__':
    unittest.main()
