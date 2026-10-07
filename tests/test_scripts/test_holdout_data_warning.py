"""The warning printed when a holdout file is used as ordinary data.

A holdout is spent by looking at it, and nothing in a CSV says it is one: the
only signal is its name. So every script that reads research data warns - it
does not refuse - when a path is named like a holdout, and the one place such a
file belongs, ``screen.py --holdout-data``, stays quiet.

Every path here is synthetic or does not exist. No test may read a real
``*_holdout.csv``.
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

from scripts import analyze, backtest, compare, optimize, screen
from scripts.common import warn_if_holdout_data
from scripts.screen import Gate, StageResult

HEADLINE = 'WARNING: this looks like a HOLDOUT file used as ordinary data.'


def make_data(start='2020-01-01', n_bars=300):
    index = pd.date_range(start, periods=n_bars, freq='D')
    closes = [100.0 + 20.0 * math.sin(i / 7.0) for i in range(n_bars)]
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


class TestWarnIfHoldoutData(unittest.TestCase):

    def warn(self, paths):
        stream = io.StringIO()
        return warn_if_holdout_data(paths, stream=stream), stream.getvalue()

    def test_a_file_named_like_a_holdout_is_warned_about(self):
        suspect, text = self.warn('data/SPY_holdout.csv')

        self.assertEqual(suspect, ['data/SPY_holdout.csv'])
        self.assertIn(HEADLINE, text)
        self.assertIn('data/SPY_holdout.csv', text)
        self.assertIn('spent by looking at it', text)
        self.assertIn('screen.py --holdout-data', text)

    def test_it_is_fenced_so_it_cannot_be_skimmed_past(self):
        _, text = self.warn('data/SPY_holdout.csv')
        lines = text.strip().splitlines()

        self.assertEqual(lines[0], '=' * 72)
        self.assertEqual(lines[-1], '=' * 72)

    def test_it_says_it_is_a_warning_and_not_a_refusal(self):
        _, text = self.warn('data/SPY_holdout.csv')

        self.assertIn('a warning and not a refusal', text)

    def test_a_research_file_prints_nothing(self):
        suspect, text = self.warn('data/SPY_research.csv')

        self.assertEqual(suspect, [])
        self.assertEqual(text, '')

    def test_the_name_is_matched_without_regard_to_case(self):
        suspect, _ = self.warn('data/SPY_HoldOut_2024.csv')

        self.assertEqual(len(suspect), 1)

    def test_only_the_file_name_counts_not_the_directory(self):
        suspect, text = self.warn(os.path.join('holdout_runs', 'SPY_research.csv'))

        self.assertEqual(suspect, [])
        self.assertEqual(text, '')

    def test_only_the_offending_paths_of_a_list_are_named(self):
        suspect, text = self.warn(['data/SPY_research.csv', 'data/QQQ_holdout.csv',
                                   'data/GLD_holdout.csv'])

        self.assertEqual(suspect, ['data/QQQ_holdout.csv', 'data/GLD_holdout.csv'])
        self.assertNotIn('SPY_research.csv', text)
        self.assertEqual(text.count(HEADLINE), 1)

    def test_it_goes_to_stderr_by_default(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            warn_if_holdout_data('data/SPY_holdout.csv')

        self.assertEqual(out.getvalue(), '')
        self.assertIn(HEADLINE, err.getvalue())


class TestEveryResearchScriptWarns(unittest.TestCase):
    """Each script warns before it reads; none of them refuses."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # Named like a holdout and deliberately absent: the warning must come
        # before the read, so the load failure that follows proves the order.
        self.absent = os.path.join(self.tmp, 'TEST_holdout.csv')

    def run_main(self, module, argv):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, 'argv', argv), redirect_stdout(out), redirect_stderr(err):
            exit_code = module.main()
        return exit_code, out.getvalue(), err.getvalue()

    def test_backtest_warns(self):
        _, _, err = self.run_main(
            backtest, ['backtest.py', '--data', self.absent, '--strategy', 'simple_ma'])

        self.assertIn(HEADLINE, err)
        self.assertIn(self.absent, err)

    def test_optimize_warns(self):
        _, _, err = self.run_main(
            optimize, ['optimize.py', '--data', self.absent, '--strategy', 'simple_ma'])

        self.assertIn(HEADLINE, err)

    def test_analyze_warns(self):
        _, _, err = self.run_main(
            analyze, ['analyze.py', '--data', self.absent, '--strategy', 'simple_ma',
                      '--analysis', 'walk_forward'])

        self.assertIn(HEADLINE, err)

    def test_compare_warns_about_the_holdout_among_its_datasets(self):
        research = os.path.join(self.tmp, 'TEST_research.csv')
        holdout = os.path.join(self.tmp, 'OTHER_holdout.csv')
        make_data().to_csv(research, index_label='timestamp')
        make_data().to_csv(holdout, index_label='timestamp')

        # Stop at the first evaluation: the warning has to be out before it.
        err = io.StringIO()
        argv = ['compare.py', '--data', research, holdout, '--strategy', 'simple_ma']
        with patch('scripts.compare.evaluate', side_effect=RuntimeError('stop here')), \
                patch.object(sys, 'argv', argv), \
                redirect_stdout(io.StringIO()), redirect_stderr(err):
            with self.assertRaises(RuntimeError):
                compare.main()

        self.assertIn(HEADLINE, err.getvalue())
        self.assertIn(holdout, err.getvalue())
        self.assertNotIn(f"    {research}", err.getvalue())

    def test_a_backtest_on_a_holdout_file_still_runs(self):
        """A warning, not a refusal: the name is only a convention."""
        path = os.path.join(self.tmp, 'REAL_holdout.csv')
        make_data().to_csv(path, index_label='timestamp')

        exit_code, out, err = self.run_main(
            backtest, ['backtest.py', '--data', path, '--strategy', 'simple_ma'])

        self.assertEqual(exit_code, 0)
        self.assertIn(HEADLINE, err)
        self.assertIn('BACKTEST RESULTS', out)

    def test_a_research_file_gets_no_warning(self):
        path = os.path.join(self.tmp, 'TEST_research.csv')
        make_data().to_csv(path, index_label='timestamp')

        exit_code, _, err = self.run_main(
            backtest, ['backtest.py', '--data', path, '--strategy', 'simple_ma'])

        self.assertEqual(exit_code, 0)
        self.assertNotIn(HEADLINE, err)


class TestScreenWarnsAboutResearchRolesOnly(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.research = self._write('TEST_research.csv', make_data('2020-01-01', 300))
        self.holdout = self._write('TEST_holdout.csv', make_data('2021-06-01', 200))
        self.misused = self._write('OTHER_holdout.csv', make_data('2020-01-01', 300))

    def _write(self, name, frame):
        path = os.path.join(self.tmp, name)
        frame.to_csv(path, index_label='timestamp')
        return path

    def run_screen(self, extra_argv, data=None):
        def stage(name, payload=None):
            def _stub(*args, **kwargs):
                return StageResult(
                    name=name,
                    gates=[Gate(stage=name, quantity='q', value=1.0, threshold=0.5,
                                flag='--flag')],
                    payload=dict(payload or {}))
            return _stub

        row = {
            'symbol': 'TEST', 'strategy': 'simple_ma', 'error': None,
            'folds': 8, 'compared_folds': 8, 'failed_folds': 0,
            'oos_sharpe': 0.9, 'median_efficiency': 0.5, 'positive_fold_pct': 60.0,
            'median_fold_pct': 3.0, 'median_bh_pct': 2.0, 'median_excess_pct': 1.0,
            'beat_bh_pct': 75.0,
        }
        argv = ['screen.py', '--data', data or self.research, '--strategy', 'simple_ma',
                *extra_argv]
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, 'argv', argv), \
                patch('scripts.screen.run_backtest_stage', side_effect=stage('backtest')), \
                patch('scripts.screen.run_optimize_stage', side_effect=stage(
                    'optimize', {'winner_parameters': {'short_window': 5,
                                                       'long_window': 20}})), \
                patch('scripts.screen.evaluate', return_value=row), \
                patch('scripts.screen.run_walk_forward_stage',
                      side_effect=stage('walk-forward')), \
                patch('scripts.screen.run_compare_stage', side_effect=stage('compare')), \
                patch('scripts.screen.render'), \
                redirect_stdout(out), redirect_stderr(err):
            exit_code = screen.main()
        return exit_code, err.getvalue()

    def test_the_holdout_stage_itself_does_not_warn(self):
        exit_code, err = self.run_screen(['--holdout-data', self.holdout])

        self.assertEqual(exit_code, screen.EXIT_OK)
        self.assertNotIn(HEADLINE, err)

    def test_a_holdout_file_as_the_primary_dataset_warns(self):
        _, err = self.run_screen([], data=self.misused)

        self.assertIn(HEADLINE, err)
        self.assertIn(self.misused, err)

    def test_a_holdout_file_among_the_compare_datasets_warns(self):
        _, err = self.run_screen(['--compare-data', self.misused])

        self.assertIn(HEADLINE, err)
        self.assertIn(self.misused, err)
        self.assertNotIn(f"    {self.research}", err)

    def test_only_the_misused_file_is_named_when_a_real_holdout_is_also_given(self):
        _, err = self.run_screen(['--compare-data', self.misused,
                                  '--holdout-data', self.holdout])

        self.assertIn(self.misused, err)
        self.assertNotIn(f"    {self.holdout}", err)


if __name__ == '__main__':
    unittest.main()
