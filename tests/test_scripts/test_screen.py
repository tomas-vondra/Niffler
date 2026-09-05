"""The screening funnel in screen.py.

Two things are pinned here. First the gates: each one must fire when the number
falls short, pass when it does not, and refuse to render a verdict when the
number could not be computed at all - ``None`` is not a pass, and it is not a
zero either. Second the plumbing: the stages must run in order, stop at the
first failure, and honour ``--force`` without pretending the failure did not
happen.

The gate arithmetic is tested against a plain ``Gate``, with no market data
anywhere, which is the whole reason the judgement is separated from the
measurement.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.run_config import RunConfig
from scripts.screen import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_STOPPED,
    Gate,
    StageResult,
    build_parser,
    main,
    pooled_beat_pct,
    run_backtest_stage,
    run_compare_stage,
    run_walk_forward_stage,
)


def make_data(n_bars=600):
    """An oscillating OHLCV frame, so a moving-average cross actually fires."""
    import math

    index = pd.date_range('2020-01-01', periods=n_bars, freq='D')
    closes = [100.0 + 20.0 * math.sin(i / 7.0) for i in range(n_bars)]
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


def fold_row(**overrides):
    """A ``compare.evaluate`` row with every field the stages read."""
    row = {
        'symbol': 'TEST', 'strategy': 'breakout', 'error': None,
        'folds': 8, 'compared_folds': 8, 'failed_folds': 0,
        'oos_sharpe': 0.9, 'median_efficiency': 0.5, 'positive_fold_pct': 60.0,
        'median_fold_pct': 3.0, 'median_bh_pct': 2.0, 'median_excess_pct': 1.0,
        'beat_bh_pct': 75.0,
    }
    row.update(overrides)
    return row


class TestGateVerdicts(unittest.TestCase):
    """A gate is arithmetic plus a sentence, and both have to be right."""

    def _gate(self, value, threshold=0.30):
        return Gate(stage='walk-forward', quantity='median efficiency',
                    value=value, threshold=threshold, flag='--min-efficiency')

    def test_a_value_below_the_threshold_stops_the_run(self):
        gate = self._gate(0.19)

        self.assertFalse(gate.passed)
        self.assertEqual(
            gate.describe(),
            "STOPPED at walk-forward: median efficiency 0.19 < 0.30 "
            "(--min-efficiency)")

    def test_a_value_above_the_threshold_passes(self):
        gate = self._gate(0.45)

        self.assertTrue(gate.passed)
        self.assertIn('passed walk-forward', gate.describe())

    def test_a_value_exactly_at_the_threshold_passes(self):
        """The threshold is a minimum, so meeting it is clearing it."""
        self.assertTrue(self._gate(0.30).passed)

    def test_the_threshold_and_its_flag_are_printed_when_the_gate_passes_too(self):
        """A gate nobody can see is not a gate."""
        text = self._gate(0.45).describe()

        self.assertIn('0.30', text)
        self.assertIn('--min-efficiency', text)

    def test_an_uncomputable_value_is_a_stop_not_a_pass(self):
        """None means "no evidence", which must never wave a strategy through."""
        gate = Gate(stage='optimize', quantity='plateau retention', value=None,
                    threshold=0.4, flag='--min-retention',
                    unknown_reason='no scored neighbour')

        self.assertFalse(gate.passed)
        text = gate.describe()
        self.assertIn('STOPPED at optimize', text)
        self.assertIn('is None', text)
        self.assertIn('no scored neighbour', text)
        # And it must not have been silently turned into a number.
        self.assertNotIn('0.00 <', text)

    def test_counts_render_as_integers(self):
        gate = Gate(stage='backtest', quantity='round trips', value=8.0,
                    threshold=30.0, flag='--min-trades-for-significance',
                    precision=0)

        self.assertEqual(
            gate.describe(),
            "STOPPED at backtest: round trips 8 < 30 "
            "(--min-trades-for-significance)")


class TestBacktestGate(unittest.TestCase):
    """Stage 1 gates on round trips, using the framework's own constant."""

    def test_a_thin_strategy_stops_at_the_first_stage(self):
        stage = run_backtest_stage(make_data(), 'TEST', 'simple_ma',
                                   RunConfig(min_trades_for_significance=10_000))

        self.assertEqual([g.passed for g in stage.gates], [False])
        self.assertIn('STOPPED at backtest', stage.gates[0].describe())

    def test_an_active_strategy_clears_it(self):
        stage = run_backtest_stage(make_data(), 'TEST', 'simple_ma',
                                   RunConfig(min_trades_for_significance=1))

        self.assertEqual([g.passed for g in stage.gates], [True])

    def test_the_threshold_is_the_engines_significance_gate(self):
        """One number, not a second one that can drift away from it."""
        stage = run_backtest_stage(make_data(), 'TEST', 'simple_ma',
                                   RunConfig(min_trades_for_significance=42))

        self.assertEqual(stage.gates[0].threshold, 42.0)
        self.assertEqual(stage.gates[0].flag, '--min-trades-for-significance')

    def test_the_gate_counts_round_trips_not_fills(self):
        """A round trip is the unit the significance test's sample size uses."""
        stage = run_backtest_stage(make_data(), 'TEST', 'simple_ma',
                                   RunConfig(min_trades_for_significance=1))

        self.assertEqual(stage.gates[0].value, float(stage.payload['round_trips']))
        self.assertLess(stage.payload['round_trips'], stage.payload['total_trades'])


class TestWalkForwardGate(unittest.TestCase):

    def test_it_fires_below_the_threshold(self):
        stage = run_walk_forward_stage(fold_row(median_efficiency=0.19), 0.30)

        self.assertEqual(
            stage.gates[0].describe(),
            "STOPPED at walk-forward: median efficiency 0.19 < 0.30 "
            "(--min-efficiency)")

    def test_it_passes_above_the_threshold(self):
        stage = run_walk_forward_stage(fold_row(median_efficiency=0.55), 0.30)

        self.assertTrue(stage.gates[0].passed)

    def test_no_defined_ratio_is_a_stop(self):
        stage = run_walk_forward_stage(fold_row(median_efficiency=None), 0.30)

        self.assertFalse(stage.gates[0].passed)
        self.assertIn('is None', stage.gates[0].describe())

    def test_a_failed_walk_forward_raises_rather_than_scoring_zero(self):
        with self.assertRaises(ValueError):
            run_walk_forward_stage(fold_row(error='boom'), 0.30)


class TestCompareGate(unittest.TestCase):

    def test_beat_pct_is_pooled_over_folds_not_averaged_over_assets(self):
        """An asset with two comparable folds is not evidence worth twelve."""
        rows = [
            fold_row(symbol='A', compared_folds=2, beat_bh_pct=100.0),
            fold_row(symbol='B', compared_folds=8, beat_bh_pct=0.0),
        ]

        # A per-asset mean would say 50%; the pooled figure is 2 of 10.
        self.assertAlmostEqual(pooled_beat_pct(rows), 20.0)

    def test_it_fires_below_the_threshold(self):
        rows = [fold_row(compared_folds=8, beat_bh_pct=25.0)]
        stage = run_compare_stage(rows, 50.0)

        self.assertIn('STOPPED at compare', stage.gates[0].describe())
        self.assertIn('BEAT% 25.0 < 50.0', stage.gates[0].describe())

    def test_it_passes_above_the_threshold(self):
        rows = [fold_row(compared_folds=8, beat_bh_pct=75.0)]

        self.assertTrue(run_compare_stage(rows, 50.0).gates[0].passed)

    def test_no_comparable_fold_anywhere_is_a_stop_not_a_zero(self):
        rows = [fold_row(compared_folds=0, beat_bh_pct=None)]

        self.assertIsNone(pooled_beat_pct(rows))
        self.assertFalse(run_compare_stage(rows, 50.0).gates[0].passed)

    def test_a_failed_dataset_is_excluded_rather_than_counted_as_a_loss(self):
        rows = [fold_row(compared_folds=8, beat_bh_pct=75.0),
                fold_row(symbol='BROKEN', error='bad file', compared_folds=None,
                         beat_bh_pct=None)]

        self.assertAlmostEqual(pooled_beat_pct(rows), 75.0)


class TestFunnelOrderAndForce(unittest.TestCase):
    """The funnel stops at the first failure, and --force does not hide it."""

    def setUp(self):
        self.data = make_data()
        self.tmp = tempfile.mkdtemp()
        self.data_path = os.path.join(self.tmp, 'TEST_research.csv')
        self.data.to_csv(self.data_path, index_label='timestamp')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, extra_argv, backtest_gate_passes):
        """Run main() with every stage stubbed except the gate arithmetic."""
        calls = []

        def stub(name, passed):
            def _stub(*args, **kwargs):
                calls.append(name)
                return StageResult(
                    name=name,
                    gates=[Gate(stage=name, quantity='q', value=1.0 if passed else 0.0,
                                threshold=0.5, flag='--flag')],
                )
            return _stub

        argv = ['screen.py', '--data', self.data_path, '--strategy', 'simple_ma']
        argv.extend(extra_argv)

        with patch.object(sys, 'argv', argv), \
                patch('scripts.screen.run_backtest_stage',
                      side_effect=stub('backtest', backtest_gate_passes)), \
                patch('scripts.screen.run_optimize_stage',
                      side_effect=stub('optimize', True)), \
                patch('scripts.screen.evaluate', return_value=fold_row()), \
                patch('scripts.screen.run_walk_forward_stage',
                      side_effect=stub('walk-forward', True)), \
                patch('scripts.screen.run_compare_stage',
                      side_effect=stub('compare', True)), \
                patch('scripts.screen.render'):
            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = main()

        return exit_code, calls, buf.getvalue()

    def test_a_failed_gate_stops_before_the_next_stage_runs(self):
        exit_code, calls, output = self._run([], backtest_gate_passes=False)

        self.assertEqual(calls, ['backtest'])
        self.assertEqual(exit_code, EXIT_STOPPED)
        self.assertIn('STOPPED at backtest', output)

    def test_force_runs_every_stage_but_still_exits_stopped(self):
        exit_code, calls, output = self._run(
            ['--force', '--compare-data', self.data_path],
            backtest_gate_passes=False)

        self.assertEqual(calls, ['backtest', 'optimize', 'walk-forward', 'compare'])
        self.assertEqual(exit_code, EXIT_STOPPED)
        self.assertIn('--force', output)

    def test_every_gate_passing_exits_zero(self):
        exit_code, calls, _ = self._run(
            ['--compare-data', self.data_path], backtest_gate_passes=True)

        self.assertEqual(calls, ['backtest', 'optimize', 'walk-forward', 'compare'])
        self.assertEqual(exit_code, EXIT_OK)

    def test_a_single_asset_skips_the_cross_asset_stage_and_says_so(self):
        """One asset is one observation; silently gating on it would be worse."""
        exit_code, calls, output = self._run([], backtest_gate_passes=True)

        self.assertEqual(calls, ['backtest', 'optimize', 'walk-forward'])
        self.assertEqual(exit_code, EXIT_OK)
        self.assertIn('SKIPPED', output)

    def test_a_missing_data_file_is_an_error_not_a_stop(self):
        argv = ['screen.py', '--data', 'no_such_file_12345.csv',
                '--strategy', 'simple_ma']
        with patch.object(sys, 'argv', argv):
            self.assertEqual(main(), EXIT_ERROR)


class TestThresholdsAreReported(unittest.TestCase):

    def test_every_gate_flag_is_documented_as_a_judgment_call(self):
        buf = io.StringIO()
        with patch.object(sys, 'argv', ['screen.py', '--help']), redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                build_parser().parse_args()
        help_text = buf.getvalue()

        for flag in ('--min-retention', '--min-grid-beat', '--min-efficiency',
                     '--min-beat-pct'):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)
        self.assertIn('judgment call', help_text)


class TestScreenOutputInvariants(unittest.TestCase):
    """The invariants every CLI in this repository has to honour."""

    def _help(self):
        buf = io.StringIO()
        with patch.object(sys, 'argv', ['screen.py', '--help']), redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                build_parser().parse_args()
        return buf.getvalue()

    def test_strategy_choices_come_from_the_registry(self):
        from niffler.strategies.registry import get_available_strategies

        help_text = self._help()
        for name in get_available_strategies():
            with self.subTest(strategy=name):
                self.assertIn(name, help_text)

    def test_optimizer_choices_come_from_the_registry(self):
        from niffler.optimization.optimizer_factory import get_available_optimizers

        help_text = self._help()
        for name in get_available_optimizers():
            with self.subTest(optimizer=name):
                self.assertIn(name, help_text)

    def test_a_frictionless_run_says_so(self):
        """A run with no cost model must label itself, like every other CLI."""
        from scripts.common import FRICTIONLESS_WARNING, report_run_config

        stream = io.StringIO()
        with redirect_stdout(io.StringIO()):
            warned = report_run_config(RunConfig(), stream=stream)

        self.assertTrue(warned)
        self.assertEqual(stream.getvalue().strip(), FRICTIONLESS_WARNING)

    def test_the_record_uses_safe_json_dump(self):
        """Raw json.dump writes a bare NaN literal, which no parser reads back."""
        import inspect

        import scripts.screen as screen

        source = inspect.getsource(screen)
        self.assertIn('safe_json_dump', source)
        self.assertNotIn('json.dump(', source)

    def test_provenance_is_collected_per_dataset_actually_read(self):
        """One record per file the run opened.

        A single block would hash one file and imply it covered all of them; a
        block for a dataset the funnel stopped short of would claim evidence
        that was never gathered.
        """
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for symbol in ('AAA', 'BBB'):
                path = os.path.join(tmp, f'{symbol}_research.csv')
                data.to_csv(path, index_label='timestamp')
                paths.append(path)
            out = os.path.join(tmp, 'screen.json')

            argv = ['screen.py', '--data', paths[0], '--strategy', 'simple_ma',
                    '--compare-data', paths[1], '--output', out]
            with patch.object(sys, 'argv', argv), \
                    patch('scripts.screen.run_backtest_stage',
                          return_value=StageResult(name='backtest')), \
                    patch('scripts.screen.run_optimize_stage',
                          return_value=StageResult(name='optimize')), \
                    patch('scripts.screen.evaluate', return_value=fold_row()), \
                    patch('scripts.screen.run_walk_forward_stage',
                          return_value=StageResult(name='walk-forward')), \
                    patch('scripts.screen.run_compare_stage',
                          return_value=StageResult(name='compare')), \
                    patch('scripts.screen.render'):
                with redirect_stdout(io.StringIO()):
                    main()

            with open(out) as handle:
                payload = json.load(handle)

        self.assertEqual(sorted(payload['provenance']), sorted(paths))

    def test_an_unreached_dataset_is_requested_but_not_provenanced(self):
        """A gate stopped the funnel before the second file was ever opened."""
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for symbol in ('AAA', 'BBB'):
                path = os.path.join(tmp, f'{symbol}_research.csv')
                data.to_csv(path, index_label='timestamp')
                paths.append(path)
            out = os.path.join(tmp, 'screen.json')

            stopped = StageResult(name='backtest', gates=[
                Gate(stage='backtest', quantity='round trips', value=1.0,
                     threshold=30.0, flag='--min-trades-for-significance')])

            argv = ['screen.py', '--data', paths[0], '--strategy', 'simple_ma',
                    '--compare-data', paths[1], '--output', out]
            with patch.object(sys, 'argv', argv), \
                    patch('scripts.screen.run_backtest_stage', return_value=stopped):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), EXIT_STOPPED)

            with open(out) as handle:
                payload = json.load(handle)

        self.assertEqual(list(payload['provenance']), [paths[0]])
        self.assertEqual(payload['requested_datasets'], paths)

    def test_the_run_config_settings_are_recorded_in_the_output(self):
        """A screening verdict is unreadable without the settings behind it."""
        data = make_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'AAA_research.csv')
            data.to_csv(path, index_label='timestamp')
            out = os.path.join(tmp, 'screen.json')

            argv = ['screen.py', '--data', path, '--strategy', 'simple_ma',
                    '--periods-per-year', '252', '--output', out]
            with patch.object(sys, 'argv', argv), \
                    patch('scripts.screen.run_backtest_stage',
                          return_value=StageResult(name='backtest')), \
                    patch('scripts.screen.run_optimize_stage',
                          return_value=StageResult(name='optimize')), \
                    patch('scripts.screen.evaluate', return_value=fold_row()), \
                    patch('scripts.screen.run_walk_forward_stage',
                          return_value=StageResult(name='walk-forward')), \
                    patch('scripts.screen.render'):
                with redirect_stdout(io.StringIO()):
                    main()

            with open(out) as handle:
                payload = json.load(handle)

        self.assertEqual(payload['settings']['periods_per_year'], 252.0)
        self.assertEqual(payload['thresholds']['min_efficiency'], 0.30)


class TestScreenThreadsTheRunConfig(unittest.TestCase):
    """The screening CLI's engine settings must reach the analyzers unchanged."""

    def test_cli_flags_become_one_config_that_every_stage_receives(self):
        data = make_data()
        seen = {}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'AAA_research.csv')
            data.to_csv(path, index_label='timestamp')

            def capture_evaluate(data_path, strategy, run_config, schedule):
                seen['evaluate'] = run_config
                return fold_row()

            def capture_backtest(frame, symbol, strategy, run_config):
                seen['backtest'] = run_config
                return StageResult(name='backtest')

            argv = ['screen.py', '--data', path, '--strategy', 'simple_ma',
                    '--periods-per-year', '252', '--min-order-value', '25',
                    '--capital', '50000']
            with patch.object(sys, 'argv', argv), \
                    patch('scripts.screen.run_backtest_stage',
                          side_effect=capture_backtest), \
                    patch('scripts.screen.run_optimize_stage',
                          return_value=StageResult(name='optimize')), \
                    patch('scripts.screen.evaluate', side_effect=capture_evaluate), \
                    patch('scripts.screen.run_walk_forward_stage',
                          return_value=StageResult(name='walk-forward')), \
                    patch('scripts.screen.render'):
                with redirect_stdout(io.StringIO()):
                    main()

        self.assertIs(seen['backtest'], seen['evaluate'])
        self.assertEqual(seen['backtest'].periods_per_year, 252.0)
        self.assertEqual(seen['backtest'].min_order_value, 25.0)
        self.assertEqual(seen['backtest'].initial_capital, 50000.0)


class TestStageResultReporting(unittest.TestCase):

    def test_a_stage_with_no_failed_gate_reports_none(self):
        stage = StageResult(name='x', gates=[
            Gate(stage='x', quantity='q', value=1.0, threshold=0.5, flag='--f')])

        self.assertEqual(stage.failed_gates, [])

    def test_failed_gates_are_reported_in_order(self):
        first = Gate(stage='x', quantity='a', value=0.0, threshold=1.0, flag='--a')
        second = Gate(stage='x', quantity='b', value=0.0, threshold=1.0, flag='--b')
        stage = StageResult(name='x', gates=[first, second])

        self.assertEqual(stage.failed_gates, [first, second])


class TestSimpleNamespaceRowsAreNotUsed(unittest.TestCase):
    """Guard against a stage quietly duck-typing an argparse namespace again."""

    def test_evaluate_is_called_with_a_run_config_and_a_schedule(self):
        from scripts.compare import FoldSchedule

        data = make_data()
        captured = {}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'AAA_research.csv')
            data.to_csv(path, index_label='timestamp')

            def capture(data_path, strategy, run_config, schedule):
                captured['run_config'] = run_config
                captured['schedule'] = schedule
                return fold_row()

            argv = ['screen.py', '--data', path, '--strategy', 'simple_ma',
                    '--test_window', '4']
            with patch.object(sys, 'argv', argv), \
                    patch('scripts.screen.run_backtest_stage',
                          return_value=StageResult(name='backtest')), \
                    patch('scripts.screen.run_optimize_stage',
                          return_value=StageResult(name='optimize')), \
                    patch('scripts.screen.evaluate', side_effect=capture), \
                    patch('scripts.screen.run_walk_forward_stage',
                          return_value=StageResult(name='walk-forward')), \
                    patch('scripts.screen.render'):
                with redirect_stdout(io.StringIO()):
                    main()

        self.assertIsInstance(captured['run_config'], RunConfig)
        self.assertIsInstance(captured['schedule'], FoldSchedule)
        # --step defaults to --test_window, so the folds do not overlap.
        self.assertEqual(captured['schedule'].effective_step_months, 4)
        self.assertNotIsInstance(captured['schedule'], SimpleNamespace)


if __name__ == '__main__':
    unittest.main()
