"""Cross-dataset comparison in compare.py.

The comparison this script draws is only meaningful if two things hold, and both are
easy to break by accident, so they are pinned here: a fold is measured against
buy-and-hold over the **same bars**, and one unusable pair does not quietly shrink the
table it appears in.
"""

import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.compare import (
    evaluate,
    main,
    paired_folds,
    render,
    symbol_from_path,
)


def base_args(**overrides) -> argparse.Namespace:
    # The cost flags stay None: build_cost_model treats a flag set alongside
    # --cost-model none as an error, which is the behaviour we want everywhere else.
    values = {
        'capital': 10000.0, 'commission': 0.001, 'cost_model': 'none',
        'slippage_bps': None, 'half_spread_bps': None, 'impact_coefficient': None,
        'max_participation': None, 'clean': False,
        'train_window': 12, 'test_window': 6, 'step': 6, 'anchored': False,
        'optimization_method': 'grid', 'optimization_metric': 'total_return',
        'n_jobs': 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestSymbolFromPath(unittest.TestCase):

    def test_takes_leading_token_upper_cased(self):
        self.assertEqual(symbol_from_path('data/BTCUSDT_research.csv'), 'BTCUSDT')
        self.assertEqual(symbol_from_path('/tmp/spy_holdout.csv'), 'SPY')

    def test_stem_without_underscore_is_used_whole(self):
        self.assertEqual(symbol_from_path('data/QQQ.csv'), 'QQQ')


class TestPairedFolds(unittest.TestCase):
    """The benchmark comes from the fold itself, never from a second computation.

    ``BacktestEngine`` defaults to buy-and-hold and ``WalkForwardAnalyzer`` runs every
    fold through it, so each fold already carries a benchmark priced over its own bars.
    Recomputing one here would be a parallel implementation free to drift from what the
    rest of the platform reports.
    """

    def test_each_fold_contributes_its_own_benchmark(self):
        folds = [
            SimpleNamespace(total_return_pct=5.0, benchmark_return_pct=2.0),
            SimpleNamespace(total_return_pct=-3.0, benchmark_return_pct=1.5),
        ]
        pairs = paired_folds(SimpleNamespace(individual_results=folds))

        self.assertEqual(pairs, [(5.0, 2.0), (-3.0, 1.5)])

    def test_fold_without_a_benchmark_is_dropped_not_zero_filled(self):
        """A missing benchmark is not a flat market, so it cannot become 0.0."""
        folds = [
            SimpleNamespace(total_return_pct=5.0, benchmark_return_pct=None),
            SimpleNamespace(total_return_pct=1.0, benchmark_return_pct=0.5),
        ]
        pairs = paired_folds(SimpleNamespace(individual_results=folds))

        self.assertEqual(pairs, [(1.0, 0.5)])

    def test_no_folds_gives_no_pairs(self):
        self.assertEqual(paired_folds(SimpleNamespace(individual_results=[])), [])


class TestEvaluateFailureHandling(unittest.TestCase):

    def test_a_failing_pair_becomes_a_row_carrying_the_error(self):
        """One broken dataset must not abort a batch, nor vanish from the table."""
        with patch('scripts.compare.load_ohlcv_csv', side_effect=ValueError('bad file')):
            row = evaluate('data/BROKEN_research.csv', 'breakout', base_args())

        self.assertEqual(row['symbol'], 'BROKEN')
        self.assertEqual(row['strategy'], 'breakout')
        self.assertIn('bad file', row['error'])

    def test_render_survives_a_table_of_only_failures(self):
        rows = [{'symbol': 'X', 'strategy': 'breakout', 'error': 'boom'}]
        render(rows)  # must not raise


class TestMainArgumentHandling(unittest.TestCase):

    def test_missing_data_file_exits_non_zero(self):
        argv = ['compare.py', '--data', 'data/does_not_exist_12345.csv']
        with patch.object(sys, 'argv', argv):
            self.assertEqual(main(), 1)

    def test_step_defaults_to_test_window_for_non_overlapping_folds(self):
        """Overlapping folds are the default in analyze.py; here they must not be.

        A comparison across assets is a count of independent evidence, so folds that
        share half their bars would inflate every row in the table equally and
        invisibly.
        """
        captured = {}

        def fake_evaluate(path, strategy, args):
            captured['step'] = args.step
            captured['test_window'] = args.test_window
            return {'symbol': 'X', 'strategy': strategy, 'error': 'stop'}

        argv = ['compare.py', '--data', __file__, '--strategy', 'breakout']
        with patch.object(sys, 'argv', argv), \
             patch('scripts.compare.evaluate', side_effect=fake_evaluate):
            main()

        self.assertEqual(captured['step'], captured['test_window'])

    def test_explicit_step_is_respected(self):
        captured = {}

        def fake_evaluate(path, strategy, args):
            captured['step'] = args.step
            return {'symbol': 'X', 'strategy': strategy, 'error': 'stop'}

        argv = ['compare.py', '--data', __file__, '--strategy', 'breakout',
                '--test_window', '6', '--step', '3']
        with patch.object(sys, 'argv', argv), \
             patch('scripts.compare.evaluate', side_effect=fake_evaluate):
            main()

        self.assertEqual(captured['step'], 3)

    def test_a_failed_pair_makes_the_run_exit_non_zero(self):
        argv = ['compare.py', '--data', __file__, '--strategy', 'breakout']
        with patch.object(sys, 'argv', argv), \
             patch('scripts.compare.evaluate',
                   return_value={'symbol': 'X', 'strategy': 'breakout',
                                 'error': 'boom'}):
            self.assertEqual(main(), 1)


class TestOutputInvariants(unittest.TestCase):
    """Invariants compare.py has to honour because every other CLI does.

    These were all missed on the first pass. They are the kind that fail silently:
    nothing raises, the run exits 0, and the damage is a malformed file or an
    unlabelled frictionless result read months later as if it were realistic.
    """

    def test_optimizer_choices_come_from_the_registry(self):
        """A hardcoded choices list goes stale the moment an optimizer is added.

        The same failure the strategy registry was built to end, one seam over.
        """
        import io
        from contextlib import redirect_stdout

        from niffler.optimization.optimizer_factory import get_available_optimizers

        buf = io.StringIO()
        argv = ['compare.py', '--data', __file__, '--help']
        # redirect_stdout must wrap assertRaises, not sit inside it: --help exits
        # via SystemExit, so anything after main() in the inner block never runs.
        with patch.object(sys, 'argv', argv), redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                main()
        help_text = buf.getvalue()

        self.assertIn('--optimization_method', help_text)
        for name in get_available_optimizers():
            with self.subTest(optimizer=name):
                self.assertIn(name, help_text)

    def test_output_json_is_standards_compliant_with_non_finite_values(self):
        """A NaN metric must land as null, not a bare NaN literal.

        `json.dump` writes `NaN`, which no strict parser will read back - so the
        record silently becomes unloadable rather than loudly failing to write.
        """
        import json
        import tempfile
        from niffler.utils.json_utils import safe_json_dump

        payload = {'rows': [{'symbol': 'X', 'median_excess_pct': float('nan'),
                             'oos_sharpe': float('inf')}]}
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            safe_json_dump(payload, f, indent=2, default=str)
            path = f.name

        with open(path) as f:
            text = f.read()
        self.assertNotIn('NaN', text)
        self.assertNotIn('Infinity', text)
        # The real test: it parses back.
        self.assertIsNone(json.loads(text)['rows'][0]['median_excess_pct'])


if __name__ == '__main__':
    unittest.main()
