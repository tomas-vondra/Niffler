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

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.compare import (
    benchmark_for_window,
    evaluate,
    main,
    paired_folds,
    render,
    symbol_from_path,
)


def make_data(bars: int = 400, start: str = '2020-01-01') -> pd.DataFrame:
    """Build a deterministic OHLCV frame."""
    rng = np.random.default_rng(11)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, bars)))
    index = pd.date_range(start, periods=bars, freq='D')
    return pd.DataFrame({
        'open': price, 'high': price * 1.01, 'low': price * 0.99,
        'close': price, 'volume': np.full(bars, 1000.0),
    }, index=index)


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


class TestBenchmarkForWindow(unittest.TestCase):

    def test_returns_a_number_for_a_normal_window(self):
        data = make_data()
        value = benchmark_for_window(data, data.index[0], data.index[-1], 'X',
                                     base_args())
        self.assertIsInstance(value, float)

    def test_single_bar_window_is_none_not_zero(self):
        """A window too short to hold a position has no benchmark.

        Zero would read as "the market went nowhere", which is a claim about the
        market rather than an admission that nothing could be computed.
        """
        data = make_data()
        self.assertIsNone(
            benchmark_for_window(data, data.index[0], data.index[0], 'X', base_args())
        )

    def test_window_is_sliced_not_taken_whole(self):
        """The benchmark must reflect the fold, not the whole file.

        This is the failure the paired comparison exists to prevent: a six-month fold
        judged against a multi-year buy-and-hold figure.
        """
        data = make_data()
        early = benchmark_for_window(data, data.index[0], data.index[50], 'X',
                                     base_args())
        whole = benchmark_for_window(data, data.index[0], data.index[-1], 'X',
                                     base_args())
        self.assertNotAlmostEqual(early, whole, places=6)


class TestPairedFolds(unittest.TestCase):

    def test_each_fold_is_paired_with_its_own_window(self):
        data = make_data()
        folds = [
            SimpleNamespace(start_date=data.index[0], end_date=data.index[100],
                            total_return_pct=5.0),
            SimpleNamespace(start_date=data.index[100], end_date=data.index[200],
                            total_return_pct=-3.0),
        ]
        result = SimpleNamespace(individual_results=folds)
        pairs = paired_folds(result, data, 'X', base_args())

        self.assertEqual(len(pairs), 2)
        self.assertEqual([p[0] for p in pairs], [5.0, -3.0])
        # Two different windows must not produce one shared benchmark figure.
        self.assertNotAlmostEqual(pairs[0][1], pairs[1][1], places=6)

    def test_fold_without_a_benchmark_is_dropped_not_zero_filled(self):
        data = make_data()
        folds = [
            SimpleNamespace(start_date=data.index[0], end_date=data.index[0],
                            total_return_pct=5.0),
            SimpleNamespace(start_date=data.index[0], end_date=data.index[100],
                            total_return_pct=1.0),
        ]
        result = SimpleNamespace(individual_results=folds)
        pairs = paired_folds(result, data, 'X', base_args())

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], 1.0)

    def test_no_folds_gives_no_pairs(self):
        result = SimpleNamespace(individual_results=[])
        self.assertEqual(paired_folds(result, make_data(), 'X', base_args()), [])


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


if __name__ == '__main__':
    unittest.main()
