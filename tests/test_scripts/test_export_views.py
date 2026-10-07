"""The summary and detail rows each script hands to a document store.

These are what a leaderboard will read, so the properties pinned are the ones
a misleading board would be built on: a truncated grid reports no grid
statistic, a trial that never traded is a state rather than a score, a fold
row carries the in-sample/out-of-sample pair, and a comparison row names its
own data.
"""

import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.exporters.run_record import (
    DETAIL_COMPARISON,
    DETAIL_FOLD,
    DETAIL_SIMULATION,
    DETAIL_TRIAL,
)
from niffler.optimization import plateau
from scripts import analyze, compare, optimize
from scripts.common import symbol_from_data_path


def optimize_args(**overrides):
    values = {'method': 'grid', 'sort_by': 'total_return', 'no_plateau': False,
              'plateau_metric': None, 'plateau_tolerance': plateau.DEFAULT_TOLERANCE}
    values.update(overrides)
    return argparse.Namespace(**values)


def trial(parameters, **metrics):
    base = {'total_return': 100.0, 'total_return_pct': 1.0, 'sharpe_ratio': 0.5,
            'max_drawdown': -5.0, 'total_trades': 12, 'win_rate': 50.0}
    base.update(metrics)
    return {'parameters': parameters, 'metrics': base}


class TestOptimizeExportViews(unittest.TestCase):

    DOCUMENT = {'results': [
        trial({'w': 20}, total_return=500.0, total_return_pct=5.0),
        trial({'w': 30}, total_trades=0, total_return=0.0),
        trial({'w': 40}, total_return=float('nan')),
    ]}
    PLATEAU = {'plateau_metric': 'total_return', 'grid_baseline': 1.0, 'grid_median': 2.0,
               'fraction_beating_baseline': 0.25, 'plateau_retention': 0.8}

    def views(self, **kwargs):
        return optimize.build_export_views(
            self.DOCUMENT, optimize_args(), plateau.SELECTION_EXHAUSTIVE,
            kwargs.get('truncated', False), kwargs.get('plateau', self.PLATEAU))

    def test_one_row_per_trial_ranked_in_result_order(self):
        _, details = self.views()
        rows = details[DETAIL_TRIAL]
        self.assertEqual([row['rank'] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]['parameters'], {'w': 20})
        self.assertEqual(rows[0]['total_return_pct'], 5.0)

    def test_a_trial_that_never_traded_or_scored_is_a_state_not_a_number(self):
        _, details = self.views()
        self.assertEqual([row['status'] for row in details[DETAIL_TRIAL]],
                         [plateau.CELL_OK, plateau.CELL_NO_TRADES, plateau.CELL_NON_FINITE])

    def test_the_summary_carries_the_winner_and_the_grid_figures(self):
        summary, _ = self.views()
        self.assertEqual(summary['n_trials'], 3)
        self.assertEqual(summary['best_parameters'], {'w': 20})
        self.assertEqual(summary['total_return_pct'], 5.0)
        self.assertEqual(summary['grid_median'], 2.0)
        self.assertEqual(summary['fraction_beating_baseline'], 0.25)
        self.assertEqual(summary['plateau_retention'], 0.8)
        self.assertIs(summary['results_truncated'], False)

    def test_an_empty_result_set_still_produces_a_summary(self):
        summary, details = optimize.build_export_views(
            {'results': []}, optimize_args(), plateau.SELECTION_EXHAUSTIVE, False,
            dict(optimize._NO_PLATEAU_SUMMARY))
        self.assertEqual(summary['n_trials'], 0)
        self.assertIsNone(summary['best_parameters'])
        self.assertEqual(details[DETAIL_TRIAL], [])


class TestPlateauSummary(unittest.TestCase):

    def report(self, reliable=True):
        report = Mock()
        report.distribution.reliable = reliable
        report.distribution.baseline = 540.48
        report.distribution.median = 438.42
        report.distribution.fraction_beating_baseline = 0.247
        report.plateau.retention = 0.6
        return report

    def test_a_complete_grid_reports_its_figures(self):
        with patch('scripts.optimize.plateau_analysis.analyse_results',
                   return_value=self.report()):
            summary = optimize.plateau_summary([], optimize_args(),
                                               plateau.SELECTION_EXHAUSTIVE)
        self.assertEqual(summary['grid_median'], 438.42)
        self.assertEqual(summary['grid_baseline'], 540.48)
        self.assertEqual(summary['fraction_beating_baseline'], 0.247)
        self.assertEqual(summary['plateau_retention'], 0.6)

    def test_a_truncated_grid_reports_no_grid_statistic(self):
        """Survivors of a keep-the-best purge flatter the grid they came from."""
        with patch('scripts.optimize.plateau_analysis.analyse_results',
                   return_value=self.report(reliable=False)):
            summary = optimize.plateau_summary([], optimize_args(),
                                               plateau.SELECTION_TRUNCATED)
        self.assertEqual(summary['plateau_metric'], 'total_return')
        for field in ('grid_baseline', 'grid_median', 'fraction_beating_baseline',
                      'plateau_retention'):
            with self.subTest(field=field):
                self.assertIsNone(summary[field])

    def test_no_plateau_skips_the_analysis(self):
        with patch('scripts.optimize.plateau_analysis.analyse_results') as analyse:
            summary = optimize.plateau_summary([], optimize_args(no_plateau=True),
                                               plateau.SELECTION_EXHAUSTIVE)
        analyse.assert_not_called()
        self.assertEqual(summary, optimize._NO_PLATEAU_SUMMARY)

    def test_a_failed_analysis_costs_the_figures_not_the_export(self):
        with patch('scripts.optimize.plateau_analysis.analyse_results',
                   side_effect=RuntimeError('boom')):
            summary = optimize.plateau_summary([], optimize_args(),
                                               plateau.SELECTION_EXHAUSTIVE)
        self.assertEqual(summary, optimize._NO_PLATEAU_SUMMARY)


class TestAnalyzeExportViews(unittest.TestCase):

    def result(self, metadata=None):
        result = Mock()
        result.metadata = metadata
        result.attempted_runs = 4
        result.failed_runs = 1
        result.failure_rate = 0.25
        return result

    def test_a_walk_forward_fold_row_carries_in_sample_and_out_of_sample(self):
        folds = [{'fold_number': 1, 'train_return_pct': 40.0, 'test_return_pct': 10.0,
                  'efficiency_ratio': 0.25, 'in_sample': False,
                  'parameters': {'w': 20}}]
        document = {'analysis_type': 'walk_forward', 'n_periods': 1,
                    'period_results': [{'period': 1, 'total_return_pct': 10.0}]}
        summary, details = analyze.build_export_views(
            self.result({'folds': folds}), document)

        self.assertEqual(details, {DETAIL_FOLD: folds})
        self.assertEqual(summary['n_periods'], 1)
        self.assertEqual(summary['failed_runs'], 1)
        self.assertEqual(summary['failure_rate'], 0.25)

    def test_without_fold_metadata_the_period_rows_are_used(self):
        periods = [{'period': 1, 'total_return_pct': 10.0}]
        document = {'analysis_type': 'walk_forward', 'period_results': periods}
        _, details = analyze.build_export_views(self.result(None), document)
        self.assertEqual(details, {DETAIL_FOLD: periods})

    def test_a_monte_carlo_run_exports_one_row_per_simulation(self):
        simulations = [{'period': 1, 'total_return_pct': -3.0},
                       {'period': 2, 'total_return_pct': 8.0}]
        document = {'analysis_type': 'monte_carlo', 'simulation_results': simulations}
        _, details = analyze.build_export_views(self.result(None), document)
        self.assertEqual(details, {DETAIL_SIMULATION: simulations})


class TestComparisonDetails(unittest.TestCase):

    def test_each_row_names_its_own_strategy_and_data(self):
        rows = [
            {'symbol': 'AAA', 'strategy': 'rsi', 'data_path': 'a.csv', 'error': None},
            {'symbol': 'BBB', 'strategy': 'breakout', 'data_path': 'b.csv',
             'error': 'boom'},
        ]
        provenance = {'a.csv': {'data': {'sha256': 'a' * 64}},
                      'b.csv': {'data': {'sha256': None}}}

        detailed = compare.comparison_details(rows, provenance)[DETAIL_COMPARISON]

        self.assertEqual(detailed[0]['strategy_key'], 'rsi')
        self.assertEqual(detailed[0]['data_sha256'], 'a' * 64)
        # A failed row is still a row: the error travels with it.
        self.assertEqual(detailed[1]['error'], 'boom')
        self.assertIsNone(detailed[1]['data_sha256'])

    def test_a_row_for_a_file_with_no_record_has_no_fingerprint(self):
        rows = [{'symbol': 'AAA', 'strategy': 'rsi', 'data_path': 'x.csv', 'error': None}]
        detailed = compare.comparison_details(rows, {})[DETAIL_COMPARISON]
        self.assertIsNone(detailed[0]['data_sha256'])


class TestSymbolFromDataPath(unittest.TestCase):

    def test_the_leading_token_of_the_file_stem(self):
        self.assertEqual(
            symbol_from_data_path('data/BTCUSDT_binance_1d_20240101.csv'), 'BTCUSDT')
        self.assertEqual(symbol_from_data_path('SPY.csv'), 'SPY')


if __name__ == '__main__':
    unittest.main()
