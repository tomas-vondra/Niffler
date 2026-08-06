"""
Unit tests for parameter plateau analysis.

The point of this module is to tell an isolated spike from a broad region, and
to describe the whole grid rather than its best row. These tests pin down the
things that would quietly destroy either: a metric read in the wrong direction,
a failed cell averaged in as a mediocre one, a score-biased sample presented as
a distribution, and a baseline that is not what its label says.
"""

import csv
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.optimization import plateau
from niffler.optimization.optimization_result import OptimizationResult


def make_result(parameters, total_return=10.0, total_trades=5, sharpe_ratio=1.0,
                max_drawdown=-10.0, win_rate=50.0, benchmark_return_pct=20.0,
                benchmark_sharpe_ratio=0.8, benchmark_max_drawdown=-25.0):
    """Build an OptimizationResult with a stubbed backtest result."""
    backtest = Mock()
    backtest.total_return_pct = total_return
    backtest.total_return = total_return
    backtest.sharpe_ratio = sharpe_ratio
    backtest.max_drawdown = max_drawdown
    backtest.win_rate = win_rate
    backtest.total_trades = total_trades
    backtest.benchmark_return_pct = benchmark_return_pct
    backtest.benchmark_sharpe_ratio = benchmark_sharpe_ratio
    backtest.benchmark_max_drawdown = benchmark_max_drawdown
    backtest.excess_return_pct = (None if benchmark_return_pct is None
                                  else total_return - benchmark_return_pct)
    return OptimizationResult(parameters=dict(parameters), backtest_result=backtest)


def plateau_grid(peak=(10, 30), height=40.0, radius=2, slope=8.0, **kwargs):
    """A broad plateau: a flat-topped region around the peak, falling away steeply.

    The top is not perfectly flat - it tilts by 0.05 per step - so the argmax is
    a single unambiguous cell while its neighbourhood still scores as well as it
    does, which is exactly the shape the plateau score is supposed to reward.
    """
    results = []
    for short_window in range(5, 16):
        for long_window in range(20, 51, 5):
            distance = max(abs(short_window - peak[0]),
                           abs(long_window - peak[1]) / 5)
            if distance <= radius:
                score = height - distance * 0.05
            else:
                score = height - radius * 0.05 - (distance - radius) * slope
            results.append(make_result(
                {'short_window': short_window, 'long_window': long_window},
                total_return=score, **kwargs))
    return results


def spike_grid(peak=(10, 30), height=40.0, floor=1.0, **kwargs):
    """A flat plain with one cell sticking out of it."""
    results = []
    for short_window in range(5, 16):
        for long_window in range(20, 51, 5):
            score = height if (short_window, long_window) == peak else floor
            results.append(make_result(
                {'short_window': short_window, 'long_window': long_window},
                total_return=score, **kwargs))
    return results


class TestBuildSurface(unittest.TestCase):
    """Turning results into a grid."""

    def test_axes_are_sorted_parameter_values(self):
        surface = plateau.build_surface(plateau_grid())

        self.assertEqual(surface.parameter_names, ['short_window', 'long_window'])
        self.assertEqual(surface.axes['short_window'], list(range(5, 16)))
        self.assertEqual(surface.axes['long_window'], list(range(20, 51, 5)))
        self.assertEqual(surface.shape, (11, 7))
        self.assertEqual(surface.size, 77)

    def test_every_cartesian_cell_is_present(self):
        surface = plateau.build_surface(plateau_grid())

        self.assertEqual(len(surface.cells), surface.size)
        self.assertEqual(surface.status_counts[plateau.CELL_OK], surface.size)
        self.assertEqual(surface.coverage, 1.0)

    def test_missing_combination_is_missing_not_zero(self):
        results = [result for result in plateau_grid()
                   if result.parameters != {'short_window': 7, 'long_window': 25}]

        surface = plateau.build_surface(results)
        cell = surface.cells[(2, 1)]

        self.assertEqual(cell.parameters, {'short_window': 7, 'long_window': 25})
        self.assertEqual(cell.status, plateau.CELL_MISSING)
        self.assertIsNone(cell.score)
        self.assertIsNone(cell.oriented_score)
        self.assertEqual(surface.status_counts[plateau.CELL_MISSING], 1)

    def test_zero_trade_combination_is_its_own_state(self):
        results = [result for result in plateau_grid()
                   if result.parameters != {'short_window': 5, 'long_window': 20}]
        results.append(make_result({'short_window': 5, 'long_window': 20},
                                   total_return=0.0, total_trades=0))

        surface = plateau.build_surface(results)
        cell = surface.cells[(0, 0)]

        self.assertEqual(cell.status, plateau.CELL_NO_TRADES)
        self.assertIsNone(cell.oriented_score)
        self.assertFalse(cell.is_ok)

    def test_non_finite_metric_is_its_own_state(self):
        excluded = ({'short_window': 5, 'long_window': 20},
                    {'short_window': 6, 'long_window': 20})
        results = [result for result in plateau_grid()
                   if result.parameters not in excluded]
        results.append(make_result({'short_window': 5, 'long_window': 20},
                                   total_return=float('inf')))
        results.append(make_result({'short_window': 6, 'long_window': 20},
                                   total_return=float('nan')))

        surface = plateau.build_surface(results)

        self.assertEqual(surface.cells[(0, 0)].status, plateau.CELL_NON_FINITE)
        self.assertEqual(surface.cells[(1, 0)].status, plateau.CELL_NON_FINITE)

    def test_none_sharpe_lands_in_non_finite_not_as_a_minimum(self):
        # The METRICS_CONFIG accessor maps a None Sharpe to -inf; that must not
        # become the worst real score on the surface.
        results = [make_result({'short_window': 5, 'long_window': 20}, sharpe_ratio=1.0),
                   make_result({'short_window': 6, 'long_window': 20}, sharpe_ratio=None)]

        surface = plateau.build_surface(results, metric='sharpe_ratio')

        self.assertEqual(surface.cells[(1, 0)].status, plateau.CELL_NON_FINITE)
        self.assertEqual([cell.score for cell in surface.ok_cells], [1.0])

    def test_duplicate_parameters_keep_the_better_cell(self):
        results = [make_result({'short_window': 5}, total_return=1.0),
                   make_result({'short_window': 5}, total_return=9.0)]

        surface = plateau.build_surface(results)

        self.assertEqual(surface.duplicates, 1)
        self.assertEqual(surface.cells[(0,)].score, 9.0)

    def test_duplicate_order_does_not_matter(self):
        forward = plateau.build_surface([make_result({'a': 1}, total_return=1.0),
                                         make_result({'a': 1}, total_return=9.0)])
        backward = plateau.build_surface([make_result({'a': 1}, total_return=9.0),
                                          make_result({'a': 1}, total_return=1.0)])

        self.assertEqual(forward.cells[(0,)].score, backward.cells[(0,)].score)

    def test_empty_results_raise(self):
        with self.assertRaises(plateau.PlateauError):
            plateau.build_surface([])

    def test_unknown_metric_raises(self):
        with self.assertRaises(plateau.PlateauError):
            plateau.build_surface(plateau_grid(), metric='calmar_ratio')

    def test_unknown_selection_raises(self):
        with self.assertRaises(plateau.PlateauError):
            plateau.build_surface(plateau_grid(), selection='whatever')

    def test_mismatched_parameter_names_raise(self):
        results = [make_result({'short_window': 5}), make_result({'long_window': 20})]

        with self.assertRaises(plateau.PlateauError):
            plateau.build_surface(results)

    def test_parameterless_results_raise(self):
        with self.assertRaises(plateau.PlateauError):
            plateau.build_surface([make_result({})])

    def test_direction_comes_from_the_optimizer(self):
        self.assertTrue(plateau.build_surface(plateau_grid()).higher_is_better)
        self.assertTrue(
            plateau.build_surface(plateau_grid(), metric='max_drawdown').higher_is_better)


class TestPlateauScore(unittest.TestCase):
    """Telling a broad region from an isolated spike."""

    def test_broad_plateau_scores_as_robust(self):
        score = plateau.analyse_plateau(plateau.build_surface(plateau_grid()))

        self.assertEqual(score.winner_parameters,
                         {'short_window': 10, 'long_window': 30})
        self.assertGreaterEqual(score.retention, plateau.BROAD_PLATEAU_RETENTION)
        self.assertEqual(score.verdict, 'broad plateau')
        self.assertGreater(score.region_size, 1)

    def test_isolated_spike_scores_as_noise(self):
        score = plateau.analyse_plateau(plateau.build_surface(spike_grid()))

        self.assertEqual(score.winner_parameters,
                         {'short_window': 10, 'long_window': 30})
        self.assertLess(score.retention, plateau.ISOLATED_SPIKE_RETENTION)
        self.assertEqual(score.verdict, 'isolated spike')
        self.assertEqual(score.region_size, 1)

    def test_spike_retention_is_zero_when_neighbours_are_typical(self):
        # Every non-peak cell scores the same, so the neighbourhood mean IS the
        # grid median and retention is exactly zero by construction.
        score = plateau.analyse_plateau(plateau.build_surface(spike_grid()))

        self.assertAlmostEqual(score.retention, 0.0, places=9)

    def test_neighbour_counts_are_reported(self):
        score = plateau.analyse_plateau(plateau.build_surface(plateau_grid()))

        # An interior winner on a 2D grid has eight neighbours.
        self.assertEqual(score.neighbours_total, 8)
        self.assertEqual(score.neighbours_ok, 8)

    def test_corner_winner_reports_fewer_neighbours(self):
        score = plateau.analyse_plateau(
            plateau.build_surface(plateau_grid(peak=(5, 20))))

        self.assertEqual(score.winner_parameters,
                         {'short_window': 5, 'long_window': 20})
        self.assertEqual(score.neighbours_total, 3)
        self.assertEqual(score.neighbours_ok, 3)

    def test_winner_with_no_scored_neighbour_gets_no_verdict(self):
        results = [make_result({'short_window': 5, 'long_window': 20}, total_return=1.0),
                   make_result({'short_window': 6, 'long_window': 20},
                               total_return=99.0, total_trades=0),
                   make_result({'short_window': 7, 'long_window': 20}, total_return=50.0)]
        # The winner at short_window=7 has one in-bounds neighbour, and it never
        # traded, so there is nothing to compute a neighbourhood from.
        score = plateau.analyse_plateau(plateau.build_surface(results))

        self.assertEqual(score.winner_parameters,
                         {'short_window': 7, 'long_window': 20})
        self.assertIsNone(score.retention)
        self.assertEqual(score.verdict, 'no verdict')
        self.assertIn('no scored neighbour', score.retention_reason)
        self.assertEqual(score.neighbours_ok, 0)

    def test_flat_surface_has_no_edge_to_retain(self):
        results = [make_result({'short_window': window}, total_return=5.0)
                   for window in range(5, 10)]

        score = plateau.analyse_plateau(plateau.build_surface(results))

        self.assertIsNone(score.retention)
        self.assertIn('no edge', score.retention_reason)
        self.assertEqual(score.edge, 0.0)

    def test_failed_neighbours_are_excluded_not_zero_filled(self):
        # A spike whose neighbours are all MISSING must not be rescued by
        # treating those neighbours as zeros, nor sunk by it: it gets no verdict.
        results = [result for result in plateau_grid()
                   if result.parameters['short_window'] not in (9, 11)
                   or result.parameters['long_window'] != 30]
        results = [result for result in results
                   if not (result.parameters['short_window'] in (9, 10, 11)
                           and result.parameters['long_window'] in (25, 35))]

        surface = plateau.build_surface(results)
        score = plateau.analyse_plateau(surface)

        self.assertEqual(score.winner_parameters,
                         {'short_window': 10, 'long_window': 30})
        self.assertEqual(score.neighbours_total, 8)
        self.assertEqual(score.neighbours_ok, 0)
        self.assertIsNone(score.retention)

    def test_single_cell_grid(self):
        surface = plateau.build_surface([make_result({'short_window': 10})])
        score = plateau.analyse_plateau(surface)

        self.assertEqual(surface.shape, (1,))
        self.assertEqual(score.neighbours_total, 0)
        self.assertIsNone(score.retention)
        self.assertEqual(score.region_size, 1)
        self.assertEqual(score.region_fraction, 1.0)

    def test_degenerate_one_by_n_grid(self):
        # One tick on the first axis, seven on the second, peaking in the middle.
        scores = {20: 1.0, 25: 2.0, 30: 3.0, 35: 9.0, 40: 3.0, 45: 2.0, 50: 1.0}
        results = [make_result({'short_window': 5, 'long_window': window},
                               total_return=score)
                   for window, score in scores.items()]

        surface = plateau.build_surface(results)
        score = plateau.analyse_plateau(surface)

        self.assertEqual(surface.shape, (1, 7))
        self.assertEqual(score.winner_parameters,
                         {'short_window': 5, 'long_window': 35})
        # A 1xN row offers two neighbours; the flat axis contributes none.
        self.assertEqual(score.neighbours_total, 2)
        self.assertIsNotNone(score.retention)

    def test_end_of_a_row_has_one_neighbour(self):
        results = [make_result({'short_window': 5, 'long_window': window},
                               total_return=float(window))
                   for window in range(20, 51, 5)]

        score = plateau.analyse_plateau(plateau.build_surface(results))

        self.assertEqual(score.winner_parameters,
                         {'short_window': 5, 'long_window': 50})
        self.assertEqual(score.neighbours_total, 1)

    def test_no_scored_cell_returns_none(self):
        results = [make_result({'short_window': 5}, total_trades=0)]

        self.assertIsNone(plateau.analyse_plateau(plateau.build_surface(results)))

    def test_tolerance_widens_the_region(self):
        # A gradual slope, so widening the band admits strictly more cells.
        results = [make_result({'short_window': window},
                               total_return=float(20 - abs(window - 10)))
                   for window in range(0, 21)]
        surface = plateau.build_surface(results)

        narrow = plateau.analyse_plateau(surface, tolerance=0.05)
        wide = plateau.analyse_plateau(surface, tolerance=0.9)

        self.assertLess(narrow.region_size, wide.region_size)

    def test_invalid_tolerance_raises(self):
        surface = plateau.build_surface(plateau_grid())

        with self.assertRaises(plateau.PlateauError):
            plateau.analyse_plateau(surface, tolerance=1.5)
        with self.assertRaises(plateau.PlateauError):
            plateau.analyse_plateau(surface, tolerance=-0.1)

    def test_region_is_contiguous_not_merely_high_scoring(self):
        # Two identical peaks, far apart, with a trough between them. Only the
        # one containing the winner may be counted.
        results = []
        for window in range(0, 11):
            score = 10.0 if window in (0, 1, 9, 10) else 0.0
            results.append(make_result({'short_window': window}, total_return=score))

        score = plateau.analyse_plateau(plateau.build_surface(results))

        self.assertEqual(score.region_size, 2)

    def test_region_extent_is_reported_per_parameter(self):
        score = plateau.analyse_plateau(plateau.build_surface(plateau_grid()))

        low, high = score.region_extent['short_window']
        self.assertLessEqual(low, 10)
        self.assertGreaterEqual(high, 10)
        self.assertIn('long_window', score.region_extent)

    def test_centre_of_an_asymmetric_region_is_not_the_argmax(self):
        # Scores rise to a peak at the edge of a wide flat shelf; the argmax and
        # the shelf's centre are different cells.
        scores = {0: 9.0, 1: 9.5, 2: 9.6, 3: 9.7, 4: 9.8, 5: 10.0,
                  6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
        results = [make_result({'short_window': window}, total_return=score)
                   for window, score in scores.items()]

        score = plateau.analyse_plateau(plateau.build_surface(results), tolerance=1.0)

        self.assertEqual(score.winner_parameters, {'short_window': 5})
        self.assertNotEqual(score.centre_parameters, score.winner_parameters)
        self.assertIsNotNone(score.centre_score)

    def test_boundary_is_flagged_when_the_region_runs_off_the_grid(self):
        score = plateau.analyse_plateau(
            plateau.build_surface(plateau_grid(peak=(5, 20))))

        self.assertEqual(sorted(score.boundary_parameters),
                         ['long_window', 'short_window'])

    def test_interior_region_is_not_flagged_as_boundary(self):
        results = [make_result({'short_window': window},
                               total_return=float(20 - abs(window - 10)))
                   for window in range(0, 21)]

        score = plateau.analyse_plateau(plateau.build_surface(results), tolerance=0.05)

        self.assertEqual(score.boundary_parameters, [])

    def test_boundary_caveat_is_rendered(self):
        surface = plateau.build_surface(plateau_grid(peak=(5, 20)))
        rendered = plateau.render_plateau(plateau.analyse_plateau(surface), surface)

        self.assertIn('edge of the parameter space', rendered)

    def test_centre_is_deterministic(self):
        surface = plateau.build_surface(plateau_grid())

        first = plateau.analyse_plateau(surface)
        second = plateau.analyse_plateau(surface)

        self.assertEqual(first.centre_index, second.centre_index)


class TestMetricDirection(unittest.TestCase):
    """max_drawdown is negative and shallower is better, everywhere."""

    def setUp(self):
        # Drawdown is shallowest (-2) at short_window=10 and deepest (-40) away
        # from it, with total_return arranged the opposite way round so a
        # direction mistake cannot pass by accident.
        self.results = []
        for window in range(5, 16):
            drawdown = -2.0 - abs(window - 10) * 5.0
            self.results.append(make_result({'short_window': window},
                                            total_return=-drawdown,
                                            max_drawdown=drawdown))

    def test_winner_is_the_shallowest_drawdown(self):
        score = plateau.analyse_plateau(
            plateau.build_surface(self.results, metric='max_drawdown'))

        self.assertEqual(score.winner_parameters, {'short_window': 10})
        self.assertEqual(score.winner_score, -2.0)

    def test_shallowest_drawdown_is_the_maximum_not_the_minimum(self):
        surface = plateau.build_surface(self.results, metric='max_drawdown')

        self.assertEqual(surface.best_cell().score, max(
            cell.score for cell in surface.ok_cells))

    def test_distribution_quartiles_stay_in_metric_units(self):
        surface = plateau.build_surface(self.results, metric='max_drawdown')
        stats = plateau.summarize_distribution(surface, self.results)

        self.assertEqual(stats.minimum, -27.0)
        self.assertEqual(stats.maximum, -2.0)
        self.assertLess(stats.median, 0.0)

    def test_beating_the_baseline_means_a_shallower_drawdown(self):
        surface = plateau.build_surface(self.results, metric='max_drawdown')
        stats = plateau.summarize_distribution(surface, self.results)

        # Benchmark drawdown is -25; only the cells shallower than that count.
        self.assertEqual(stats.baseline, -25.0)
        expected = sum(1 for result in self.results
                       if result.backtest_result.max_drawdown > -25.0)
        self.assertEqual(stats.beat_count, expected)

    def test_plateau_retention_is_positive_for_a_drawdown_cone(self):
        score = plateau.analyse_plateau(
            plateau.build_surface(self.results, metric='max_drawdown'))

        self.assertGreater(score.retention, 0.0)


class TestDistributionStats(unittest.TestCase):
    """Describing the whole grid, not the top row."""

    def test_quartiles_cover_every_scored_cell(self):
        results = [make_result({'short_window': window}, total_return=float(window))
                   for window in range(0, 101)]

        stats = plateau.summarize_distribution(
            plateau.build_surface(results), results)

        self.assertEqual(stats.count, 101)
        self.assertEqual(stats.minimum, 0.0)
        self.assertEqual(stats.median, 50.0)
        self.assertEqual(stats.maximum, 100.0)
        self.assertAlmostEqual(stats.q1, 25.0)
        self.assertAlmostEqual(stats.q3, 75.0)

    def test_failed_cells_are_excluded_from_statistics(self):
        results = [make_result({'short_window': 1}, total_return=10.0),
                   make_result({'short_window': 2}, total_return=20.0),
                   make_result({'short_window': 3}, total_return=0.0, total_trades=0)]

        stats = plateau.summarize_distribution(
            plateau.build_surface(results), results)

        self.assertEqual(stats.count, 2)
        self.assertEqual(stats.mean, 15.0)
        self.assertEqual(stats.minimum, 10.0)
        self.assertEqual(stats.status_counts[plateau.CELL_NO_TRADES], 1)

    def test_zero_filling_a_failed_cell_would_change_the_mean(self):
        # Guards the invariant directly: were the no-trade cell folded in as
        # 0.0, the mean would be 10.0 rather than 15.0.
        results = [make_result({'short_window': 1}, total_return=10.0),
                   make_result({'short_window': 2}, total_return=20.0),
                   make_result({'short_window': 3}, total_return=0.0, total_trades=0)]

        stats = plateau.summarize_distribution(
            plateau.build_surface(results), results)

        self.assertNotEqual(stats.mean, 10.0)

    def test_fraction_beating_the_baseline(self):
        results = [make_result({'short_window': window}, total_return=float(window),
                               benchmark_return_pct=95.0)
                   for window in range(0, 100)]

        stats = plateau.summarize_distribution(
            plateau.build_surface(results), results)

        self.assertEqual(stats.baseline, 95.0)
        self.assertEqual(stats.baseline_label, 'buy-and-hold total return')
        self.assertEqual(stats.beat_count, 4)  # 96..99
        self.assertAlmostEqual(stats.fraction_beating_baseline, 0.04)

    def test_missing_benchmark_falls_back_to_a_labelled_zero(self):
        results = [make_result({'short_window': 1}, total_return=10.0,
                               benchmark_return_pct=None)]

        stats = plateau.summarize_distribution(
            plateau.build_surface(results), results)

        self.assertEqual(stats.baseline, 0.0)
        self.assertIn('no benchmark available', stats.baseline_label)
        self.assertNotIn('buy-and-hold', stats.baseline_label)

    def test_missing_benchmark_gives_max_drawdown_no_baseline_at_all(self):
        results = [make_result({'short_window': 1}, benchmark_max_drawdown=None)]

        stats = plateau.summarize_distribution(
            plateau.build_surface(results, metric='max_drawdown'), results)

        self.assertIsNone(stats.baseline)
        self.assertIsNone(stats.beat_count)

    def test_metric_without_a_do_nothing_equivalent_has_no_baseline(self):
        results = [make_result({'short_window': 1}, win_rate=60.0)]

        stats = plateau.summarize_distribution(
            plateau.build_surface(results, metric='win_rate'), results)

        self.assertIsNone(stats.baseline)
        self.assertIsNone(stats.fraction_beating_baseline)

    def test_excess_return_baseline_is_zero_by_definition(self):
        results = [make_result({'short_window': 1}, total_return=30.0)]

        stats = plateau.summarize_distribution(
            plateau.build_surface(results, metric='excess_return_pct'), results)

        self.assertEqual(stats.baseline, 0.0)
        self.assertIn('buy-and-hold', stats.baseline_label)

    def test_truncated_selection_withholds_the_distribution(self):
        results = [make_result({'short_window': window}, total_return=float(window))
                   for window in range(0, 10)]

        surface = plateau.build_surface(results, selection=plateau.SELECTION_TRUNCATED)
        stats = plateau.summarize_distribution(surface, results)

        self.assertFalse(stats.reliable)
        self.assertIsNone(stats.median)
        self.assertIsNone(stats.fraction_beating_baseline)
        self.assertIn('score-biased', stats.unreliable_reason)

    def test_sampled_selection_still_reports_statistics(self):
        results = [make_result({'short_window': window}, total_return=float(window))
                   for window in range(0, 10)]

        surface = plateau.build_surface(results, selection=plateau.SELECTION_SAMPLED)
        stats = plateau.summarize_distribution(surface, results)

        self.assertTrue(stats.reliable)
        self.assertIsNotNone(stats.median)

    def test_inconsistent_benchmarks_are_reported(self):
        # Another test module disables logging globally; assertLogs would see
        # nothing and this test would pass alone and fail under discovery.
        previous = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, previous)

        results = [make_result({'short_window': 1}, benchmark_return_pct=10.0),
                   make_result({'short_window': 2}, benchmark_return_pct=90.0)]

        with self.assertLogs('niffler.optimization.plateau', level='WARNING') as logs:
            baseline, _ = plateau.resolve_baseline(results, 'total_return')

        self.assertEqual(baseline, 50.0)
        self.assertIn('not constant', logs.output[0])


class TestRendering(unittest.TestCase):
    """The console output."""

    def test_heatmap_marks_winner_and_states(self):
        results = [result for result in plateau_grid()
                   if result.parameters != {'short_window': 5, 'long_window': 20}]
        results.append(make_result({'short_window': 6, 'long_window': 20},
                                   total_return=0.0, total_trades=0))

        surface = plateau.build_surface(results)
        score = plateau.analyse_plateau(surface)
        rendered = plateau.render_heatmap(surface, score)

        self.assertIn(plateau.HEATMAP_WINNER, rendered)
        self.assertIn(plateau.HEATMAP_MISSING, rendered)
        self.assertIn(plateau.HEATMAP_NO_TRADES, rendered)
        self.assertIn('short_window', rendered)
        self.assertIn('long_window', rendered)

    def test_heatmap_is_pure_ascii(self):
        surface = plateau.build_surface(plateau_grid())
        rendered = plateau.render_heatmap(surface, plateau.analyse_plateau(surface))

        rendered.encode('ascii')  # raises if a non-ASCII character crept in

    def test_heatmap_slices_a_three_parameter_surface_at_the_winner(self):
        results = []
        for short_window in range(5, 10):
            for long_window in range(20, 41, 5):
                for size in (0.5, 1.0):
                    score = 10.0 + size + short_window
                    results.append(make_result(
                        {'short_window': short_window, 'long_window': long_window,
                         'position_size': size},
                        total_return=score))

        surface = plateau.build_surface(results)
        score = plateau.analyse_plateau(surface)
        rendered = plateau.render_heatmap(surface, score)

        self.assertIn('slice:', rendered)
        self.assertIn('position_size=1', rendered)
        self.assertIn('cells not shown', rendered)

    def test_display_axes_prefer_the_longest_parameters(self):
        results = []
        for short_window in range(5, 10):
            for long_window in range(20, 41, 5):
                for size in (0.5, 1.0):
                    results.append(make_result(
                        {'short_window': short_window, 'long_window': long_window,
                         'position_size': size}))

        surface = plateau.build_surface(results)

        self.assertEqual(plateau.choose_display_axes(surface),
                         ('short_window', 'long_window'))

    def test_heatmap_of_a_single_parameter_surface(self):
        results = [make_result({'short_window': window}, total_return=float(window))
                   for window in range(5, 10)]

        surface = plateau.build_surface(results)
        rendered = plateau.render_heatmap(surface, plateau.analyse_plateau(surface))

        self.assertIn('single row', rendered)

    def test_heatmap_rejects_an_unknown_parameter(self):
        surface = plateau.build_surface(plateau_grid())

        with self.assertRaises(plateau.PlateauError):
            plateau.render_heatmap(surface, x_parameter='nope')

    def test_distribution_block_states_the_best_of_n_reading(self):
        results = [make_result({'short_window': window}, total_return=float(window),
                               benchmark_return_pct=95.0)
                   for window in range(0, 100)]
        surface = plateau.build_surface(results)

        rendered = plateau.render_distribution(
            plateau.summarize_distribution(surface, results), surface)

        self.assertIn('maximum of', rendered)
        self.assertIn('beating the baseline', rendered)
        self.assertIn('by chance', rendered)

    def test_distribution_block_reports_every_cell_state(self):
        results = [make_result({'short_window': 1}, total_return=5.0),
                   make_result({'short_window': 2}, total_trades=0),
                   make_result({'short_window': 3}, total_return=float('nan'))]
        surface = plateau.build_surface(results)

        rendered = plateau.render_distribution(
            plateau.summarize_distribution(surface, results), surface)

        self.assertIn('never traded', rendered)
        self.assertIn('non-finite', rendered)
        self.assertIn('no result', rendered)

    def test_truncated_distribution_block_refuses_to_print_quartiles(self):
        results = [make_result({'short_window': window}, total_return=float(window))
                   for window in range(0, 10)]
        surface = plateau.build_surface(results, selection=plateau.SELECTION_TRUNCATED)

        rendered = plateau.render_distribution(
            plateau.summarize_distribution(surface, results), surface)

        self.assertIn('NO DISTRIBUTION REPORTED', rendered)
        self.assertNotIn('median', rendered)

    def test_plateau_block_reports_neighbour_coverage(self):
        surface = plateau.build_surface(plateau_grid())
        rendered = plateau.render_plateau(plateau.analyse_plateau(surface), surface)

        self.assertIn('plateau score', rendered)
        self.assertIn('computed from 8 of 8', rendered)

    def test_plateau_centre_is_opt_in_and_labelled(self):
        surface = plateau.build_surface(plateau_grid())
        score = plateau.analyse_plateau(surface)

        without = plateau.render_plateau(score, surface)
        with_centre = plateau.render_plateau(score, surface, show_centre=True)

        self.assertNotIn('plateau centre', without)
        self.assertIn('plateau centre', with_centre)
        self.assertIn("NOT the optimizer's winner", with_centre)

    def test_plateau_block_without_a_score(self):
        surface = plateau.build_surface([make_result({'a': 1}, total_trades=0)])

        rendered = plateau.render_plateau(None, surface)

        self.assertIn('no surface to read', rendered)

    def test_render_report_includes_the_heatmap_only_on_request(self):
        report = plateau.analyse_results(plateau_grid())

        self.assertNotIn('PARAMETER SURFACE', plateau.render_report(report))
        self.assertIn('PARAMETER SURFACE',
                      plateau.render_report(report, show_heatmap=True))


class TestSurfaceCsv(unittest.TestCase):
    """The machine-readable surface."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, 'surface.csv')

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        os.rmdir(self.directory)

    def _rows(self):
        with open(self.path, newline='') as handle:
            return list(csv.DictReader(handle))

    def test_every_cell_is_written(self):
        surface = plateau.build_surface(plateau_grid())

        written = plateau.write_surface_csv(surface, self.path)

        self.assertEqual(written, surface.size)
        self.assertEqual(len(self._rows()), surface.size)

    def test_missing_cells_are_blank_not_zero(self):
        results = [result for result in plateau_grid()
                   if result.parameters != {'short_window': 7, 'long_window': 25}]
        surface = plateau.build_surface(results)

        plateau.write_surface_csv(surface, self.path)
        row = next(row for row in self._rows()
                   if row['short_window'] == '7' and row['long_window'] == '25')

        self.assertEqual(row['status'], plateau.CELL_MISSING)
        self.assertEqual(row['metric_value'], '')
        self.assertEqual(row['total_return_pct'], '')

    def test_non_finite_values_are_blank(self):
        results = [make_result({'short_window': 1}, total_return=float('inf'))]
        surface = plateau.build_surface(results)

        plateau.write_surface_csv(surface, self.path)
        row = self._rows()[0]

        self.assertEqual(row['status'], plateau.CELL_NON_FINITE)
        self.assertEqual(row['metric_value'], '')

    def test_winner_and_plateau_flags(self):
        surface = plateau.build_surface(plateau_grid())
        score = plateau.analyse_plateau(surface)

        plateau.write_surface_csv(surface, self.path, score)
        rows = self._rows()

        winners = [row for row in rows if row['is_winner'] == 'True']
        self.assertEqual(len(winners), 1)
        self.assertEqual(int(winners[0]['short_window']), 10)
        self.assertTrue(any(row['within_plateau_band'] == 'True' for row in rows))
        self.assertEqual(
            sum(1 for row in rows if row['is_plateau_centre'] == 'True'), 1)


class TestAnalyseResults(unittest.TestCase):
    """The one-call entry point."""

    def test_returns_surface_distribution_and_plateau(self):
        report = plateau.analyse_results(plateau_grid())

        self.assertEqual(report.surface.metric, 'total_return')
        self.assertEqual(report.distribution.count, report.surface.size)
        self.assertIsNotNone(report.plateau)

    def test_metric_is_threaded_through(self):
        report = plateau.analyse_results(plateau_grid(), metric='sharpe_ratio')

        self.assertEqual(report.surface.metric, 'sharpe_ratio')
        self.assertEqual(report.distribution.metric, 'sharpe_ratio')


if __name__ == '__main__':
    unittest.main()
