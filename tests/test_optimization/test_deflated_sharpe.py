"""
Unit tests for the deflated Sharpe ratio of a search's winner.

The reference values are the worked example in Bailey & Lopez de Prado, "The
Deflated Sharpe Ratio" (2014): an annualised Sharpe of 2.5 over 1250 daily
observations, selected from N=100 trials whose annualised Sharpe ratios have
variance 1/2, with skewness -3 and kurtosis 10. The paper reports an expected
maximum of 0.1132 per day and a deflated Sharpe of 0.9004.

The property test is the reason the module exists: the best of many zero-edge
trials must come out near chance, where the uncorrected reading calls it a
near-certainty.
"""

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.optimization import deflated_sharpe as ds
from niffler.optimization import plateau
from niffler.optimization.grid_search_optimizer import GridSearchOptimizer
from niffler.optimization.parameter_space import ParameterSpace
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy

PAPER_TRIALS = 100
PAPER_TRIAL_STD = math.sqrt(0.5 / 250)
PAPER_SHARPE = 2.5 / math.sqrt(250)
PAPER_OBSERVATIONS = 1250
PAPER_SKEWNESS = -3.0
PAPER_KURTOSIS = 10.0


def equity_from_returns(returns):
    """An equity curve whose per-bar returns are exactly ``returns``."""
    values = [100.0]
    for value in returns:
        values.append(values[-1] * (1.0 + value))
    return pd.Series(values, index=pd.date_range('2024-01-01', periods=len(values), freq='D'))


def result_with(returns):
    """A stand-in optimisation result carrying one equity curve."""
    return SimpleNamespace(
        parameters={},
        backtest_result=SimpleNamespace(portfolio_values=equity_from_returns(returns)))


def moments(sharpe, observations=500, skewness=0.0, kurtosis=3.0):
    return ds.ReturnMoments(sharpe=sharpe, skewness=skewness, kurtosis=kurtosis,
                            observations=observations)


class TestExpectedMaxSharpe(unittest.TestCase):
    """The luck line."""

    def test_matches_the_published_example(self):
        self.assertAlmostEqual(
            ds.expected_max_sharpe(PAPER_TRIALS, PAPER_TRIAL_STD), 0.1132, places=4)

    def test_one_trial_selects_nothing(self):
        self.assertEqual(ds.expected_max_sharpe(1, 0.3), 0.0)

    def test_more_trials_raise_the_line(self):
        lines = [ds.expected_max_sharpe(n, 0.1) for n in (2, 10, 100, 1000)]
        self.assertEqual(lines, sorted(lines))
        self.assertGreater(lines[0], 0.0)

    def test_scales_with_the_spread(self):
        self.assertAlmostEqual(ds.expected_max_sharpe(50, 0.4),
                               2 * ds.expected_max_sharpe(50, 0.2))

    def test_fewer_than_one_trial_is_an_error(self):
        for trials in (0, 0.5, -3, float('nan')):
            with self.subTest(trials=trials):
                with self.assertRaises(ValueError):
                    ds.expected_max_sharpe(trials, 0.1)

    def test_a_negative_spread_is_an_error(self):
        with self.assertRaises(ValueError):
            ds.expected_max_sharpe(10, -0.1)


class TestProbabilisticSharpe(unittest.TestCase):
    """The probability that a true Sharpe is above a line."""

    def test_matches_the_published_example(self):
        luck_line = ds.expected_max_sharpe(PAPER_TRIALS, PAPER_TRIAL_STD)
        probability = ds.probabilistic_sharpe(
            PAPER_SHARPE, luck_line, PAPER_OBSERVATIONS, PAPER_SKEWNESS, PAPER_KURTOSIS)
        self.assertAlmostEqual(probability, 0.9004, places=4)

    def test_a_sharpe_on_the_line_is_a_coin_flip(self):
        self.assertAlmostEqual(ds.probabilistic_sharpe(0.1, 0.1, 400, 0.0, 3.0), 0.5)

    def test_hand_computed_for_normal_returns(self):
        # z = 0.1 * sqrt(100) / sqrt(1 + 0.5 * 0.01) = 0.997509, and
        # Phi(z) = Phi(1) - 0.002491 * pdf(1) = 0.841345 - 0.000603.
        probability = ds.probabilistic_sharpe(0.1, 0.0, 101, 0.0, 3.0)
        self.assertAlmostEqual(probability, 0.84074, places=5)

    def test_negative_skew_and_fat_tails_lower_it(self):
        clean = ds.probabilistic_sharpe(0.1, 0.0, 500, 0.0, 3.0)
        ugly = ds.probabilistic_sharpe(0.1, 0.0, 500, -2.0, 12.0)
        self.assertLess(ugly, clean)

    def test_fewer_than_two_observations_is_undefined(self):
        self.assertIsNone(ds.probabilistic_sharpe(0.1, 0.0, 1, 0.0, 3.0))

    def test_a_non_positive_estimator_variance_is_undefined(self):
        # 1 - 50 * 0.5 is negative: no square root to take.
        self.assertIsNone(ds.probabilistic_sharpe(0.5, 0.0, 500, 50.0, 1.0))


class TestReturnMoments(unittest.TestCase):

    def test_hand_computed(self):
        # mean 0.005, sample std 0.01, m2 7.5e-5, m3 7.5e-7, m4 1.3125e-8
        result = ds.return_moments([0.02, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(result.sharpe, 0.5)
        self.assertAlmostEqual(result.skewness, 2 / math.sqrt(3))
        self.assertAlmostEqual(result.kurtosis, 7 / 3)
        self.assertEqual(result.observations, 4)

    def test_a_flat_series_has_no_sharpe(self):
        self.assertIsNone(ds.return_moments([0.0, 0.0, 0.0]))

    def test_one_return_is_not_enough(self):
        self.assertIsNone(ds.return_moments([0.01]))

    def test_matches_the_engine_sharpe_convention(self):
        from niffler.backtesting import metrics

        returns = pd.Series([0.01, -0.02, 0.015, 0.003, -0.004])
        self.assertAlmostEqual(ds.return_moments(returns).sharpe,
                               metrics.sharpe_ratio_of_returns(returns, 1.0))


class TestDeflatedSharpe(unittest.TestCase):
    """The winner against the search."""

    TRIALS = [0.02, 0.05, 0.08, 0.03, 0.07]

    def test_an_ordinary_search_reports_both_luck_lines(self):
        result = ds.deflated_sharpe(moments(0.08), self.TRIALS, trials_evaluated=5)

        spread = float(np.std(self.TRIALS, ddof=1))
        line = ds.expected_max_sharpe(5, spread)
        self.assertEqual(result.status, ds.STATUS_OK)
        self.assertEqual(result.trials, 5.0)
        self.assertEqual(result.trials_source, ds.TRIALS_EVALUATED)
        self.assertAlmostEqual(result.trial_sharpe_std, spread)
        self.assertAlmostEqual(result.expected_max_sharpe, line)
        self.assertAlmostEqual(result.grid_relative_luck_line, 0.05 + line)
        self.assertAlmostEqual(
            result.probability, ds.probabilistic_sharpe(0.08, line, 500, 0.0, 3.0))
        self.assertAlmostEqual(
            result.grid_relative_probability,
            ds.probabilistic_sharpe(0.08, 0.05 + line, 500, 0.0, 3.0))

    def test_a_grid_above_zero_is_judged_more_strictly_against_itself(self):
        result = ds.deflated_sharpe(moments(0.08), self.TRIALS, trials_evaluated=5)
        self.assertLess(result.grid_relative_probability, result.probability)

    def test_the_count_is_every_evaluated_combination_not_only_the_scored_ones(self):
        scored = ds.deflated_sharpe(moments(0.08), self.TRIALS, trials_evaluated=5)
        with_blanks = ds.deflated_sharpe(moments(0.08), self.TRIALS, trials_evaluated=50)

        self.assertEqual(with_blanks.trials, 50.0)
        self.assertEqual(with_blanks.trials_with_sharpe, 5)
        self.assertGreater(with_blanks.expected_max_sharpe, scored.expected_max_sharpe)

    def test_effective_trials_overrides_the_count_and_says_so(self):
        raw = ds.deflated_sharpe(moments(0.08), self.TRIALS, trials_evaluated=5)
        fewer = ds.deflated_sharpe(moments(0.08), self.TRIALS, trials_evaluated=5,
                                   effective_trials=2)

        self.assertEqual(fewer.trials, 2.0)
        self.assertEqual(fewer.trials_source, ds.TRIALS_OVERRIDE)
        self.assertEqual(fewer.trials_evaluated, 5)
        self.assertLess(fewer.expected_max_sharpe, raw.expected_max_sharpe)
        self.assertGreater(fewer.probability, raw.probability)

    def test_effective_trials_below_one_is_an_error(self):
        for value in (0, 0.9, -1, float('nan')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ds.deflated_sharpe(moments(0.08), self.TRIALS, 5, effective_trials=value)

    def test_one_scored_trial_has_no_spread(self):
        result = ds.deflated_sharpe(moments(0.08), [0.08], trials_evaluated=1)

        self.assertEqual(result.status, ds.STATUS_TOO_FEW_TRIALS)
        self.assertIsNone(result.probability)
        self.assertIsNone(result.expected_max_sharpe)
        self.assertIsNone(result.trials)

    def test_identical_trials_have_no_spread(self):
        result = ds.deflated_sharpe(moments(0.08), [0.08, 0.08, 0.08], trials_evaluated=3)

        self.assertEqual(result.status, ds.STATUS_NO_DISPERSION)
        self.assertIsNone(result.probability)
        self.assertIsNone(result.grid_relative_probability)

    def test_extreme_moments_report_the_lines_but_no_probability(self):
        winner = moments(0.5, skewness=50.0, kurtosis=1.0)
        result = ds.deflated_sharpe(winner, self.TRIALS, trials_evaluated=5)

        self.assertEqual(result.status, ds.STATUS_UNDEFINED)
        self.assertIsNone(result.probability)
        self.assertIsNotNone(result.expected_max_sharpe)

    def test_annualisation_plays_no_part_in_the_probability(self):
        daily = ds.deflated_sharpe(moments(0.08), self.TRIALS, 5, periods_per_year=252)
        hourly = ds.deflated_sharpe(moments(0.08), self.TRIALS, 5, periods_per_year=8760)

        self.assertEqual(daily.probability, hourly.probability)
        self.assertAlmostEqual(daily.annualised(0.08), 0.08 * math.sqrt(252))
        self.assertAlmostEqual(hourly.annualised(0.08), 0.08 * math.sqrt(8760))

    def test_no_annualisation_factor_means_no_annualised_figure(self):
        result = ds.deflated_sharpe(moments(0.08), self.TRIALS, 5)
        self.assertIsNone(result.annualised(0.08))


class TestPureNoise(unittest.TestCase):
    """The best of many zero-edge trials is not evidence of an edge."""

    TRIALS = 200
    BARS = 500
    SEEDS = 40

    def _search(self, seed, edge=0.0):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0, 0.01, size=(self.TRIALS, self.BARS))
        returns[0] += edge
        trial_moments = [ds.return_moments(row) for row in returns]
        sharpes = [moment.sharpe for moment in trial_moments]
        winner = trial_moments[int(np.argmax(sharpes))]
        return winner, sharpes

    def test_the_winner_of_a_noise_search_is_near_chance(self):
        deflated, uncorrected = [], []
        for seed in range(self.SEEDS):
            winner, sharpes = self._search(seed)
            deflated.append(ds.deflated_sharpe(winner, sharpes, self.TRIALS).probability)
            uncorrected.append(ds.probabilistic_sharpe(
                winner.sharpe, 0.0, winner.observations, winner.skewness, winner.kurtosis))

        # Without the correction every one of these winners looks like a find.
        self.assertGreater(min(uncorrected), 0.95)
        self.assertGreater(np.mean(deflated), 0.35)
        self.assertLess(np.mean(deflated), 0.65)
        self.assertLess(np.mean(np.array(deflated) >= ds.DEFAULT_CONFIDENCE), 0.10)

    def test_a_real_edge_among_the_noise_still_clears_the_line(self):
        # 0.003 a bar on 0.01 volatility: a per-bar Sharpe near 0.3.
        winner, sharpes = self._search(seed=1, edge=0.003)
        result = ds.deflated_sharpe(winner, sharpes, self.TRIALS)

        self.assertGreater(result.probability, 0.99)
        self.assertGreater(result.grid_relative_probability, 0.99)


class TestAnalyseResults(unittest.TestCase):
    """Reading the optimizer's own result list."""

    def results(self):
        rng = np.random.default_rng(3)
        return [result_with(rng.normal(mean, 0.01, 300))
                for mean in (0.002, 0.001, 0.0005, 0.0)]

    def test_the_winner_is_the_first_result(self):
        results = self.results()
        analysis = ds.analyse_results(results, plateau.SELECTION_EXHAUSTIVE)

        first = ds.return_moments(results[0].backtest_result.portfolio_values.pct_change().dropna())
        self.assertEqual(analysis.status, ds.STATUS_OK)
        self.assertAlmostEqual(analysis.winner.sharpe, first.sharpe)
        self.assertEqual(analysis.winner.observations, 300)
        self.assertEqual(analysis.trials, 4.0)

    def test_a_truncated_result_set_is_refused(self):
        analysis = ds.analyse_results(self.results(), plateau.SELECTION_TRUNCATED)

        self.assertEqual(analysis.status, ds.STATUS_TRUNCATED)
        for field in ('trials', 'trial_sharpe_std', 'expected_max_sharpe',
                      'grid_relative_luck_line', 'probability', 'grid_relative_probability',
                      'winner'):
            with self.subTest(field=field):
                self.assertIsNone(getattr(analysis, field))

    def test_a_sampled_search_is_not_truncated(self):
        analysis = ds.analyse_results(self.results(), plateau.SELECTION_SAMPLED)
        self.assertEqual(analysis.status, ds.STATUS_OK)

    def test_a_trial_that_never_traded_counts_as_a_trial_but_has_no_sharpe(self):
        results = self.results() + [result_with([0.0] * 300)]
        analysis = ds.analyse_results(results, plateau.SELECTION_EXHAUSTIVE)

        self.assertEqual(analysis.trials, 5.0)
        self.assertEqual(analysis.trials_evaluated, 5)
        self.assertEqual(analysis.trials_with_sharpe, 4)

    def test_a_winner_that_never_traded_has_nothing_to_deflate(self):
        results = [result_with([0.0] * 300)] + self.results()
        analysis = ds.analyse_results(results, plateau.SELECTION_EXHAUSTIVE)

        self.assertEqual(analysis.status, ds.STATUS_NO_RETURNS)
        self.assertIsNone(analysis.probability)

    def test_effective_trials_below_one_is_an_error_even_when_truncated(self):
        with self.assertRaises(ValueError):
            ds.analyse_results(self.results(), plateau.SELECTION_TRUNCATED,
                               effective_trials=0.5)


class TestRealSearch(unittest.TestCase):
    """The optimizer keeps what the analysis needs, at any worker count."""

    def _search(self, n_jobs):
        index = pd.date_range('2024-01-01', periods=120, freq='D')
        closes = [100.0 + (i % 17) * 1.5 - (i % 7) * 2.0 + i * 0.3 for i in range(120)]
        data = pd.DataFrame({
            'open': closes,
            'high': [close * 1.01 for close in closes],
            'low': [close * 0.99 for close in closes],
            'close': closes,
            'volume': [10_000.0] * 120,
        }, index=index)
        optimizer = GridSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=ParameterSpace({
                'short_window': {'type': 'int', 'min': 3, 'max': 8, 'step': 1},
            }),
            data=data,
            n_jobs=n_jobs,
        )
        return optimizer.optimize()

    def test_equity_curves_survive_the_search(self):
        results = self._search(n_jobs=1)
        analysis = ds.analyse_results(results, plateau.SELECTION_EXHAUSTIVE)

        self.assertEqual(analysis.trials_evaluated, len(results))
        self.assertGreaterEqual(analysis.trials_with_sharpe, 2)
        self.assertIsNotNone(analysis.winner)

    def test_a_parallel_search_gives_the_same_answer(self):
        sequential = ds.analyse_results(self._search(n_jobs=1), plateau.SELECTION_EXHAUSTIVE)
        parallel = ds.analyse_results(self._search(n_jobs=2), plateau.SELECTION_EXHAUSTIVE)

        self.assertEqual(parallel, sequential)


class TestSummaryFields(unittest.TestCase):
    """What the exported run summary carries."""

    def test_no_analysis_is_all_none(self):
        fields = ds.summary_fields(None)

        self.assertEqual(tuple(fields), ds.SUMMARY_FIELDS)
        self.assertTrue(all(value is None for value in fields.values()))

    def test_sharpe_figures_are_annualised_and_probabilities_are_not(self):
        result = ds.deflated_sharpe(moments(0.08), [0.02, 0.05, 0.08], 3,
                                    periods_per_year=365)
        fields = ds.summary_fields(result)

        self.assertEqual(tuple(fields), ds.SUMMARY_FIELDS)
        self.assertEqual(fields['deflated_sharpe'], result.probability)
        self.assertEqual(fields['grid_relative_probability'], result.grid_relative_probability)
        self.assertEqual(fields['search_luck_status'], ds.STATUS_OK)
        self.assertEqual(fields['search_luck_trials'], 3.0)
        self.assertEqual(fields['search_luck_trials_source'], ds.TRIALS_EVALUATED)
        self.assertAlmostEqual(fields['expected_max_sharpe'],
                               result.expected_max_sharpe * math.sqrt(365))
        self.assertAlmostEqual(fields['trial_sharpe_std'],
                               result.trial_sharpe_std * math.sqrt(365))

    def test_a_refused_analysis_exports_its_status_and_nothing_else(self):
        result = ds.analyse_results([], plateau.SELECTION_TRUNCATED, periods_per_year=365)
        fields = ds.summary_fields(result)

        self.assertEqual(fields.pop('search_luck_status'), ds.STATUS_TRUNCATED)
        self.assertTrue(all(value is None for value in fields.values()), fields)

    def test_every_field_is_mapped_in_the_runs_index(self):
        mapping = json.loads(
            (project_root / 'config/elasticsearch/mappings/runs.json').read_text())
        properties = mapping['mappings']['properties']

        for field in ds.SUMMARY_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, properties)


class TestRenderReport(unittest.TestCase):

    def ok(self, winner_sharpe=0.08, trials=(0.02, 0.05, 0.08, 0.03, 0.07), **kwargs):
        return ds.deflated_sharpe(moments(winner_sharpe), list(trials), len(trials),
                                  periods_per_year=365, **kwargs)

    def test_a_refused_analysis_says_why_and_prints_no_number(self):
        text = ds.render_report(ds.analyse_results([], plateau.SELECTION_TRUNCATED))

        self.assertIn('NOT COMPUTED', text)
        self.assertIn('score-biased', text)
        self.assertNotIn('%', text)
        self.assertNotIn('winner:', text)

    def test_both_luck_lines_and_probabilities_are_shown(self):
        result = self.ok()
        text = ds.render_report(result)

        self.assertIn(f'{result.annualised(result.expected_max_sharpe):.3f} annualised', text)
        self.assertIn(f'{result.annualised(result.grid_relative_luck_line):.3f} annualised',
                      text)
        self.assertIn(f'{result.probability * 100:.1f}%', text)
        self.assertIn(f'{result.grid_relative_probability * 100:.1f}%', text)

    def test_the_default_count_is_explained(self):
        text = ds.render_report(self.ok())

        self.assertIn('trials counted: 5 - every combination evaluated', text)
        self.assertIn('over-corrects', text)

    def test_an_overridden_count_names_the_flag_and_the_real_count(self):
        text = ds.render_report(self.ok(effective_trials=2))

        self.assertIn('trials counted: 2 (--effective-trials; 5 combinations were evaluated)',
                      text)

    def test_the_grid_relative_figure_leads_and_the_published_one_follows(self):
        text = ds.render_report(self.ok())

        self.assertLess(text.index('GRID-RELATIVE'), text.index('DEFLATED SHARPE RATIO'))
        self.assertTrue(text.startswith(ds.BLOCK_TITLE))

    def test_only_the_published_figure_is_called_deflated(self):
        text = ds.render_report(self.ok())

        self.assertEqual(text.count('DEFLATED'), 1)
        self.assertIn('Bailey & Lopez de Prado', text)
        for field in ('grid_relative_probability', 'grid_relative_luck_line'):
            self.assertIn(field, ds.SUMMARY_FIELDS)
            self.assertNotIn('deflated', field)

    def test_a_winner_above_its_grid_clears_the_bar(self):
        text = ds.render_report(self.ok(winner_sharpe=0.6))
        self.assertIn('This clears the conventional 95% bar', text)

    def test_the_verdict_follows_the_grid_relative_figure_not_the_published_one(self):
        trials = (0.30, 0.31, 0.32, 0.33, 0.34)
        result = self.ok(winner_sharpe=0.34, trials=trials)
        self.assertGreaterEqual(result.probability, 0.95)
        self.assertLess(result.grid_relative_probability, 0.95)

        text = ds.render_report(result)
        self.assertIn('does NOT clear the conventional 95% bar', text)
        self.assertIn('no evidence for these particular parameters', text)
        self.assertNotIn('This clears', text)

    def test_a_winner_below_the_line_does_not_clear_it(self):
        text = ds.render_report(self.ok(winner_sharpe=0.01))
        self.assertIn('does NOT clear the conventional 95% bar', text)

    def test_trials_that_never_traded_are_said_to_count_without_a_sharpe(self):
        result = ds.deflated_sharpe(moments(0.08), [0.02, 0.05, 0.08], 7,
                                    periods_per_year=365)
        text = ds.render_report(result)

        self.assertIn('trials counted: 7', text)
        self.assertIn('4 of them never traded: they count as trials but have no Sharpe', text)

    def test_a_grid_where_everything_traded_says_nothing_about_blanks(self):
        self.assertNotIn('never traded', ds.render_report(self.ok()))

    def test_per_bar_figures_are_labelled_as_such_without_an_annualisation(self):
        result = ds.deflated_sharpe(moments(0.08), [0.02, 0.05, 0.08], 3)
        text = ds.render_report(result)

        self.assertIn('per bar', text)
        self.assertNotIn('annualised', text)

    def test_extreme_moments_print_no_probability(self):
        winner = moments(0.5, skewness=50.0, kurtosis=1.0)
        result = ds.deflated_sharpe(winner, [0.02, 0.05, 0.08], 3)
        text = ds.render_report(result)

        self.assertIn('NO PROBABILITY', text)
        self.assertNotIn('%', text)


if __name__ == '__main__':
    unittest.main()
