import logging
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from niffler.analysis.monte_carlo_analyzer import MonteCarloAnalyzer, MIN_SAMPLE_BARS
from niffler.strategies.base_strategy import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, short_window=10, long_window=20):
        super().__init__("MockStrategy")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        signals = data.copy()
        signals['signal'] = 0
        signals['position_size'] = 1.0
        if len(signals) > self.short_window:
            short_ma = signals['close'].rolling(self.short_window).mean()
            long_ma = signals['close'].rolling(self.long_window).mean()
            signals.loc[short_ma > long_ma, 'signal'] = 1
            signals.loc[short_ma < long_ma, 'signal'] = -1
        return signals

    def get_description(self):
        return f"Mock strategy with short_window={self.short_window}, long_window={self.long_window}"


class ScriptedRNG:
    """Minimal numpy-Generator stand-in that yields scripted block start positions."""

    def __init__(self, starts):
        self._starts = list(starts)
        self._calls = 0

    def integers(self, low, high):
        value = self._starts[self._calls % len(self._starts)]
        self._calls += 1
        return value


def make_price_data(periods=365, seed=42):
    """Build a deterministic OHLCV frame with valid OHLC relationships."""
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0008, 0.02, periods)
    prices = 100 * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'open': prices * (1 + rng.normal(0, 0.001, periods)),
        'high': prices * (1 + np.abs(rng.normal(0.002, 0.005, periods))),
        'low': prices * (1 - np.abs(rng.normal(0.002, 0.005, periods))),
        'close': prices,
        'volume': rng.integers(1000, 10000, periods)
    }, index=dates)

    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    return data


class MonteCarloTestBase(unittest.TestCase):
    """Shared fixtures."""

    def setUp(self):
        # Other test modules disable logging globally at import time, which would make
        # every assertLogs assertion here vacuously fail. Re-enable for the duration of
        # each test and restore whatever level was in force afterwards.
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, previous_disable_level)

        self.test_data = make_price_data()
        self.optimal_parameters = {'short_window': 10, 'long_window': 20}
        self.strategy_class = MockStrategy

    def make_analyzer(self, **overrides):
        kwargs = {
            'strategy_class': self.strategy_class,
            'optimal_parameters': self.optimal_parameters,
            'n_simulations': 5,
            'n_jobs': 1,
        }
        kwargs.update(overrides)
        return MonteCarloAnalyzer(**kwargs)


class TestMonteCarloConfiguration(MonteCarloTestBase):
    """Initialisation and validation."""

    def test_init_valid_parameters(self):
        analyzer = self.make_analyzer(n_simulations=100, bootstrap_sample_pct=0.8,
                                      block_size_days=30, initial_capital=10000,
                                      commission=0.001, random_seed=11)

        self.assertEqual(analyzer.strategy_class, self.strategy_class)
        self.assertEqual(analyzer.n_simulations, 100)
        self.assertEqual(analyzer.bootstrap_sample_pct, 0.8)
        self.assertEqual(analyzer.block_size_days, 30)
        self.assertEqual(analyzer.random_seed, 11)

    def test_init_invalid_parameters(self):
        for kwargs in (
            {'optimal_parameters': {}},
            {'n_simulations': -1},
            {'bootstrap_sample_pct': 1.5},
            {'block_size_days': -1},
            {'initial_capital': -1000},
            {'commission': -0.001},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    self.make_analyzer(**kwargs)

    def test_strategy_parameter_validation(self):
        self.make_analyzer()
        with self.assertRaises(ValueError):
            self.make_analyzer(optimal_parameters={'invalid_param': 'invalid_value'})

    def test_analyze_insufficient_data(self):
        analyzer = self.make_analyzer(n_simulations=10)

        with self.assertRaises(ValueError):
            analyzer.analyze(pd.DataFrame())

        with self.assertRaises(ValueError):
            analyzer.analyze(self.test_data.head(50))

    def test_analyze_invalid_data_format(self):
        analyzer = self.make_analyzer(n_simulations=10)
        with self.assertRaises(ValueError):
            analyzer.analyze(self.test_data.reset_index())

    def test_block_size_adjustment(self):
        analyzer = self.make_analyzer(block_size_days=400, n_simulations=1)
        small_data = self.test_data.head(200)

        with patch.object(analyzer, '_run_simulations_sequential') as mock_run:
            mock_run.return_value = ([self._mock_result()], 0)
            with self.assertLogs(level=logging.WARNING):
                analyzer.analyze(small_data, "TEST")

        self.assertLess(analyzer.block_size_days, 200)

    def _mock_result(self):
        result = Mock()
        result.total_return = 1000
        result.total_return_pct = 10
        result.sharpe_ratio = 1.5
        result.max_drawdown = -5.0
        result.win_rate = 65.0
        result.total_trades = 20
        return result


class TestBlockBootstrap(MonteCarloTestBase):
    """The bootstrap must produce alternative price PATHS, not reordered history."""

    def test_sample_has_a_fresh_monotonic_index_without_duplicates(self):
        analyzer = self.make_analyzer(block_size_days=30, bootstrap_sample_pct=0.5)
        sample = analyzer._block_bootstrap_sample(self.test_data,
                                                  rng=np.random.default_rng(1))

        self.assertIsInstance(sample.index, pd.DatetimeIndex)
        self.assertTrue(sample.index.is_monotonic_increasing)
        self.assertFalse(sample.index.has_duplicates)
        # Evenly spaced: no gaps left behind by the resampling.
        self.assertEqual(len(set(np.diff(sample.index.to_numpy()))), 1)
        self.assertListEqual(list(sample.columns), list(self.test_data.columns))

    def test_sample_size_follows_bootstrap_pct(self):
        analyzer = self.make_analyzer(block_size_days=30, bootstrap_sample_pct=0.5)
        sample = analyzer._block_bootstrap_sample(self.test_data,
                                                  rng=np.random.default_rng(1))
        self.assertEqual(len(sample), int(len(self.test_data) * 0.5))

    def test_sample_is_not_a_chronological_subset_of_history(self):
        """
        The old implementation sorted the drawn blocks back into chronological order,
        which produced a slice of real history rather than a resampled path.
        """
        analyzer = self.make_analyzer(block_size_days=20, bootstrap_sample_pct=0.8)
        sample = analyzer._block_bootstrap_sample(self.test_data,
                                                  rng=np.random.default_rng(3))

        original_closes = set(np.round(self.test_data['close'].to_numpy(), 8))
        synthetic_closes = np.round(sample['close'].to_numpy()[1:], 8)
        overlap = sum(1 for value in synthetic_closes if value in original_closes)

        # A reconstructed path essentially never lands back on the historical levels.
        self.assertLess(overlap, len(synthetic_closes) * 0.05)

    def test_path_starts_at_the_real_starting_price(self):
        analyzer = self.make_analyzer(block_size_days=20)
        sample = analyzer._block_bootstrap_sample(self.test_data,
                                                  rng=np.random.default_rng(5))
        self.assertAlmostEqual(sample['close'].iloc[0], self.test_data['close'].iloc[0])

    def test_no_artificial_gap_returns_from_glued_price_levels(self):
        """
        Concatenating raw price LEVELS created huge fake jumps between blocks. Compounding
        returns instead means every synthetic return is a real historical return.
        """
        analyzer = self.make_analyzer(block_size_days=20, bootstrap_sample_pct=0.9)
        sample = analyzer._block_bootstrap_sample(self.test_data,
                                                  rng=np.random.default_rng(9))

        original_returns = self.test_data['close'].pct_change().dropna()
        synthetic_returns = sample['close'].pct_change().dropna()

        self.assertLessEqual(synthetic_returns.abs().max(),
                             original_returns.abs().max() * 1.000001)

    def test_blocks_preserve_intra_block_autocorrelation(self):
        """Consecutive returns inside a drawn block must stay consecutive."""
        data = self.test_data.head(40)
        analyzer = self.make_analyzer(block_size_days=5, bootstrap_sample_pct=1.0)

        starts = [10, 0, 25, 5, 30, 15, 20, 2]
        sample = analyzer._block_bootstrap_sample(data, rng=ScriptedRNG(starts))

        returns = data['close'].to_numpy()[1:] / data['close'].to_numpy()[:-1] - 1.0
        expected_idx = np.concatenate([np.arange(s, s + 5) for s in starts])[:len(sample) - 1]
        expected_close = data['close'].iloc[0] * np.cumprod(1 + returns[expected_idx])

        np.testing.assert_allclose(sample['close'].to_numpy()[1:], expected_close)

    def test_blocks_are_not_resorted_into_chronological_order(self):
        """The drawn order IS the resampling; re-sorting would neutralise it."""
        data = self.test_data.head(40)
        analyzer = self.make_analyzer(block_size_days=5, bootstrap_sample_pct=1.0)

        later_first = analyzer._block_bootstrap_sample(data, rng=ScriptedRNG([25, 0] * 8))
        earlier_first = analyzer._block_bootstrap_sample(data, rng=ScriptedRNG([0, 25] * 8))

        self.assertFalse(np.allclose(later_first['close'].to_numpy(),
                                     earlier_first['close'].to_numpy()))

    def test_ohlc_relationships_are_preserved(self):
        analyzer = self.make_analyzer(block_size_days=15, bootstrap_sample_pct=0.9)

        for seed in range(5):
            with self.subTest(seed=seed):
                sample = analyzer._block_bootstrap_sample(self.test_data,
                                                          rng=np.random.default_rng(seed))
                self.assertFalse((sample['high'] < sample['low']).any())
                self.assertFalse((sample['high'] < sample['open']).any())
                self.assertFalse((sample['high'] < sample['close']).any())
                self.assertFalse((sample['low'] > sample['open']).any())
                self.assertFalse((sample['low'] > sample['close']).any())
                self.assertTrue((sample[['open', 'high', 'low', 'close']] > 0).all().all())

    def test_sample_is_backtestable_by_the_engine(self):
        """The engine validates OHLC, index ordering and positivity - the path must pass."""
        from niffler.backtesting.backtest_engine import BacktestEngine

        analyzer = self.make_analyzer(block_size_days=20, bootstrap_sample_pct=0.8)
        sample = analyzer._block_bootstrap_sample(self.test_data,
                                                  rng=np.random.default_rng(4))

        engine = BacktestEngine(initial_capital=10000, commission=0.001)
        result = engine.run_backtest(MockStrategy(), sample, "TEST")
        self.assertEqual(len(result.portfolio_values), len(sample))

    def test_same_rng_seed_gives_identical_sample(self):
        analyzer = self.make_analyzer(block_size_days=20)

        first = analyzer._block_bootstrap_sample(self.test_data, rng=np.random.default_rng(77))
        second = analyzer._block_bootstrap_sample(self.test_data, rng=np.random.default_rng(77))
        other = analyzer._block_bootstrap_sample(self.test_data, rng=np.random.default_rng(78))

        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first['close'].equals(other['close']))

    def test_data_shorter_than_block_size(self):
        analyzer = self.make_analyzer(block_size_days=100)
        small_data = self.test_data.head(50)

        sample = analyzer._block_bootstrap_sample(small_data, rng=np.random.default_rng(2))

        self.assertLessEqual(len(sample), len(small_data))
        self.assertFalse(sample.index.has_duplicates)
        self.assertFalse((sample['high'] < sample['low']).any())


class TestMonteCarloSeeding(MonteCarloTestBase):
    """F5: the documented random_seed must actually reach the workers."""

    def test_simulation_seed_is_derived_deterministically(self):
        analyzer = self.make_analyzer(random_seed=100)
        self.assertEqual(analyzer._simulation_seed(0), 100)
        self.assertEqual(analyzer._simulation_seed(7), 107)

    def test_simulation_seed_is_none_without_base_seed(self):
        analyzer = self.make_analyzer()
        self.assertIsNone(analyzer._simulation_seed(3))

    def test_each_simulation_uses_a_different_seed(self):
        analyzer = self.make_analyzer(random_seed=100)
        seeds = {analyzer._simulation_seed(i) for i in range(10)}
        self.assertEqual(len(seeds), 10)

    def test_parallel_submission_passes_a_per_simulation_seed(self):
        analyzer = self.make_analyzer(n_simulations=3, n_jobs=2, random_seed=500)

        executor = Mock()
        futures = [Mock() for _ in range(3)]
        for future in futures:
            future.result.return_value = Mock()
        executor.submit.side_effect = futures

        with patch('niffler.analysis.monte_carlo_analyzer.ProcessPoolExecutor') as pool:
            pool.return_value.__enter__.return_value = executor
            with patch('niffler.analysis.monte_carlo_analyzer.as_completed',
                       return_value=futures):
                analyzer._run_simulations_parallel(self.test_data, "TEST")

        # The cost model is appended after the seed, so the seed is second to last.
        submitted_seeds = [call.args[-2] for call in executor.submit.call_args_list]
        self.assertEqual(submitted_seeds, [500, 501, 502])

        submitted_cost_models = [call.args[-1] for call in executor.submit.call_args_list]
        self.assertEqual(submitted_cost_models, [analyzer.cost_model] * 3)

    def test_static_worker_is_seeded_by_its_argument(self):
        """
        On spawn platforms the worker inherits nothing, so the seed must arrive as an
        argument. Same seed in, same path out.
        """
        def run(seed):
            return MonteCarloAnalyzer._run_single_simulation_static(
                self.test_data, "TEST", 0, self.strategy_class, self.optimal_parameters,
                0.8, 30, 10000.0, 0.001, seed
            )

        first = run(4242)
        second = run(4242)
        different = run(9999)

        self.assertIsNotNone(first)
        self.assertEqual(first.total_return, second.total_return)
        self.assertNotEqual(first.total_return, different.total_return)
        self.assertEqual(first.metadata['random_seed'], 4242)

    def test_sequential_runs_with_same_seed_are_identical(self):
        first = self.make_analyzer(n_simulations=4, random_seed=321).analyze(self.test_data, "TEST")
        second = self.make_analyzer(n_simulations=4, random_seed=321).analyze(self.test_data, "TEST")
        other = self.make_analyzer(n_simulations=4, random_seed=654).analyze(self.test_data, "TEST")

        self.assertEqual(self._returns(first), self._returns(second))
        self.assertNotEqual(self._returns(first), self._returns(other))

    def test_parallel_runs_with_same_seed_are_identical(self):
        first = self.make_analyzer(n_simulations=4, n_jobs=2, random_seed=321).analyze(self.test_data, "TEST")
        second = self.make_analyzer(n_simulations=4, n_jobs=2, random_seed=321).analyze(self.test_data, "TEST")
        other = self.make_analyzer(n_simulations=4, n_jobs=2, random_seed=654).analyze(self.test_data, "TEST")

        self.assertEqual(self._returns(first), self._returns(second))
        self.assertNotEqual(self._returns(first), self._returns(other))

    def test_parallel_matches_sequential_for_the_same_seed(self):
        sequential = self.make_analyzer(n_simulations=4, n_jobs=1, random_seed=321).analyze(self.test_data, "TEST")
        parallel = self.make_analyzer(n_simulations=4, n_jobs=2, random_seed=321).analyze(self.test_data, "TEST")

        self.assertEqual(self._returns(sequential), self._returns(parallel))

    def test_parallel_results_are_ordered_by_simulation_id(self):
        """Workers finish in arbitrary order; the reported results must not.

        Regression test: results used to be appended in ``as_completed`` order, so a
        seeded parallel run produced a shuffled per-simulation table, aggregates that
        differed in their last floating-point digits between runs, and an arbitrary
        retained subset once memory trimming kicked in.
        """
        result = self.make_analyzer(
            n_simulations=6, n_jobs=3, random_seed=321
        ).analyze(self.test_data, "TEST")

        sim_ids = [r.metadata['simulation_id'] for r in result.individual_results]
        self.assertEqual(sim_ids, sorted(sim_ids))

    def test_parallel_preserves_sequential_ordering(self):
        """The parallel path yields the same results in the same order as sequential."""
        sequential = self.make_analyzer(n_simulations=6, n_jobs=1, random_seed=321).analyze(self.test_data, "TEST")
        parallel = self.make_analyzer(n_simulations=6, n_jobs=3, random_seed=321).analyze(self.test_data, "TEST")

        self.assertEqual(self._ordered_returns(sequential), self._ordered_returns(parallel))

    @staticmethod
    def _returns(result):
        return sorted(round(r.total_return, 8) for r in result.individual_results)

    @staticmethod
    def _ordered_returns(result):
        """Returns in reported order - sensitive to result ordering, unlike _returns."""
        return [round(r.total_return, 8) for r in result.individual_results]


class TestMonteCarloSurvivorship(MonteCarloTestBase):
    """F6: dropped simulations must show up in the reported result."""

    def _mock_result(self):
        result = Mock()
        result.total_return = 100.0
        result.total_return_pct = 1.0
        result.sharpe_ratio = 0.5
        result.max_drawdown = -2.0
        result.win_rate = 55.0
        result.total_trades = 4
        result.metadata = {}
        return result

    def test_failed_simulations_are_reported(self):
        analyzer = self.make_analyzer(n_simulations=10)
        side_effect = [self._mock_result(), self._mock_result()] + [None] * 8

        with patch.object(analyzer, '_run_single_simulation', side_effect=side_effect):
            result = analyzer.analyze(self.test_data, "TEST")

        self.assertEqual(result.successful_runs, 2)
        self.assertEqual(result.attempted_runs, 10)
        self.assertEqual(result.failed_runs, 8)
        self.assertAlmostEqual(result.failure_rate, 0.8)
        self.assertTrue(result.is_survivorship_biased)

        self.assertEqual(result.combined_metrics['attempted_simulations'], 10.0)
        self.assertEqual(result.combined_metrics['failed_simulations'], 8.0)
        self.assertAlmostEqual(result.combined_metrics['failure_rate_pct'], 80.0)
        self.assertEqual(result.analysis_parameters['failed_simulations'], 8)

    def test_high_failure_rate_logs_a_warning(self):
        analyzer = self.make_analyzer(n_simulations=10)
        side_effect = [self._mock_result()] + [None] * 9

        with patch.object(analyzer, '_run_single_simulation', side_effect=side_effect):
            with self.assertLogs(level=logging.WARNING) as captured:
                analyzer.analyze(self.test_data, "TEST")

        self.assertTrue(any('survivorship' in message for message in captured.output))

    def test_clean_run_reports_zero_failures(self):
        analyzer = self.make_analyzer(n_simulations=4, random_seed=1)
        result = analyzer.analyze(self.test_data, "TEST")

        self.assertEqual(result.failed_runs, 0)
        self.assertEqual(result.failure_rate, 0.0)
        self.assertFalse(result.is_survivorship_biased)
        self.assertEqual(result.combined_metrics['failure_rate_pct'], 0.0)

    def test_discarded_short_sample_is_logged_not_silent(self):
        analyzer = self.make_analyzer()
        short_sample = self.test_data.head(MIN_SAMPLE_BARS - 1)

        with patch.object(analyzer, '_block_bootstrap_sample', return_value=short_sample):
            with self.assertLogs(level=logging.WARNING) as captured:
                result = analyzer._run_single_simulation(self.test_data, "TEST", 0)

        self.assertIsNone(result)
        self.assertTrue(any('discarded' in message for message in captured.output))

    def test_analyze_raises_when_every_simulation_fails(self):
        analyzer = self.make_analyzer(n_simulations=5)
        with patch.object(analyzer, '_run_single_simulation', return_value=None):
            with self.assertRaises(ValueError) as ctx:
                analyzer.analyze(self.test_data, "TEST")
        self.assertIn('failed', str(ctx.exception))

    def test_memory_management_keeps_recent_results_not_best_results(self):
        analyzer = self.make_analyzer(max_results_in_memory=4)

        results = []
        for i in range(10):
            result = Mock()
            result.total_return = (10 - i) * 100  # earliest results are the "best"
            results.append(result)

        with self.assertLogs(level=logging.WARNING):
            managed = analyzer._manage_memory_efficient_results(results)

        self.assertEqual(len(managed), 2)
        # Keeping the best half would amputate the left tail of the distribution.
        self.assertEqual([r.total_return for r in managed], [200, 100])

    def test_memory_management_no_op_below_limit(self):
        analyzer = self.make_analyzer(max_results_in_memory=100)
        results = [Mock(total_return=i) for i in range(10)]
        self.assertEqual(analyzer._manage_memory_efficient_results(results), results)


class TestMonteCarloExecution(MonteCarloTestBase):
    """Simulation plumbing."""

    @patch('niffler.analysis.monte_carlo_analyzer.BacktestEngine')
    def test_run_single_simulation(self, mock_engine_class):
        mock_result = Mock()
        engine = Mock()
        engine.run_backtest.return_value = mock_result
        mock_engine_class.return_value = engine

        analyzer = self.make_analyzer(n_simulations=1, random_seed=8)
        result = analyzer._run_single_simulation(self.test_data, "TEST", 0)

        self.assertIsNotNone(result)
        self.assertEqual(result.metadata['simulation_id'], 0)
        self.assertEqual(result.metadata['random_seed'], 8)
        self.assertEqual(result.metadata['parameters_used'], self.optimal_parameters)
        engine.run_backtest.assert_called_once()

    def test_sequential_simulations(self):
        analyzer = self.make_analyzer(n_simulations=5)

        with patch.object(analyzer, '_run_single_simulation', return_value=Mock()) as mock_sim:
            results, failed = analyzer._run_simulations_sequential(self.test_data, "TEST")

        self.assertEqual(len(results), 5)
        self.assertEqual(failed, 0)
        self.assertEqual(mock_sim.call_count, 5)

    def test_sequential_simulations_with_failures(self):
        analyzer = self.make_analyzer(n_simulations=5)
        mock_result = Mock()

        with patch.object(analyzer, '_run_single_simulation') as mock_sim:
            mock_sim.side_effect = [mock_result, None, mock_result,
                                    RuntimeError("boom"), mock_result]
            results, failed = analyzer._run_simulations_sequential(self.test_data, "TEST")

        self.assertEqual(len(results), 3)
        self.assertEqual(failed, 2)

    def test_parallel_simulations(self):
        analyzer = self.make_analyzer(n_simulations=3, n_jobs=2)

        executor = Mock()
        futures = [Mock(), Mock(), Mock()]
        futures[0].result.return_value = Mock()
        futures[1].result.return_value = Mock()
        futures[2].result.return_value = None
        executor.submit.side_effect = futures

        with patch('niffler.analysis.monte_carlo_analyzer.ProcessPoolExecutor') as pool:
            pool.return_value.__enter__.return_value = executor
            with patch('niffler.analysis.monte_carlo_analyzer.as_completed',
                       return_value=futures):
                results, failed = analyzer._run_simulations_parallel(self.test_data, "TEST")

        self.assertEqual(len(results), 2)
        self.assertEqual(failed, 1)
        self.assertEqual(executor.submit.call_count, 3)

    def test_analyze_integration(self):
        analyzer = self.make_analyzer(n_simulations=5, random_seed=42)
        result = analyzer.analyze(self.test_data, "TEST")

        self.assertEqual(result.analysis_type, 'monte_carlo')
        self.assertEqual(result.strategy_name, 'MockStrategy')
        self.assertEqual(result.symbol, 'TEST')
        self.assertEqual(len(result.individual_results), 5)
        self.assertEqual(result.analysis_parameters['random_seed'], 42)
        self.assertEqual(result.analysis_parameters['optimal_parameters'],
                         self.optimal_parameters)
        for key in ('n_simulations', 'bootstrap_sample_pct', 'block_size_days'):
            self.assertIn(key, result.analysis_parameters)


class TestMonteCarloMetrics(MonteCarloTestBase):
    """Aggregate and distribution statistics."""

    def test_combined_metrics_calculation(self):
        analyzer = self.make_analyzer()

        results = []
        for i in range(5):
            result = Mock()
            result.total_return = (i + 1) * 100
            result.total_return_pct = (i + 1) * 10
            result.sharpe_ratio = (i + 1) * 0.5
            result.max_drawdown = -(i + 1) * 2
            result.win_rate = 50 + (i + 1) * 5
            result.total_trades = (i + 1) * 10
            results.append(result)

        metrics = analyzer._calculate_combined_metrics(results)

        self.assertEqual(metrics['total_simulations'], 5)
        self.assertEqual(metrics['mean_return'], 300)
        self.assertEqual(metrics['profitable_simulations_pct'], 100)

    def test_distribution_statistics_calculation(self):
        analyzer = self.make_analyzer()

        results = []
        returns = [-200, -100, 0, 100, 200, 300, 400, 500, 600, 700]
        for i, ret in enumerate(returns):
            result = Mock()
            result.total_return = ret
            result.total_return_pct = ret / 10
            result.sharpe_ratio = i * 0.1
            results.append(result)

        stats = analyzer._calculate_distribution_statistics(results)

        for key in ('return_var_5pct', 'return_var_1pct', 'return_cvar_5pct',
                    'return_cvar_1pct', 'return_ci_95_lower', 'return_ci_95_upper'):
            self.assertIn(key, stats)

        self.assertAlmostEqual(stats['return_var_5pct'], -200, delta=50)

    def test_percentile_results(self):
        analyzer = self.make_analyzer()

        results = []
        for i in range(10):
            result = Mock()
            result.total_return = i * 100
            result.total_return_pct = i * 10
            result.sharpe_ratio = i * 0.2
            result.max_drawdown = -i * 5
            result.win_rate = 30 + i * 5
            results.append(result)

        percentiles = analyzer.get_percentile_results(results)

        for metric in ('total_return', 'total_return_pct', 'sharpe_ratio',
                       'max_drawdown', 'win_rate'):
            self.assertIn(metric, percentiles)
            for p in ('p5', 'p25', 'p50', 'p75', 'p95'):
                self.assertIn(p, percentiles[metric])


class TestMonteCarloDrawdownAggregation(MonteCarloTestBase):
    """max_drawdown is negative, so the tail is the minimum, not the maximum."""

    @staticmethod
    def _results(drawdowns):
        results = []
        for i, drawdown in enumerate(drawdowns):
            result = Mock()
            result.total_return = 100.0 * i
            result.total_return_pct = 1.0 * i
            result.sharpe_ratio = 0.5
            result.max_drawdown = drawdown
            result.win_rate = 50.0
            result.total_trades = 3
            results.append(result)
        return results

    def test_worst_max_drawdown_is_the_deepest_one(self):
        analyzer = self.make_analyzer()

        metrics = analyzer._calculate_combined_metrics(self._results([-5.0, -40.0]))

        self.assertEqual(metrics['worst_max_drawdown'], -40.0)
        self.assertEqual(metrics['best_max_drawdown'], -5.0)

    def test_worst_bounds_the_mean_from_below(self):
        analyzer = self.make_analyzer()

        metrics = analyzer._calculate_combined_metrics(
            self._results([-3.2, -18.0, -45.0])
        )

        self.assertLessEqual(metrics['worst_max_drawdown'], metrics['mean_max_drawdown'])
        self.assertGreaterEqual(metrics['best_max_drawdown'], metrics['mean_max_drawdown'])


class TestMonteCarloBlockSizeGuard(MonteCarloTestBase):
    """A block covering most of the sample collapses the distribution to one path."""

    def test_oversized_block_is_clamped_before_simulating(self):
        data = make_price_data(periods=400)
        analyzer = self.make_analyzer(block_size_days=len(data) - 1, n_simulations=2)

        with self.assertLogs(level='WARNING') as logs:
            analyzer.analyze(data, "TEST")

        self.assertLessEqual(analyzer.block_size_days, (len(data) - 1) // 2)
        self.assertTrue(any('Block size' in message for message in logs.output))

    def test_different_seeds_produce_different_paths(self):
        """The old guard let block_size_days = len(data) - 1 through, and every
        simulation then replayed the exact same contiguous history."""
        data = make_price_data(periods=400)

        paths = []
        for seed in (1, 2, 3):
            analyzer = self.make_analyzer(block_size_days=len(data) - 1,
                                          random_seed=seed, n_simulations=1)
            analyzer.analyze(data, "TEST")  # applies the block-size guard
            paths.append(analyzer._block_bootstrap_sample(data)['close'].to_numpy())

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                self.assertFalse(np.array_equal(paths[i], paths[j]),
                                 f"seeds {i} and {j} produced identical paths")

    def test_degenerate_block_size_warns(self):
        """Reaching _block_bootstrap_sample with a single start position is loud."""
        data = make_price_data(periods=120)
        analyzer = self.make_analyzer(block_size_days=10)
        # Bypass analyze()'s clamp to reach the sampler's own guard.
        analyzer.block_size_days = len(data) - 1

        with self.assertLogs(level='WARNING') as logs:
            analyzer._block_bootstrap_sample(data)

        self.assertTrue(any('single possible start position' in message
                            for message in logs.output))


if __name__ == '__main__':
    unittest.main()
