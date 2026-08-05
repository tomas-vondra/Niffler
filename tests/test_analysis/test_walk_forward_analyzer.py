import logging
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
from dateutil.relativedelta import relativedelta

from niffler.analysis.walk_forward_analyzer import (
    WalkForwardAnalyzer,
    WalkForwardWindow,
    WalkForwardFold,
    MODE_WALK_FORWARD,
    MODE_SEGMENTED_IN_SAMPLE,
)
from niffler.optimization.parameter_space import ParameterSpace
from niffler.strategies.base_strategy import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy whose signals actually depend on its parameters."""

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


def make_price_data(start='2021-01-01', end='2023-12-31', seed=42):
    """Build a deterministic OHLCV frame with valid OHLC relationships."""
    dates = pd.date_range(start=start, end=end, freq='D')
    n_days = len(dates)

    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0008, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'open': prices * (1 + rng.normal(0, 0.001, n_days)),
        'high': prices * (1 + np.abs(rng.normal(0.002, 0.005, n_days))),
        'low': prices * (1 - np.abs(rng.normal(0.002, 0.005, n_days))),
        'close': prices,
        'volume': rng.integers(1000, 10000, n_days)
    }, index=dates)

    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    return data


class WalkForwardTestBase(unittest.TestCase):
    """Shared fixtures."""

    def setUp(self):
        # Other test modules disable logging globally at import time, which would make
        # every assertLogs assertion here vacuously fail. Re-enable for the duration of
        # each test and restore whatever level was in force afterwards.
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, previous_disable_level)

        self.test_data = make_price_data()
        self.strategy_class = MockStrategy
        self.optimal_parameters = {'short_window': 10, 'long_window': 20}
        self.parameter_space = ParameterSpace({
            'short_window': {'type': 'int', 'min': 5, 'max': 9, 'step': 4},
            'long_window': {'type': 'int', 'min': 20, 'max': 30, 'step': 10},
        })

    def make_analyzer(self, **overrides):
        kwargs = {
            'strategy_class': self.strategy_class,
            'parameter_space': self.parameter_space,
            'train_window_months': 6,
            'test_window_months': 3,
            'step_months': 3,
            'n_jobs': 1,
        }
        kwargs.update(overrides)
        return WalkForwardAnalyzer(**kwargs)


class TestWalkForwardConfiguration(WalkForwardTestBase):
    """Configuration and validation."""

    def test_init_valid_parameters(self):
        analyzer = self.make_analyzer(initial_capital=25000, commission=0.002,
                                      max_results_in_memory=100)

        self.assertEqual(analyzer.strategy_class, self.strategy_class)
        self.assertEqual(analyzer.mode, MODE_WALK_FORWARD)
        self.assertEqual(analyzer.train_window_months, 6)
        self.assertEqual(analyzer.test_window_months, 3)
        self.assertEqual(analyzer.step_months, 3)
        self.assertEqual(analyzer.initial_capital, 25000)
        self.assertEqual(analyzer.commission, 0.002)
        self.assertEqual(analyzer.max_results_in_memory, 100)
        self.assertFalse(analyzer.anchored)

    def test_walk_forward_is_the_default_mode(self):
        """The honest mode must be the default one."""
        analyzer = self.make_analyzer()
        self.assertEqual(analyzer.mode, MODE_WALK_FORWARD)

    def test_walk_forward_mode_requires_parameter_space(self):
        """Without a search space there is nothing to fit, so nothing is out-of-sample."""
        with self.assertRaises(ValueError) as ctx:
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
            )

        message = str(ctx.exception)
        self.assertIn('parameter_space', message)
        self.assertIn(MODE_SEGMENTED_IN_SAMPLE, message)

    def test_segmented_mode_requires_optimal_parameters(self):
        with self.assertRaises(ValueError):
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                mode=MODE_SEGMENTED_IN_SAMPLE,
            )

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                parameter_space=self.parameter_space,
                mode='definitely_not_a_mode',
            )

    def test_unknown_optimization_method_rejected(self):
        with self.assertRaises(ValueError):
            self.make_analyzer(optimization_method='bogus')

    def test_init_invalid_numeric_parameters(self):
        for kwargs in (
            {'train_window_months': -1},
            {'train_window_months': 0},
            {'test_window_months': -1},
            {'step_months': -1},
            {'initial_capital': -1000},
            {'commission': -0.001},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    self.make_analyzer(**kwargs)

    def test_segmented_mode_validates_strategy_parameters(self):
        WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            mode=MODE_SEGMENTED_IN_SAMPLE,
        )

        with self.assertRaises(ValueError):
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters={'invalid_param': 'invalid_value'},
                mode=MODE_SEGMENTED_IN_SAMPLE,
            )

    def test_analyze_insufficient_data(self):
        analyzer = self.make_analyzer()

        with self.assertRaises(ValueError):
            analyzer.analyze(pd.DataFrame())

        with self.assertRaises(ValueError):
            analyzer.analyze(self.test_data.head(50))

        # Enough rows to pass the 100-row floor but not enough for train + test.
        with self.assertRaises(ValueError):
            analyzer.analyze(self.test_data.head(120))

    def test_analyze_invalid_data_format(self):
        analyzer = self.make_analyzer()
        with self.assertRaises(ValueError):
            analyzer.analyze(self.test_data.reset_index())


class TestWalkForwardWindows(WalkForwardTestBase):
    """Schedule generation - this is where the missing training window used to hide."""

    def test_windows_have_a_training_period_preceding_the_test_period(self):
        analyzer = self.make_analyzer()
        windows = analyzer._generate_walk_forward_periods(self.test_data)

        self.assertGreater(len(windows), 0)
        for window in windows:
            with self.subTest(window=window.window_number):
                self.assertTrue(window.has_training_window)
                self.assertLess(window.train_start, window.train_end)
                self.assertLess(window.test_start, window.test_end)
                # The test window starts exactly where the training window ends,
                # so no bar is ever both trained on and tested on.
                self.assertEqual(window.train_end, window.test_start)

    def test_training_window_never_overlaps_test_window(self):
        analyzer = self.make_analyzer()
        windows = analyzer._generate_walk_forward_periods(self.test_data)

        for window in windows:
            train = self.test_data[(self.test_data.index >= window.train_start) &
                                   (self.test_data.index < window.train_end)]
            test = self.test_data[(self.test_data.index >= window.test_start) &
                                  (self.test_data.index < window.test_end)]
            self.assertGreater(len(train), 0)
            self.assertGreater(len(test), 0)
            self.assertEqual(len(train.index.intersection(test.index)), 0)
            self.assertLess(train.index.max(), test.index.min())

    def test_rolling_windows_step_forward(self):
        analyzer = self.make_analyzer(step_months=3, anchored=False)
        windows = analyzer._generate_walk_forward_periods(self.test_data)

        self.assertGreater(len(windows), 1)
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(current.train_start,
                             previous.train_start + relativedelta(months=3))
            self.assertEqual(current.test_start,
                             previous.test_start + relativedelta(months=3))

    def test_anchored_windows_keep_the_same_training_start(self):
        analyzer = self.make_analyzer(anchored=True)
        windows = analyzer._generate_walk_forward_periods(self.test_data)

        self.assertGreater(len(windows), 1)
        for window in windows:
            self.assertEqual(window.train_start, self.test_data.index[0])
        # ...but the training window still grows and the test window still moves.
        self.assertLess(windows[0].train_end, windows[-1].train_end)

    def test_window_too_large_for_data_yields_nothing(self):
        analyzer = self.make_analyzer(train_window_months=36, test_window_months=24)
        windows = analyzer._generate_walk_forward_periods(self.test_data)
        self.assertEqual(len(windows), 0)

    def test_segmented_mode_windows_have_no_training_period(self):
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            mode=MODE_SEGMENTED_IN_SAMPLE,
            test_window_months=6,
            step_months=3,
            n_jobs=1,
        )
        windows = analyzer._generate_walk_forward_periods(self.test_data)

        self.assertGreater(len(windows), 0)
        for window in windows:
            self.assertFalse(window.has_training_window)


class TestWalkForwardExecution(WalkForwardTestBase):
    """Fold execution: fit in-sample, evaluate out-of-sample."""

    def test_optimizer_only_ever_sees_training_data(self):
        """
        The regression test for circular validation: the optimizer must never be handed a
        single bar from the window it is graded on.
        """
        analyzer = self.make_analyzer(train_window_months=6, test_window_months=3,
                                      step_months=6)
        windows = analyzer._generate_walk_forward_periods(self.test_data)
        self.assertGreater(len(windows), 0)

        seen_data = []

        with patch('niffler.analysis.walk_forward_analyzer.create_optimizer') as mock_create:
            def capture(**kwargs):
                seen_data.append(kwargs['data'])
                optimizer = Mock()
                backtest_result = Mock()
                backtest_result.total_return_pct = 5.0
                optimizer.optimize.return_value = [
                    Mock(parameters={'short_window': 5, 'long_window': 20},
                         backtest_result=backtest_result)
                ]
                return optimizer

            mock_create.side_effect = capture
            result = analyzer.analyze(self.test_data, "TEST")

        self.assertEqual(len(seen_data), len(result.individual_results))
        for train_data, fold in zip(seen_data, result.metadata['folds']):
            with self.subTest(fold=fold['fold_number']):
                self.assertGreater(len(train_data), 0)
                self.assertLess(train_data.index.max(), fold['test_start'])
                self.assertGreaterEqual(train_data.index.min(), fold['train_start'])

    def test_analyze_reports_train_test_parameters_and_efficiency(self):
        analyzer = self.make_analyzer(train_window_months=6, test_window_months=3,
                                      step_months=6)
        result = analyzer.analyze(self.test_data, "TEST")

        self.assertEqual(result.analysis_type, 'walk_forward')
        self.assertEqual(result.strategy_name, 'MockStrategy')
        self.assertEqual(result.symbol, 'TEST')
        self.assertGreater(result.n_periods, 0)

        self.assertEqual(result.analysis_parameters['mode'], MODE_WALK_FORWARD)
        self.assertEqual(result.analysis_parameters['train_window_months'], 6)
        self.assertIn('parameter_space', result.analysis_parameters)

        valid_short = {5, 9}
        valid_long = {20, 30}

        for fold in result.metadata['folds']:
            with self.subTest(fold=fold['fold_number']):
                self.assertIsNotNone(fold['train_start'])
                self.assertIsNotNone(fold['train_end'])
                self.assertLessEqual(fold['train_end'], fold['test_start'])
                self.assertGreater(fold['n_train_bars'], 0)
                self.assertGreater(fold['n_test_bars'], 0)
                self.assertIn(fold['parameters']['short_window'], valid_short)
                self.assertIn(fold['parameters']['long_window'], valid_long)
                self.assertIsNotNone(fold['train_return_pct'])
                self.assertFalse(fold['in_sample'])

        self.assertIn('folds_with_efficiency_ratio', result.combined_metrics)
        self.assertIn('mean_efficiency_ratio', result.combined_metrics)

    def test_fold_metadata_is_attached_to_each_backtest_result(self):
        analyzer = self.make_analyzer(train_window_months=6, test_window_months=3,
                                      step_months=6)
        result = analyzer.analyze(self.test_data, "TEST")

        for backtest_result in result.individual_results:
            metadata = backtest_result.metadata
            self.assertIn('train_start', metadata)
            self.assertIn('test_start', metadata)
            self.assertIn('parameters_used', metadata)
            self.assertIn('efficiency_ratio', metadata)
            self.assertFalse(metadata['in_sample'])

    def test_segmented_mode_marks_results_as_in_sample(self):
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            mode=MODE_SEGMENTED_IN_SAMPLE,
            test_window_months=6,
            step_months=6,
            n_jobs=1,
        )
        result = analyzer.analyze(self.test_data, "TEST")

        self.assertEqual(result.analysis_parameters['mode'], MODE_SEGMENTED_IN_SAMPLE)
        self.assertIsNone(result.analysis_parameters['train_window_months'])
        for fold in result.metadata['folds']:
            self.assertTrue(fold['in_sample'])
            self.assertIsNone(fold['train_start'])
            self.assertIsNone(fold['efficiency_ratio'])
            self.assertEqual(fold['parameters'], self.optimal_parameters)

    def test_run_single_fold_returns_none_for_short_test_window(self):
        analyzer = self.make_analyzer()
        window = WalkForwardWindow(
            window_number=1,
            train_start=self.test_data.index[0],
            train_end=self.test_data.index[200],
            test_start=self.test_data.index[200],
            test_end=self.test_data.index[205],
        )
        self.assertIsNone(analyzer._run_single_fold(self.test_data, window, "TEST"))

    def test_run_single_fold_returns_none_for_short_training_window(self):
        analyzer = self.make_analyzer()
        window = WalkForwardWindow(
            window_number=1,
            train_start=self.test_data.index[0],
            train_end=self.test_data.index[5],
            test_start=self.test_data.index[5],
            test_end=self.test_data.index[200],
        )
        self.assertIsNone(analyzer._run_single_fold(self.test_data, window, "TEST"))

    def test_sequential_execution_counts_failures(self):
        analyzer = self.make_analyzer()
        windows = analyzer._generate_walk_forward_periods(self.test_data)[:5]
        self.assertEqual(len(windows), 5)

        fold = Mock(spec=WalkForwardFold)
        fold.fold_number = 1

        with patch.object(analyzer, '_run_single_fold') as mock_fold:
            mock_fold.side_effect = [fold, None, fold, RuntimeError("boom"), fold]
            folds, failed = analyzer._run_folds_sequential(self.test_data, "TEST", windows)

        self.assertEqual(len(folds), 3)
        self.assertEqual(failed, 2)


class TestWalkForwardEfficiencyRatio(WalkForwardTestBase):
    """The efficiency ratio is the point of the exercise."""

    def test_ratio_is_one_when_oos_matches_is_per_bar(self):
        ratio = WalkForwardAnalyzer._calculate_efficiency_ratio(10.0, 100, 5.0, 50)
        self.assertAlmostEqual(ratio, 1.0)

    def test_ratio_halves_when_oos_performance_halves(self):
        ratio = WalkForwardAnalyzer._calculate_efficiency_ratio(10.0, 100, 2.5, 50)
        self.assertAlmostEqual(ratio, 0.5)

    def test_ratio_is_negative_when_oos_loses_money(self):
        ratio = WalkForwardAnalyzer._calculate_efficiency_ratio(10.0, 100, -5.0, 50)
        self.assertLess(ratio, 0)

    def test_ratio_undefined_without_training_window(self):
        self.assertIsNone(
            WalkForwardAnalyzer._calculate_efficiency_ratio(None, 0, 5.0, 50)
        )

    def test_ratio_undefined_when_in_sample_performance_not_positive(self):
        # Dividing by a non-positive in-sample result would flip losses into
        # flattering positive "efficiency".
        self.assertIsNone(
            WalkForwardAnalyzer._calculate_efficiency_ratio(-10.0, 100, -5.0, 50)
        )
        self.assertIsNone(
            WalkForwardAnalyzer._calculate_efficiency_ratio(0.0, 100, 5.0, 50)
        )

    def test_efficiency_metrics_aggregate_defined_ratios_only(self):
        analyzer = self.make_analyzer()
        folds = [
            self._fold(1, 0.5),
            self._fold(2, 1.5),
            self._fold(3, None),
        ]

        metrics = analyzer._calculate_efficiency_metrics(folds)

        self.assertEqual(metrics['folds_with_efficiency_ratio'], 2.0)
        self.assertAlmostEqual(metrics['mean_efficiency_ratio'], 1.0)
        self.assertAlmostEqual(metrics['worst_efficiency_ratio'], 0.5)
        self.assertAlmostEqual(metrics['best_efficiency_ratio'], 1.5)
        self.assertAlmostEqual(metrics['folds_above_half_efficiency_pct'], 100.0)

    def test_efficiency_metrics_without_any_ratio(self):
        analyzer = self.make_analyzer()
        metrics = analyzer._calculate_efficiency_metrics([self._fold(1, None)])
        self.assertEqual(metrics['folds_with_efficiency_ratio'], 0.0)
        self.assertNotIn('mean_efficiency_ratio', metrics)

    def _fold(self, number, ratio):
        return WalkForwardFold(
            fold_number=number,
            test_start=pd.Timestamp('2022-01-01'),
            test_end=pd.Timestamp('2022-04-01'),
            parameters={'short_window': 5, 'long_window': 20},
            test_result=Mock(),
            n_test_bars=90,
            train_return_pct=5.0 if ratio is not None else None,
            efficiency_ratio=ratio,
        )


class TestWalkForwardSurvivorship(WalkForwardTestBase):
    """Failed folds must be surfaced, never silently dropped."""

    def _fold_stub(self, number):
        fold = Mock(spec=WalkForwardFold)
        fold.fold_number = number
        fold.train_start = pd.Timestamp('2021-01-01')
        fold.train_end = pd.Timestamp('2021-07-01')
        fold.test_start = pd.Timestamp('2021-07-01')
        fold.test_end = pd.Timestamp('2021-10-01')
        fold.efficiency_ratio = 0.5
        fold.train_return_pct = 4.0
        fold.to_dict.return_value = {'fold_number': number}

        result = Mock()
        result.total_return = 100.0
        result.total_return_pct = 1.0
        result.sharpe_ratio = 0.5
        result.max_drawdown = -2.0
        result.win_rate = 55.0
        result.total_trades = 5
        result.portfolio_values = None
        fold.test_result = result
        return fold

    def test_failed_folds_are_reported_in_the_result(self):
        analyzer = self.make_analyzer(train_window_months=6, test_window_months=3,
                                      step_months=3)
        windows = analyzer._generate_walk_forward_periods(self.test_data)
        n_windows = len(windows)
        self.assertGreater(n_windows, 4)

        # Only two folds survive; everything else dies.
        side_effect = [self._fold_stub(1), self._fold_stub(2)] + [None] * (n_windows - 2)

        with patch.object(analyzer, '_run_single_fold', side_effect=side_effect):
            result = analyzer.analyze(self.test_data, "TEST")

        self.assertEqual(result.successful_runs, 2)
        self.assertEqual(result.attempted_runs, n_windows)
        self.assertEqual(result.failed_runs, n_windows - 2)
        self.assertAlmostEqual(result.failure_rate, (n_windows - 2) / n_windows)
        self.assertTrue(result.is_survivorship_biased)

        self.assertEqual(result.combined_metrics['failed_folds'], float(n_windows - 2))
        self.assertEqual(result.combined_metrics['attempted_folds'], float(n_windows))
        self.assertGreater(result.combined_metrics['failure_rate_pct'], 5.0)
        self.assertEqual(result.analysis_parameters['failed_folds'], n_windows - 2)

    def test_high_failure_rate_logs_a_warning(self):
        analyzer = self.make_analyzer(train_window_months=6, test_window_months=3,
                                      step_months=3)
        windows = analyzer._generate_walk_forward_periods(self.test_data)
        side_effect = [self._fold_stub(1)] + [None] * (len(windows) - 1)

        with patch.object(analyzer, '_run_single_fold', side_effect=side_effect):
            with self.assertLogs(level=logging.WARNING) as captured:
                analyzer.analyze(self.test_data, "TEST")

        self.assertTrue(any('survivorship' in message for message in captured.output))

    def test_analyze_raises_when_every_fold_fails(self):
        analyzer = self.make_analyzer()
        with patch.object(analyzer, '_run_single_fold', return_value=None):
            with self.assertRaises(ValueError) as ctx:
                analyzer.analyze(self.test_data, "TEST")
        self.assertIn('failed', str(ctx.exception))

    def test_memory_management_keeps_recent_folds_not_best_folds(self):
        analyzer = self.make_analyzer(max_results_in_memory=4)

        folds = []
        for i in range(10):
            fold = Mock(spec=WalkForwardFold)
            fold.fold_number = i
            fold.test_result = Mock(total_return=(10 - i) * 100)  # earliest is "best"
            folds.append(fold)

        managed = analyzer._manage_memory_efficient_results(folds)

        self.assertEqual(len(managed), 2)
        # Keeping the best folds would have kept 0 and 1 and biased every metric.
        self.assertEqual([f.fold_number for f in managed], [8, 9])


class TestWalkForwardMetrics(WalkForwardTestBase):
    """Aggregate and stability metrics."""

    def _results(self, returns):
        results = []
        for i, ret in enumerate(returns):
            result = Mock()
            result.total_return = ret
            result.total_return_pct = ret / 10
            result.sharpe_ratio = i * 0.2
            result.max_drawdown = -(i + 1) * 2
            result.win_rate = 50 + i
            result.total_trades = (i + 1) * 10
            result.portfolio_values = pd.Series([10000 + j * 10 for j in range(30)])
            results.append(result)
        return results

    def test_combined_metrics_calculation(self):
        analyzer = self.make_analyzer()
        metrics = analyzer._calculate_combined_metrics(self._results([100, 200, 300, 400, 500]))

        self.assertEqual(metrics['total_periods'], 5)
        self.assertEqual(metrics['avg_return'], 300)
        self.assertEqual(metrics['profitable_periods_pct'], 100)
        self.assertIn('combined_sharpe_ratio', metrics)

    def test_stability_metrics_calculation(self):
        analyzer = self.make_analyzer()
        metrics = analyzer._calculate_stability_metrics(
            self._results([100, 200, -50, 150, 300, -100, 250])
        )

        for key in ('return_volatility', 'return_pct_volatility', 'sharpe_volatility',
                    'return_consistency', 'temporal_stability'):
            self.assertIn(key, metrics)

        self.assertGreaterEqual(metrics['temporal_stability'], 0)
        self.assertLessEqual(metrics['temporal_stability'], 1)

    def test_temporal_stability_calculation(self):
        analyzer = self.make_analyzer()

        rising = self._results([0, 100, 200, 300, 400])
        self.assertEqual(analyzer._calculate_temporal_stability(rising), 1.0)

        alternating = self._results([100, -50, 200, -100, 150])
        self.assertLess(analyzer._calculate_temporal_stability(alternating), 0.5)

    def test_rolling_stability_calculation(self):
        analyzer = self.make_analyzer()
        rolling = analyzer._calculate_rolling_stability(
            self._results([i * 50 for i in range(10)])
        )

        self.assertIn('rolling_mean_stability', rolling)
        self.assertIn('trend_consistency', rolling)
        self.assertGreaterEqual(rolling['rolling_mean_stability'], 0)
        self.assertGreaterEqual(rolling['trend_consistency'], 0)
        self.assertLessEqual(rolling['trend_consistency'], 1)

    def test_trend_consistency_edge_cases(self):
        analyzer = self.make_analyzer()
        self.assertEqual(analyzer._calculate_trend_consistency([100]), 1.0)
        self.assertEqual(analyzer._calculate_trend_consistency([100, 200]), 1.0)
        self.assertEqual(analyzer._calculate_trend_consistency([100, 100, 100, 100]), 1.0)


def make_result(max_drawdown=-10.0, portfolio_values=None, total_return=100.0,
                sharpe_ratio=1.0):
    """Build a minimal stand-in for a BacktestResult."""
    result = Mock()
    result.total_return = total_return
    result.total_return_pct = total_return / 100.0
    result.sharpe_ratio = sharpe_ratio
    result.max_drawdown = max_drawdown
    result.win_rate = 50.0
    result.total_trades = 5
    result.portfolio_values = portfolio_values
    return result


class TestWalkForwardDrawdownAggregation(WalkForwardTestBase):
    """max_drawdown is negative, so the worst outcome is the minimum."""

    def test_worst_max_drawdown_is_the_deepest_one(self):
        analyzer = self.make_analyzer()

        metrics = analyzer._calculate_combined_metrics([
            make_result(max_drawdown=-5.0),
            make_result(max_drawdown=-40.0),
        ])

        self.assertEqual(metrics['worst_max_drawdown'], -40.0)
        self.assertEqual(metrics['best_max_drawdown'], -5.0)

    def test_worst_is_never_milder_than_the_average(self):
        analyzer = self.make_analyzer()

        metrics = analyzer._calculate_combined_metrics([
            make_result(max_drawdown=dd) for dd in (-3.0, -12.0, -45.0)
        ])

        self.assertLessEqual(metrics['worst_max_drawdown'], metrics['avg_max_drawdown'])
        self.assertGreaterEqual(metrics['best_max_drawdown'], metrics['avg_max_drawdown'])


class TestWalkForwardCombinedSharpe(WalkForwardTestBase):
    """The pooled Sharpe must share the per-fold annualisation convention."""

    @staticmethod
    def _series(index):
        values = 10000.0 * np.cumprod(1 + np.tile([0.001, -0.0005], len(index) // 2))
        return pd.Series(values[:len(index)], index=index)

    def _expected_sharpe(self, series, periods_per_year):
        returns = series.pct_change().dropna()
        return np.sqrt(periods_per_year) * returns.mean() / returns.std()

    def test_hourly_folds_annualise_with_8760_not_252(self):
        analyzer = self.make_analyzer()
        index = pd.date_range('2024-01-01', periods=200, freq='h')
        series = self._series(index)

        metrics = analyzer._calculate_combined_metrics([make_result(portfolio_values=series)])

        self.assertAlmostEqual(metrics['combined_sharpe_ratio'],
                               self._expected_sharpe(series, 8760.0), places=6)
        self.assertNotAlmostEqual(metrics['combined_sharpe_ratio'],
                                  self._expected_sharpe(series, 252.0), places=3)

    def test_daily_crypto_folds_annualise_with_365(self):
        analyzer = self.make_analyzer()
        index = pd.date_range('2024-01-01', periods=200, freq='D')
        series = self._series(index)

        metrics = analyzer._calculate_combined_metrics([make_result(portfolio_values=series)])

        self.assertAlmostEqual(metrics['combined_sharpe_ratio'],
                               self._expected_sharpe(series, 365.0), places=6)


class TestWalkForwardOutOfSampleOverlap(WalkForwardTestBase):
    """Overlapping folds must not be pooled as independent observations."""

    @staticmethod
    def _series(start, periods):
        index = pd.date_range(start, periods=periods, freq='D')
        values = 10000.0 * np.cumprod(1 + np.tile([0.002, -0.001], periods // 2))
        return pd.Series(values[:periods], index=index)

    def test_repeated_bars_are_counted_once(self):
        analyzer = self.make_analyzer()
        first = self._series('2024-01-01', 100)
        # Second fold restarts halfway through the first one's window.
        second = self._series('2024-02-20', 100)

        metrics = analyzer._calculate_combined_metrics([
            make_result(portfolio_values=first),
            make_result(portfolio_values=second),
        ])

        # Bar 0 of each fold has no return, so overlap is measured on the return bars.
        overlap = len(first.index[1:].intersection(second.index[1:]))
        self.assertGreater(overlap, 0)
        self.assertEqual(metrics['overlapping_oos_bars'], float(overlap))
        self.assertEqual(metrics['independent_oos_bars'],
                         float((len(first) - 1) + (len(second) - 1) - overlap))
        self.assertGreater(metrics['oos_overlap_pct'], 0.0)

    def test_disjoint_folds_report_no_overlap(self):
        analyzer = self.make_analyzer()

        metrics = analyzer._calculate_combined_metrics([
            make_result(portfolio_values=self._series('2024-01-01', 60)),
            make_result(portfolio_values=self._series('2024-06-01', 60)),
        ])

        self.assertEqual(metrics['overlapping_oos_bars'], 0.0)
        self.assertEqual(metrics['oos_overlap_pct'], 0.0)
        self.assertEqual(metrics['independent_oos_bars'], float(59 + 59))

    def test_overlapping_schedule_warns(self):
        analyzer = self.make_analyzer(test_window_months=6, step_months=3,
                                      train_window_months=6)

        with self.assertLogs(level='WARNING') as logs:
            analyzer.analyze(self.test_data, "TEST")

        self.assertTrue(any('overlap' in message for message in logs.output), logs.output)

    def test_non_overlapping_schedule_does_not_warn_about_overlap(self):
        analyzer = self.make_analyzer(test_window_months=3, step_months=3)

        with self.assertLogs(level='INFO') as logs:
            analyzer.analyze(self.test_data, "TEST")

        overlap_warnings = [m for m in logs.output
                            if 'WARNING' in m and 'windows overlap' in m]
        self.assertEqual(overlap_warnings, [])


class TestWalkForwardParallelExecution(WalkForwardTestBase):
    """The parallel fold path must stay wired to _run_single_fold_static."""

    def _windows(self, count=3):
        return [
            WalkForwardWindow(
                window_number=i + 1,
                test_start=pd.Timestamp('2021-01-01') + relativedelta(months=3 * i),
                test_end=pd.Timestamp('2021-04-01') + relativedelta(months=3 * i),
                train_start=pd.Timestamp('2020-01-01') + relativedelta(months=3 * i),
                train_end=pd.Timestamp('2021-01-01') + relativedelta(months=3 * i),
            )
            for i in range(count)
        ]

    def _fold(self, number):
        return WalkForwardFold(
            fold_number=number,
            test_start=pd.Timestamp('2021-01-01'),
            test_end=pd.Timestamp('2021-04-01'),
            parameters={'short_window': 5, 'long_window': 20},
            test_result=make_result(),
            n_test_bars=90,
        )

    @patch('niffler.analysis.walk_forward_analyzer.as_completed')
    @patch('niffler.analysis.walk_forward_analyzer.ProcessPoolExecutor')
    def test_submit_arguments_match_the_static_entry_point(self, mock_executor_cls,
                                                           mock_as_completed):
        analyzer = self.make_analyzer(n_jobs=2)
        windows = self._windows(2)

        executor = mock_executor_cls.return_value.__enter__.return_value
        futures = [Mock(), Mock()]
        for future, number in zip(futures, (1, 2)):
            future.result.return_value = self._fold(number)
        executor.submit.side_effect = futures
        mock_as_completed.side_effect = lambda mapping: list(mapping)

        folds, failed = analyzer._run_folds_parallel(self.test_data, "TEST", windows)

        self.assertEqual(failed, 0)
        self.assertEqual([f.fold_number for f in folds], [1, 2])
        self.assertEqual(executor.submit.call_count, 2)

        args = executor.submit.call_args_list[0][0]
        self.assertIs(args[0], WalkForwardAnalyzer._run_single_fold_static)
        # The positional payload must line up with _run_single_fold_static's signature.
        self.assertEqual(len(args) - 1, 12)
        self.assertIs(args[1], self.test_data)
        self.assertIs(args[2], windows[0])
        self.assertEqual(args[3], "TEST")
        self.assertIs(args[4], analyzer.strategy_class)
        self.assertIs(args[5], analyzer.parameter_space)
        self.assertEqual(args[7], analyzer.mode)
        self.assertEqual(args[8], analyzer.optimization_method)
        self.assertEqual(args[9], analyzer.optimization_metric)
        self.assertEqual(args[10], analyzer.initial_capital)
        self.assertEqual(args[11], analyzer.commission)
        self.assertIs(args[12], analyzer.cost_model)

    @patch('niffler.analysis.walk_forward_analyzer.as_completed')
    @patch('niffler.analysis.walk_forward_analyzer.ProcessPoolExecutor')
    def test_folds_are_sorted_and_failures_counted(self, mock_executor_cls,
                                                   mock_as_completed):
        analyzer = self.make_analyzer(n_jobs=2)
        windows = self._windows(3)

        executor = mock_executor_cls.return_value.__enter__.return_value
        out_of_order = Mock()
        out_of_order.result.return_value = self._fold(3)
        first = Mock()
        first.result.return_value = self._fold(1)
        none_result = Mock()
        none_result.result.return_value = None
        # as_completed yields them out of order on purpose.
        executor.submit.side_effect = [out_of_order, none_result, first]
        mock_as_completed.side_effect = lambda mapping: list(mapping)

        folds, failed = analyzer._run_folds_parallel(self.test_data, "TEST", windows)

        self.assertEqual([f.fold_number for f in folds], [1, 3])
        self.assertEqual(failed, 1)

    @patch('niffler.analysis.walk_forward_analyzer.as_completed')
    @patch('niffler.analysis.walk_forward_analyzer.ProcessPoolExecutor')
    def test_result_errors_are_counted_as_failures(self, mock_executor_cls,
                                                   mock_as_completed):
        analyzer = self.make_analyzer(n_jobs=2)
        windows = self._windows(1)

        executor = mock_executor_cls.return_value.__enter__.return_value
        broken = Mock()
        broken.result.side_effect = RuntimeError("boom")
        executor.submit.side_effect = [broken]
        mock_as_completed.side_effect = lambda mapping: list(mapping)

        folds, failed = analyzer._run_folds_parallel(self.test_data, "TEST", windows)

        self.assertEqual(folds, [])
        self.assertEqual(failed, 1)


if __name__ == '__main__':
    unittest.main()
