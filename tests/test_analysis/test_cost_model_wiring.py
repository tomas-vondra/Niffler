"""
Both analyzers must validate a strategy in the market it will be traded in.

A walk-forward fold that optimises frictionlessly and then measures out-of-sample
performance with costs (or the reverse) is comparing two different worlds, and a
Monte Carlo distribution built without costs describes a market nobody trades.
"""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.analysis.monte_carlo_analyzer import MonteCarloAnalyzer
from niffler.analysis.walk_forward_analyzer import WalkForwardAnalyzer
from niffler.backtesting.cost_model import FixedSlippageModel, ZeroCostModel
from niffler.backtesting.run_config import RunConfig
from niffler.optimization.parameter_space import ParameterSpace
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy

PARAMETER_SPACE = ParameterSpace({
    'short_window': {'type': 'int', 'min': 5, 'max': 6},
    'long_window': {'type': 'int', 'min': 20, 'max': 21},
})

OPTIMAL_PARAMETERS = {'short_window': 5, 'long_window': 20}


def make_data(n_bars=400):
    """Build an oscillating OHLCV frame, so a moving-average cross actually fires."""
    index = pd.date_range('2022-01-01', periods=n_bars, freq='D')
    closes = [100.0 + 20.0 * math.sin(i / 9.0) for i in range(n_bars)]
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


class TestWalkForwardCarriesTheCostModel(unittest.TestCase):
    """Both halves of a fold - the fit and the grade - must use the same costs."""

    def setUp(self):
        self.cost_model = FixedSlippageModel(slippage_bps=25.0)

    def _analyzer(self, cost_model, n_jobs=1):
        return WalkForwardAnalyzer(
            strategy_class=SimpleMAStrategy,
            parameter_space=PARAMETER_SPACE,
            train_window_months=6,
            test_window_months=3,
            step_months=3,
            n_jobs=n_jobs,
            run_config=RunConfig(cost_model=cost_model),
        )

    def test_the_reusable_engine_gets_it(self):
        analyzer = self._analyzer(self.cost_model)

        self.assertIs(analyzer._backtest_engine.cost_model, self.cost_model)

    def test_the_default_stays_frictionless(self):
        self.assertIsInstance(self._analyzer(None)._backtest_engine.cost_model,
                              ZeroCostModel)

    def test_the_per_fold_optimizer_gets_it(self):
        with patch('niffler.analysis.walk_forward_analyzer.create_optimizer') as factory:
            factory.return_value.optimize.return_value = []

            WalkForwardAnalyzer._optimize_on_training_window(
                train_data=make_data(120),
                strategy_class=SimpleMAStrategy,
                parameter_space=PARAMETER_SPACE,
                optimization_method='grid',
                optimization_metric='total_return',
                run_config=RunConfig(cost_model=self.cost_model),
            )

        self.assertIs(factory.call_args.kwargs['run_config'].cost_model, self.cost_model)

    def test_a_fold_without_a_shared_engine_still_gets_it(self):
        """The parallel path builds a fresh engine per fold."""
        with patch('niffler.analysis.walk_forward_analyzer.BacktestEngine') as engine_cls:
            engine_cls.from_config.return_value.run_backtest.side_effect = RuntimeError(
                'stop here')

            with self.assertRaises(RuntimeError):
                WalkForwardAnalyzer._execute_fold(
                    data=make_data(),
                    window=self._window(),
                    symbol='TEST',
                    strategy_class=SimpleMAStrategy,
                    parameter_space=None,
                    optimal_parameters=OPTIMAL_PARAMETERS,
                    mode='segmented_in_sample',
                    optimization_method='grid',
                    optimization_metric='total_return',
                    run_config=RunConfig(cost_model=self.cost_model),
                    backtest_engine=None,
                )

        self.assertIs(engine_cls.from_config.call_args.args[0].cost_model, self.cost_model)

    def test_the_cost_model_is_reported_in_the_analysis_parameters(self):
        analyzer = self._analyzer(self.cost_model)
        result = analyzer.analyze(make_data(), 'TEST')

        self.assertEqual(result.analysis_parameters['cost_model'],
                         self.cost_model.description)

    @staticmethod
    def _window():
        from niffler.analysis.walk_forward_analyzer import WalkForwardWindow

        return WalkForwardWindow(
            window_number=1,
            test_start=pd.Timestamp('2022-04-01'),
            test_end=pd.Timestamp('2022-10-01'),
            train_start=None,
            train_end=None,
        )


class TestMonteCarloCarriesTheCostModel(unittest.TestCase):
    """Every simulated path pays the same costs the real one would."""

    def setUp(self):
        self.cost_model = FixedSlippageModel(slippage_bps=25.0)
        self.data = make_data(200)

    def _analyzer(self, cost_model, n_simulations=2, n_jobs=1):
        return MonteCarloAnalyzer(
            strategy_class=SimpleMAStrategy,
            optimal_parameters=OPTIMAL_PARAMETERS,
            n_simulations=n_simulations,
            n_jobs=n_jobs,
            random_seed=7,
            run_config=RunConfig(cost_model=cost_model),
        )

    def test_the_reusable_engine_gets_it(self):
        self.assertIs(self._analyzer(self.cost_model)._backtest_engine.cost_model,
                      self.cost_model)

    def test_the_default_stays_frictionless(self):
        self.assertIsInstance(self._analyzer(None)._backtest_engine.cost_model,
                              ZeroCostModel)

    def test_the_static_worker_applies_it(self):
        free = MonteCarloAnalyzer._run_single_simulation_static(
            self.data, 'TEST', 0, SimpleMAStrategy, OPTIMAL_PARAMETERS,
            0.8, 30, 7, RunConfig()
        )
        costed = MonteCarloAnalyzer._run_single_simulation_static(
            self.data, 'TEST', 0, SimpleMAStrategy, OPTIMAL_PARAMETERS,
            0.8, 30, 7, RunConfig(cost_model=self.cost_model)
        )

        self.assertIsNotNone(free)
        self.assertIsNotNone(costed)
        self.assertGreater(costed.total_slippage, 0.0)
        self.assertEqual(free.total_slippage, 0.0)
        self.assertLess(costed.final_capital, free.final_capital)

    def test_the_parallel_submission_passes_it(self):
        analyzer = self._analyzer(self.cost_model, n_simulations=2, n_jobs=2)

        executor = Mock()
        futures = [Mock(), Mock()]
        for future in futures:
            future.result.return_value = None
        executor.submit.side_effect = futures

        with patch('niffler.analysis.monte_carlo_analyzer.ProcessPoolExecutor') as pool:
            pool.return_value.__enter__.return_value = executor
            with patch('niffler.analysis.monte_carlo_analyzer.as_completed',
                       return_value=futures):
                analyzer._run_simulations_parallel(self.data, 'TEST')

        for call in executor.submit.call_args_list:
            self.assertIs(call.args[-1].cost_model, self.cost_model)

    def test_the_cost_model_is_reported_in_the_analysis_parameters(self):
        analyzer = self._analyzer(self.cost_model)
        result = analyzer.analyze(self.data, 'TEST')

        self.assertEqual(result.analysis_parameters['cost_model'],
                         self.cost_model.description)


if __name__ == '__main__':
    unittest.main()
