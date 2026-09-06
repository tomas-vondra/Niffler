"""
Both analyzers must validate a strategy under the risk rules it would be traded with.

The analyzers build their own strategy objects, so a risk manager attached to a
strategy could never reach them: walk-forward and Monte Carlo ran with risk
management off unconditionally. A Monte Carlo distribution is a risk
measurement, so taking it with the risk layer disabled measures nothing.
"""

import math
import pickle
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
from niffler.backtesting.run_config import RunConfig
from niffler.optimization.parameter_space import ParameterSpace
from niffler.risk import FixedRiskManager, RiskDecision
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy

PARAMETER_SPACE = ParameterSpace({
    'short_window': {'type': 'int', 'min': 5, 'max': 6},
    'long_window': {'type': 'int', 'min': 20, 'max': 21},
})

OPTIMAL_PARAMETERS = {'short_window': 5, 'long_window': 20}


class VetoRiskManager(FixedRiskManager):
    """Refuses every trade, so its presence shows up as zero trades.

    Defined at module level because a spawned worker unpickles it by import
    path; a Mock or a local class could not cross that boundary at all, which is
    what makes this a real test of the spawn path.
    """

    def evaluate_trade(self, signal, current_price, portfolio_value,
                       historical_data, portfolio):
        return RiskDecision(position_size=0.0, stop_loss_price=None,
                            max_risk_per_trade=0.0, allow_trade=False,
                            reason='vetoed by test')


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


class TestTheConfigSurvivesPickling(unittest.TestCase):
    """A manager that cannot be pickled cannot reach a worker process."""

    def test_a_configured_run_config_round_trips(self):
        config = RunConfig(risk_manager=VetoRiskManager(position_size_pct=0.05))

        restored = pickle.loads(pickle.dumps(config))

        self.assertIsInstance(restored.risk_manager, VetoRiskManager)
        self.assertEqual(restored.risk_manager.position_size_pct, 0.05)


class TestWalkForwardCarriesTheRiskManager(unittest.TestCase):
    """Both halves of a fold - the fit and the grade - must use the same rules."""

    def setUp(self):
        self.risk_manager = VetoRiskManager()

    def _analyzer(self, risk_manager, n_jobs=1):
        return WalkForwardAnalyzer(
            strategy_class=SimpleMAStrategy,
            parameter_space=PARAMETER_SPACE,
            train_window_months=6,
            test_window_months=3,
            step_months=3,
            n_jobs=n_jobs,
            run_config=RunConfig(risk_manager=risk_manager),
        )

    def test_the_reusable_engine_gets_it(self):
        analyzer = self._analyzer(self.risk_manager)

        self.assertIs(analyzer._backtest_engine.risk_manager, self.risk_manager)

    def test_the_default_is_still_no_risk_management(self):
        self.assertIsNone(self._analyzer(None)._backtest_engine.risk_manager)

    def test_the_per_fold_optimizer_gets_it(self):
        with patch('niffler.analysis.walk_forward_analyzer.create_optimizer') as factory:
            factory.return_value.optimize.return_value = []

            WalkForwardAnalyzer._optimize_on_training_window(
                train_data=make_data(120),
                strategy_class=SimpleMAStrategy,
                parameter_space=PARAMETER_SPACE,
                optimization_method='grid',
                optimization_metric='total_return',
                run_config=RunConfig(risk_manager=self.risk_manager),
            )

        self.assertIs(factory.call_args.kwargs['run_config'].risk_manager,
                      self.risk_manager)

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
                    run_config=RunConfig(risk_manager=self.risk_manager),
                    backtest_engine=None,
                )

        self.assertIs(engine_cls.from_config.call_args.args[0].risk_manager,
                      self.risk_manager)

    def test_it_reaches_folds_across_the_spawn_boundary(self):
        """Two processes, a vetoing manager, and therefore no trades anywhere."""
        data = make_data()

        traded = self._analyzer(None, n_jobs=2).analyze(data, 'TEST')
        vetoed = self._analyzer(self.risk_manager, n_jobs=2).analyze(data, 'TEST')

        self.assertGreater(sum(r.total_trades for r in traded.individual_results), 0)
        self.assertEqual(sum(r.total_trades for r in vetoed.individual_results), 0)
        self.assertTrue(vetoed.individual_results)

    def test_it_is_recorded_in_the_analysis_parameters(self):
        analyzer = self._analyzer(self.risk_manager)
        result = analyzer.analyze(make_data(), 'TEST')

        self.assertEqual(result.analysis_parameters['risk_manager']['class'],
                         'VetoRiskManager')

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


class TestMonteCarloCarriesTheRiskManager(unittest.TestCase):
    """Every simulated path runs the risk layer the real one would."""

    def setUp(self):
        self.risk_manager = VetoRiskManager()
        self.data = make_data(200)

    def _analyzer(self, risk_manager, n_simulations=2, n_jobs=1):
        return MonteCarloAnalyzer(
            strategy_class=SimpleMAStrategy,
            optimal_parameters=OPTIMAL_PARAMETERS,
            n_simulations=n_simulations,
            n_jobs=n_jobs,
            random_seed=7,
            run_config=RunConfig(risk_manager=risk_manager),
        )

    def test_the_reusable_engine_gets_it(self):
        self.assertIs(self._analyzer(self.risk_manager)._backtest_engine.risk_manager,
                      self.risk_manager)

    def test_the_default_is_still_no_risk_management(self):
        self.assertIsNone(self._analyzer(None)._backtest_engine.risk_manager)

    def test_the_static_worker_applies_it(self):
        unmanaged = MonteCarloAnalyzer._run_single_simulation_static(
            self.data, 'TEST', 0, SimpleMAStrategy, OPTIMAL_PARAMETERS,
            0.8, 30, 7, RunConfig()
        )
        vetoed = MonteCarloAnalyzer._run_single_simulation_static(
            self.data, 'TEST', 0, SimpleMAStrategy, OPTIMAL_PARAMETERS,
            0.8, 30, 7, RunConfig(risk_manager=self.risk_manager)
        )

        self.assertIsNotNone(unmanaged)
        self.assertIsNotNone(vetoed)
        self.assertGreater(unmanaged.total_trades, 0)
        self.assertEqual(vetoed.total_trades, 0)

    def test_the_parallel_submission_passes_it(self):
        analyzer = self._analyzer(self.risk_manager, n_simulations=2, n_jobs=2)

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
            self.assertIs(call.args[-1].risk_manager, self.risk_manager)

    def test_it_is_recorded_in_the_analysis_parameters(self):
        analyzer = self._analyzer(self.risk_manager)
        result = analyzer.analyze(self.data, 'TEST')

        self.assertEqual(result.analysis_parameters['risk_manager']['class'],
                         'VetoRiskManager')


if __name__ == '__main__':
    unittest.main()
