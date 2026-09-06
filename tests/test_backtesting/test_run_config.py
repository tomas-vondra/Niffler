"""One run configuration, carried whole from the command line to the engine.

``BacktestEngine`` takes ten settings. It used to be constructed at six places
and five of them passed three: ``initial_capital``, ``commission`` and
``cost_model``. Everything else - the benchmark, the annualisation factor, the
execution timing, the significance gate, the order floor - silently reverted to
its default inside every walk-forward fold, every Monte Carlo path and every
grid cell, with nothing anywhere reporting that it had.

These tests pin the two halves of the fix. First that ``RunConfig``'s defaults
*are* the engine's defaults, field for field, so nothing moved. Second that a
non-default setting actually arrives at the engine that does the work -
including on the fresh-engine paths a spawned worker takes, which is where the
settings used to be dropped.
"""

import math
import unittest

import pandas as pd

from niffler.analysis.walk_forward_analyzer import (
    MODE_SEGMENTED_IN_SAMPLE,
    WalkForwardAnalyzer,
    WalkForwardWindow,
)
from niffler.analysis.monte_carlo_analyzer import MonteCarloAnalyzer
from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.cost_model import FixedSlippageModel, ZeroCostModel
from niffler.backtesting.run_config import (
    EXECUTION_TIMINGS,
    RunConfig,
    resolve_run_config,
)
from niffler.optimization.optimizer_factory import create_optimizer
from niffler.optimization.parameter_space import ParameterSpace
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy

PARAMETER_SPACE = ParameterSpace({
    'short_window': {'type': 'int', 'min': 5, 'max': 6},
    'long_window': {'type': 'int', 'min': 20, 'max': 21},
})

OPTIMAL_PARAMETERS = {'short_window': 5, 'long_window': 20}

#: Every field of RunConfig, paired with the engine attribute it sets.
ENGINE_FIELDS = (
    'initial_capital', 'commission', 'min_order_value', 'execution_timing',
    'periods_per_year', 'benchmark', 'min_trades_for_significance',
    'bootstrap_samples', 'bootstrap_seed',
)


def make_data(n_bars=500):
    """An oscillating OHLCV frame, so a moving-average cross actually fires."""
    index = pd.date_range('2021-01-01', periods=n_bars, freq='D')
    closes = [100.0 + 20.0 * math.sin(i / 9.0) for i in range(n_bars)]
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.01 for c in closes],
        'low': [c * 0.99 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


class TestDefaultsAreUnchanged(unittest.TestCase):
    """A default RunConfig must configure exactly the engine that existed before."""

    def test_every_field_matches_the_engines_own_default(self):
        engine = BacktestEngine()
        config = RunConfig()

        for field in ENGINE_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(getattr(config, field), getattr(engine, field))

    def test_the_default_cost_model_is_frictionless(self):
        """None is kept as None on the config so 'unconfigured' stays visible,
        and becomes a ZeroCostModel on the engine, as it always did."""
        self.assertIsNone(RunConfig().cost_model)
        self.assertIsInstance(BacktestEngine.from_config(RunConfig()).cost_model,
                              ZeroCostModel)

    def test_the_bootstrap_is_off_by_default(self):
        """The interval is the expensive part; nothing in a loop reads it."""
        self.assertEqual(RunConfig().bootstrap_samples, 0)

    def test_from_config_reproduces_a_directly_constructed_engine(self):
        direct = BacktestEngine(initial_capital=25000.0, commission=0.002,
                                periods_per_year=252.0, benchmark='none')
        via_config = BacktestEngine.from_config(RunConfig(
            initial_capital=25000.0, commission=0.002,
            periods_per_year=252.0, benchmark='none'))

        for field in ENGINE_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(getattr(direct, field), getattr(via_config, field))


class TestValidationLivesInOnePlace(unittest.TestCase):
    """The range checks moved to the config; the engine delegates to them."""

    def test_the_engine_still_rejects_what_it_always_rejected(self):
        for kwargs, message in (
            ({'initial_capital': -1}, 'Initial capital must be positive'),
            ({'commission': -0.1}, 'Commission cannot be negative'),
            ({'min_order_value': -1}, 'Minimum order value cannot be negative'),
            ({'execution_timing': 'nope'}, 'Execution timing must be one of'),
            ({'periods_per_year': 0}, 'Periods per year must be positive'),
            ({'benchmark': 'spx'}, 'Benchmark must be one of'),
            ({'min_trades_for_significance': -1},
             'Minimum trades for significance cannot be negative'),
            ({'bootstrap_samples': -1}, 'Bootstrap samples cannot be negative'),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as engine_ctx:
                    BacktestEngine(**kwargs)
                with self.assertRaises(ValueError) as config_ctx:
                    RunConfig(**kwargs)

                self.assertIn(message, str(engine_ctx.exception))
                self.assertEqual(str(engine_ctx.exception), str(config_ctx.exception))

    def test_a_non_cost_model_is_a_type_error(self):
        with self.assertRaises(TypeError):
            RunConfig(cost_model='fixed')

    def test_a_none_benchmark_is_normalised_not_rejected(self):
        """The engine has always taken None to mean 'no benchmark'."""
        self.assertEqual(RunConfig(benchmark=None).benchmark, 'none')

    def test_the_execution_timings_tuple_is_defined_once(self):
        self.assertIs(BacktestEngine.EXECUTION_TIMINGS, EXECUTION_TIMINGS)

    def test_the_config_is_immutable(self):
        """A config handed to an analyzer cannot change under the backtests
        that already used it."""
        config = RunConfig()
        with self.assertRaises(Exception):
            config.commission = 0.5


class TestResolveRunConfig(unittest.TestCase):

    def test_none_becomes_the_default_config(self):
        self.assertEqual(resolve_run_config(None), RunConfig())

    def test_a_given_config_is_returned_unchanged(self):
        config = RunConfig(initial_capital=1234.0)
        self.assertIs(resolve_run_config(config), config)


class TestTheConfigReachesTheEngine(unittest.TestCase):
    """The regression this whole change exists to prevent.

    Each assertion sets a knob no call site used to forward and checks that the
    engine that actually runs a backtest received it.
    """

    def setUp(self):
        self.config = RunConfig(periods_per_year=252.0, benchmark='none',
                                min_order_value=250.0,
                                cost_model=FixedSlippageModel(slippage_bps=25.0))

    def test_the_optimizers_reusable_engine_gets_every_setting(self):
        optimizer = create_optimizer(
            method='grid',
            strategy_class=SimpleMAStrategy,
            parameter_space=PARAMETER_SPACE,
            data=make_data(200),
            n_jobs=1,
            run_config=self.config,
        )

        engine = optimizer._backtest_engine
        self.assertEqual(engine.periods_per_year, 252.0)
        self.assertEqual(engine.benchmark, 'none')
        self.assertEqual(engine.min_order_value, 250.0)
        self.assertIs(engine.cost_model, self.config.cost_model)

    def test_the_optimizers_worker_engine_gets_every_setting(self):
        """The spawn path builds its own engine, which is where the settings
        used to be lost."""
        result = type(create_optimizer(
            method='grid', strategy_class=SimpleMAStrategy,
            parameter_space=PARAMETER_SPACE, data=make_data(200), n_jobs=1,
        ))._evaluate_single_combination_static(
            OPTIMAL_PARAMETERS, SimpleMAStrategy, make_data(200), self.config)

        self.assertIsNotNone(result)
        # benchmark='none' reached the worker, so there is no comparison at all.
        self.assertIsNone(result.backtest_result.benchmark_return_pct)

    def test_the_walk_forward_analyzers_reusable_engine_gets_every_setting(self):
        analyzer = WalkForwardAnalyzer(
            strategy_class=SimpleMAStrategy,
            parameter_space=PARAMETER_SPACE,
            train_window_months=6,
            test_window_months=3,
            step_months=3,
            n_jobs=1,
            run_config=self.config,
        )

        engine = analyzer._backtest_engine
        self.assertEqual(engine.periods_per_year, 252.0)
        self.assertEqual(engine.benchmark, 'none')
        self.assertEqual(engine.min_order_value, 250.0)

    def test_a_fold_that_builds_its_own_engine_gets_every_setting(self):
        """The parallel fold path. Break _execute_fold so it ignores
        run_config.benchmark and this is the assertion that fails."""
        fold = WalkForwardAnalyzer._execute_fold(
            data=make_data(),
            window=WalkForwardWindow(
                window_number=1,
                test_start=pd.Timestamp('2021-04-01'),
                test_end=pd.Timestamp('2021-10-01'),
            ),
            symbol='TEST',
            strategy_class=SimpleMAStrategy,
            parameter_space=None,
            optimal_parameters=OPTIMAL_PARAMETERS,
            mode=MODE_SEGMENTED_IN_SAMPLE,
            optimization_method='grid',
            optimization_metric='total_return',
            run_config=RunConfig(benchmark='none'),
            backtest_engine=None,
        )

        self.assertIsNotNone(fold)
        self.assertIsNone(fold.test_result.benchmark_return_pct)

    def test_a_fold_keeps_the_benchmark_when_the_config_asks_for_one(self):
        """The positive control: the assertion above is not passing by accident."""
        fold = WalkForwardAnalyzer._execute_fold(
            data=make_data(),
            window=WalkForwardWindow(
                window_number=1,
                test_start=pd.Timestamp('2021-04-01'),
                test_end=pd.Timestamp('2021-10-01'),
            ),
            symbol='TEST',
            strategy_class=SimpleMAStrategy,
            parameter_space=None,
            optimal_parameters=OPTIMAL_PARAMETERS,
            mode=MODE_SEGMENTED_IN_SAMPLE,
            optimization_method='grid',
            optimization_metric='total_return',
            run_config=RunConfig(),
            backtest_engine=None,
        )

        self.assertIsNotNone(fold)
        self.assertIsNotNone(fold.test_result.benchmark_return_pct)

    def test_the_monte_carlo_analyzers_engine_gets_every_setting(self):
        analyzer = MonteCarloAnalyzer(
            strategy_class=SimpleMAStrategy,
            optimal_parameters=OPTIMAL_PARAMETERS,
            n_simulations=1,
            n_jobs=1,
            random_seed=7,
            run_config=self.config,
        )

        engine = analyzer._backtest_engine
        self.assertEqual(engine.periods_per_year, 252.0)
        self.assertEqual(engine.benchmark, 'none')
        self.assertEqual(engine.min_order_value, 250.0)

    def test_a_monte_carlo_worker_gets_every_setting(self):
        result = MonteCarloAnalyzer._run_single_simulation_static(
            make_data(200), 'TEST', 0, SimpleMAStrategy, OPTIMAL_PARAMETERS,
            0.8, 30, 7, RunConfig(benchmark='none'))

        self.assertIsNotNone(result)
        self.assertIsNone(result.benchmark_return_pct)

    def test_the_config_survives_pickling(self):
        """Worker processes are spawned on Windows, so the config crosses a
        pickle boundary before it reaches any engine."""
        import pickle

        restored = pickle.loads(pickle.dumps(self.config))

        self.assertEqual(restored.periods_per_year, 252.0)
        self.assertEqual(restored.benchmark, 'none')
        self.assertEqual(restored.cost_model.description,
                         self.config.cost_model.description)


class TestConfigMetadata(unittest.TestCase):

    def test_to_metadata_covers_every_field(self):
        metadata = RunConfig().to_metadata()

        for field in ENGINE_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, metadata)
        self.assertIn('cost_model', metadata)

    def test_the_cost_model_is_reduced_to_its_description(self):
        metadata = RunConfig(cost_model=FixedSlippageModel(slippage_bps=5.0))\
            .to_metadata()

        self.assertIsInstance(metadata['cost_model'], str)

    def test_an_unconfigured_cost_model_stays_none_rather_than_naming_a_default(self):
        self.assertIsNone(RunConfig().to_metadata()['cost_model'])

    def test_the_risk_manager_is_reduced_to_a_reconstructable_record(self):
        from niffler.risk import FixedRiskManager

        metadata = RunConfig(
            risk_manager=FixedRiskManager(position_size_pct=0.3)).to_metadata()

        self.assertEqual(metadata['risk_manager']['name'], 'fixed')
        self.assertEqual(metadata['risk_manager']['parameters']['position_size_pct'],
                         0.3)

    def test_no_risk_management_is_recorded_as_such(self):
        self.assertEqual(RunConfig().to_metadata()['risk_manager']['name'], 'none')

    def test_the_analyzers_record_the_whole_configuration(self):
        analyzer = WalkForwardAnalyzer(
            strategy_class=SimpleMAStrategy,
            parameter_space=PARAMETER_SPACE,
            train_window_months=6,
            test_window_months=3,
            step_months=3,
            n_jobs=1,
            run_config=RunConfig(periods_per_year=252.0),
        )
        result = analyzer.analyze(make_data(), 'TEST')

        self.assertEqual(result.analysis_parameters['periods_per_year'], 252.0)
        self.assertEqual(result.analysis_parameters['benchmark'], 'buy_and_hold')


if __name__ == '__main__':
    unittest.main()
