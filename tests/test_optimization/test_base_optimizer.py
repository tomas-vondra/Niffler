import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import threading
import signal
from typing import List

from niffler.optimization.base_optimizer import BaseOptimizer
from niffler.optimization.parameter_space import ParameterSpace
from niffler.optimization.optimization_result import OptimizationResult
from niffler.strategies.base_strategy import BaseStrategy


# Concrete implementation for testing
class TestOptimizer(BaseOptimizer):
    """Concrete implementation of BaseOptimizer for testing."""
    
    def optimize(self) -> List[OptimizationResult]:
        """Simple implementation for testing."""
        # Generate a few test combinations
        combinations = [
            {'param1': 10, 'param2': 0.5},
            {'param1': 15, 'param2': 0.7},
            {'param1': 20, 'param2': 0.9}
        ]
        return self._evaluate_combinations(combinations)


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, param1=10, param2=0.5):
        super().__init__("MockStrategy", {"param1": param1, "param2": param2})
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data):
        """Mock signal generation."""
        signals_df = pd.DataFrame(index=data.index)
        signals_df['signal'] = 0
        signals_df['position_size'] = 0.0
        return signals_df
    
    def get_description(self) -> str:
        """Return a description of the strategy."""
        return "Mock strategy for testing purposes"


class TestBaseOptimizer(unittest.TestCase):
    """Test cases for BaseOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data with DatetimeIndex
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        self.test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 4,
            'high': [101, 102, 103, 104, 105] * 4,
            'low': [99, 100, 101, 102, 103] * 4,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 4,
            'volume': [1000, 1100, 1200, 1300, 1400] * 4
        }, index=dates)
        
        # Create parameter space
        self.parameter_space = ParameterSpace({
            'param1': {'type': 'int', 'min': 10, 'max': 20, 'step': 5},
            'param2': {'type': 'float', 'min': 0.1, 'max': 1.0, 'step': 0.2}
        })
        
        # Mock backtest result
        self.mock_backtest_result = Mock()
        self.mock_backtest_result.total_return = 0.15
        self.mock_backtest_result.sharpe_ratio = 1.5
        self.mock_backtest_result.max_drawdown = 0.05
        self.mock_backtest_result.win_rate = 0.6
        self.mock_backtest_result.total_trades = 10
        self.mock_backtest_result.total_profits = 1000
        self.mock_backtest_result.total_return_pct = 15.0
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization with valid parameters."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data,
            initial_capital=10000,
            commission=0.001
        )
        
        self.assertEqual(optimizer.strategy_class, MockStrategy)
        self.assertEqual(optimizer.parameter_space, self.parameter_space)
        self.assertEqual(optimizer.initial_capital, 10000)
        self.assertEqual(optimizer.commission, 0.001)
        self.assertEqual(optimizer.sort_by, 'total_return')
    
    def test_optimizer_initialization_with_custom_sort(self):
        """Test optimizer initialization with custom sort metric."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data,
            sort_by='sharpe_ratio'
        )
        
        self.assertEqual(optimizer.sort_by, 'sharpe_ratio')
    
    def test_invalid_initial_capital(self):
        """Test validation fails with invalid initial capital."""
        with self.assertRaises(ValueError) as context:
            TestOptimizer(
                strategy_class=MockStrategy,
                parameter_space=self.parameter_space,
                data=self.test_data,
                initial_capital=-1000
            )
        
        self.assertIn("initial_capital must be positive", str(context.exception))
    
    def test_invalid_commission(self):
        """Test validation fails with invalid commission."""
        with self.assertRaises(ValueError) as context:
            TestOptimizer(
                strategy_class=MockStrategy,
                parameter_space=self.parameter_space,
                data=self.test_data,
                commission=-0.1
            )
        
        self.assertIn("commission cannot be negative", str(context.exception))
    
    def test_invalid_sort_by(self):
        """Test validation fails with invalid sort metric."""
        with self.assertRaises(ValueError) as context:
            TestOptimizer(
                strategy_class=MockStrategy,
                parameter_space=self.parameter_space,
                data=self.test_data,
                sort_by='invalid_metric'
            )
        
        self.assertIn("sort_by must be one of", str(context.exception))
    
    def test_empty_data(self):
        """Test validation fails with empty data."""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            TestOptimizer(
                strategy_class=MockStrategy,
                parameter_space=self.parameter_space,
                data=empty_data
            )
        
        self.assertIn("data cannot be empty", str(context.exception))
    
    def test_missing_data_columns(self):
        """Test validation fails with missing required columns."""
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103]
            # Missing low, close, volume
        })
        
        with self.assertRaises(ValueError) as context:
            TestOptimizer(
                strategy_class=MockStrategy,
                parameter_space=self.parameter_space,
                data=incomplete_data
            )
        
        self.assertIn("data is missing required columns", str(context.exception))
    
    def test_signal_handler_setup(self):
        """Test signal handler setup and shutdown functionality."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Initially, shutdown should not be requested
        self.assertFalse(optimizer._check_shutdown())
        
        # Simulate signal
        with optimizer._shutdown_lock:
            optimizer._shutdown_requested = True
        
        # Now shutdown should be requested
        self.assertTrue(optimizer._check_shutdown())
    
    def test_generate_float_range(self):
        """Test float range generation with decimal precision."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        float_range = optimizer._generate_float_range(0.1, 0.5, 0.1)
        expected = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        self.assertEqual(len(float_range), len(expected))
        for i, value in enumerate(float_range):
            self.assertAlmostEqual(value, expected[i], places=10)
    
    def test_generate_int_range(self):
        """Test integer range generation."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        int_range = optimizer._generate_int_range(1, 10, 2)
        expected = [1, 3, 5, 7, 9]
        
        self.assertEqual(int_range, expected)
    
    def test_calculate_float_steps(self):
        """Test float step calculation."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        min_steps, max_steps = optimizer._calculate_float_steps(0.1, 0.5, 0.1)
        
        self.assertEqual(min_steps, 1)
        self.assertEqual(max_steps, 5)
    
    def test_steps_to_float(self):
        """Test converting steps back to float value."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        result = optimizer._steps_to_float(3, 0.1)
        self.assertAlmostEqual(result, 0.3, places=10)
    
    def test_count_parameter_combinations(self):
        """Test counting parameter combinations."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Test int parameter
        int_config = {'type': 'int', 'min': 1, 'max': 10, 'step': 2}
        count = optimizer._count_parameter_combinations('test', int_config)
        self.assertEqual(count, 5)  # [1, 3, 5, 7, 9]
        
        # Test float parameter with step
        float_config = {'type': 'float', 'min': 0.1, 'max': 0.5, 'step': 0.1}
        count = optimizer._count_parameter_combinations('test', float_config)
        self.assertEqual(count, 5)  # [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Test float parameter without step (continuous)
        continuous_config = {'type': 'float', 'min': 0.0, 'max': 1.0}
        count = optimizer._count_parameter_combinations('test', continuous_config)
        self.assertEqual(count, float('inf'))
        
        # Test choice parameter
        choice_config = {'type': 'choice', 'choices': ['a', 'b', 'c']}
        count = optimizer._count_parameter_combinations('test', choice_config)
        self.assertEqual(count, 3)
    
    def test_generate_parameter_values(self):
        """Test generating parameter values for grid search."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Test int parameter
        int_config = {'type': 'int', 'min': 1, 'max': 5, 'step': 2}
        values = optimizer._generate_parameter_values('test', int_config)
        self.assertEqual(values, [1, 3, 5])
        
        # Test choice parameter
        choice_config = {'type': 'choice', 'choices': ['a', 'b', 'c']}
        values = optimizer._generate_parameter_values('test', choice_config)
        self.assertEqual(values, ['a', 'b', 'c'])
    
    def test_generate_random_parameter_value(self):
        """Test generating random parameter values."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Test int parameter
        int_config = {'type': 'int', 'min': 1, 'max': 10}
        value = optimizer._generate_random_parameter_value('test', int_config)
        self.assertIsInstance(value, int)
        self.assertGreaterEqual(value, 1)
        self.assertLessEqual(value, 10)
        
        # Test choice parameter
        choice_config = {'type': 'choice', 'choices': ['a', 'b', 'c']}
        value = optimizer._generate_random_parameter_value('test', choice_config)
        self.assertIn(value, ['a', 'b', 'c'])
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_evaluate_single_combination(self, mock_engine_class):
        """Test evaluating a single parameter combination."""
        # Setup mock
        mock_engine = Mock()
        mock_engine.run_backtest.return_value = self.mock_backtest_result
        mock_engine_class.return_value = mock_engine
        
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Mock the reusable engine
        optimizer._backtest_engine = mock_engine
        
        parameters = {'param1': 10, 'param2': 0.5}
        result = optimizer._evaluate_single_combination(parameters)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.parameters, parameters)
        self.assertEqual(result.backtest_result, self.mock_backtest_result)
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_evaluate_single_combination_with_error(self, mock_engine_class):
        """Test evaluating combination that raises an error."""
        # Setup mock to raise exception
        mock_engine = Mock()
        mock_engine.run_backtest.side_effect = Exception("Test error")
        mock_engine_class.return_value = mock_engine
        
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        optimizer._backtest_engine = mock_engine
        
        parameters = {'param1': 10, 'param2': 0.5}
        result = optimizer._evaluate_single_combination(parameters)
        
        self.assertIsNone(result)
    
    def test_memory_efficient_results_management(self):
        """Test memory efficient results management."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Create mock results with different performance
        results = []
        for i in range(5):
            mock_result = Mock()
            mock_result.total_return_pct = 10.0 + i * 5.0  # 10.0, 15.0, 20.0, 25.0, 30.0
            result = OptimizationResult(
                parameters={'param': i},
                backtest_result=mock_result
            )
            results.append(result)
        
        # Simulate adding results one by one with memory limit
        optimizer.MAX_RESULTS_IN_MEMORY = 3
        managed_results = []
        
        for result in results:
            managed_results = optimizer._manage_memory_efficient_results(managed_results, result)
        
        # Should keep only the best results when limit is exceeded
        self.assertLessEqual(len(managed_results), optimizer.MAX_RESULTS_IN_MEMORY)
    
    def test_analyze_best_metrics(self):
        """Test analyzing best metrics across results."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Create mock results with different metrics
        results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.total_return = 0.1 + i * 0.05
            mock_result.sharpe_ratio = 1.0 + i * 0.5
            mock_result.max_drawdown = 0.1 - i * 0.02
            mock_result.win_rate = 0.5 + i * 0.1
            mock_result.total_trades = 10 + i * 5
            mock_result.total_profits = 1000 + i * 200
            mock_result.total_return_pct = (0.1 + i * 0.05) * 100
            
            result = OptimizationResult(
                parameters={'param': i},
                backtest_result=mock_result
            )
            results.append(result)
        
        best_metrics = optimizer.analyze_best_metrics(results)
        
        # Should have entries for all metrics
        expected_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']
        for metric in expected_metrics:
            self.assertIn(metric, best_metrics)
            self.assertIn('parameters', best_metrics[metric])
            self.assertIn('value', best_metrics[metric])
    
    def test_analyze_best_metrics_empty_results(self):
        """Test analyzing best metrics with empty results."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        best_metrics = optimizer.analyze_best_metrics([])
        self.assertEqual(best_metrics, {})
    
    @patch('builtins.open', unittest.mock.mock_open())
    @patch('json.dump')
    def test_save_results(self, mock_json_dump):
        """Test saving optimization results to file."""
        optimizer = TestOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Create mock result
        result = OptimizationResult(
            parameters={'param1': 10},
            backtest_result=self.mock_backtest_result
        )
        
        optimizer.save_results([result], 'test_results.json')
        
        # Verify that json.dump was called
        mock_json_dump.assert_called_once()
        
        # Check the structure of saved data
        saved_data = mock_json_dump.call_args[0][0]
        self.assertIn('metadata', saved_data)
        self.assertIn('results', saved_data)
        self.assertEqual(len(saved_data['results']), 1)


if __name__ == '__main__':
    unittest.main()