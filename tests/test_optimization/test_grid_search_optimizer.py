import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from typing import List

from niffler.optimization.grid_search_optimizer import GridSearchOptimizer
from niffler.optimization.parameter_space import ParameterSpace
from niffler.optimization.optimization_result import OptimizationResult
from niffler.strategies.base_strategy import BaseStrategy


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


class TestGridSearchOptimizer(unittest.TestCase):
    """Test cases for GridSearchOptimizer."""
    
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
        
        # Create small parameter space for testing
        self.small_parameter_space = ParameterSpace({
            'param1': {'type': 'int', 'min': 10, 'max': 12, 'step': 1},
            'param2': {'type': 'float', 'min': 0.1, 'max': 0.3, 'step': 0.1}
        })
        
        # Create parameter space with choice
        self.choice_parameter_space = ParameterSpace({
            'param1': {'type': 'int', 'min': 10, 'max': 11, 'step': 1},
            'param2': {'type': 'choice', 'choices': ['a', 'b']}
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
    
    def test_grid_search_optimizer_initialization(self):
        """Test grid search optimizer initialization."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        self.assertEqual(optimizer.strategy_class, MockStrategy)
        self.assertEqual(optimizer.parameter_space, self.small_parameter_space)
        self.assertEqual(optimizer.MAX_COMBINATIONS_WARNING, 1000000)
    
    def test_estimate_combinations_count_small(self):
        """Test estimating combinations count for small parameter space."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        count = optimizer._estimate_combinations_count()
        # param1: 10, 11, 12 (3 values)
        # param2: 0.1, 0.2, 0.3 (3 values)
        # Total: 3 * 3 = 9
        self.assertEqual(count, 9)
    
    def test_estimate_combinations_count_with_choice(self):
        """Test estimating combinations count with choice parameter."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.choice_parameter_space,
            data=self.test_data
        )
        
        count = optimizer._estimate_combinations_count()
        # param1: 10, 11 (2 values)
        # param2: 'a', 'b' (2 values)
        # Total: 2 * 2 = 4
        self.assertEqual(count, 4)
    
    def test_estimate_combinations_count_continuous_parameter(self):
        """Test estimating combinations count with continuous parameter raises error."""
        continuous_space = ParameterSpace({
            'param1': {'type': 'float', 'min': 0.0, 'max': 1.0}  # No step
        })
        
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=continuous_space,
            data=self.test_data
        )
        
        with self.assertRaises(ValueError) as context:
            optimizer._estimate_combinations_count()
        
        self.assertIn("Grid search requires step size for float parameter", str(context.exception))
    
    @patch('niffler.optimization.grid_search_optimizer.logging')
    def test_estimate_combinations_count_large_warns(self, mock_logging):
        """Test that large combination count triggers warning."""
        large_space = ParameterSpace({
            'param1': {'type': 'int', 'min': 1, 'max': 1001, 'step': 1},
            'param2': {'type': 'int', 'min': 1, 'max': 1000, 'step': 1}
        })
        
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=large_space,
            data=self.test_data
        )
        
        count = optimizer._estimate_combinations_count()
        
        # Should have called warning
        mock_logging.warning.assert_called()
        warning_call = mock_logging.warning.call_args[0][0]
        self.assertIn("Grid search will generate", warning_call)
    
    def test_build_param_values(self):
        """Test building parameter value lists."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        param_values = optimizer._build_param_values()
        
        self.assertIn('param1', param_values)
        self.assertIn('param2', param_values)
        
        # Check int parameter values
        self.assertEqual(param_values['param1'], [10, 11, 12])
        
        # Check float parameter values
        expected_float = [0.1, 0.2, 0.3]
        for i, value in enumerate(param_values['param2']):
            self.assertAlmostEqual(value, expected_float[i], places=10)
    
    def test_build_param_values_with_choice(self):
        """Test building parameter values with choice parameter."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.choice_parameter_space,
            data=self.test_data
        )
        
        param_values = optimizer._build_param_values()
        
        self.assertEqual(param_values['param1'], [10, 11])
        self.assertEqual(param_values['param2'], ['a', 'b'])
    
    def test_generate_grid_combinations_lazy(self):
        """Test lazy generation of grid combinations."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        combinations_generator = optimizer._generate_grid_combinations_lazy()
        combinations = list(combinations_generator)
        
        # Should generate 9 combinations (3 * 3)
        self.assertEqual(len(combinations), 9)
        
        # Check that all combinations are present
        param1_values = {10, 11, 12}
        param2_values = {0.1, 0.2, 0.3}
        
        for combo in combinations:
            self.assertIn('param1', combo)
            self.assertIn('param2', combo)
            self.assertIn(combo['param1'], param1_values)
            self.assertAlmostEqual(combo['param2'], 
                                 min(param2_values, key=lambda x: abs(x - combo['param2'])), 
                                 places=1)
    
    def test_evaluate_combinations_lazy(self):
        """Test lazy evaluation of combinations."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        # Mock the evaluation method
        mock_result = OptimizationResult(
            parameters={'param1': 10, 'param2': 0.1},
            backtest_result=self.mock_backtest_result
        )
        
        with patch.object(optimizer, '_evaluate_single_combination', return_value=mock_result):
            with patch.object(optimizer, '_sort_and_log_results', return_value=[mock_result]):
                combinations_gen = iter([{'param1': 10, 'param2': 0.1}])
                
                results = optimizer._evaluate_combinations_lazy(combinations_gen, 1)
                
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0], mock_result)
    
    def test_evaluate_combinations_lazy_with_shutdown(self):
        """Test lazy evaluation stops on shutdown signal."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        # Mock shutdown after first evaluation
        call_count = 0
        def mock_check_shutdown():
            nonlocal call_count
            call_count += 1
            return call_count > 1  # Return True after first call
        
        with patch.object(optimizer, '_check_shutdown', side_effect=mock_check_shutdown):
            with patch.object(optimizer, '_evaluate_single_combination', return_value=None):
                with patch.object(optimizer, '_sort_and_log_results', return_value=[]):
                    combinations_gen = iter([
                        {'param1': 10, 'param2': 0.1},
                        {'param1': 11, 'param2': 0.2}
                    ])
                    
                    results = optimizer._evaluate_combinations_lazy(combinations_gen, 2)
                    
                    # Should stop after first iteration due to shutdown
                    self.assertEqual(len(results), 0)
    
    @patch('niffler.optimization.grid_search_optimizer.logging')
    def test_optimize_complete_flow(self, mock_logging):
        """Test complete optimization flow."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        # Mock the evaluation to return successful results
        mock_result = OptimizationResult(
            parameters={'param1': 10, 'param2': 0.1},
            backtest_result=self.mock_backtest_result
        )
        
        with patch.object(optimizer, '_evaluate_single_combination', return_value=mock_result):
            results = optimizer.optimize()
            
            # Should have results
            self.assertGreater(len(results), 0)
            
            # Should have logged the start of optimization
            mock_logging.info.assert_called()
            log_calls = [call[0][0] for call in mock_logging.info.call_args_list]
            start_messages = [msg for msg in log_calls if "Starting grid search" in msg]
            self.assertTrue(any(start_messages))
    
    def test_optimize_with_no_results(self):
        """Test optimization when all evaluations fail."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        # Mock all evaluations to return None (failure)
        with patch.object(optimizer, '_evaluate_single_combination', return_value=None):
            results = optimizer.optimize()
            
            # Should return empty list
            self.assertEqual(len(results), 0)
    
    def test_optimize_with_mixed_results(self):
        """Test optimization with some successful and some failed evaluations."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        # Mock evaluation to return success/failure alternately
        call_count = 0
        def mock_evaluate(params):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return None  # Failure
            else:
                return OptimizationResult(
                    parameters=params,
                    backtest_result=self.mock_backtest_result
                )
        
        with patch.object(optimizer, '_evaluate_single_combination', side_effect=mock_evaluate):
            results = optimizer.optimize()
            
            # Should have some results (approximately half)
            self.assertGreater(len(results), 0)
            self.assertLess(len(results), 9)  # Less than total combinations
    
    def test_optimizer_inheritance(self):
        """Test that GridSearchOptimizer properly inherits from BaseOptimizer."""
        optimizer = GridSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        # Should have inherited methods
        self.assertTrue(hasattr(optimizer, '_evaluate_combinations'))
        self.assertTrue(hasattr(optimizer, '_sort_and_log_results'))
        self.assertTrue(hasattr(optimizer, 'analyze_best_metrics'))
        
        # Should have access to configuration constants
        self.assertTrue(hasattr(optimizer, 'METRICS_CONFIG'))
        self.assertTrue(hasattr(optimizer, 'DEFAULT_INITIAL_CAPITAL'))


if __name__ == '__main__':
    unittest.main()