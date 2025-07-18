import unittest
from unittest.mock import Mock
from niffler.optimization.optimization_result import OptimizationResult


class TestOptimizationResult(unittest.TestCase):
    """Test cases for OptimizationResult."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_backtest_result = Mock()
        self.mock_backtest_result.total_return = 0.15
        self.mock_backtest_result.sharpe_ratio = 1.5
        self.mock_backtest_result.max_drawdown = 0.05
        
        self.valid_parameters = {
            'short_window': 10,
            'long_window': 20,
            'threshold': 0.02
        }
    
    def test_valid_optimization_result_creation(self):
        """Test creating a valid optimization result."""
        result = OptimizationResult(
            parameters=self.valid_parameters,
            backtest_result=self.mock_backtest_result
        )
        
        self.assertEqual(result.parameters, self.valid_parameters)
        self.assertEqual(result.backtest_result, self.mock_backtest_result)
    
    def test_empty_parameters_dict(self):
        """Test creating result with empty parameters dict."""
        result = OptimizationResult(
            parameters={},
            backtest_result=self.mock_backtest_result
        )
        
        self.assertEqual(result.parameters, {})
        self.assertEqual(result.backtest_result, self.mock_backtest_result)
    
    def test_invalid_parameters_type(self):
        """Test validation fails with non-dict parameters."""
        with self.assertRaises(TypeError) as context:
            OptimizationResult(
                parameters="invalid",
                backtest_result=self.mock_backtest_result
            )
        
        self.assertIn("parameters must be a dictionary", str(context.exception))
    
    def test_invalid_parameters_list(self):
        """Test validation fails with list parameters."""
        with self.assertRaises(TypeError) as context:
            OptimizationResult(
                parameters=['param1', 'param2'],
                backtest_result=self.mock_backtest_result
            )
        
        self.assertIn("parameters must be a dictionary", str(context.exception))
    
    def test_none_backtest_result(self):
        """Test validation fails with None backtest result."""
        with self.assertRaises(ValueError) as context:
            OptimizationResult(
                parameters=self.valid_parameters,
                backtest_result=None
            )
        
        self.assertIn("backtest_result cannot be None", str(context.exception))
    
    def test_parameters_with_various_types(self):
        """Test parameters with different value types."""
        mixed_parameters = {
            'int_param': 10,
            'float_param': 1.5,
            'string_param': 'test',
            'bool_param': True,
            'list_param': [1, 2, 3]
        }
        
        result = OptimizationResult(
            parameters=mixed_parameters,
            backtest_result=self.mock_backtest_result
        )
        
        self.assertEqual(result.parameters, mixed_parameters)
    
    def test_dataclass_equality(self):
        """Test that two results with same data are equal."""
        result1 = OptimizationResult(
            parameters=self.valid_parameters,
            backtest_result=self.mock_backtest_result
        )
        
        result2 = OptimizationResult(
            parameters=self.valid_parameters.copy(),
            backtest_result=self.mock_backtest_result
        )
        
        self.assertEqual(result1, result2)
    
    def test_dataclass_inequality(self):
        """Test that results with different parameters are not equal."""
        result1 = OptimizationResult(
            parameters=self.valid_parameters,
            backtest_result=self.mock_backtest_result
        )
        
        different_parameters = self.valid_parameters.copy()
        different_parameters['short_window'] = 15
        
        result2 = OptimizationResult(
            parameters=different_parameters,
            backtest_result=self.mock_backtest_result
        )
        
        self.assertNotEqual(result1, result2)
    
    def test_repr_and_str(self):
        """Test string representation of result."""
        result = OptimizationResult(
            parameters=self.valid_parameters,
            backtest_result=self.mock_backtest_result
        )
        
        result_str = str(result)
        result_repr = repr(result)
        
        # Should contain the class name and key information
        self.assertIn("OptimizationResult", result_repr)
        self.assertIn("short_window", result_str)


if __name__ == '__main__':
    unittest.main()