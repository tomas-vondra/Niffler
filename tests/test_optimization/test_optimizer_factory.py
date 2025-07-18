import unittest
from unittest.mock import Mock, patch
import pandas as pd

from niffler.optimization.optimizer_factory import create_optimizer
from niffler.optimization.grid_search_optimizer import GridSearchOptimizer
from niffler.optimization.random_search_optimizer import RandomSearchOptimizer
from niffler.optimization.parameter_space import ParameterSpace, SIMPLE_MA_PARAMETER_SPACE
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
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


class TestOptimizerFactory(unittest.TestCase):
    """Test cases for optimizer factory functions."""
    
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
        
        # Create test parameter space
        self.test_parameter_space = ParameterSpace({
            'param1': {'type': 'int', 'min': 10, 'max': 20, 'step': 1},
            'param2': {'type': 'float', 'min': 0.1, 'max': 1.0, 'step': 0.1}
        })
    
    def test_create_grid_search_optimizer(self):
        """Test creating a grid search optimizer."""
        optimizer = create_optimizer(
            method='grid',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data,
            initial_capital=10000,
            commission=0.001,
            sort_by='total_return',
            n_jobs=1
        )
        
        self.assertIsInstance(optimizer, GridSearchOptimizer)
        self.assertEqual(optimizer.strategy_class, MockStrategy)
        self.assertEqual(optimizer.initial_capital, 10000)
        self.assertEqual(optimizer.commission, 0.001)
        self.assertEqual(optimizer.sort_by, 'total_return')
        self.assertEqual(optimizer.n_jobs, 1)
    
    def test_create_random_search_optimizer(self):
        """Test creating a random search optimizer."""
        optimizer = create_optimizer(
            method='random',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data,
            initial_capital=5000,
            commission=0.002,
            sort_by='sharpe_ratio',
            n_jobs=2
        )
        
        self.assertIsInstance(optimizer, RandomSearchOptimizer)
        self.assertEqual(optimizer.strategy_class, MockStrategy)
        self.assertEqual(optimizer.initial_capital, 5000)
        self.assertEqual(optimizer.commission, 0.002)
        self.assertEqual(optimizer.sort_by, 'sharpe_ratio')
        self.assertEqual(optimizer.n_jobs, 2)
    
    def test_create_optimizer_with_defaults(self):
        """Test creating optimizer with default parameters."""
        optimizer = create_optimizer(
            method='grid',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data
        )
        
        self.assertIsInstance(optimizer, GridSearchOptimizer)
        self.assertEqual(optimizer.initial_capital, 10000.0)  # Default
        self.assertEqual(optimizer.commission, 0.001)  # Default
        self.assertEqual(optimizer.sort_by, 'total_return')  # Default
        # n_jobs gets auto-detected to system CPU count, so just check it's positive
        self.assertIsInstance(optimizer.n_jobs, int)
        self.assertGreater(optimizer.n_jobs, 0)
    
    def test_invalid_optimizer_type(self):
        """Test creating optimizer with invalid type."""
        with self.assertRaises(ValueError) as context:
            create_optimizer(
                method='invalid_type',
                strategy_class=MockStrategy,
                parameter_space=self.test_parameter_space,
                data=self.test_data
            )
        
        self.assertIn("Unknown optimization method", str(context.exception))
    
    def test_strategy_class_assignment(self):
        """Test that strategy class is correctly assigned."""
        optimizer = create_optimizer(
            method='grid',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data
        )
        
        # Verify strategy class is assigned correctly
        self.assertEqual(optimizer.strategy_class, MockStrategy)
    
    def test_optimizer_methods_work(self):
        """Test that the exact method names work."""
        optimizer1 = create_optimizer(
            method='grid',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data
        )
        
        optimizer2 = create_optimizer(
            method='random',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data
        )
        
        self.assertIsInstance(optimizer1, GridSearchOptimizer)
        self.assertIsInstance(optimizer2, RandomSearchOptimizer)
    
    def test_strategy_class_passed_correctly(self):
        """Test that strategy class is passed correctly."""
        optimizer1 = create_optimizer(
            method='grid',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data
        )
        
        optimizer2 = create_optimizer(
            method='grid',
            strategy_class=SimpleMAStrategy,
            parameter_space=SIMPLE_MA_PARAMETER_SPACE,
            data=self.test_data
        )
        
        self.assertEqual(optimizer1.strategy_class, MockStrategy)
        self.assertEqual(optimizer2.strategy_class, SimpleMAStrategy)
    
    def test_parameter_space_mapping(self):
        """Test that correct parameter space is assigned to strategy."""
        optimizer = create_optimizer(
            method='grid',
            strategy_class=SimpleMAStrategy,
            parameter_space=SIMPLE_MA_PARAMETER_SPACE,
            data=self.test_data
        )
        
        # Should use the passed SIMPLE_MA_PARAMETER_SPACE
        self.assertEqual(optimizer.parameter_space, SIMPLE_MA_PARAMETER_SPACE)
        
        # Check that it has the expected parameters
        self.assertIn('short_window', optimizer.parameter_space.parameters)
        self.assertIn('long_window', optimizer.parameter_space.parameters)
    
    def test_all_optimizer_parameters_passed(self):
        """Test that all parameters are correctly passed to optimizer."""
        test_params = {
            'method': 'random',
            'strategy_class': MockStrategy,
            'parameter_space': self.test_parameter_space,
            'data': self.test_data,
            'initial_capital': 25000,
            'commission': 0.005,
            'sort_by': 'max_drawdown',
            'n_jobs': 4
        }
        
        optimizer = create_optimizer(**test_params)
        
        self.assertIsInstance(optimizer, RandomSearchOptimizer)
        self.assertEqual(optimizer.initial_capital, 25000)
        self.assertEqual(optimizer.commission, 0.005)
        self.assertEqual(optimizer.sort_by, 'max_drawdown')
        self.assertEqual(optimizer.n_jobs, 4)
    
    def test_grid_search_specific_creation(self):
        """Test that grid search optimizer gets correct configuration."""
        optimizer = create_optimizer(
            method='grid',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data
        )
        
        # Should have grid search specific attributes
        self.assertTrue(hasattr(optimizer, 'MAX_COMBINATIONS_WARNING'))
        self.assertTrue(hasattr(optimizer, '_estimate_combinations_count'))
        self.assertTrue(hasattr(optimizer, '_generate_grid_combinations_lazy'))
    
    def test_random_search_specific_creation(self):
        """Test that random search optimizer gets correct configuration."""
        optimizer = create_optimizer(
            method='random',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data
        )
        
        # Should have random search specific attributes
        self.assertTrue(hasattr(optimizer, 'MAX_ATTEMPTS_MULTIPLIER'))
        self.assertTrue(hasattr(optimizer, 'DEFAULT_SAMPLE_RATIO'))
        self.assertTrue(hasattr(optimizer, 'estimate_optimal_samples'))
        self.assertTrue(hasattr(optimizer, '_generate_random_combinations'))
    
    def test_data_validation_integration(self):
        """Test that data validation works through factory."""
        # Create invalid data
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102]
            # Missing required columns
        })
        
        with self.assertRaises(ValueError) as context:
            create_optimizer(
                method='grid',
                strategy_class=MockStrategy,
                parameter_space=self.test_parameter_space,
                data=invalid_data
            )
        
        self.assertIn("data is missing required columns", str(context.exception))
    
    def test_invalid_sort_metric_validation(self):
        """Test validation of sort metric through factory."""
        with self.assertRaises(ValueError) as context:
            create_optimizer(
                method='grid',
                strategy_class=MockStrategy,
                parameter_space=self.test_parameter_space,
                data=self.test_data,
                sort_by='invalid_metric'
            )
        
        self.assertIn("sort_by must be one of", str(context.exception))
    
    def test_invalid_capital_validation(self):
        """Test validation of initial capital through factory."""
        with self.assertRaises(ValueError) as context:
            create_optimizer(
                method='grid',
                strategy_class=MockStrategy,
                parameter_space=self.test_parameter_space,
                data=self.test_data,
                initial_capital=-1000
            )
        
        self.assertIn("initial_capital must be positive", str(context.exception))
    
    def test_invalid_commission_validation(self):
        """Test validation of commission through factory."""
        with self.assertRaises(ValueError) as context:
            create_optimizer(
                method='grid',
                strategy_class=MockStrategy,
                parameter_space=self.test_parameter_space,
                data=self.test_data,
                commission=-0.1
            )
        
        self.assertIn("commission cannot be negative", str(context.exception))
    
    def test_factory_preserves_data_reference(self):
        """Test that factory preserves the original data reference."""
        optimizer = create_optimizer(
            method='grid',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data
        )
        
        # Should be the same object reference
        self.assertIs(optimizer.data, self.test_data)
    
    def test_multiple_optimizer_creation_independence(self):
        """Test that creating multiple optimizers doesn't interfere."""
        optimizer1 = create_optimizer(
            method='grid',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data,
            initial_capital=10000
        )
        
        optimizer2 = create_optimizer(
            method='random',
            strategy_class=MockStrategy,
            parameter_space=self.test_parameter_space,
            data=self.test_data,
            initial_capital=20000
        )
        
        # Should be different instances with different configurations
        self.assertIsInstance(optimizer1, GridSearchOptimizer)
        self.assertIsInstance(optimizer2, RandomSearchOptimizer)
        self.assertEqual(optimizer1.initial_capital, 10000)
        self.assertEqual(optimizer2.initial_capital, 20000)
        self.assertIsNot(optimizer1, optimizer2)


if __name__ == '__main__':
    unittest.main()