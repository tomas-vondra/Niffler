import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import random
from typing import List

from niffler.optimization.random_search_optimizer import RandomSearchOptimizer
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


class TestRandomSearchOptimizer(unittest.TestCase):
    """Test cases for RandomSearchOptimizer."""
    
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
        
        # Create parameter space for testing
        self.parameter_space = ParameterSpace({
            'param1': {'type': 'int', 'min': 10, 'max': 20, 'step': 1},
            'param2': {'type': 'float', 'min': 0.1, 'max': 1.0, 'step': 0.1}
        })
        
        # Create small parameter space for duplicate testing
        self.small_parameter_space = ParameterSpace({
            'param1': {'type': 'int', 'min': 1, 'max': 2, 'step': 1},
            'param2': {'type': 'choice', 'choices': ['a', 'b']}
        })
        
        # Create continuous parameter space
        self.continuous_space = ParameterSpace({
            'param1': {'type': 'float', 'min': 0.0, 'max': 1.0},  # No step
            'param2': {'type': 'int', 'min': 1, 'max': 10, 'step': 1}
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
    
    def test_random_search_optimizer_initialization(self):
        """Test random search optimizer initialization."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        self.assertEqual(optimizer.strategy_class, MockStrategy)
        self.assertEqual(optimizer.parameter_space, self.parameter_space)
        self.assertEqual(optimizer.MAX_ATTEMPTS_MULTIPLIER, 10)
        self.assertEqual(optimizer.DEFAULT_SAMPLE_RATIO, 0.1)
        self.assertEqual(optimizer.MIN_SAMPLES, 10)
        self.assertEqual(optimizer.MAX_SAMPLES, 10000)
        self.assertEqual(optimizer.DUPLICATE_RATE_THRESHOLD, 0.8)
    
    def test_generate_single_combination(self):
        """Test generating a single random parameter combination."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Set seed for reproducibility
        random.seed(42)
        combination = optimizer._generate_single_combination()
        
        self.assertIn('param1', combination)
        self.assertIn('param2', combination)
        
        # Check value ranges
        self.assertIsInstance(combination['param1'], int)
        self.assertGreaterEqual(combination['param1'], 10)
        self.assertLessEqual(combination['param1'], 20)
        
        self.assertIsInstance(combination['param2'], float)
        self.assertGreaterEqual(combination['param2'], 0.1)
        self.assertLessEqual(combination['param2'], 1.0)
    
    def test_generate_single_combination_continuous(self):
        """Test generating combination with continuous parameter."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.continuous_space,
            data=self.test_data
        )
        
        combination = optimizer._generate_single_combination()
        
        self.assertIn('param1', combination)
        self.assertIn('param2', combination)
        
        # Continuous float parameter
        self.assertIsInstance(combination['param1'], float)
        self.assertGreaterEqual(combination['param1'], 0.0)
        self.assertLessEqual(combination['param1'], 1.0)
        
        # Discrete int parameter
        self.assertIsInstance(combination['param2'], int)
        self.assertGreaterEqual(combination['param2'], 1)
        self.assertLessEqual(combination['param2'], 10)
    
    def test_hash_combination(self):
        """Test hashing parameter combinations."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        combo1 = {'param1': 10, 'param2': 0.5}
        combo2 = {'param1': 10, 'param2': 0.5}
        combo3 = {'param1': 15, 'param2': 0.5}
        
        hash1 = optimizer._hash_combination(combo1)
        hash2 = optimizer._hash_combination(combo2)
        hash3 = optimizer._hash_combination(combo3)
        
        # Same combinations should have same hash
        self.assertEqual(hash1, hash2)
        
        # Different combinations should have different hash (with high probability)
        self.assertNotEqual(hash1, hash3)
        
        # Hash should be an integer
        self.assertIsInstance(hash1, int)
    
    def test_hash_combination_order_independence(self):
        """Test that hash is independent of parameter order."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        combo1 = {'param1': 10, 'param2': 0.5}
        combo2 = {'param2': 0.5, 'param1': 10}  # Different order
        
        hash1 = optimizer._hash_combination(combo1)
        hash2 = optimizer._hash_combination(combo2)
        
        self.assertEqual(hash1, hash2)
    
    def test_generate_random_combinations_basic(self):
        """Test generating random combinations with basic parameters."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        combinations = optimizer._generate_random_combinations(5, seed=42)
        
        self.assertEqual(len(combinations), 5)
        
        # All combinations should be valid
        for combo in combinations:
            self.assertIn('param1', combo)
            self.assertIn('param2', combo)
    
    def test_generate_random_combinations_duplicates(self):
        """Test handling of duplicate combinations."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,  # Only 4 possible combinations
            data=self.test_data
        )
        
        # Request more combinations than possible
        combinations = optimizer._generate_random_combinations(10, seed=42)
        
        # Should get at most 4 unique combinations
        self.assertLessEqual(len(combinations), 4)
        
        # All combinations should be unique
        hashes = [optimizer._hash_combination(combo) for combo in combinations]
        self.assertEqual(len(hashes), len(set(hashes)))
    
    @patch('niffler.optimization.random_search_optimizer.logging')
    def test_generate_random_combinations_high_duplicate_rate(self, mock_logging):
        """Test early termination due to high duplicate rate."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        # This should trigger high duplicate rate warning
        combinations = optimizer._generate_random_combinations(1000, seed=42)
        
        # Should have stopped early
        self.assertLess(len(combinations), 1000)
        
        # Should have logged warning about duplicate rate
        mock_logging.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logging.warning.call_args_list]
        duplicate_warnings = [msg for msg in warning_calls if "high duplicate rate" in msg]
        self.assertTrue(any(duplicate_warnings))
    
    @patch('niffler.optimization.random_search_optimizer.logging')
    def test_generate_random_combinations_max_attempts(self, mock_logging):
        """Test max attempts reached warning."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,
            data=self.test_data
        )
        
        # Force max attempts by setting very low threshold
        optimizer.DUPLICATE_RATE_THRESHOLD = 1.0  # Never trigger early termination
        
        combinations = optimizer._generate_random_combinations(100, seed=42)
        
        # Should have logged max attempts warning
        mock_logging.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logging.warning.call_args_list]
        max_attempts_warnings = [msg for msg in warning_calls if "maximum attempts" in msg]
        self.assertTrue(any(max_attempts_warnings))
    
    def test_estimate_total_combinations_discrete(self):
        """Test estimating total combinations for discrete parameters."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        total = optimizer._estimate_total_combinations()
        
        # param1: 11 values (10-20 inclusive)
        # param2: 10 values (0.1-1.0 with step 0.1)
        # Total: 11 * 10 = 110
        self.assertEqual(total, 110)
    
    def test_estimate_total_combinations_continuous(self):
        """Test estimating total combinations with continuous parameters."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.continuous_space,
            data=self.test_data
        )
        
        total = optimizer._estimate_total_combinations()
        
        # Should return infinity for continuous parameters
        self.assertEqual(total, float('inf'))
    
    def test_estimate_optimal_samples_discrete(self):
        """Test estimating optimal samples for discrete parameter space."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        optimal = optimizer.estimate_optimal_samples()
        
        # Should be 10% of 110 = 11, bounded by MIN_SAMPLES (10) and MAX_SAMPLES (10000)
        expected = max(optimizer.MIN_SAMPLES, int(110 * optimizer.DEFAULT_SAMPLE_RATIO))
        self.assertEqual(optimal, expected)
    
    def test_estimate_optimal_samples_continuous(self):
        """Test estimating optimal samples for continuous parameter space."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.continuous_space,
            data=self.test_data
        )
        
        optimal = optimizer.estimate_optimal_samples()
        
        # Should return MAX_SAMPLES for continuous parameters
        self.assertEqual(optimal, optimizer.MAX_SAMPLES)
    
    def test_estimate_optimal_samples_very_small_space(self):
        """Test estimating optimal samples for very small parameter space."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.small_parameter_space,  # Only 4 combinations
            data=self.test_data
        )
        
        optimal = optimizer.estimate_optimal_samples()
        
        # Should be at least MIN_SAMPLES even for small spaces
        self.assertEqual(optimal, optimizer.MIN_SAMPLES)
    
    @patch('niffler.optimization.random_search_optimizer.logging')
    def test_optimize_with_auto_estimation(self, mock_logging):
        """Test optimization with automatic sample size estimation."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data,
            n_jobs=1  # Disable parallel processing for mocking to work
        )
        
        # Mock the evaluation to return successful results
        mock_result = OptimizationResult(
            parameters={'param1': 10, 'param2': 0.1},
            backtest_result=self.mock_backtest_result
        )
        
        with patch.object(optimizer, '_evaluate_single_combination', return_value=mock_result):
            results = optimizer.optimize(n_trials=None)  # Auto-estimate
            
            # Should have results
            self.assertGreater(len(results), 0)
            
            # Should have logged about using estimated sample size
            mock_logging.info.assert_called()
            log_calls = [call[0][0] for call in mock_logging.info.call_args_list]
            estimation_messages = [msg for msg in log_calls if "estimated optimal sample size" in msg]
            self.assertTrue(any(estimation_messages))
    
    @patch('niffler.optimization.random_search_optimizer.logging')
    def test_optimize_with_suggestion(self, mock_logging):
        """Test optimization with suboptimal sample size suggestion."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Mock the evaluation to return successful results
        mock_result = OptimizationResult(
            parameters={'param1': 10, 'param2': 0.1},
            backtest_result=self.mock_backtest_result
        )
        
        with patch.object(optimizer, '_evaluate_single_combination', return_value=mock_result):
            # Use a sample size that's very different from optimal
            results = optimizer.optimize(n_trials=1000)
            
            # Should have logged suggestion
            mock_logging.info.assert_called()
            log_calls = [call[0][0] for call in mock_logging.info.call_args_list]
            suggestion_messages = [msg for msg in log_calls if "Consider using" in msg and "samples instead of" in msg]
            self.assertTrue(any(suggestion_messages))
    
    def test_optimize_with_explicit_trials(self):
        """Test optimization with explicitly specified number of trials."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data,
            n_jobs=1  # Disable parallel processing for mocking to work
        )
        
        # Mock the evaluation to return successful results
        mock_result = OptimizationResult(
            parameters={'param1': 10, 'param2': 0.1},
            backtest_result=self.mock_backtest_result
        )
        
        with patch.object(optimizer, '_evaluate_single_combination', return_value=mock_result):
            results = optimizer.optimize(n_trials=5, seed=42)
            
            # Should have results
            self.assertEqual(len(results), 5)
    
    def test_optimize_with_failures(self):
        """Test optimization when some evaluations fail."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data,
            n_jobs=1  # Disable parallel processing for mocking to work
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
            results = optimizer.optimize(n_trials=10, seed=42)
            
            # Should have approximately half the results
            self.assertGreater(len(results), 0)
            self.assertLess(len(results), 10)
    
    def test_seeded_reproducibility(self):
        """Test that using the same seed produces reproducible results."""
        optimizer1 = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        optimizer2 = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        combinations1 = optimizer1._generate_random_combinations(5, seed=42)
        combinations2 = optimizer2._generate_random_combinations(5, seed=42)
        
        # Should be identical
        self.assertEqual(combinations1, combinations2)
    
    def test_optimizer_inheritance(self):
        """Test that RandomSearchOptimizer properly inherits from BaseOptimizer."""
        optimizer = RandomSearchOptimizer(
            strategy_class=MockStrategy,
            parameter_space=self.parameter_space,
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