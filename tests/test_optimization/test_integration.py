import unittest
from unittest.mock import Mock, patch
import pandas as pd
import tempfile
import json
import os

from niffler.optimization import (
    GridSearchOptimizer, 
    RandomSearchOptimizer, 
    ParameterSpace, 
    OptimizationResult,
    create_optimizer
)
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
from niffler.backtesting.backtest_result import BacktestResult


class TestOptimizationIntegration(unittest.TestCase):
    """Integration tests for the optimization package."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create realistic test data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 + pd.Series(range(100)).apply(lambda x: x * 0.1 + (x % 10) * 0.5)
        
        self.test_data = pd.DataFrame({
            'open': prices + 0.1,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)
        
        # Create realistic parameter space
        self.parameter_space = ParameterSpace({
            'short_window': {'type': 'int', 'min': 5, 'max': 10, 'step': 1},
            'long_window': {'type': 'int', 'min': 15, 'max': 20, 'step': 1}
        })
        
        # Mock realistic backtest results
        self.mock_backtest_results = []
        for i in range(10):
            result = Mock()
            result.total_return = 0.05 + i * 0.01  # 0.05 to 0.14
            result.sharpe_ratio = 0.5 + i * 0.1    # 0.5 to 1.4
            result.max_drawdown = 0.02 + i * 0.005  # 0.02 to 0.065
            result.win_rate = 0.4 + i * 0.02       # 0.4 to 0.58
            result.total_trades = 10 + i           # 10 to 19
            result.total_profits = 1000 + i * 100  # 1000 to 1900
            result.total_losses = -500 - i * 50    # -500 to -950
            self.mock_backtest_results.append(result)
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_grid_search_end_to_end(self, mock_engine_class):
        """Test complete grid search optimization workflow."""
        # Setup mock backtest engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        call_count = 0
        def mock_run_backtest(strategy, data):
            nonlocal call_count
            result = self.mock_backtest_results[call_count % len(self.mock_backtest_results)]
            call_count += 1
            return result
        
        mock_engine.run_backtest.side_effect = mock_run_backtest
        
        # Create and run optimizer
        optimizer = GridSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data,
            initial_capital=10000,
            commission=0.001
        )
        
        # Mock the reusable engine
        optimizer._backtest_engine = mock_engine
        
        results = optimizer.optimize()
        
        # Verify results
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 6 * 6)  # Max combinations
        
        # All results should be OptimizationResult instances
        for result in results:
            self.assertIsInstance(result, OptimizationResult)
            self.assertIn('short_window', result.parameters)
            self.assertIn('long_window', result.parameters)
            
            # Verify parameter ranges
            self.assertGreaterEqual(result.parameters['short_window'], 5)
            self.assertLessEqual(result.parameters['short_window'], 10)
            self.assertGreaterEqual(result.parameters['long_window'], 15)
            self.assertLessEqual(result.parameters['long_window'], 20)
        
        # Results should be sorted by total return (default)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].backtest_result.total_return,
                results[i + 1].backtest_result.total_return
            )
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_random_search_end_to_end(self, mock_engine_class):
        """Test complete random search optimization workflow."""
        # Setup mock backtest engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        call_count = 0
        def mock_run_backtest(strategy, data):
            nonlocal call_count
            result = self.mock_backtest_results[call_count % len(self.mock_backtest_results)]
            call_count += 1
            return result
        
        mock_engine.run_backtest.side_effect = mock_run_backtest
        
        # Create and run optimizer
        optimizer = RandomSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data,
            initial_capital=10000,
            commission=0.001,
            n_jobs=1  # Disable parallel processing for mocking to work
        )
        
        # Mock the reusable engine
        optimizer._backtest_engine = mock_engine
        
        results = optimizer.optimize(n_trials=10, seed=42)
        
        # Verify results
        self.assertEqual(len(results), 10)  # Should have exactly 10 results
        
        # All results should be OptimizationResult instances
        for result in results:
            self.assertIsInstance(result, OptimizationResult)
            self.assertIn('short_window', result.parameters)
            self.assertIn('long_window', result.parameters)
            
            # Verify parameter ranges
            self.assertGreaterEqual(result.parameters['short_window'], 5)
            self.assertLessEqual(result.parameters['short_window'], 10)
            self.assertGreaterEqual(result.parameters['long_window'], 15)
            self.assertLessEqual(result.parameters['long_window'], 20)
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_factory_integration(self, mock_engine_class):
        """Test optimization through factory function."""
        # Setup mock
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run_backtest.return_value = self.mock_backtest_results[0]
        
        # Create optimizer through factory
        from niffler.optimization.optimizer_factory import get_strategy_class, get_parameter_space
        optimizer = create_optimizer(
            method='grid',
            strategy_class=get_strategy_class('simple_ma'),
            parameter_space=get_parameter_space('simple_ma'),
            data=self.test_data,
            initial_capital=5000,
            commission=0.002
        )
        
        # Mock the reusable engine
        optimizer._backtest_engine = mock_engine
        
        # Should be able to run optimization
        results = optimizer.optimize()
        
        self.assertIsInstance(optimizer, GridSearchOptimizer)
        self.assertGreater(len(results), 0)
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_metrics_analysis_integration(self, mock_engine_class):
        """Test complete workflow including metrics analysis."""
        # Setup mock
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        call_count = 0
        def mock_run_backtest(strategy, data):
            nonlocal call_count
            result = self.mock_backtest_results[call_count % len(self.mock_backtest_results)]
            call_count += 1
            return result
        
        mock_engine.run_backtest.side_effect = mock_run_backtest
        
        # Create optimizer
        optimizer = RandomSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data,
            n_jobs=1  # Disable parallel processing for mocking to work
        )
        
        # Mock the reusable engine
        optimizer._backtest_engine = mock_engine
        
        # Run optimization
        results = optimizer.optimize(n_trials=5, seed=42)
        
        # Analyze best metrics
        best_metrics = optimizer.analyze_best_metrics(results)
        
        # Verify metrics analysis
        expected_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades', 'profit_factor']
        for metric in expected_metrics:
            self.assertIn(metric, best_metrics)
            self.assertIn('parameters', best_metrics[metric])
            self.assertIn('value', best_metrics[metric])
            self.assertIn('higher_is_better', best_metrics[metric])
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_results_saving_integration(self, mock_engine_class):
        """Test complete workflow including saving results."""
        # Setup mock
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run_backtest.return_value = self.mock_backtest_results[0]
        
        # Create optimizer
        optimizer = GridSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Mock the reusable engine
        optimizer._backtest_engine = mock_engine
        
        # Run optimization
        results = optimizer.optimize()
        
        # Save results to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            optimizer.save_results(results, temp_filename)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(temp_filename))
            
            with open(temp_filename, 'r') as f:
                saved_data = json.load(f)
            
            # Verify structure
            self.assertIn('metadata', saved_data)
            self.assertIn('results', saved_data)
            
            # Verify metadata
            metadata = saved_data['metadata']
            self.assertEqual(metadata['optimizer_class'], 'GridSearchOptimizer')
            self.assertEqual(metadata['strategy_class'], 'SimpleMAStrategy')
            self.assertIn('timestamp', metadata)
            
            # Verify results
            saved_results = saved_data['results']
            self.assertEqual(len(saved_results), len(results))
            
            for saved_result in saved_results:
                self.assertIn('parameters', saved_result)
                self.assertIn('metrics', saved_result)
                
                # Check metrics
                metrics = saved_result['metrics']
                self.assertIn('total_return', metrics)
                self.assertIn('sharpe_ratio', metrics)
                self.assertIn('max_drawdown', metrics)
        
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_memory_management_integration(self, mock_engine_class):
        """Test memory management during large optimization."""
        # Setup mock
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        call_count = 0
        def mock_run_backtest(strategy, data):
            nonlocal call_count
            result = self.mock_backtest_results[call_count % len(self.mock_backtest_results)]
            call_count += 1
            return result
        
        mock_engine.run_backtest.side_effect = mock_run_backtest
        
        # Create optimizer with small memory limit
        optimizer = RandomSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data
        )
        
        # Mock the reusable engine
        optimizer._backtest_engine = mock_engine
        
        # Set small memory limit for testing
        original_limit = optimizer.MAX_RESULTS_IN_MEMORY
        optimizer.MAX_RESULTS_IN_MEMORY = 5
        
        try:
            # Run optimization with more trials than memory limit
            results = optimizer.optimize(n_trials=20, seed=42)
            
            # Should have limited results due to memory management
            self.assertLessEqual(len(results), optimizer.MAX_RESULTS_IN_MEMORY)
            
        finally:
            # Restore original limit
            optimizer.MAX_RESULTS_IN_MEMORY = original_limit
    
    def test_parameter_space_validation_integration(self):
        """Test parameter space validation through complete workflow."""
        # Test that invalid parameter space raises error during construction
        with self.assertRaises(ValueError):
            invalid_space = ParameterSpace({
                'param1': {'type': 'int', 'min': 10}  # Missing max
            })
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_error_handling_integration(self, mock_engine_class):
        """Test error handling throughout optimization workflow."""
        # Setup mock to occasionally fail
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        call_count = 0
        def mock_run_backtest_with_errors(strategy, data):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Mock backtest error")
            return self.mock_backtest_results[call_count % len(self.mock_backtest_results)]
        
        mock_engine.run_backtest.side_effect = mock_run_backtest_with_errors
        
        # Create optimizer
        optimizer = RandomSearchOptimizer(
            strategy_class=SimpleMAStrategy,
            parameter_space=self.parameter_space,
            data=self.test_data,
            n_jobs=1  # Disable parallel processing for mocking to work
        )
        
        # Mock the reusable engine
        optimizer._backtest_engine = mock_engine
        
        # Should handle errors gracefully and return partial results
        results = optimizer.optimize(n_trials=10, seed=42)
        
        # Should have some results despite errors
        self.assertGreater(len(results), 0)
        self.assertLess(len(results), 10)  # Should be less than total due to failures
    
    @patch('niffler.optimization.base_optimizer.BacktestEngine')
    def test_different_sort_metrics_integration(self, mock_engine_class):
        """Test optimization with different sort metrics."""
        # Setup mock
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        call_count = 0
        def mock_run_backtest(strategy, data):
            nonlocal call_count
            result = self.mock_backtest_results[call_count % len(self.mock_backtest_results)]
            call_count += 1
            return result
        
        mock_engine.run_backtest.side_effect = mock_run_backtest
        
        # Test different sort metrics
        sort_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for sort_by in sort_metrics:
            with self.subTest(sort_by=sort_by):
                optimizer = RandomSearchOptimizer(
                    strategy_class=SimpleMAStrategy,
                    parameter_space=self.parameter_space,
                    data=self.test_data,
                    sort_by=sort_by
                )
                
                # Mock the reusable engine
                optimizer._backtest_engine = mock_engine
                
                results = optimizer.optimize(n_trials=5, seed=42)
                
                # Verify results are sorted correctly
                if sort_by == 'max_drawdown':  # Lower is better
                    for i in range(len(results) - 1):
                        self.assertLessEqual(
                            getattr(results[i].backtest_result, sort_by),
                            getattr(results[i + 1].backtest_result, sort_by)
                        )
                else:  # Higher is better
                    for i in range(len(results) - 1):
                        self.assertGreaterEqual(
                            getattr(results[i].backtest_result, sort_by),
                            getattr(results[i + 1].backtest_result, sort_by)
                        )


if __name__ == '__main__':
    unittest.main()