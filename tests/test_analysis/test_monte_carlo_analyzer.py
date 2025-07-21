import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from niffler.analysis.monte_carlo_analyzer import MonteCarloAnalyzer
from niffler.strategies.base_strategy import BaseStrategy
from niffler.backtesting.backtest_result import BacktestResult


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, short_window=10, long_window=20):
        super().__init__("MockStrategy")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        signals = data.copy()
        signals['signal'] = 0
        signals['position_size'] = 1.0
        # Simple mock signals - use iloc for positional indexing
        if len(signals) > 10:
            signals.iloc[::10, signals.columns.get_loc('signal')] = 1  # Buy every 10th day
            signals.iloc[5::10, signals.columns.get_loc('signal')] = -1  # Sell every 10th day (offset)
        return signals
    
    def get_description(self):
        return f"Mock strategy with short_window={self.short_window}, long_window={self.long_window}"


class TestMonteCarloAnalyzer(unittest.TestCase):
    """Test cases for MonteCarloAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, n_days)  # Daily returns
        prices = 100 * np.cumprod(1 + returns)
        
        self.test_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0.002, 0.005, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0.002, 0.005, n_days))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        self.test_data['high'] = np.maximum(
            self.test_data['high'],
            np.maximum(self.test_data['open'], self.test_data['close'])
        )
        self.test_data['low'] = np.minimum(
            self.test_data['low'],
            np.minimum(self.test_data['open'], self.test_data['close'])
        )
        
        self.optimal_parameters = {'short_window': 10, 'long_window': 20}
        self.strategy_class = MockStrategy
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_simulations=100,
            bootstrap_sample_pct=0.8,
            block_size_days=30,
            initial_capital=10000,
            commission=0.001,
            n_jobs=1
        )
        
        self.assertEqual(analyzer.strategy_class, self.strategy_class)
        self.assertEqual(analyzer.optimal_parameters, self.optimal_parameters)
        self.assertEqual(analyzer.n_simulations, 100)
        self.assertEqual(analyzer.bootstrap_sample_pct, 0.8)
        self.assertEqual(analyzer.block_size_days, 30)
        self.assertEqual(analyzer.initial_capital, 10000)
        self.assertEqual(analyzer.commission, 0.001)
        self.assertEqual(analyzer.n_jobs, 1)
    
    def test_init_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Empty parameters
        with self.assertRaises(ValueError):
            MonteCarloAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters={},
                n_simulations=100
            )
        
        # Negative simulations
        with self.assertRaises(ValueError):
            MonteCarloAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
                n_simulations=-1
            )
        
        # Invalid bootstrap percentage
        with self.assertRaises(ValueError):
            MonteCarloAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
                bootstrap_sample_pct=1.5
            )
        
        # Negative block size
        with self.assertRaises(ValueError):
            MonteCarloAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
                block_size_days=-1
            )
        
        # Negative initial capital
        with self.assertRaises(ValueError):
            MonteCarloAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
                initial_capital=-1000
            )
        
        # Negative commission
        with self.assertRaises(ValueError):
            MonteCarloAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
                commission=-0.001
            )
    
    def test_strategy_parameter_validation(self):
        """Test strategy parameter validation."""
        # Valid parameters should not raise
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Invalid parameters should raise
        with self.assertRaises(ValueError):
            MonteCarloAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters={'invalid_param': 'invalid_value'}
            )
    
    def test_analyze_insufficient_data(self):
        """Test analyze with insufficient data."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_simulations=10
        )
        
        # Empty data
        with self.assertRaises(ValueError):
            analyzer.analyze(pd.DataFrame())
        
        # Too little data
        small_data = self.test_data.head(50)
        with self.assertRaises(ValueError):
            analyzer.analyze(small_data)
    
    def test_analyze_invalid_data_format(self):
        """Test analyze with invalid data format."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_simulations=10
        )
        
        # Non-datetime index
        bad_data = self.test_data.reset_index()
        with self.assertRaises(ValueError):
            analyzer.analyze(bad_data)
    
    def test_block_size_adjustment(self):
        """Test automatic block size adjustment for small datasets."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            block_size_days=400,  # Larger than data
            n_simulations=1,  # Just one simulation for test speed
            n_jobs=1  # Force sequential execution
        )
        
        # Test that block size gets adjusted when data is smaller
        small_data = self.test_data.head(200)
        
        # Should automatically adjust block size during analyze
        with patch('niffler.analysis.monte_carlo_analyzer.logging') as mock_logging:
            with patch.object(analyzer, '_run_simulations_sequential') as mock_run_sims:
                # Mock successful simulation result
                mock_result = Mock()
                mock_result.total_return = 1000
                mock_result.total_return_pct = 10
                mock_result.sharpe_ratio = 1.5
                mock_result.max_drawdown = -5.0
                mock_result.win_rate = 65.0
                mock_result.total_trades = 20
                mock_run_sims.return_value = [mock_result]
                
                result = analyzer.analyze(small_data, "TEST")
                
                # Check that warning was logged about block size adjustment
                mock_logging.warning.assert_called()
                # Block size should be adjusted to be smaller than data length
                self.assertTrue(analyzer.block_size_days < 200)
    
    def test_block_bootstrap_sample(self):
        """Test block bootstrap sampling."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            block_size_days=30,
            bootstrap_sample_pct=0.5
        )
        
        sampled_data = analyzer._block_bootstrap_sample(self.test_data)
        
        # Check sample size
        expected_size = int(len(self.test_data) * 0.5)
        self.assertAlmostEqual(len(sampled_data), expected_size, delta=30)
        
        # Check data structure preservation
        self.assertListEqual(list(sampled_data.columns), list(self.test_data.columns))
        self.assertTrue(isinstance(sampled_data.index, pd.DatetimeIndex))
        
        # Check data is sorted
        self.assertTrue(sampled_data.index.is_monotonic_increasing)
    
    def test_block_bootstrap_small_data(self):
        """Test block bootstrap with data smaller than block size."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            block_size_days=100
        )
        
        small_data = self.test_data.head(50)
        sampled_data = analyzer._block_bootstrap_sample(small_data)
        
        # Should return sample of original data
        self.assertLessEqual(len(sampled_data), len(small_data))
    
    @patch('niffler.analysis.monte_carlo_analyzer.BacktestEngine')
    def test_run_single_simulation(self, mock_backtest_engine):
        """Test single simulation execution."""
        # Mock backtest result
        mock_result = Mock()
        mock_result.total_return = 1000
        mock_result.total_return_pct = 10
        mock_result.sharpe_ratio = 1.5
        mock_result.max_drawdown = -5.0
        mock_result.win_rate = 65.0
        mock_result.total_trades = 20
        
        mock_engine_instance = Mock()
        mock_engine_instance.run_backtest.return_value = mock_result
        mock_backtest_engine.return_value = mock_engine_instance
        
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_simulations=1
        )
        
        result = analyzer._run_single_simulation(self.test_data, "TEST", 0)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.metadata['simulation_id'], 0)
        self.assertEqual(result.metadata['parameters_used'], self.optimal_parameters)
        mock_engine_instance.run_backtest.assert_called_once()
    
    def test_run_single_simulation_insufficient_data(self):
        """Test single simulation with insufficient sampled data."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            bootstrap_sample_pct=0.1  # Valid minimum sample
        )
        
        # Should return None for insufficient data
        with patch.object(analyzer, '_block_bootstrap_sample') as mock_bootstrap:
            mock_bootstrap.return_value = pd.DataFrame()  # Empty sample
            result = analyzer._run_single_simulation(self.test_data, "TEST", 0)
            self.assertIsNone(result)
    
    def test_memory_management(self):
        """Test memory management functionality."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            max_results_in_memory=4  # Set to 4 so that with 10 results, we get 2 kept (4//2)
        )
        
        # Create mock results with different returns
        results = []
        for i in range(10):
            mock_result = Mock()
            mock_result.total_return = i * 100  # Increasing returns
            results.append(mock_result)
        
        managed_results = analyzer._manage_memory_efficient_results(results)
        
        # Should keep only the best half (top 2 when max_results_in_memory=4)
        self.assertEqual(len(managed_results), 2)
        
        # Should be sorted by total return (best first)
        returns = [r.total_return for r in managed_results]
        self.assertEqual(returns, [900, 800])
    
    def test_sequential_simulations(self):
        """Test sequential simulation execution."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_simulations=5,
            n_jobs=1
        )
        
        with patch.object(analyzer, '_run_single_simulation') as mock_single_sim:
            # Mock successful simulations
            mock_result = Mock()
            mock_single_sim.return_value = mock_result
            
            results = analyzer._run_simulations_sequential(self.test_data, "TEST")
            
            self.assertEqual(len(results), 5)
            self.assertEqual(mock_single_sim.call_count, 5)
    
    def test_sequential_simulations_with_failures(self):
        """Test sequential simulations with some failures."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_simulations=5,
            n_jobs=1
        )
        
        with patch.object(analyzer, '_run_single_simulation') as mock_single_sim:
            # Mock some failures
            mock_result = Mock()
            mock_single_sim.side_effect = [mock_result, None, mock_result, Exception("Test error"), mock_result]
            
            results = analyzer._run_simulations_sequential(self.test_data, "TEST")
            
            # Should have 3 successful results
            self.assertEqual(len(results), 3)
    
    def test_combined_metrics_calculation(self):
        """Test combined metrics calculation."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Create mock results
        results = []
        for i in range(5):
            mock_result = Mock()
            mock_result.total_return = (i + 1) * 100
            mock_result.total_return_pct = (i + 1) * 10
            mock_result.sharpe_ratio = (i + 1) * 0.5
            mock_result.max_drawdown = -(i + 1) * 2
            mock_result.win_rate = 50 + (i + 1) * 5
            mock_result.total_trades = (i + 1) * 10
            results.append(mock_result)
        
        metrics = analyzer._calculate_combined_metrics(results)
        
        self.assertIn('total_simulations', metrics)
        self.assertIn('mean_return', metrics)
        self.assertIn('median_return', metrics)
        self.assertIn('std_return', metrics)
        self.assertIn('profitable_simulations_pct', metrics)
        
        self.assertEqual(metrics['total_simulations'], 5)
        self.assertEqual(metrics['mean_return'], 300)  # Average of 100,200,300,400,500
        self.assertEqual(metrics['profitable_simulations_pct'], 100)  # All positive
    
    def test_distribution_statistics_calculation(self):
        """Test distribution statistics calculation."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Create mock results with known distribution
        results = []
        returns = [-200, -100, 0, 100, 200, 300, 400, 500, 600, 700]
        return_pcts = [-20, -10, 0, 10, 20, 30, 40, 50, 60, 70]
        
        for i, (ret, ret_pct) in enumerate(zip(returns, return_pcts)):
            mock_result = Mock()
            mock_result.total_return = ret
            mock_result.total_return_pct = ret_pct
            mock_result.sharpe_ratio = i * 0.1
            results.append(mock_result)
        
        stats = analyzer._calculate_distribution_statistics(results)
        
        # Check VaR calculations
        self.assertIn('return_var_5pct', stats)
        self.assertIn('return_var_1pct', stats)
        self.assertIn('return_cvar_5pct', stats)
        self.assertIn('return_cvar_1pct', stats)
        
        # Check confidence intervals
        self.assertIn('return_ci_95_lower', stats)
        self.assertIn('return_ci_95_upper', stats)
        
        # 5% VaR should be around the 5th percentile
        self.assertAlmostEqual(stats['return_var_5pct'], -200, delta=50)
    
    def test_percentile_results(self):
        """Test percentile results calculation."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Create mock results
        results = []
        for i in range(10):
            mock_result = Mock()
            mock_result.total_return = i * 100
            mock_result.total_return_pct = i * 10
            mock_result.sharpe_ratio = i * 0.2
            mock_result.max_drawdown = -i * 5
            mock_result.win_rate = 30 + i * 5
            results.append(mock_result)
        
        percentiles = analyzer.get_percentile_results(results)
        
        self.assertIn('total_return', percentiles)
        self.assertIn('total_return_pct', percentiles)
        self.assertIn('sharpe_ratio', percentiles)
        self.assertIn('max_drawdown', percentiles)
        self.assertIn('win_rate', percentiles)
        
        # Check default percentiles
        for metric in percentiles:
            self.assertIn('p5', percentiles[metric])
            self.assertIn('p25', percentiles[metric])
            self.assertIn('p50', percentiles[metric])
            self.assertIn('p75', percentiles[metric])
            self.assertIn('p95', percentiles[metric])
    
    @patch('niffler.analysis.monte_carlo_analyzer.ProcessPoolExecutor')
    def test_parallel_simulations(self, mock_executor):
        """Test parallel simulation execution."""
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_simulations=3,
            n_jobs=2
        )
        
        # Mock parallel execution
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        mock_future1 = Mock()
        mock_future2 = Mock()
        mock_future3 = Mock()
        
        mock_result = Mock()
        mock_future1.result.return_value = mock_result
        mock_future2.result.return_value = mock_result
        mock_future3.result.return_value = None  # One failure
        
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2, mock_future3]
        
        # Mock as_completed
        with patch('niffler.analysis.monte_carlo_analyzer.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2, mock_future3]
            
            results = analyzer._run_simulations_parallel(self.test_data, "TEST")
            
            # Should have 2 successful results
            self.assertEqual(len(results), 2)
            self.assertEqual(mock_executor_instance.submit.call_count, 3)
    
    @patch('niffler.analysis.monte_carlo_analyzer.BacktestEngine')
    def test_analyze_integration(self, mock_backtest_engine):
        """Test full analyze method integration."""
        # Mock successful backtest results
        mock_result = Mock()
        mock_result.total_return = 1000
        mock_result.total_return_pct = 10
        mock_result.sharpe_ratio = 1.5
        mock_result.max_drawdown = -5.0
        mock_result.win_rate = 65.0
        mock_result.total_trades = 20
        
        mock_engine_instance = Mock()
        mock_engine_instance.run_backtest.return_value = mock_result
        mock_backtest_engine.return_value = mock_engine_instance
        
        analyzer = MonteCarloAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_simulations=5,
            n_jobs=1,
            random_seed=42
        )
        
        result = analyzer.analyze(self.test_data, "TEST")
        
        # Check result structure
        self.assertEqual(result.analysis_type, 'monte_carlo')
        self.assertEqual(result.strategy_name, 'MockStrategy')
        self.assertEqual(result.symbol, 'TEST')
        self.assertIsNotNone(result.individual_results)
        self.assertIsNotNone(result.combined_metrics)
        self.assertIsNotNone(result.stability_metrics)
        
        # Check analysis parameters
        self.assertEqual(result.analysis_parameters['optimal_parameters'], self.optimal_parameters)
        self.assertIn('n_simulations', result.analysis_parameters)
        self.assertIn('bootstrap_sample_pct', result.analysis_parameters)
        self.assertIn('block_size_days', result.analysis_parameters)


if __name__ == '__main__':
    unittest.main()