import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from niffler.analysis.walk_forward_analyzer import WalkForwardAnalyzer
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


class TestWalkForwardAnalyzer(unittest.TestCase):
    """Test cases for WalkForwardAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data spanning 2 years for meaningful walk-forward
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Generate realistic price data with trend
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
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            test_window_months=6,
            step_months=3,
            initial_capital=10000,
            commission=0.001,
            n_jobs=1,
            max_results_in_memory=100
        )
        
        self.assertEqual(analyzer.strategy_class, self.strategy_class)
        self.assertEqual(analyzer.optimal_parameters, self.optimal_parameters)
        self.assertEqual(analyzer.test_window_months, 6)
        self.assertEqual(analyzer.step_months, 3)
        self.assertEqual(analyzer.initial_capital, 10000)
        self.assertEqual(analyzer.commission, 0.001)
        self.assertEqual(analyzer.n_jobs, 1)
        self.assertEqual(analyzer.max_results_in_memory, 100)
    
    def test_init_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Empty parameters
        with self.assertRaises(ValueError):
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters={},
                test_window_months=6
            )
        
        # Negative test window
        with self.assertRaises(ValueError):
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
                test_window_months=-1
            )
        
        # Negative step months
        with self.assertRaises(ValueError):
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
                step_months=-1
            )
        
        # Negative initial capital
        with self.assertRaises(ValueError):
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
                initial_capital=-1000
            )
        
        # Negative commission
        with self.assertRaises(ValueError):
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters=self.optimal_parameters,
                commission=-0.001
            )
    
    def test_strategy_parameter_validation(self):
        """Test strategy parameter validation."""
        # Valid parameters should not raise
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Invalid parameters should raise
        with self.assertRaises(ValueError):
            WalkForwardAnalyzer(
                strategy_class=self.strategy_class,
                optimal_parameters={'invalid_param': 'invalid_value'}
            )
    
    def test_analyze_insufficient_data(self):
        """Test analyze with insufficient data."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            test_window_months=6
        )
        
        # Empty data
        with self.assertRaises(ValueError):
            analyzer.analyze(pd.DataFrame())
        
        # Too little data
        small_data = self.test_data.head(50)
        with self.assertRaises(ValueError):
            analyzer.analyze(small_data)
        
        # Insufficient data for window size
        with self.assertRaises(ValueError):
            analyzer.analyze(self.test_data.head(100))  # Not enough for 6-month windows
    
    def test_analyze_invalid_data_format(self):
        """Test analyze with invalid data format."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Non-datetime index
        bad_data = self.test_data.reset_index()
        with self.assertRaises(ValueError):
            analyzer.analyze(bad_data)
    
    def test_generate_walk_forward_periods(self):
        """Test walk-forward period generation."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            test_window_months=6,
            step_months=3
        )
        
        periods = analyzer._generate_walk_forward_periods(self.test_data)
        
        # Should have multiple periods
        self.assertGreater(len(periods), 0)
        
        # Each period should be a tuple of (start, end)
        for period in periods:
            self.assertIsInstance(period, tuple)
            self.assertEqual(len(period), 2)
            start, end = period
            self.assertIsInstance(start, datetime)
            self.assertIsInstance(end, datetime)
            self.assertLess(start, end)
        
        # Periods should be properly spaced
        if len(periods) > 1:
            first_start = periods[0][0]
            second_start = periods[1][0]
            # Should be approximately 3 months apart (step_months)
            delta_months = relativedelta(second_start, first_start).months
            self.assertAlmostEqual(delta_months, 3, delta=1)
    
    def test_generate_walk_forward_periods_edge_cases(self):
        """Test walk-forward period generation edge cases."""
        # Very large window size
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            test_window_months=24,  # Larger than data span
            step_months=6
        )
        
        periods = analyzer._generate_walk_forward_periods(self.test_data)
        # Should return empty list or very few periods
        self.assertLessEqual(len(periods), 1)
        
        # Very small step size
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            test_window_months=3,
            step_months=1
        )
        
        periods = analyzer._generate_walk_forward_periods(self.test_data)
        # Should generate many overlapping periods
        self.assertGreater(len(periods), 10)
    
    @patch('niffler.analysis.walk_forward_analyzer.BacktestEngine')
    def test_run_single_period(self, mock_backtest_engine):
        """Test single period execution."""
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
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        test_start = self.test_data.index[0]
        test_end = self.test_data.index[0] + relativedelta(months=3)
        
        result = analyzer._run_single_period(self.test_data, test_start, test_end, "TEST", 1)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.metadata['period_number'], 1)
        self.assertEqual(result.metadata['parameters_used'], self.optimal_parameters)
        self.assertEqual(result.metadata['test_start'], test_start)
        self.assertEqual(result.metadata['test_end'], test_end)
        mock_engine_instance.run_backtest.assert_called_once()
    
    def test_run_single_period_insufficient_data(self):
        """Test single period with insufficient data in window."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Create a very narrow time window with little data
        test_start = self.test_data.index[-10]  # Near end of data
        test_end = self.test_data.index[-1]
        
        result = analyzer._run_single_period(self.test_data, test_start, test_end, "TEST", 1)
        
        # Should return None for insufficient data
        self.assertIsNone(result)
    
    def test_memory_management(self):
        """Test memory management functionality."""
        analyzer = WalkForwardAnalyzer(
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
    
    def test_sequential_periods(self):
        """Test sequential period execution."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            test_window_months=3,
            step_months=3,
            n_jobs=1
        )
        
        # Generate a few test periods
        periods = [(self.test_data.index[0] + relativedelta(months=i*3),
                   self.test_data.index[0] + relativedelta(months=i*3+3)) 
                  for i in range(3)]
        
        with patch.object(analyzer, '_run_single_period') as mock_single_period:
            # Mock successful periods
            mock_result = Mock()
            mock_single_period.return_value = mock_result
            
            results = analyzer._run_periods_sequential(self.test_data, "TEST", periods)
            
            self.assertEqual(len(results), 3)
            self.assertEqual(mock_single_period.call_count, 3)
    
    def test_sequential_periods_with_failures(self):
        """Test sequential periods with some failures."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_jobs=1
        )
        
        periods = [(self.test_data.index[0] + relativedelta(months=i*3),
                   self.test_data.index[0] + relativedelta(months=i*3+3)) 
                  for i in range(5)]
        
        with patch.object(analyzer, '_run_single_period') as mock_single_period:
            # Mock some failures
            mock_result = Mock()
            mock_single_period.side_effect = [mock_result, None, mock_result, Exception("Test error"), mock_result]
            
            results = analyzer._run_periods_sequential(self.test_data, "TEST", periods)
            
            # Should have 3 successful results
            self.assertEqual(len(results), 3)
    
    def test_combined_metrics_calculation(self):
        """Test combined metrics calculation."""
        analyzer = WalkForwardAnalyzer(
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
            # Mock portfolio values for combined Sharpe calculation
            mock_result.portfolio_values = pd.Series([10000 + j*10 for j in range(30)])
            results.append(mock_result)
        
        metrics = analyzer._calculate_combined_metrics(results)
        
        self.assertIn('total_periods', metrics)
        self.assertIn('avg_return', metrics)
        self.assertIn('median_return', metrics)
        self.assertIn('std_return', metrics)
        self.assertIn('profitable_periods_pct', metrics)
        self.assertIn('combined_sharpe_ratio', metrics)
        
        self.assertEqual(metrics['total_periods'], 5)
        self.assertEqual(metrics['avg_return'], 300)  # Average of 100,200,300,400,500
        self.assertEqual(metrics['profitable_periods_pct'], 100)  # All positive
    
    def test_stability_metrics_calculation(self):
        """Test stability metrics calculation."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Create mock results with varying performance
        results = []
        returns = [100, 200, -50, 150, 300, -100, 250]  # Mix of positive/negative
        for i, ret in enumerate(returns):
            mock_result = Mock()
            mock_result.total_return = ret
            mock_result.total_return_pct = ret / 10
            mock_result.sharpe_ratio = i * 0.2
            results.append(mock_result)
        
        stability_metrics = analyzer._calculate_stability_metrics(results)
        
        self.assertIn('return_volatility', stability_metrics)
        self.assertIn('return_pct_volatility', stability_metrics)
        self.assertIn('sharpe_volatility', stability_metrics)
        self.assertIn('return_consistency', stability_metrics)
        self.assertIn('temporal_stability', stability_metrics)
        
        # Temporal stability should be between 0 and 1
        self.assertGreaterEqual(stability_metrics['temporal_stability'], 0)
        self.assertLessEqual(stability_metrics['temporal_stability'], 1)
    
    def test_temporal_stability_calculation(self):
        """Test temporal stability calculation specifically."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Create results with perfect increasing trend (high stability)
        results = []
        for i in range(5):
            mock_result = Mock()
            mock_result.total_return = i * 100  # Perfect increasing
            results.append(mock_result)
        
        stability = analyzer._calculate_temporal_stability(results)
        self.assertEqual(stability, 1.0)  # Perfect stability
        
        # Create results with alternating pattern (low stability)
        results = []
        alternating_returns = [100, -50, 200, -100, 150]
        for ret in alternating_returns:
            mock_result = Mock()
            mock_result.total_return = ret
            results.append(mock_result)
        
        stability = analyzer._calculate_temporal_stability(results)
        self.assertLess(stability, 0.5)  # Low stability due to alternating pattern
    
    def test_rolling_stability_calculation(self):
        """Test rolling stability calculations."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Create enough results for rolling calculations
        results = []
        for i in range(10):
            mock_result = Mock()
            mock_result.total_return = i * 50 + np.random.normal(0, 10)  # Trend with noise
            results.append(mock_result)
        
        rolling_stability = analyzer._calculate_rolling_stability(results)
        
        self.assertIn('rolling_mean_stability', rolling_stability)
        self.assertIn('trend_consistency', rolling_stability)
        
        # Values should be between 0 and some reasonable upper bound
        self.assertGreaterEqual(rolling_stability['rolling_mean_stability'], 0)
        self.assertGreaterEqual(rolling_stability['trend_consistency'], 0)
        self.assertLessEqual(rolling_stability['trend_consistency'], 1)
    
    @patch('niffler.analysis.walk_forward_analyzer.ProcessPoolExecutor')
    def test_parallel_periods(self, mock_executor):
        """Test parallel period execution."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            n_jobs=2
        )
        
        periods = [(self.test_data.index[0] + relativedelta(months=i*3),
                   self.test_data.index[0] + relativedelta(months=i*3+3)) 
                  for i in range(3)]
        
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
        with patch('niffler.analysis.walk_forward_analyzer.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2, mock_future3]
            
            results = analyzer._run_periods_parallel(self.test_data, "TEST", periods)
            
            # Should have 2 successful results
            self.assertEqual(len(results), 2)
            self.assertEqual(mock_executor_instance.submit.call_count, 3)
    
    @patch('niffler.analysis.walk_forward_analyzer.BacktestEngine')
    def test_analyze_integration(self, mock_backtest_engine):
        """Test full analyze method integration."""
        # Mock backtest result
        mock_result = Mock()
        mock_result.total_return = 1000
        mock_result.total_return_pct = 10
        mock_result.sharpe_ratio = 1.5
        mock_result.max_drawdown = -5.0
        mock_result.win_rate = 65.0
        mock_result.total_trades = 20
        mock_result.portfolio_values = pd.Series([10000 + i*10 for i in range(30)])
        
        mock_engine_instance = Mock()
        mock_engine_instance.run_backtest.return_value = mock_result
        mock_backtest_engine.return_value = mock_engine_instance
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters,
            test_window_months=6,
            step_months=6,
            n_jobs=1
        )
        
        result = analyzer.analyze(self.test_data, "TEST")
        
        # Check result structure
        self.assertEqual(result.analysis_type, 'walk_forward')
        self.assertEqual(result.strategy_name, 'MockStrategy')
        self.assertEqual(result.symbol, 'TEST')
        self.assertIsNotNone(result.individual_results)
        self.assertIsNotNone(result.combined_metrics)
        self.assertIsNotNone(result.stability_metrics)
        
        # Check analysis parameters
        self.assertEqual(result.analysis_parameters['optimal_parameters'], self.optimal_parameters)
        self.assertIn('test_window_months', result.analysis_parameters)
        self.assertIn('step_months', result.analysis_parameters)
        self.assertIn('n_periods', result.analysis_parameters)
        
        # Should have some successful periods
        self.assertGreater(len(result.individual_results), 0)
    
    def test_trend_consistency_edge_cases(self):
        """Test trend consistency calculation edge cases."""
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            optimal_parameters=self.optimal_parameters
        )
        
        # Test with too few data points
        consistency = analyzer._calculate_trend_consistency([100])
        self.assertEqual(consistency, 1.0)
        
        consistency = analyzer._calculate_trend_consistency([100, 200])
        self.assertEqual(consistency, 1.0)
        
        # Test with constant values
        consistency = analyzer._calculate_trend_consistency([100, 100, 100, 100])
        self.assertEqual(consistency, 1.0)  # No changes = perfect consistency


if __name__ == '__main__':
    unittest.main()