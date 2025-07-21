import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from io import StringIO
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts'))

import analyze
from niffler.analysis.analysis_result import AnalysisResult
from niffler.backtesting.backtest_result import BacktestResult


class TestAnalyzeScript(unittest.TestCase):
    """Test cases for analyze.py script."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, n_days)
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
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_csv.name)
        self.temp_csv.close()
        
        # Test parameters
        self.test_params = {'short_window': 10, 'long_window': 20}
        
        # Create temporary params file
        self.temp_params_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_params, self.temp_params_file)
        self.temp_params_file.close()
        
        # Create optimization results file format
        self.temp_opt_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        opt_data = {
            'results': [
                {'parameters': self.test_params, 'total_return': 1000},
                {'parameters': {'short_window': 15, 'long_window': 25}, 'total_return': 800}
            ]
        }
        json.dump(opt_data, self.temp_opt_file)
        self.temp_opt_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_csv.name)
        os.unlink(self.temp_params_file.name)
        os.unlink(self.temp_opt_file.name)
    
    def test_create_parser(self):
        """Test command line parser creation."""
        parser = analyze.create_parser()
        
        # Test required arguments
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        
        # Test valid arguments
        args = parser.parse_args([
            '--data', 'test.csv',
            '--analysis', 'walk_forward',
            '--strategy', 'simple_ma',
            '--params', '{"short_window": 10, "long_window": 20}'
        ])
        
        self.assertEqual(args.data, 'test.csv')
        self.assertEqual(args.analysis, 'walk_forward')
        self.assertEqual(args.strategy, 'simple_ma')
        self.assertEqual(args.params, '{"short_window": 10, "long_window": 20}')
    
    def test_load_data_valid_csv(self):
        """Test loading valid CSV data."""
        data = analyze.load_data(self.temp_csv.name)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(data.index, pd.DatetimeIndex)
        self.assertTrue(all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        self.assertEqual(len(data), len(self.test_data))
    
    def test_load_data_with_timestamp_column(self):
        """Test loading CSV with timestamp column."""
        # Create data with timestamp column
        data_with_ts = self.test_data.reset_index()
        data_with_ts.rename(columns={'index': 'timestamp'}, inplace=True)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data_with_ts.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            loaded_data = analyze.load_data(temp_file.name)
            self.assertIsInstance(loaded_data.index, pd.DatetimeIndex)
        finally:
            os.unlink(temp_file.name)
    
    def test_load_data_missing_columns(self):
        """Test loading CSV with missing required columns."""
        # Create data missing volume column
        incomplete_data = self.test_data.drop('volume', axis=1)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        incomplete_data.to_csv(temp_file.name)
        temp_file.close()
        
        try:
            with self.assertRaises(ValueError) as context:
                analyze.load_data(temp_file.name)
            self.assertIn("Missing required columns", str(context.exception))
        finally:
            os.unlink(temp_file.name)
    
    def test_load_data_invalid_file(self):
        """Test loading non-existent file."""
        with self.assertRaises(Exception):
            analyze.load_data('non_existent_file.csv')
    
    def test_load_parameters_from_json_string(self):
        """Test loading parameters from JSON string."""
        # Mock args with params
        args = Mock()
        args.params = '{"short_window": 10, "long_window": 20}'
        args.params_file = None
        
        params = analyze.load_parameters(args)
        self.assertEqual(params, {'short_window': 10, 'long_window': 20})
    
    def test_load_parameters_from_file(self):
        """Test loading parameters from file."""
        args = Mock()
        args.params = None
        args.params_file = self.temp_params_file.name
        
        params = analyze.load_parameters(args)
        self.assertEqual(params, self.test_params)
    
    def test_load_parameters_from_optimization_file(self):
        """Test loading parameters from optimization results file."""
        args = Mock()
        args.params = None
        args.params_file = self.temp_opt_file.name
        
        params = analyze.load_parameters(args)
        self.assertEqual(params, self.test_params)  # Should get best result
    
    def test_load_parameters_invalid_json(self):
        """Test loading invalid JSON parameters."""
        args = Mock()
        args.params = '{"invalid": json}'
        args.params_file = None
        
        with self.assertRaises(ValueError):
            analyze.load_parameters(args)
    
    def test_load_parameters_file_not_found(self):
        """Test loading from non-existent file."""
        args = Mock()
        args.params = None
        args.params_file = 'non_existent.json'
        
        with self.assertRaises(ValueError):
            analyze.load_parameters(args)
    
    def test_get_strategy_class(self):
        """Test strategy class retrieval."""
        from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
        
        strategy_class = analyze.get_strategy_class('simple_ma')
        self.assertEqual(strategy_class, SimpleMAStrategy)
        
        with self.assertRaises(ValueError):
            analyze.get_strategy_class('unknown_strategy')
    
    def test_validate_parameters_valid(self):
        """Test parameter validation with valid parameters."""
        from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
        
        # Should not raise any exception
        analyze.validate_parameters(SimpleMAStrategy, self.test_params)
    
    def test_validate_parameters_invalid(self):
        """Test parameter validation with invalid parameters."""
        from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
        
        with self.assertRaises(ValueError):
            analyze.validate_parameters(SimpleMAStrategy, {'invalid_param': 'value'})
    
    @patch('analyze.WalkForwardAnalyzer')
    def test_run_walk_forward_analysis(self, mock_analyzer_class):
        """Test walk-forward analysis execution."""
        # Mock analyzer and result
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create mock result
        mock_result = Mock()
        mock_result.analysis_type = 'walk_forward'
        mock_result.strategy_name = 'SimpleMAStrategy'
        mock_result.symbol = 'TEST'
        mock_result.analysis_start_date = datetime(2023, 1, 1)
        mock_result.analysis_end_date = datetime(2023, 12, 31)
        mock_result.n_periods = 3
        mock_result.combined_metrics = {'avg_return': 100.0, 'std_return': 50.0}
        mock_result.stability_metrics = {'temporal_stability': 0.8}
        mock_result.to_dataframe.return_value = pd.DataFrame({
            'start_date': [datetime(2023, 1, 1), datetime(2023, 7, 1)],
            'end_date': [datetime(2023, 6, 30), datetime(2023, 12, 31)],
            'total_return': [100, 150],
            'total_return_pct': [10, 15],
            'sharpe_ratio': [1.2, 1.5],
            'max_drawdown': [-5, -3],
            'win_rate': [60, 65]
        })
        
        mock_analyzer.analyze.return_value = mock_result
        
        # Mock args
        args = Mock()
        args.strategy = 'simple_ma'
        args.test_window = 6
        args.step = 3
        args.initial_capital = 10000
        args.commission = 0.001
        args.n_jobs = 1
        args.symbol = 'TEST'
        
        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = analyze.run_walk_forward_analysis(args, self.test_data, self.test_params)
        
        # Verify analyzer was called correctly
        mock_analyzer_class.assert_called_once()
        mock_analyzer.analyze.assert_called_once_with(self.test_data, 'TEST')
        
        # Verify result
        self.assertEqual(result, mock_result)
        
        # Verify output contains expected text
        output = mock_stdout.getvalue()
        self.assertIn('WALK-FORWARD ANALYSIS RESULTS', output)
        self.assertIn('Strategy: SimpleMAStrategy', output)
        self.assertIn('Symbol: TEST', output)
    
    @patch('analyze.MonteCarloAnalyzer')
    def test_run_monte_carlo_analysis(self, mock_analyzer_class):
        """Test Monte Carlo analysis execution."""
        # Mock analyzer and result
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create mock result
        mock_result = Mock()
        mock_result.analysis_type = 'monte_carlo'
        mock_result.strategy_name = 'SimpleMAStrategy'
        mock_result.symbol = 'TEST'
        mock_result.analysis_start_date = datetime(2023, 1, 1)
        mock_result.analysis_end_date = datetime(2023, 12, 31)
        mock_result.individual_results = [Mock(), Mock(), Mock()]  # 3 simulations
        mock_result.combined_metrics = {'mean_return': 100.0, 'std_return': 30.0}
        mock_result.stability_metrics = {'return_var_5pct': -50.0}
        
        mock_analyzer.analyze.return_value = mock_result
        mock_analyzer.get_percentile_results.return_value = {
            'total_return': {'p5': 50.0, 'p50': 100.0, 'p95': 150.0}
        }
        
        # Mock args
        args = Mock()
        args.strategy = 'simple_ma'
        args.simulations = 100
        args.bootstrap_pct = 0.8
        args.block_size = 30
        args.initial_capital = 10000
        args.commission = 0.001
        args.n_jobs = 1
        args.random_seed = 42
        args.symbol = 'TEST'
        
        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = analyze.run_monte_carlo_analysis(args, self.test_data, self.test_params)
        
        # Verify analyzer was called correctly
        mock_analyzer_class.assert_called_once()
        mock_analyzer.analyze.assert_called_once_with(self.test_data, 'TEST')
        
        # Verify result
        self.assertEqual(result, mock_result)
        
        # Verify output contains expected text
        output = mock_stdout.getvalue()
        self.assertIn('MONTE CARLO ANALYSIS RESULTS', output)
        self.assertIn('Strategy: SimpleMAStrategy', output)
        self.assertIn('Successful Simulations: 3', output)
    
    def test_save_results_walk_forward(self):
        """Test saving walk-forward analysis results."""
        # Create mock result
        mock_result = Mock()
        mock_result.analysis_type = 'walk_forward'
        mock_result.strategy_name = 'SimpleMAStrategy'
        mock_result.symbol = 'TEST'
        mock_result.analysis_start_date = datetime(2023, 1, 1)
        mock_result.analysis_end_date = datetime(2023, 12, 31)
        mock_result.n_periods = 2
        mock_result.combined_metrics = {'avg_return': 100.0}
        mock_result.stability_metrics = {'temporal_stability': 0.8}
        mock_result.analysis_parameters = {'test_window_months': 6}
        mock_result.metadata = {'test_info': 'test_value'}
        mock_result.get_summary_statistics.return_value = {'total_return': {'mean': 100.0}}
        mock_result.get_performance_consistency.return_value = {'consistency_ratio': 0.5}
        mock_result.to_dataframe.return_value = pd.DataFrame({
            'period': [1, 2],
            'total_return': [100, 150]
        })
        
        # Save to temporary file
        temp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_output.close()
        
        try:
            analyze.save_results(mock_result, temp_output.name)
            
            # Load and verify saved data
            with open(temp_output.name, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data['analysis_type'], 'walk_forward')
            self.assertEqual(saved_data['strategy_name'], 'SimpleMAStrategy')
            self.assertEqual(saved_data['symbol'], 'TEST')
            self.assertEqual(saved_data['n_periods'], 2)
            self.assertIn('period_results', saved_data)
            self.assertIn('metadata', saved_data)
            
        finally:
            os.unlink(temp_output.name)
    
    def test_save_results_monte_carlo(self):
        """Test saving Monte Carlo analysis results."""
        # Create mock result
        mock_result = Mock()
        mock_result.analysis_type = 'monte_carlo'
        mock_result.strategy_name = 'SimpleMAStrategy'
        mock_result.symbol = 'TEST'
        mock_result.analysis_start_date = datetime(2023, 1, 1)
        mock_result.analysis_end_date = datetime(2023, 12, 31)
        mock_result.n_periods = 100
        mock_result.combined_metrics = {'mean_return': 100.0}
        mock_result.stability_metrics = {'return_var_5pct': -50.0}
        mock_result.analysis_parameters = {'n_simulations': 100}
        mock_result.metadata = None
        mock_result.get_summary_statistics.return_value = {'total_return': {'mean': 100.0}}
        mock_result.get_performance_consistency.return_value = {'consistency_ratio': 0.5}
        mock_result.to_dataframe.return_value = pd.DataFrame({
            'simulation': [1, 2],
            'total_return': [90, 110]
        })
        
        # Save to temporary file
        temp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_output.close()
        
        try:
            analyze.save_results(mock_result, temp_output.name)
            
            # Load and verify saved data
            with open(temp_output.name, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data['analysis_type'], 'monte_carlo')
            self.assertIn('simulation_results', saved_data)
            self.assertNotIn('metadata', saved_data)  # Should not include null metadata
            
        finally:
            os.unlink(temp_output.name)
    
    @patch('analyze.setup_logging')
    @patch('analyze.load_data')
    @patch('analyze.load_parameters')
    @patch('analyze.run_walk_forward_analysis')
    @patch('sys.argv', ['analyze.py', '--data', 'test.csv', '--analysis', 'walk_forward', 
                       '--strategy', 'simple_ma', '--params', '{"short_window": 10, "long_window": 20}'])
    def test_main_walk_forward_success(self, mock_run_wf, mock_load_params, mock_load_data, mock_setup_logging):
        """Test main function with successful walk-forward analysis."""
        # Setup mocks
        mock_load_data.return_value = self.test_data
        mock_load_params.return_value = self.test_params
        mock_result = Mock()
        mock_run_wf.return_value = mock_result
        
        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            analyze.main()
        
        # Verify calls
        mock_setup_logging.assert_called_once()
        mock_load_data.assert_called_once_with('test.csv')
        mock_load_params.assert_called_once()
        mock_run_wf.assert_called_once()
        
        # Verify success message
        output = mock_stdout.getvalue()
        self.assertIn('Analysis completed successfully!', output)
    
    @patch('analyze.setup_logging')
    @patch('analyze.load_data')
    @patch('analyze.load_parameters')
    @patch('analyze.run_monte_carlo_analysis')
    @patch('sys.argv', ['analyze.py', '--data', 'test.csv', '--analysis', 'monte_carlo', 
                       '--strategy', 'simple_ma', '--params', '{"short_window": 10, "long_window": 20}',
                       '--simulations', '10'])
    def test_main_monte_carlo_success(self, mock_run_mc, mock_load_params, mock_load_data, mock_setup_logging):
        """Test main function with successful Monte Carlo analysis."""
        # Setup mocks
        mock_load_data.return_value = self.test_data
        mock_load_params.return_value = self.test_params
        mock_result = Mock()
        mock_run_mc.return_value = mock_result
        
        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            analyze.main()
        
        # Verify calls
        mock_setup_logging.assert_called_once()
        mock_load_data.assert_called_once_with('test.csv')
        mock_load_params.assert_called_once()
        mock_run_mc.assert_called_once()
        
        # Verify success message
        output = mock_stdout.getvalue()
        self.assertIn('Analysis completed successfully!', output)
    
    @patch('analyze.setup_logging')
    @patch('analyze.load_data')
    @patch('analyze.save_results')
    @patch('analyze.run_walk_forward_analysis')
    @patch('sys.argv', ['analyze.py', '--data', 'test.csv', '--analysis', 'walk_forward', 
                       '--strategy', 'simple_ma', '--params', '{"short_window": 10, "long_window": 20}',
                       '--output', 'results.json'])
    def test_main_with_output_file(self, mock_run_wf, mock_save_results, mock_load_data, mock_setup_logging):
        """Test main function with output file specified."""
        # Setup mocks
        mock_load_data.return_value = self.test_data
        mock_result = Mock()
        mock_run_wf.return_value = mock_result
        
        with patch('analyze.load_parameters') as mock_load_params:
            mock_load_params.return_value = self.test_params
            analyze.main()
        
        # Verify save_results was called
        mock_save_results.assert_called_once_with(mock_result, 'results.json')
    
    @patch('analyze.setup_logging')
    @patch('analyze.load_data')
    @patch('sys.argv', ['analyze.py', '--data', 'test.csv', '--analysis', 'walk_forward', 
                       '--strategy', 'simple_ma', '--params', '{"short_window": 10, "long_window": 20}'])
    def test_main_with_error(self, mock_load_data, mock_setup_logging):
        """Test main function with error handling."""
        # Make load_data raise an exception
        mock_load_data.side_effect = Exception("Test error")
        
        with self.assertRaises(SystemExit):
            with patch('sys.stderr', new_callable=StringIO):
                analyze.main()
    
    def test_main_invalid_analysis_type(self):
        """Test main function with invalid analysis type."""
        with patch('sys.argv', ['analyze.py', '--data', 'test.csv', '--analysis', 'invalid', 
                                '--strategy', 'simple_ma', '--params', '{"short_window": 10}']):
            with self.assertRaises(SystemExit):
                analyze.main()


class TestAnalyzeIntegration(unittest.TestCase):
    """Integration tests for analyze.py script."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create sample data file
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        n_days = len(dates)
        
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = 100 * np.cumprod(1 + returns)
        
        test_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0.002, 0.005, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0.002, 0.005, n_days))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        # Ensure OHLC relationships
        test_data['high'] = np.maximum(test_data['high'], 
                                     np.maximum(test_data['open'], test_data['close']))
        test_data['low'] = np.minimum(test_data['low'], 
                                    np.minimum(test_data['open'], test_data['close']))
        
        # Save to temporary file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data.to_csv(self.temp_csv.name)
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        os.unlink(self.temp_csv.name)
    
    @patch('analyze.setup_logging')
    @patch('analyze.WalkForwardAnalyzer')
    def test_full_walk_forward_pipeline(self, mock_analyzer_class, mock_setup_logging):
        """Test complete walk-forward analysis pipeline."""
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock result with proper methods
        mock_result = Mock()
        mock_result.analysis_type = 'walk_forward'
        mock_result.strategy_name = 'SimpleMAStrategy'
        mock_result.symbol = 'TESTDATA'
        mock_result.analysis_start_date = datetime(2023, 1, 1)
        mock_result.analysis_end_date = datetime(2023, 6, 30)
        mock_result.n_periods = 1
        mock_result.combined_metrics = {'avg_return': 150.0, 'profitable_periods_pct': 100.0}
        mock_result.stability_metrics = {'temporal_stability': 0.85, 'return_volatility': 25.0}
        
        # Mock dataframe
        df = pd.DataFrame({
            'start_date': [datetime(2023, 1, 1)],
            'end_date': [datetime(2023, 6, 30)],
            'total_return': [150.0],
            'total_return_pct': [15.0],
            'sharpe_ratio': [1.2],
            'max_drawdown': [-5.0],
            'win_rate': [65.0]
        })
        mock_result.to_dataframe.return_value = df
        mock_analyzer.analyze.return_value = mock_result
        
        # Test command line arguments
        test_args = [
            'analyze.py',
            '--data', self.temp_csv.name,
            '--analysis', 'walk_forward',
            '--strategy', 'simple_ma',
            '--params', '{"short_window": 5, "long_window": 15}',
            '--test_window', '3',
            '--step', '1',
            '--symbol', 'TESTDATA'
        ]
        
        with patch('sys.argv', test_args):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                analyze.main()
        
        # Verify output
        output = mock_stdout.getvalue()
        self.assertIn('WALK-FORWARD ANALYSIS RESULTS', output)
        self.assertIn('TESTDATA', output)
        self.assertIn('150.0000', output)  # Should show the return value
    
    @patch('analyze.setup_logging')
    @patch('analyze.MonteCarloAnalyzer')
    def test_full_monte_carlo_pipeline(self, mock_analyzer_class, mock_setup_logging):
        """Test complete Monte Carlo analysis pipeline."""
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock result
        mock_result = Mock()
        mock_result.analysis_type = 'monte_carlo'
        mock_result.strategy_name = 'SimpleMAStrategy'
        mock_result.symbol = 'TESTDATA'
        mock_result.analysis_start_date = datetime(2023, 1, 1)
        mock_result.analysis_end_date = datetime(2023, 6, 30)
        mock_result.individual_results = [Mock() for _ in range(50)]  # 50 simulations
        mock_result.combined_metrics = {'mean_return': 125.0, 'profitable_simulations_pct': 85.0}
        mock_result.stability_metrics = {'return_var_5pct': -45.0, 'return_skewness': 0.2}
        
        # Mock percentile results
        mock_analyzer.get_percentile_results.return_value = {
            'total_return': {'p5': 50.0, 'p25': 90.0, 'p50': 125.0, 'p75': 160.0, 'p95': 200.0},
            'sharpe_ratio': {'p5': 0.5, 'p25': 0.8, 'p50': 1.1, 'p75': 1.4, 'p95': 1.8}
        }
        
        mock_analyzer.analyze.return_value = mock_result
        
        # Test command line arguments
        test_args = [
            'analyze.py',
            '--data', self.temp_csv.name,
            '--analysis', 'monte_carlo',
            '--strategy', 'simple_ma',
            '--params', '{"short_window": 8, "long_window": 21}',
            '--simulations', '50',
            '--bootstrap_pct', '0.75',
            '--block_size', '20',
            '--random_seed', '123',
            '--symbol', 'TESTDATA'
        ]
        
        with patch('sys.argv', test_args):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                analyze.main()
        
        # Verify output
        output = mock_stdout.getvalue()
        self.assertIn('MONTE CARLO ANALYSIS RESULTS', output)
        self.assertIn('TESTDATA', output)
        self.assertIn('50', output)  # Should show number of simulations
        self.assertIn('Percentile Analysis', output)


if __name__ == '__main__':
    unittest.main()