import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import sys
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the backtest script functions
from scripts.backtest import load_data, print_backtest_results, main
from niffler.backtesting import BacktestEngine, BacktestResult, Trade, TradeSide


class TestBacktestScript(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Create sample CSV data
        self.sample_data_path = os.path.join(self.temp_dir, "test_data.csv")
        self.create_sample_csv_data()
        
        # Create sample cleaned CSV data
        self.cleaned_data_path = os.path.join(self.temp_dir, "test_data_cleaned.csv")
        self.create_sample_cleaned_csv_data()
        
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_sample_csv_data(self):
        """Create sample CSV data for testing."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(105, 115, 50),
            'low': np.random.uniform(95, 105, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.uniform(1000, 2000, 50)
        })
        
        # Ensure OHLC relationships are valid
        for i in range(len(data)):
            low = data.iloc[i]['low']
            high = data.iloc[i]['high']
            data.iloc[i, data.columns.get_loc('open')] = np.random.uniform(low, high)
            data.iloc[i, data.columns.get_loc('close')] = np.random.uniform(low, high)
            
        data.to_csv(self.sample_data_path, index=False)
        
    def create_sample_cleaned_csv_data(self):
        """Create sample cleaned CSV data for testing."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.linspace(100, 150, 50),  # Trending upward
            'high': np.linspace(105, 155, 50),
            'low': np.linspace(95, 145, 50),
            'close': np.linspace(102, 152, 50),
            'volume': [1500.0] * 50
        })
        
        data.to_csv(self.cleaned_data_path, index=False)
        
    def test_load_data_cleaned_file(self):
        """Test loading cleaned CSV data."""
        data = load_data(self.cleaned_data_path, clean=False)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 50)
        self.assertTrue(all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        self.assertIsInstance(data.index, pd.DatetimeIndex)
        
    def test_load_data_file_not_found(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data("/nonexistent/path/file.csv")
            
    def test_load_data_missing_columns(self):
        """Test loading data with missing columns."""
        # Create CSV with missing volume column
        incomplete_data_path = os.path.join(self.temp_dir, "incomplete.csv")
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10
            # Missing volume column
        })
        data.to_csv(incomplete_data_path, index=False)
        
        with self.assertRaises(ValueError) as context:
            load_data(incomplete_data_path)
        self.assertIn("Missing required columns", str(context.exception))
        
    @patch('scripts.preprocessor.load_and_clean_csv')
    def test_load_data_with_clean_flag(self, mock_load_clean):
        """Test loading data with clean flag."""
        # Mock the load_and_clean_csv function
        mock_data = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000.0] * 10
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        mock_load_clean.return_value = mock_data
        
        data = load_data(self.sample_data_path, clean=True)
        
        self.assertIsInstance(data, pd.DataFrame)
        mock_load_clean.assert_called_once_with(self.sample_data_path)
        
    @patch('scripts.preprocessor.load_and_clean_csv')
    def test_load_data_clean_returns_none(self, mock_load_clean):
        """Test loading data when clean function returns None."""
        mock_load_clean.return_value = None
        
        with self.assertRaises(ValueError) as context:
            load_data(self.sample_data_path, clean=True)
        self.assertIn("Failed to load and clean data", str(context.exception))
        
    def test_print_backtest_results(self):
        """Test print_backtest_results function."""
        # Create mock BacktestResult
        trades = [
            Trade(pd.Timestamp('2024-01-05'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0),
            Trade(pd.Timestamp('2024-01-15'), 'TEST', TradeSide.SELL, 110.0, 10.0, 1100.0)
        ]
        
        portfolio_values = pd.Series([10000, 10500, 11000], 
                                   index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        result = BacktestResult(
            strategy_name="TestStrategy",
            symbol="TEST",
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2024-01-31'),
            initial_capital=10000.0,
            final_capital=11000.0,
            total_return=1000.0,
            total_return_pct=10.0,
            trades=trades,
            portfolio_values=portfolio_values,
            max_drawdown=-2.0,
            sharpe_ratio=1.5,
            win_rate=100.0,
            total_trades=2
        )
        
        # Capture stdout
        with patch('builtins.print') as mock_print:
            print_backtest_results(result)
            
            # Check that print was called multiple times
            self.assertTrue(mock_print.called)
            
            # Check that key information was printed
            printed_text = ' '.join([str(call[0][0]) for call in mock_print.call_args_list])
            self.assertIn("TestStrategy", printed_text)
            self.assertIn("TEST", printed_text)
            self.assertIn("10.00%", printed_text)
            self.assertIn("100.0%", printed_text)  # Win rate
            
    def test_print_backtest_results_no_trades(self):
        """Test print_backtest_results with no trades."""
        result = BacktestResult(
            strategy_name="TestStrategy",
            symbol="TEST",
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2024-01-31'),
            initial_capital=10000.0,
            final_capital=10000.0,
            total_return=0.0,
            total_return_pct=0.0,
            trades=[],
            portfolio_values=pd.Series([10000] * 3, 
                                     index=pd.date_range('2024-01-01', periods=3, freq='D')),
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            total_trades=0
        )
        
        # Should not raise any errors
        with patch('builtins.print'):
            print_backtest_results(result)
            
    def test_print_backtest_results_many_trades(self):
        """Test print_backtest_results with many trades (tests trade limiting)."""
        trades = [
            Trade(pd.Timestamp(f'2024-01-{i:02d}'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0)
            for i in range(1, 11)  # 10 trades
        ]
        
        result = BacktestResult(
            strategy_name="TestStrategy",
            symbol="TEST",
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2024-01-31'),
            initial_capital=10000.0,
            final_capital=11000.0,
            total_return=1000.0,
            total_return_pct=10.0,
            trades=trades,
            portfolio_values=pd.Series([10000] * 3, 
                                     index=pd.date_range('2024-01-01', periods=3, freq='D')),
            max_drawdown=-2.0,
            sharpe_ratio=1.5,
            win_rate=100.0,
            total_trades=10
        )
        
        with patch('builtins.print') as mock_print:
            print_backtest_results(result)
            
            # Check that "and X more trades" message appears
            printed_text = ' '.join([str(call[0][0]) for call in mock_print.call_args_list])
            self.assertIn("and 5 more trades", printed_text)
            
    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.BacktestEngine')
    @patch('scripts.backtest.SimpleMAStrategy')
    @patch('scripts.backtest.load_data')
    @patch('scripts.backtest.print_backtest_results')
    @patch('sys.argv', ['backtest.py', '--data', 'test.csv', '--symbol', 'TEST'])
    def test_main_basic_execution(self, mock_print_results, mock_load_data, 
                                 mock_strategy_class, mock_engine_class, mock_setup_logging):
        """Test main function basic execution."""
        # Setup mocks
        mock_data = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000.0] * 10
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        mock_load_data.return_value = mock_data
        
        mock_strategy = MagicMock()
        mock_strategy.get_description.return_value = "Test Strategy Description"
        mock_strategy_class.return_value = mock_strategy
        
        mock_engine = MagicMock()
        mock_result = BacktestResult(
            strategy_name="TestStrategy",
            symbol="TEST",
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2024-01-10'),
            initial_capital=10000.0,
            final_capital=10000.0,
            total_return=0.0,
            total_return_pct=0.0,
            trades=[],
            portfolio_values=pd.Series([10000] * 10, index=pd.date_range('2024-01-01', periods=10, freq='D')),
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            total_trades=0
        )
        mock_engine.run_backtest.return_value = mock_result
        mock_engine_class.return_value = mock_engine
        
        # Run main
        main()
        
        # Check that components were called
        mock_setup_logging.assert_called_once()
        mock_load_data.assert_called_once()
        mock_strategy_class.assert_called_once()
        mock_engine_class.assert_called_once()
        mock_engine.run_backtest.assert_called_once()
        mock_print_results.assert_called_once()
        
    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.load_data')
    @patch('sys.argv', ['backtest.py', '--data', 'nonexistent.csv'])
    def test_main_file_not_found(self, mock_load_data, mock_setup_logging):
        """Test main function with non-existent file."""
        mock_load_data.side_effect = FileNotFoundError("Data file not found")
        
        with patch('builtins.print') as mock_print:
            with patch('sys.exit') as mock_exit:
                main()
                
                # Check that error was printed and exit was called
                mock_print.assert_called_with("Error: Data file not found")
                mock_exit.assert_called_with(1)
                
    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.load_data')
    @patch('sys.argv', ['backtest.py', '--data', 'test.csv', '--output', 'results.csv'])
    def test_main_with_output_file(self, mock_load_data, mock_setup_logging):
        """Test main function with output file."""
        # Setup mocks
        mock_data = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000.0] * 10
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        mock_load_data.return_value = mock_data
        
        # Create mock result with trades
        trades = [
            Trade(pd.Timestamp('2024-01-05'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0),
            Trade(pd.Timestamp('2024-01-15'), 'TEST', TradeSide.SELL, 110.0, 10.0, 1100.0)
        ]
        
        portfolio_values = pd.Series([10000, 10500, 11000], 
                                   index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        mock_result = BacktestResult(
            strategy_name="TestStrategy",
            symbol="TEST",
            start_date=pd.Timestamp('2024-01-01'),
            end_date=pd.Timestamp('2024-01-31'),
            initial_capital=10000.0,
            final_capital=11000.0,
            total_return=1000.0,
            total_return_pct=10.0,
            trades=trades,
            portfolio_values=portfolio_values,
            max_drawdown=-2.0,
            sharpe_ratio=1.5,
            win_rate=100.0,
            total_trades=2
        )
        
        with patch('scripts.backtest.BacktestEngine') as mock_engine_class:
            with patch('scripts.backtest.SimpleMAStrategy') as mock_strategy_class:
                with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                    mock_engine = MagicMock()
                    mock_engine.run_backtest.return_value = mock_result
                    mock_engine_class.return_value = mock_engine
                    
                    mock_strategy = MagicMock()
                    mock_strategy.get_description.return_value = "Test Strategy"
                    mock_strategy_class.return_value = mock_strategy
                    
                    main()
                    
                    # Check that CSV files were saved
                    self.assertTrue(mock_to_csv.called)
                    
    @patch('sys.argv', ['backtest.py', '--data', 'test.csv', '--log-level', 'DEBUG'])
    def test_main_command_line_arguments(self):
        """Test main function command line argument parsing."""
        with patch('scripts.backtest.setup_logging') as mock_setup_logging:
            with patch('scripts.backtest.load_data') as mock_load_data:
                with patch('scripts.backtest.BacktestEngine') as mock_engine_class:
                    with patch('scripts.backtest.SimpleMAStrategy') as mock_strategy_class:
                        # Mock to prevent actual execution
                        mock_load_data.return_value = pd.DataFrame({
                            'open': [100.0] * 10,
                            'high': [105.0] * 10,
                            'low': [95.0] * 10,
                            'close': [102.0] * 10,
                            'volume': [1000.0] * 10
                        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
                        
                        mock_engine = MagicMock()
                        mock_result = BacktestResult(
                            strategy_name="TestStrategy",
                            symbol="TEST",
                            start_date=pd.Timestamp('2024-01-01'),
                            end_date=pd.Timestamp('2024-01-10'),
                            initial_capital=10000.0,
                            final_capital=10000.0,
                            total_return=0.0,
                            total_return_pct=0.0,
                            trades=[],
                            portfolio_values=pd.Series([10000] * 10, index=pd.date_range('2024-01-01', periods=10, freq='D')),
                            max_drawdown=0.0,
                            sharpe_ratio=0.0,
                            win_rate=0.0,
                            total_trades=0
                        )
                        mock_engine.run_backtest.return_value = mock_result
                        mock_engine_class.return_value = mock_engine
                        
                        mock_strategy = MagicMock()
                        mock_strategy.get_description.return_value = "Test Strategy"
                        mock_strategy_class.return_value = mock_strategy
                        
                        main()
                        
                        # Check that setup_logging was called with DEBUG level
                        mock_setup_logging.assert_called_once_with(level='DEBUG')
                        
    def test_main_invalid_strategy(self):
        """Test main function with invalid strategy (argparse should handle)."""
        # This test just ensures argparse handles invalid strategy without crashing
        # argparse will call sys.exit() when invalid choice is provided
        
        # Create a mock argv with invalid strategy
        test_argv = ['backtest.py', '--data', 'test.csv', '--strategy', 'invalid_strategy']
        
        # Test that argparse catches the invalid strategy choice
        with patch('sys.argv', test_argv):
            with patch('argparse.ArgumentParser.error') as mock_error:
                with patch('scripts.backtest.setup_logging'):
                    try:
                        main()
                    except SystemExit:
                        pass  # Expected when argparse encounters invalid choice
                        
                    # The test passes if argparse handles the error gracefully


if __name__ == '__main__':
    unittest.main()