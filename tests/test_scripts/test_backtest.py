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
from scripts.backtest import load_data, main, report_export_outcome
from niffler.backtesting import BacktestEngine, BacktestResult, Trade, TradeSide
from niffler.exporters.exporter_manager import ExportSummary


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
        
    @patch('niffler.data.create_default_manager')
    def test_load_data_with_clean_flag(self, mock_create_manager):
        """Test loading data with clean flag runs the preprocessing pipeline."""
        mock_data = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000.0] * 10
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))

        mock_manager = MagicMock()
        mock_manager.run.return_value = mock_data
        mock_create_manager.return_value = mock_manager

        data = load_data(self.sample_data_path, clean=True)

        self.assertIsInstance(data, pd.DataFrame)
        mock_create_manager.assert_called_once_with()
        mock_manager.run.assert_called_once()

    @patch('niffler.data.create_default_manager')
    def test_load_data_clean_returns_none(self, mock_create_manager):
        """Test loading data when the cleaning pipeline discards everything."""
        mock_manager = MagicMock()
        mock_manager.run.return_value = None
        mock_create_manager.return_value = mock_manager

        with self.assertRaises(ValueError) as context:
            load_data(self.sample_data_path, clean=True)
        self.assertIn("Data cleaning removed all rows", str(context.exception))

    def test_load_data_sorts_unsorted_timestamps(self):
        """Test that rows are returned in chronological order."""
        unsorted_path = os.path.join(self.temp_dir, "unsorted.csv")
        dates = pd.date_range('2024-01-01', periods=5, freq='D')[::-1]
        pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 5,
            'high': [105.0] * 5,
            'low': [95.0] * 5,
            'close': [102.0] * 5,
            'volume': [1000.0] * 5
        }).to_csv(unsorted_path, index=False)

        data = load_data(unsorted_path)

        self.assertTrue(data.index.is_monotonic_increasing)


    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.BacktestEngine')
    @patch('scripts.backtest.SimpleMAStrategy')
    @patch('scripts.backtest.load_data')
    @patch('sys.argv', ['backtest.py', '--data', 'test.csv', '--symbol', 'TEST'])
    def test_main_basic_execution(self, mock_load_data, 
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
            total_trades=0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            num_winning_trades=0,
            num_losing_trades=0
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
        
    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.load_data')
    @patch('sys.argv', ['backtest.py', '--data', 'nonexistent.csv'])
    def test_main_file_not_found(self, mock_load_data, mock_setup_logging):
        """Test main function with non-existent file."""
        mock_load_data.side_effect = FileNotFoundError("Data file not found")

        with patch('builtins.print') as mock_print:
            exit_code = main()

        # Check that the error was reported and a non-zero code returned
        self.assertEqual(exit_code, 1)
        mock_print.assert_called_with("Error: Data file not found", file=sys.stderr)


    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.load_data')
    @patch('sys.argv', ['backtest.py', '--data', 'test.csv', '--exporters', 'csv', '--csv-output-dir', '/tmp'])
    def test_main_with_csv_export(self, mock_load_data, mock_setup_logging):
        """Test main function with CSV export."""
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
            total_trades=2,
            profit_factor=2.5,
            average_win=100.0,
            average_loss=0.0,
            largest_win=100.0,
            largest_loss=0.0,
            num_winning_trades=1,
            num_losing_trades=0
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
                            total_trades=0,
                            profit_factor=0.0,
                            average_win=0.0,
                            average_loss=0.0,
                            largest_win=0.0,
                            largest_loss=0.0,
                            num_winning_trades=0,
                            num_losing_trades=0
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


class TestBacktestExportReporting(unittest.TestCase):
    """Tests for turning export outcomes into a report and an exit code."""

    def _make_result(self) -> BacktestResult:
        """Build a minimal BacktestResult for main() to export."""
        return BacktestResult(
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
            total_trades=0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            num_winning_trades=0,
            num_losing_trades=0
        )

    def test_report_export_outcome_all_successful(self):
        """A summary without failures reports success and exits zero."""
        summary = ExportSummary(successes=['CSVExporter'], failures=[], backtest_id='abc')

        with patch('builtins.print') as mock_print:
            exit_code = report_export_outcome(summary, ['CSVExporter'])

        self.assertEqual(exit_code, 0)
        printed = ' '.join(str(call_args) for call_args in mock_print.call_args_list)
        self.assertIn('abc', printed)
        self.assertIn('CSVExporter', printed)

    def test_report_export_outcome_with_failures(self):
        """A summary with failures reports them and exits non-zero."""
        summary = ExportSummary(
            successes=['ConsoleExporter'],
            failures=[('CSVExporter', 'disk full')],
            backtest_id='abc'
        )

        with patch('builtins.print') as mock_print:
            exit_code = report_export_outcome(summary, ['ConsoleExporter', 'CSVExporter'])

        self.assertEqual(exit_code, 1)
        printed = ' '.join(str(call_args) for call_args in mock_print.call_args_list)
        self.assertIn('disk full', printed)
        self.assertIn('CSVExporter', printed)

    def test_report_export_outcome_all_exporters_failed(self):
        """Every exporter failing must not look like a successful run."""
        summary = ExportSummary(
            successes=[],
            failures=[('CSVExporter', 'boom'), ('ConsoleExporter', 'boom')],
            backtest_id='abc'
        )

        with patch('builtins.print'):
            exit_code = report_export_outcome(summary, ['CSVExporter', 'ConsoleExporter'])

        self.assertEqual(exit_code, 1)

    def test_report_export_outcome_legacy_string_result(self):
        """A manager that only returns a backtest id still works."""
        with patch('builtins.print') as mock_print:
            exit_code = report_export_outcome('legacy-id', ['CSVExporter'])

        self.assertEqual(exit_code, 0)
        printed = ' '.join(str(call_args) for call_args in mock_print.call_args_list)
        self.assertIn('legacy-id', printed)

    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.ExporterManager')
    @patch('scripts.backtest.BacktestEngine')
    @patch('scripts.backtest.SimpleMAStrategy')
    @patch('scripts.backtest.load_data')
    @patch('sys.argv', ['backtest.py', '--data', 'test.csv', '--exporters', 'csv'])
    def test_main_returns_non_zero_when_export_fails(self, mock_load_data, mock_strategy_class,
                                                     mock_engine_class, mock_manager_class,
                                                     mock_setup_logging):
        """main() must not report success when every exporter failed."""
        mock_load_data.return_value = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000.0] * 10
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))

        mock_strategy = MagicMock()
        mock_strategy.get_description.return_value = "Test Strategy"
        mock_strategy_class.return_value = mock_strategy

        mock_engine = MagicMock()
        mock_engine.run_backtest.return_value = self._make_result()
        mock_engine_class.return_value = mock_engine

        mock_manager = MagicMock()
        mock_manager.get_exporter_count.return_value = 1
        mock_manager.get_exporter_names.return_value = ['CSVExporter']
        mock_manager.export_backtest_result.return_value = ExportSummary(
            successes=[],
            failures=[('CSVExporter', 'permission denied')],
            backtest_id='abc'
        )
        mock_manager_class.return_value = mock_manager

        exit_code = main()

        self.assertEqual(exit_code, 1)

    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.ExporterManager')
    @patch('scripts.backtest.BacktestEngine')
    @patch('scripts.backtest.SimpleMAStrategy')
    @patch('scripts.backtest.load_data')
    @patch('sys.argv', ['backtest.py', '--data', 'test.csv', '--exporters', 'csv'])
    def test_main_returns_zero_when_export_succeeds(self, mock_load_data, mock_strategy_class,
                                                    mock_engine_class, mock_manager_class,
                                                    mock_setup_logging):
        """main() returns 0 when every exporter succeeded."""
        mock_load_data.return_value = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000.0] * 10
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))

        mock_strategy = MagicMock()
        mock_strategy.get_description.return_value = "Test Strategy"
        mock_strategy_class.return_value = mock_strategy

        mock_engine = MagicMock()
        mock_engine.run_backtest.return_value = self._make_result()
        mock_engine_class.return_value = mock_engine

        mock_manager = MagicMock()
        mock_manager.get_exporter_count.return_value = 1
        mock_manager.get_exporter_names.return_value = ['CSVExporter']
        mock_manager.export_backtest_result.return_value = ExportSummary(
            successes=['CSVExporter'], failures=[], backtest_id='abc'
        )
        mock_manager_class.return_value = mock_manager

        exit_code = main()

        self.assertEqual(exit_code, 0)

    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.ExporterManager')
    @patch('scripts.backtest.BacktestEngine')
    @patch('scripts.backtest.SimpleMAStrategy')
    @patch('scripts.backtest.load_data')
    @patch('sys.argv', ['backtest.py', '--data', 'test.csv', '--exporters', 'does-not-exist'])
    def test_main_returns_non_zero_when_no_exporter_created(self, mock_load_data, mock_strategy_class,
                                                            mock_engine_class, mock_manager_class,
                                                            mock_setup_logging):
        """An unusable --exporters value must not silently write nothing."""
        mock_load_data.return_value = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000.0] * 10
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))

        mock_strategy = MagicMock()
        mock_strategy.get_description.return_value = "Test Strategy"
        mock_strategy_class.return_value = mock_strategy

        mock_engine = MagicMock()
        mock_engine.run_backtest.return_value = self._make_result()
        mock_engine_class.return_value = mock_engine

        mock_manager = MagicMock()
        mock_manager.get_exporter_count.return_value = 0
        mock_manager_class.return_value = mock_manager

        exit_code = main()

        self.assertEqual(exit_code, 1)
        mock_manager.export_backtest_result.assert_not_called()

    @patch('scripts.backtest.setup_logging')
    @patch('scripts.backtest.collect_provenance')
    @patch('scripts.backtest.ExporterManager')
    @patch('scripts.backtest.BacktestEngine')
    @patch('scripts.backtest.SimpleMAStrategy')
    @patch('scripts.backtest.load_data')
    @patch('sys.argv', ['backtest.py', '--data', 'test.csv', '--exporters', 'csv'])
    def test_main_collects_provenance_once_and_passes_it_to_the_manager(
            self, mock_load_data, mock_strategy_class, mock_engine_class,
            mock_manager_class, mock_collect_provenance, mock_setup_logging):
        """Provenance is collected once per run and shared by every exporter.

        Collecting it per exporter would hash the input CSV once per destination.
        """
        mock_load_data.return_value = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000.0] * 10
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))

        mock_strategy = MagicMock()
        mock_strategy.get_description.return_value = "Test Strategy"
        mock_strategy_class.return_value = mock_strategy

        mock_engine = MagicMock()
        mock_engine.run_backtest.return_value = self._make_result()
        mock_engine_class.return_value = mock_engine

        provenance = {'code': {'git_sha': 'a' * 40}}
        mock_collect_provenance.return_value = provenance

        mock_manager = MagicMock()
        mock_manager.get_exporter_count.return_value = 1
        mock_manager.get_exporter_names.return_value = ['CSVExporter']
        mock_manager.export_backtest_result.return_value = ExportSummary(
            successes=['CSVExporter'], failures=[], backtest_id='abc'
        )
        mock_manager_class.return_value = mock_manager

        exit_code = main()

        self.assertEqual(exit_code, 0)
        # Fingerprinted against the file named on the command line.
        mock_collect_provenance.assert_called_once_with('test.csv')
        self.assertIs(
            mock_manager.export_backtest_result.call_args.kwargs['provenance'],
            provenance
        )


class TestBenchmarkCommandLine(unittest.TestCase):
    """The benchmark and significance flags must reach the engine."""

    def _run(self, argv):
        """Run main() with everything but the engine construction mocked out."""
        data = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000.0] * 10
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))

        result = BacktestResult(
            strategy_name="TestStrategy", symbol="TEST",
            start_date=data.index[0], end_date=data.index[-1],
            initial_capital=10000.0, final_capital=10000.0,
            total_return=0.0, total_return_pct=0.0, trades=[],
            portfolio_values=pd.Series([10000.0] * 10, index=data.index),
            max_drawdown=0.0, sharpe_ratio=0.0, win_rate=0.0,
            total_trades=0, profit_factor=0.0
        )

        with patch('sys.argv', argv), \
                patch('scripts.backtest.setup_logging'), \
                patch('scripts.backtest.load_data', return_value=data), \
                patch('scripts.backtest.SimpleMAStrategy') as strategy_class, \
                patch('scripts.backtest.ExporterManager') as manager_class, \
                patch('scripts.backtest.BacktestEngine') as engine_class:
            strategy_class.return_value = MagicMock()
            engine = MagicMock()
            engine.run_backtest.return_value = result
            engine_class.return_value = engine

            manager = MagicMock()
            manager.get_exporter_count.return_value = 1
            manager.export_backtest_result.return_value = ExportSummary(
                successes=['ConsoleExporter'], failures=[], backtest_id='abc'
            )
            manager_class.return_value = manager

            with patch('builtins.print'):
                exit_code = main()

        return exit_code, engine_class.call_args.kwargs

    def test_defaults_are_buy_and_hold_and_thirty_trades(self):
        exit_code, kwargs = self._run(['backtest.py', '--data', 'test.csv'])

        self.assertEqual(exit_code, 0)
        self.assertEqual(kwargs['benchmark'], 'buy_and_hold')
        self.assertEqual(kwargs['min_trades_for_significance'], 30)

    def test_benchmark_can_be_switched_off(self):
        _, kwargs = self._run(['backtest.py', '--data', 'test.csv',
                               '--benchmark', 'none'])

        self.assertEqual(kwargs['benchmark'], 'none')

    def test_significance_gate_is_configurable(self):
        _, kwargs = self._run(['backtest.py', '--data', 'test.csv',
                               '--min-trades-for-significance', '50'])

        self.assertEqual(kwargs['min_trades_for_significance'], 50)

    def test_bootstrap_flags_reach_the_engine(self):
        _, kwargs = self._run(['backtest.py', '--data', 'test.csv',
                               '--bootstrap-samples', '250',
                               '--bootstrap-seed', '7'])

        self.assertEqual(kwargs['bootstrap_samples'], 250)
        self.assertEqual(kwargs['bootstrap_seed'], 7)

    def test_an_unknown_benchmark_is_rejected_by_argparse(self):
        with patch('sys.argv', ['backtest.py', '--data', 'test.csv',
                                '--benchmark', 'spx']):
            with self.assertRaises(SystemExit):
                main()


if __name__ == '__main__':
    unittest.main()