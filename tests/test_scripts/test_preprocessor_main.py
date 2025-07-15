import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import sys
from pathlib import Path

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from scripts import preprocessor
from niffler.data import create_default_manager


class TestPreprocessorMain(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_file = os.path.join(self.temp_dir, 'test_data.csv')
        
        # Create test CSV data
        self.test_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'volume': [1000, 1100, 1200]
        })
        self.test_data.to_csv(self.test_csv_file, index=False)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)

    def test_load_and_clean_csv_success(self):
        """Test successful CSV loading and cleaning."""
        result = preprocessor.load_and_clean_csv(self.test_csv_file)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertTrue(isinstance(result.index, pd.DatetimeIndex))

    def test_load_and_clean_csv_with_timestamp_column(self):
        """Test CSV loading with specific timestamp column."""
        result = preprocessor.load_and_clean_csv(self.test_csv_file, timestamp_column='Date')
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(isinstance(result.index, pd.DatetimeIndex))

    def test_load_and_clean_csv_file_not_found(self):
        """Test CSV loading with non-existent file."""
        non_existent_file = os.path.join(self.temp_dir, 'non_existent.csv')
        
        with patch('scripts.preprocessor.logging.error') as mock_log_error:
            result = preprocessor.load_and_clean_csv(non_existent_file)
            
            self.assertIsNone(result)
            mock_log_error.assert_called_once()

    def test_load_and_clean_csv_invalid_data(self):
        """Test CSV loading with invalid data."""
        invalid_csv = os.path.join(self.temp_dir, 'invalid.csv')
        with open(invalid_csv, 'w') as f:
            f.write("invalid,csv,data\n1,2,invalid_date")
        
        # The function actually handles this gracefully and returns a DataFrame
        # It doesn't fail on invalid data - it just processes what it can
        result = preprocessor.load_and_clean_csv(invalid_csv)
        
        # Should return a DataFrame even with invalid data
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    def test_process_file_success(self):
        """Test successful file processing."""
        output_path = os.path.join(self.temp_dir, 'output.csv')
        
        with patch('scripts.preprocessor.logging.info') as mock_log_info:
            result = preprocessor.process_file(self.test_csv_file, output_path)
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(os.path.exists(output_path))
            
            # Check that logging was called
            mock_log_info.assert_any_call(f"Processing file: {self.test_csv_file}")
            mock_log_info.assert_any_call(f"Cleaned data saved to: {output_path}")

    def test_process_file_no_output_path(self):
        """Test file processing without output path."""
        result = preprocessor.process_file(self.test_csv_file)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    def test_process_file_not_found(self):
        """Test file processing with non-existent file."""
        non_existent_file = os.path.join(self.temp_dir, 'non_existent.csv')
        
        with patch('scripts.preprocessor.logging.error') as mock_log_error:
            result = preprocessor.process_file(non_existent_file)
            
            self.assertIsNone(result)
            mock_log_error.assert_called_with(f"Input file not found: {non_existent_file}")

    @patch('sys.argv', ['script.py', '--input', 'test_file.csv'])
    def test_main_single_file_with_default_output(self):
        """Test main function with single file and default output."""
        test_file = os.path.join(self.temp_dir, 'test_file.csv')
        self.test_data.to_csv(test_file, index=False)
        
        with patch('scripts.preprocessor.Path') as mock_path_class:
            mock_path = Mock()
            mock_path.is_file.return_value = True
            mock_path.is_dir.return_value = False
            mock_path.parent = Path(self.temp_dir)
            mock_path.stem = 'test_file'
            mock_path.suffix = '.csv'
            mock_path_class.return_value = mock_path
            
            with patch('scripts.preprocessor.process_file') as mock_process_file:
                mock_process_file.return_value = self.test_data
                
                with patch('scripts.preprocessor.logging.info') as mock_log_info:
                    preprocessor.main()
                    
                    mock_process_file.assert_called_once()
                    mock_log_info.assert_called_with("File processing completed successfully")

    @patch('sys.argv', ['script.py', '--input', 'test_file.csv', '--output', 'output_file.csv'])
    def test_main_single_file_with_custom_output(self):
        """Test main function with single file and custom output."""
        test_file = os.path.join(self.temp_dir, 'test_file.csv')
        self.test_data.to_csv(test_file, index=False)
        
        with patch('scripts.preprocessor.Path') as mock_path_class:
            mock_path = Mock()
            mock_path.is_file.return_value = True
            mock_path.is_dir.return_value = False
            mock_path_class.return_value = mock_path
            
            with patch('scripts.preprocessor.process_file') as mock_process_file:
                mock_process_file.return_value = self.test_data
                
                with patch('scripts.preprocessor.logging.info') as mock_log_info:
                    preprocessor.main()
                    
                    mock_process_file.assert_called_once_with(str(mock_path), 'output_file.csv')
                    mock_log_info.assert_called_with("File processing completed successfully")

    @patch('sys.argv', ['script.py', '--input', 'test_file.csv'])
    def test_main_single_file_processing_failed(self):
        """Test main function with single file processing failure."""
        with patch('scripts.preprocessor.Path') as mock_path_class:
            mock_path = Mock()
            mock_path.is_file.return_value = True
            mock_path.is_dir.return_value = False
            mock_path.parent = Path(self.temp_dir)
            mock_path.stem = 'test_file'
            mock_path.suffix = '.csv'
            mock_path_class.return_value = mock_path
            
            with patch('scripts.preprocessor.process_file') as mock_process_file:
                mock_process_file.return_value = None
                
                with patch('scripts.preprocessor.logging.error') as mock_log_error:
                    preprocessor.main()
                    
                    mock_log_error.assert_called_with("File processing failed")

    def test_main_directory_processing(self):
        """Test main function with directory processing."""
        # Create multiple CSV files in directory
        test_dir = os.path.join(self.temp_dir, 'test_dir')
        os.makedirs(test_dir)
        
        file1 = os.path.join(test_dir, 'file1.csv')
        file2 = os.path.join(test_dir, 'file2.csv')
        self.test_data.to_csv(file1, index=False)
        self.test_data.to_csv(file2, index=False)
        
        with patch('sys.argv', ['script.py', '--input', test_dir]):
            with patch('scripts.preprocessor.process_file') as mock_process_file:
                mock_process_file.return_value = self.test_data
                
                with patch('scripts.preprocessor.logging.info') as mock_log_info:
                    preprocessor.main()
                    
                    self.assertEqual(mock_process_file.call_count, 2)
                    mock_log_info.assert_any_call(f"Processing 2 CSV files in directory: {test_dir}")

    @patch('sys.argv', ['script.py', '--input', 'test_dir'])
    def test_main_directory_no_csv_files(self):
        """Test main function with directory containing no CSV files."""
        with patch('scripts.preprocessor.Path') as mock_path_class:
            mock_path = Mock()
            mock_path.is_file.return_value = False
            mock_path.is_dir.return_value = True
            mock_path.glob.return_value = []
            mock_path.__str__ = lambda self: 'test_dir'
            mock_path_class.return_value = mock_path
            
            with patch('scripts.preprocessor.logging.error') as mock_log_error:
                preprocessor.main()
                
                mock_log_error.assert_called_with("No CSV files found in directory: test_dir")

    def test_main_directory_processing_with_failure(self):
        """Test main function with directory processing where one file fails."""
        # Create directory with CSV files
        test_dir = os.path.join(self.temp_dir, 'test_dir')
        os.makedirs(test_dir)
        
        file1 = os.path.join(test_dir, 'file1.csv')
        file2 = os.path.join(test_dir, 'file2.csv')
        self.test_data.to_csv(file1, index=False)
        self.test_data.to_csv(file2, index=False)
        
        with patch('sys.argv', ['script.py', '--input', test_dir]):
            with patch('scripts.preprocessor.process_file') as mock_process_file:
                mock_process_file.side_effect = [self.test_data, None]  # Second file fails
                
                with patch('scripts.preprocessor.logging.error') as mock_log_error:
                    preprocessor.main()
                    
                    # Should log error for failed file
                    self.assertTrue(mock_log_error.called)
                    args, kwargs = mock_log_error.call_args
                    self.assertIn('Failed to process:', args[0])

    @patch('sys.argv', ['script.py', '--input', 'non_existent_path'])
    def test_main_path_not_exists(self):
        """Test main function with non-existent path."""
        with patch('scripts.preprocessor.Path') as mock_path_class:
            mock_path = Mock()
            mock_path.is_file.return_value = False
            mock_path.is_dir.return_value = False
            mock_path.__str__ = lambda self: 'non_existent_path'
            mock_path_class.return_value = mock_path
            
            with patch('scripts.preprocessor.logging.error') as mock_log_error:
                preprocessor.main()
                
                mock_log_error.assert_called_with("Input path does not exist: non_existent_path")

    def test_main_directory_custom_suffix(self):
        """Test main function with directory processing and custom suffix."""
        # Create directory with CSV file
        test_dir = os.path.join(self.temp_dir, 'test_dir')
        os.makedirs(test_dir)
        
        file1 = os.path.join(test_dir, 'file1.csv')
        self.test_data.to_csv(file1, index=False)
        
        with patch('sys.argv', ['script.py', '--input', test_dir, '--suffix', '_processed']):
            with patch('scripts.preprocessor.process_file') as mock_process_file:
                mock_process_file.return_value = self.test_data
                
                preprocessor.main()
                
                # Check that process_file was called with correct output path containing custom suffix
                mock_process_file.assert_called_once()
                args, kwargs = mock_process_file.call_args
                self.assertIn('_processed', args[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)