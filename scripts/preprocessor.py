import argparse
import pandas as pd
import numpy as np
import os
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from niffler.data import create_default_manager
from config.logging import setup_logging

# Configure logging
setup_logging(level="INFO")


def load_and_clean_csv(file_path, timestamp_column=None):
    """
    Load a CSV file and apply standard cleaning pipeline using PreprocessorManager.
    
    Args:
        file_path (str): Path to the CSV file
        timestamp_column (str): Name of timestamp column (auto-detected if None)
        
    Returns:
        pd.DataFrame: Cleaned DataFrame or None if failed
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Try to parse timestamp/date column as index
        if timestamp_column:
            timestamp_cols = [timestamp_column]
        else:
            timestamp_cols = ['timestamp', 'date', 'Date', 'Timestamp']
        
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break
        
        # Clean the data using PreprocessorManager
        manager = create_default_manager()
        return manager.run(df)
        
    except Exception as e:
        logging.error(f"Error loading and cleaning file {file_path}: {e}")
        return None


def process_file(input_path, output_path=None):
    """
    Process a single CSV file with trading data cleaning.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to output CSV file (optional)
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return None
    
    logging.info(f"Processing file: {input_path}")
    
    # Use the updated load_and_clean_csv function
    df_clean = load_and_clean_csv(input_path)
    
    if df_clean is not None and output_path:
        df_clean.to_csv(output_path)
        logging.info(f"Cleaned data saved to: {output_path}")
    
    return df_clean

def main():
    parser = argparse.ArgumentParser(description='Clean and preprocess trading data.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file or directory containing CSV files.')
    parser.add_argument('--output', type=str,
                        help='Path to output CSV file or directory. If not specified, creates cleaned_ prefix.')
    parser.add_argument('--suffix', type=str, default='_cleaned',
                        help='Suffix to add to output files when processing directory (default: _cleaned).')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        if args.output:
            output_path = args.output
        else:
            # Create output filename with suffix
            output_path = input_path.parent / f"{input_path.stem}{args.suffix}{input_path.suffix}"
        
        result = process_file(str(input_path), str(output_path))
        if result is not None:
            logging.info("File processing completed successfully")
        else:
            logging.error("File processing failed")
    
    elif input_path.is_dir():
        # Process directory
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            logging.error(f"No CSV files found in directory: {input_path}")
            return
        
        output_dir = Path(args.output) if args.output else input_path
        output_dir.mkdir(exist_ok=True)
        
        logging.info(f"Processing {len(csv_files)} CSV files in directory: {input_path}")
        
        for csv_file in csv_files:
            output_file = output_dir / f"{csv_file.stem}{args.suffix}{csv_file.suffix}"
            result = process_file(str(csv_file), str(output_file))
            if result is None:
                logging.error(f"Failed to process: {csv_file}")
    
    else:
        logging.error(f"Input path does not exist: {input_path}")

if __name__ == '__main__':
    main()