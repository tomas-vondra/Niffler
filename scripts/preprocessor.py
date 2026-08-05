import argparse
import pandas as pd
import os
import logging
import sys
from pathlib import Path
from typing import Optional

# Running "python scripts/preprocessor.py" puts scripts/ on sys.path but not
# the repository root, so the root has to be added for "import niffler" to work.
# When imported as scripts.preprocessor the root is already importable.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niffler.config.logging import setup_logging
from scripts.common import load_ohlcv_csv

# Configure logging
setup_logging(level="INFO")


def load_and_clean_csv(file_path: str, timestamp_column: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load a CSV file and apply the standard cleaning pipeline.

    Args:
        file_path: Path to the CSV file
        timestamp_column: Name of timestamp column (auto-detected if None)

    Returns:
        Cleaned DataFrame, or None if the file could not be loaded or cleaned.
    """
    try:
        # This CLI also cleans non-OHLCV files, so column and index
        # requirements are relaxed and duplicates are only reported.
        return load_ohlcv_csv(
            file_path,
            clean=True,
            timestamp_column=timestamp_column,
            required_columns=(),
            require_datetime_index=False,
            on_duplicates='warn'
        )
    except (FileNotFoundError, ValueError, OSError) as e:
        logging.error(f"Error loading and cleaning file {file_path}: {e}")
        return None


def process_file(input_path: str, output_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Process a single CSV file with trading data cleaning.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (optional)

    Returns:
        Cleaned DataFrame, or None if processing failed.
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return None
    
    logging.info(f"Processing file: {input_path}")
    
    # Use the updated load_and_clean_csv function
    df_clean = load_and_clean_csv(input_path)
    
    if df_clean is not None and output_path:
        try:
            df_clean.to_csv(output_path)
        except OSError as e:
            logging.error(f"Could not write cleaned data to {output_path}: {e}")
            return None
        logging.info(f"Cleaned data saved to: {output_path}")

    return df_clean


def main() -> int:
    """Clean and preprocess trading data files.

    Returns:
        Process exit code: 0 on success, 1 if any file failed to process.
    """
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
            return 0

        logging.error("File processing failed")
        return 1

    elif input_path.is_dir():
        # Process directory
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            logging.error(f"No CSV files found in directory: {input_path}")
            return 1

        output_dir = Path(args.output) if args.output else input_path
        output_dir.mkdir(exist_ok=True)

        logging.info(f"Processing {len(csv_files)} CSV files in directory: {input_path}")

        failures = 0
        for csv_file in csv_files:
            output_file = output_dir / f"{csv_file.stem}{args.suffix}{csv_file.suffix}"
            result = process_file(str(csv_file), str(output_file))
            if result is None:
                failures += 1
                logging.error(f"Failed to process: {csv_file}")

        return 1 if failures else 0

    logging.error(f"Input path does not exist: {input_path}")
    return 1


if __name__ == '__main__':
    sys.exit(main())