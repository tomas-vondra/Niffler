import argparse
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_trading_data(df):
    """
    Clean trading data by removing infinite values and handling NaN data.
    
    Args:
        df (pd.DataFrame): DataFrame with trading data
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        logging.warning("Empty DataFrame provided for cleaning")
        return df
    
    original_rows = len(df)
    logging.info(f"Starting data cleaning with {original_rows} rows")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove infinite values
    df_clean = remove_infinite_values(df_clean)
    
    # Handle NaN values with forward-fill
    df_clean = handle_nan_values(df_clean)
    
    # Validate OHLC data integrity
    df_clean = validate_ohlc_data(df_clean)
    
    # Validate data quality
    df_clean = validate_data_quality(df_clean)
    
    # Detect time gaps
    df_clean = detect_time_gaps(df_clean)
    
    final_rows = len(df_clean)
    logging.info(f"Data cleaning completed. Rows: {original_rows} -> {final_rows}")
    
    return df_clean

def remove_infinite_values(df):
    """
    Remove infinite values from the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with potential infinite values
        
    Returns:
        pd.DataFrame: DataFrame with infinite values removed
    """
    # Check for infinite values
    inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
    inf_count = inf_mask.sum().sum()
    
    if inf_count > 0:
        logging.warning(f"Found {inf_count} infinite values, replacing with NaN")
        
        # Replace infinite values with NaN
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        
        # Log which columns had infinite values
        inf_columns = inf_mask.sum()
        inf_columns = inf_columns[inf_columns > 0]
        for col, count in inf_columns.items():
            logging.info(f"Column '{col}': {count} infinite values replaced")
    else:
        logging.info("No infinite values found")
        df_clean = df
    
    return df_clean

def handle_nan_values(df):
    """
    Handle NaN values using forward-fill method.
    
    Args:
        df (pd.DataFrame): DataFrame with potential NaN values
        
    Returns:
        pd.DataFrame: DataFrame with NaN values handled
    """
    # Check for NaN values
    nan_count = df.isnull().sum().sum()
    
    if nan_count > 0:
        logging.warning(f"Found {nan_count} NaN values, applying forward-fill")
        
        # Log which columns have NaN values
        nan_columns = df.isnull().sum()
        nan_columns = nan_columns[nan_columns > 0]
        for col, count in nan_columns.items():
            logging.info(f"Column '{col}': {count} NaN values")
        
        # Apply forward-fill
        df_clean = df.ffill()
        
        # Check if any NaN values remain (at the beginning of the series)
        remaining_nan = df_clean.isnull().sum().sum()
        if remaining_nan > 0:
            logging.warning(f"{remaining_nan} NaN values remain at the beginning of series")
            # For leading NaN values, use backward fill
            df_clean = df_clean.bfill()
            
            # If still NaN values remain, drop those rows
            final_nan = df_clean.isnull().sum().sum()
            if final_nan > 0:
                logging.warning(f"Dropping {final_nan} rows with persistent NaN values")
                df_clean = df_clean.dropna()
    else:
        logging.info("No NaN values found")
        df_clean = df
    
    return df_clean

def validate_ohlc_data(df):
    """
    Validate OHLC (Open, High, Low, Close) data integrity.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with invalid OHLC rows removed
    """
    if df.empty:
        logging.warning("DataFrame is empty for OHLC validation")
        return df
    
    # Find OHLC columns (case-insensitive)
    ohlc_cols = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'high', 'low', 'close']:
            ohlc_cols[col_lower] = col
    
    # Check if we have the required OHLC columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in ohlc_cols]
    
    if missing_cols:
        logging.info(f"OHLC validation skipped - missing columns: {missing_cols}")
        return df
    
    logging.info("Validating OHLC data integrity")
    
    # Get actual column names
    open_col = ohlc_cols['open']
    high_col = ohlc_cols['high']
    low_col = ohlc_cols['low']
    close_col = ohlc_cols['close']
    
    original_rows = len(df)
    invalid_rows = pd.Series([False] * len(df), index=df.index)
    
    # Rule 1: High should be >= Low
    high_low_invalid = df[high_col] < df[low_col]
    if high_low_invalid.any():
        count = high_low_invalid.sum()
        logging.warning(f"Found {count} rows where High < Low")
        invalid_rows |= high_low_invalid
    
    # Rule 2: High should be >= Open
    high_open_invalid = df[high_col] < df[open_col]
    if high_open_invalid.any():
        count = high_open_invalid.sum()
        logging.warning(f"Found {count} rows where High < Open")
        invalid_rows |= high_open_invalid
    
    # Rule 3: High should be >= Close
    high_close_invalid = df[high_col] < df[close_col]
    if high_close_invalid.any():
        count = high_close_invalid.sum()
        logging.warning(f"Found {count} rows where High < Close")
        invalid_rows |= high_close_invalid
    
    # Rule 4: Low should be <= Open
    low_open_invalid = df[low_col] > df[open_col]
    if low_open_invalid.any():
        count = low_open_invalid.sum()
        logging.warning(f"Found {count} rows where Low > Open")
        invalid_rows |= low_open_invalid
    
    # Rule 5: Low should be <= Close
    low_close_invalid = df[low_col] > df[close_col]
    if low_close_invalid.any():
        count = low_close_invalid.sum()
        logging.warning(f"Found {count} rows where Low > Close")
        invalid_rows |= low_close_invalid
    
    # Remove invalid rows
    if invalid_rows.any():
        invalid_count = invalid_rows.sum()
        logging.warning(f"Removing {invalid_count} rows with invalid OHLC data")
        df = df[~invalid_rows]
    else:
        logging.info("All OHLC data is valid")
    
    final_rows = len(df)
    logging.info(f"OHLC validation completed. Rows: {original_rows} -> {final_rows}")
    
    return df

def detect_time_gaps(df):
    """
    Detect time gaps in the trading data sequence.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index
        
    Returns:
        pd.DataFrame: Original DataFrame (gaps are logged but not filled)
    """
    if df.empty or len(df) < 2:
        logging.info("Insufficient data for time gap detection")
        return df
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.warning("Index is not DatetimeIndex, skipping time gap detection")
        return df
    
    logging.info("Detecting time gaps in data")
    
    # Calculate time differences between consecutive rows
    time_diffs = df.index.to_series().diff().dropna()
    
    # Determine expected frequency
    # Use the most common time difference as the expected frequency
    mode_diff = time_diffs.mode()
    
    if mode_diff.empty:
        logging.warning("Could not determine expected frequency")
        return df
    
    expected_freq = mode_diff.iloc[0]
    logging.info(f"Expected frequency: {expected_freq}")
    
    # Define gap threshold (e.g., 1.5x the expected frequency)
    gap_threshold = expected_freq * 1.5
    
    # Find gaps
    gaps = time_diffs[time_diffs > gap_threshold]
    
    if not gaps.empty:
        logging.warning(f"Found {len(gaps)} time gaps in data:")
        for timestamp, gap_size in gaps.items():
            gap_start = df.index[df.index < timestamp][-1] if len(df.index[df.index < timestamp]) > 0 else None
            gap_end = timestamp
            logging.warning(f"  Gap from {gap_start} to {gap_end} (duration: {gap_size})")
    else:
        logging.info("No significant time gaps detected")
    
    # Calculate data completeness
    if len(df) > 1:
        total_expected_periods = (df.index[-1] - df.index[0]) / expected_freq + 1
        completeness = len(df) / total_expected_periods * 100
        logging.info(f"Data completeness: {completeness:.1f}%")
    
    return df

def validate_data_quality(df):
    """
    Validate data quality and ensure trading data integrity.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        pd.DataFrame: Validated DataFrame
    """
    if df.empty:
        logging.warning("DataFrame is empty after cleaning")
        return df
    
    # Check for negative values in price and volume columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    price_cols = [col for col in numeric_cols if col.lower() in ['open', 'high', 'low', 'close']]
    volume_cols = [col for col in numeric_cols if col.lower() in ['volume']]
    
    # Validate price columns (should be positive)
    for col in price_cols:
        if col in df.columns:
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                logging.warning(f"Column '{col}': {negative_count} non-positive values found")
                # Remove rows with non-positive prices
                df = df[df[col] > 0]
    
    # Validate volume columns (should be non-negative)
    for col in volume_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                logging.warning(f"Column '{col}': {negative_count} negative values found")
                # Remove rows with negative volume
                df = df[df[col] >= 0]
    
    # Check for duplicate timestamps/index
    if df.index.duplicated().any():
        duplicate_count = df.index.duplicated().sum()
        logging.warning(f"Found {duplicate_count} duplicate timestamps, removing duplicates")
        df = df[~df.index.duplicated(keep='first')]
    
    # Ensure data is sorted by timestamp
    if not df.index.is_monotonic_increasing:
        logging.info("Sorting data by timestamp")
        df = df.sort_index()
    
    # Final validation summary
    logging.info(f"Data validation completed. Final shape: {df.shape}")
    
    return df

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
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        
        # Try to parse timestamp/date column as index
        timestamp_cols = ['timestamp', 'date', 'Date', 'Timestamp']
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break
        
        # Clean the data
        df_clean = clean_trading_data(df)
        
        # Save cleaned data if output path is specified
        if output_path:
            df_clean.to_csv(output_path)
            logging.info(f"Cleaned data saved to: {output_path}")
        
        return df_clean
        
    except Exception as e:
        logging.error(f"Error processing file {input_path}: {e}")
        return None

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