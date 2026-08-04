import pandas as pd
import logging
from typing import Any, Dict, Optional
from .base_preprocessor import BasePreprocessor


class OhlcValidatorPreprocessor(BasePreprocessor):
    """
    Preprocessor that validates OHLC (Open, High, Low, Close) data integrity.

    Rows that violate the OHLC invariants used to be deleted silently, which
    punched holes in the time series that only a later gap detector would
    notice. The behaviour is now explicit and configurable:

    * ``mode='drop'`` (default) removes the offending rows, exactly as before,
      but reports precisely how many rows and which rules were violated.
    * ``mode='repair'`` keeps every row and clamps ``high``/``low`` to the true
      extremes of the bar, which preserves the sampling grid.
    * ``mode='flag'`` changes nothing and only reports.

    Whatever the mode, when the affected fraction reaches
    ``warn_threshold`` a loud WARNING is emitted so a large slice of the data
    can never vanish (or be rewritten) unnoticed. Statistics are exposed on
    :attr:`last_stats` and on ``result.attrs['ohlc_validation']``.
    """

    VALID_MODES = ('drop', 'repair', 'flag')

    def __init__(self, mode: str = 'drop', warn_threshold: float = 0.01):
        """
        Initialize the preprocessor.

        Args:
            mode: What to do with rows that fail validation - 'drop', 'repair'
                or 'flag'.
            warn_threshold: Fraction of affected rows (0..1) at or above which
                the report is escalated to a loud WARNING.

        Raises:
            ValueError: An unknown mode was requested.
        """
        super().__init__("OhlcValidatorPreprocessor")
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode='{mode}'. Supported modes: {self.VALID_MODES}")
        self.mode = mode
        self.warn_threshold = warn_threshold
        self.last_stats: Dict[str, Any] = {}

    def can_process(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has OHLC columns."""
        if df.empty:
            return False

        return self._find_ohlc_columns(df) is not None

    def _find_ohlc_columns(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """
        Locate the OHLC columns case-insensitively.

        Args:
            df: DataFrame to inspect.

        Returns:
            Mapping of 'open'/'high'/'low'/'close' to the actual column names,
            or None if any of them is missing.
        """
        ohlc_cols = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in ['open', 'high', 'low', 'close']:
                ohlc_cols[col_lower] = col

        required_cols = ['open', 'high', 'low', 'close']
        if any(col not in ohlc_cols for col in required_cols):
            return None
        return ohlc_cols

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLC (Open, High, Low, Close) data integrity.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame processed according to the configured mode: invalid rows
            removed ('drop'), clamped ('repair') or left untouched ('flag').
            ``result.attrs['ohlc_validation']`` carries the per-rule statistics.
        """
        if df.empty:
            logging.warning("DataFrame is empty for OHLC validation")
            return df

        ohlc_cols = self._find_ohlc_columns(df)
        if ohlc_cols is None:
            present = {str(c).lower() for c in df.columns}
            missing_cols = [c for c in ['open', 'high', 'low', 'close'] if c not in present]
            logging.info(f"OHLC validation skipped - missing columns: {missing_cols}")
            return df

        logging.info(f"Validating OHLC data integrity (mode={self.mode})")

        open_col = ohlc_cols['open']
        high_col = ohlc_cols['high']
        low_col = ohlc_cols['low']
        close_col = ohlc_cols['close']

        original_rows = len(df)
        invalid_rows = pd.Series(False, index=df.index)
        rule_violations: Dict[str, int] = {}

        rules = (
            ('high_lt_low', df[high_col] < df[low_col], "High < Low"),
            ('high_lt_open', df[high_col] < df[open_col], "High < Open"),
            ('high_lt_close', df[high_col] < df[close_col], "High < Close"),
            ('low_gt_open', df[low_col] > df[open_col], "Low > Open"),
            ('low_gt_close', df[low_col] > df[close_col], "Low > Close"),
        )

        for rule_name, violated, description in rules:
            violated = violated.fillna(False).astype(bool)
            count = int(violated.sum())
            rule_violations[rule_name] = count
            if count:
                logging.warning(f"Found {count} rows where {description}")
                invalid_rows |= violated

        invalid_count = int(invalid_rows.sum())
        invalid_ratio = invalid_count / original_rows if original_rows else 0.0
        dropped_rows = 0
        repaired_rows = 0
        result = df

        if invalid_count:
            if self.mode == 'drop':
                dropped_rows = invalid_count
                logging.warning(
                    f"Removing {invalid_count} of {original_rows} rows ({invalid_ratio:.2%}) "
                    f"with invalid OHLC data - this punches holes in the time series"
                )
                result = df[~invalid_rows]
            elif self.mode == 'repair':
                repaired_rows = invalid_count
                logging.warning(
                    f"Repairing {invalid_count} of {original_rows} rows ({invalid_ratio:.2%}) "
                    f"with invalid OHLC data by clamping High/Low to the bar extremes"
                )
                result = df.copy()
                bar = result.loc[invalid_rows, [open_col, high_col, low_col, close_col]]
                result.loc[invalid_rows, high_col] = bar.max(axis=1)
                result.loc[invalid_rows, low_col] = bar.min(axis=1)
            else:  # 'flag'
                logging.warning(
                    f"Flagged {invalid_count} of {original_rows} rows ({invalid_ratio:.2%}) "
                    f"with invalid OHLC data (mode='flag', data left untouched)"
                )

            if invalid_ratio >= self.warn_threshold:
                logging.warning(
                    f"HIGH OHLC INVALID RATE: {invalid_ratio:.2%} of the series failed OHLC "
                    f"validation (mode={self.mode}); the input data source is suspect"
                )
        else:
            logging.info("All OHLC data is valid")

        if result is df:
            # Never attach stats to the caller's frame.
            result = df.copy(deep=False)

        final_rows = len(result)
        self.last_stats = {
            'mode': self.mode,
            'original_rows': original_rows,
            'final_rows': final_rows,
            'invalid_rows': invalid_count,
            'invalid_ratio': invalid_ratio,
            'dropped_rows': dropped_rows,
            'repaired_rows': repaired_rows,
            'rule_violations': rule_violations,
        }
        result.attrs['ohlc_validation'] = self.last_stats

        logging.info(f"OHLC validation completed. Rows: {original_rows} -> {final_rows}")

        return result
