import pandas as pd
import logging
from typing import Any, Dict, Optional, Sequence
from .base_preprocessor import BasePreprocessor


class NanValuePreprocessor(BasePreprocessor):
    """
    Preprocessor that resolves NaN values with an explicit, per-column policy.

    A blanket ``ffill().bfill()`` over an OHLCV frame is not safe:

    * Forward-filling **prices** is defensible (the last traded price is still
      the best estimate), but forward-filling **volume** invents trades that
      never happened, so volume is filled with ``0`` instead.
    * Forward-filled bars are flat, and flat bars have zero return. They deflate
      measured volatility and therefore *inflate* the Sharpe ratio downstream,
      so the number of fabricated bars is counted, exposed and logged.
    * Backward-filling copies a later bar into an earlier timestamp, which is
      look-ahead bias. It is therefore **off by default** and must be opted into
      explicitly; unresolvable (leading) rows are dropped instead.

    After :meth:`process`, the statistics are available both on the instance
    (:attr:`last_stats`) and on the returned frame (``df.attrs['nan_fill']``).
    """

    #: Columns treated as prices (forward-fill is acceptable), lower-cased.
    PRICE_COLUMNS = ('open', 'high', 'low', 'close', 'adj close', 'adj_close',
                     'vwap', 'price', 'bid', 'ask')

    #: Columns treated as flow/volume quantities (missing means "nothing traded").
    VOLUME_COLUMNS = ('volume', 'vol', 'quote_volume', 'quote_asset_volume',
                      'base_volume', 'trades', 'num_trades', 'number_of_trades',
                      'turnover', 'open_interest')

    #: Fraction of fabricated rows above which the warning is escalated.
    HIGH_SYNTHETIC_RATIO = 0.05

    _VALID_POLICIES = ('ffill', 'zero', 'none')

    def __init__(self, price_fill: str = 'ffill', volume_fill: str = 'zero',
                 other_fill: str = 'ffill', allow_backward_fill: bool = False,
                 max_fill_gap: Optional[int] = None,
                 add_synthetic_column: bool = False,
                 price_columns: Optional[Sequence[str]] = None,
                 volume_columns: Optional[Sequence[str]] = None):
        """
        Initialize the preprocessor.

        Args:
            price_fill: Policy for price columns: 'ffill', 'zero' or 'none'.
            volume_fill: Policy for volume/flow columns: 'ffill', 'zero' or 'none'.
            other_fill: Policy for every other column.
            allow_backward_fill: Opt in to backward-filling values that remain
                NaN after the per-column pass. This imports future information
                into the past and is off by default.
            max_fill_gap: Maximum consecutive NaNs a forward-fill may bridge.
                ``None`` means unlimited (pandas default).
            add_synthetic_column: Add a boolean ``is_synthetic`` column marking
                rows that contain at least one fabricated value. Off by default
                so the frame's shape is preserved for downstream consumers.
            price_columns: Override the price column names (lower-cased).
            volume_columns: Override the volume column names (lower-cased).

        Raises:
            ValueError: An unknown fill policy was requested.
        """
        super().__init__("NanValuePreprocessor")

        for label, policy in (('price_fill', price_fill), ('volume_fill', volume_fill),
                              ('other_fill', other_fill)):
            if policy not in self._VALID_POLICIES:
                raise ValueError(
                    f"Invalid {label}='{policy}'. Supported policies: {self._VALID_POLICIES}"
                )

        self.price_fill = price_fill
        self.volume_fill = volume_fill
        self.other_fill = other_fill
        self.allow_backward_fill = allow_backward_fill
        self.max_fill_gap = max_fill_gap
        self.add_synthetic_column = add_synthetic_column
        self.price_columns = tuple(price_columns) if price_columns is not None else self.PRICE_COLUMNS
        self.volume_columns = tuple(volume_columns) if volume_columns is not None else self.VOLUME_COLUMNS
        self.last_stats: Dict[str, Any] = {}

    def column_policy(self, column: str) -> str:
        """
        Resolve the fill policy for a single column.

        Args:
            column: Column name (matched case-insensitively).

        Returns:
            One of 'ffill', 'zero', 'none'.
        """
        key = str(column).strip().lower()
        if key in self.volume_columns:
            return self.volume_fill
        if key in self.price_columns:
            return self.price_fill
        return self.other_fill

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve NaN values with the configured per-column policy.

        Args:
            df: DataFrame with potential NaN values

        Returns:
            DataFrame with NaN values resolved. Rows that could not be resolved
            (typically leading rows, since backward-fill is disabled) are
            dropped. ``result.attrs['nan_fill']`` carries the fill statistics.
        """
        if df.empty:
            logging.warning("Empty DataFrame provided for NaN value processing")
            return df

        original_rows = len(df)
        nan_mask = df.isnull()
        nan_count = int(nan_mask.to_numpy().sum())

        if nan_count == 0:
            logging.info("No NaN values found")
            result = df.copy(deep=False)
            self.last_stats = self._build_stats(
                original_rows=original_rows, total_nan=0, per_column={},
                filled_forward=0, filled_zero=0, filled_backward=0,
                dropped_rows=0, synthetic_rows=0,
            )
            result.attrs['nan_fill'] = self.last_stats
            return result

        logging.warning(f"Found {nan_count} NaN values, applying per-column fill policy")

        nan_per_column = nan_mask.sum()
        for col, count in nan_per_column[nan_per_column > 0].items():
            logging.info(f"Column '{col}': {count} NaN values (policy={self.column_policy(col)})")

        result = df.copy()
        synthetic_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        per_column: Dict[str, int] = {}
        filled_forward = 0
        filled_zero = 0

        for col in df.columns:
            col_nan = nan_mask[col]
            if not col_nan.any():
                continue

            policy = self.column_policy(col)
            if policy == 'ffill':
                filled = result[col].ffill(limit=self.max_fill_gap)
            elif policy == 'zero':
                filled = result[col].fillna(0)
            else:  # 'none'
                continue

            resolved = col_nan & filled.notna()
            resolved_count = int(resolved.sum())
            if resolved_count == 0:
                continue

            result[col] = filled
            synthetic_mask[col] = resolved
            per_column[str(col)] = resolved_count
            if policy == 'ffill':
                filled_forward += resolved_count
            else:
                filled_zero += resolved_count

        filled_backward = 0
        remaining_mask = result.isnull()
        remaining = int(remaining_mask.to_numpy().sum())

        if remaining > 0 and self.allow_backward_fill:
            logging.warning(
                f"LOOK-AHEAD: backward-filling {remaining} value(s) copies future bars into "
                f"earlier timestamps - only valid for offline data inspection, never for backtests"
            )
            backfilled = result.bfill()
            resolved = remaining_mask & backfilled.notna()
            filled_backward = int(resolved.to_numpy().sum())
            synthetic_mask |= resolved
            for col in df.columns:
                col_resolved = int(resolved[col].sum())
                if col_resolved:
                    per_column[str(col)] = per_column.get(str(col), 0) + col_resolved
            result = backfilled

        unresolved_rows = result.isnull().any(axis=1)
        dropped_rows = int(unresolved_rows.sum())
        if dropped_rows > 0:
            reason = "" if self.allow_backward_fill else \
                " (backward-fill is disabled to avoid look-ahead)"
            logging.warning(
                f"Dropping {dropped_rows} row(s) with unresolvable NaN values{reason}"
            )
            result = result[~unresolved_rows]
            synthetic_mask = synthetic_mask[~unresolved_rows]

        synthetic_rows_mask = synthetic_mask.any(axis=1) if not synthetic_mask.empty \
            else pd.Series(False, index=result.index, dtype=bool)
        synthetic_rows = int(synthetic_rows_mask.sum())

        self.last_stats = self._build_stats(
            original_rows=original_rows, total_nan=nan_count, per_column=per_column,
            filled_forward=filled_forward, filled_zero=filled_zero,
            filled_backward=filled_backward, dropped_rows=dropped_rows,
            synthetic_rows=synthetic_rows,
        )

        if synthetic_rows > 0:
            ratio = self.last_stats['synthetic_row_ratio']
            message = (
                f"FABRICATED DATA: {synthetic_rows}/{original_rows} bars ({ratio:.2%}) contain "
                f"at least one filled value (forward={filled_forward}, zero={filled_zero}, "
                f"backward={filled_backward}). Flat fabricated bars deflate volatility and "
                f"inflate downstream Sharpe ratios."
            )
            if ratio >= self.HIGH_SYNTHETIC_RATIO:
                logging.warning(f"HIGH {message}")
            else:
                logging.warning(message)

        if self.add_synthetic_column:
            result = result.copy()
            result['is_synthetic'] = synthetic_rows_mask.reindex(result.index, fill_value=False)

        result.attrs['nan_fill'] = self.last_stats
        return result

    def _build_stats(self, original_rows: int, total_nan: int, per_column: Dict[str, int],
                     filled_forward: int, filled_zero: int, filled_backward: int,
                     dropped_rows: int, synthetic_rows: int) -> Dict[str, Any]:
        """
        Assemble the statistics dictionary for a single ``process`` call.

        Args:
            original_rows: Row count before processing.
            total_nan: Number of NaN cells found.
            per_column: Filled-cell count per column.
            filled_forward: Cells resolved by forward-fill.
            filled_zero: Cells resolved by zero-fill.
            filled_backward: Cells resolved by backward-fill.
            dropped_rows: Rows dropped because they stayed unresolvable.
            synthetic_rows: Rows containing at least one fabricated value.

        Returns:
            Dictionary of fill statistics.
        """
        return {
            'original_rows': original_rows,
            'total_nan': total_nan,
            'per_column': per_column,
            'filled_forward': filled_forward,
            'filled_zero': filled_zero,
            'filled_backward': filled_backward,
            'dropped_rows': dropped_rows,
            'synthetic_rows': synthetic_rows,
            'synthetic_row_ratio': (synthetic_rows / original_rows) if original_rows else 0.0,
        }
