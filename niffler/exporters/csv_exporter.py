"""
CSV Exporter

Exports backtest results to CSV files for analysis and external tools.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .base_exporter import BaseExporter
from ..utils.json_utils import safe_json_dump
from ..backtesting.backtest_result import BacktestResult

# Characters that are illegal in Windows filenames plus the POSIX path separator.
_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RUN = re.compile(r'\s+')
_UNDERSCORE_RUN = re.compile(r'_{2,}')

# Windows reserved device names (case-insensitive), which cannot be used as filenames.
_RESERVED_NAMES = frozenset(
    ['con', 'prn', 'aux', 'nul']
    + [f'com{i}' for i in range(1, 10)]
    + [f'lpt{i}' for i in range(1, 10)]
)

# Keeps generated filenames well below the typical 255 byte per-component limit.
_MAX_COMPONENT_LENGTH = 64


def sanitize_path_component(value: Any, fallback: str = "unknown") -> str:
    """
    Convert an arbitrary user-derived value into a filesystem-safe, readable slug.

    Path separators and characters that are illegal on Windows are replaced with
    underscores, whitespace runs are collapsed, and the result is trimmed so that it
    never resolves to a nested path, a reserved device name or an empty string.

    Args:
        value: Value to turn into a filename component (e.g. "BTC/USDT")
        fallback: Value returned when sanitisation leaves nothing usable

    Returns:
        A safe, human-readable filename component (e.g. "BTC_USDT")
    """
    text = str(value) if value is not None else ""

    text = _WHITESPACE_RUN.sub('_', text.strip())
    text = _ILLEGAL_FILENAME_CHARS.sub('_', text)
    text = _UNDERSCORE_RUN.sub('_', text)
    # Leading/trailing dots and spaces are stripped by Windows and break round-tripping.
    text = text.strip('._ ')

    if len(text) > _MAX_COMPONENT_LENGTH:
        text = text[:_MAX_COMPONENT_LENGTH].rstrip('._ ')

    if not text or text.lower() in _RESERVED_NAMES:
        return fallback

    return text


class CSVExporter(BaseExporter):
    """Exporter that saves backtest results to CSV files."""

    def __init__(self, output_dir: str = ".", config: Dict[str, Any] = None):
        """
        Initialize CSV exporter.

        Args:
            output_dir: Directory where CSV files will be saved
            config: Additional configuration options
        """
        super().__init__(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_backtest_result(self, result: BacktestResult, backtest_id: str,
                              metadata: Dict[str, Any]) -> None:
        """
        Export backtest results to CSV files.

        Args:
            result: BacktestResult object containing all backtest data
            backtest_id: Unique identifier for this backtest run
            metadata: Additional metadata about the backtest

        Raises:
            ExportError: If the result does not contain exportable data
            OSError: If the CSV files cannot be written
        """
        self.require_valid_result(result, "CSV")

        base_filename = self._generate_filename(result, backtest_id)

        try:
            # Export metadata
            self._export_metadata(metadata, backtest_id, base_filename)

            # Export portfolio values
            portfolio_file = self._export_portfolio_values(result, backtest_id, base_filename)

            # Export trades
            trades_file = self._export_trades(result, backtest_id, base_filename)

            self.logger.info(f"CSV export completed:")
            self.logger.info(f"  Portfolio values: {portfolio_file}")
            if trades_file:
                self.logger.info(f"  Trades: {trades_file}")

        except Exception as e:
            self.logger.error(f"Failed to export CSV files: {e}")
            raise

    def _generate_filename(self, result: BacktestResult, backtest_id: str) -> str:
        """
        Generate base filename for CSV files.

        Symbols such as "BTC/USDT" and strategy names such as "Simple MA Crossover" are
        slugified so they cannot introduce directory separators or characters that are
        illegal on the target filesystem.

        Args:
            result: BacktestResult providing symbol, strategy name and date range
            backtest_id: Unique identifier for this backtest run

        Returns:
            A safe base filename without extension
        """
        symbol = sanitize_path_component(getattr(result, 'symbol', None), fallback="unknown_symbol")
        strategy_name = sanitize_path_component(
            getattr(result, 'strategy_name', None), fallback="unknown_strategy"
        )
        start_date = result.start_date.strftime('%Y%m%d')
        end_date = result.end_date.strftime('%Y%m%d')
        short_id = sanitize_path_component(str(backtest_id)[:8], fallback="00000000")
        return f"{symbol}_{strategy_name}_{start_date}_{end_date}_{short_id}"

    def _export_metadata(self, metadata: Dict[str, Any], backtest_id: str, base_filename: str) -> str:
        """Export backtest metadata to JSON file."""
        metadata_file = self.output_dir / f"{base_filename}_metadata.json"

        # Add backtest_id to metadata
        metadata_with_id = {**metadata, 'backtest_id': backtest_id}

        # Non-finite metrics (inf/NaN) are written as null so the file stays valid JSON.
        with open(metadata_file, 'w') as f:
            safe_json_dump(metadata_with_id, f, indent=2, default=str)

        return str(metadata_file)

    def _export_portfolio_values(self, result: BacktestResult, backtest_id: str, base_filename: str) -> str:
        """Export portfolio values to CSV."""
        portfolio_file = self.output_dir / f"{base_filename}_portfolio.csv"

        # Create DataFrame with portfolio values
        portfolio_df = pd.DataFrame({
            'timestamp': result.portfolio_values.index,
            'portfolio_value': result.portfolio_values.values,
            'backtest_id': backtest_id
        })

        portfolio_df.to_csv(portfolio_file, index=False)
        return str(portfolio_file)

    def _export_trades(self, result: BacktestResult, backtest_id: str, base_filename: str) -> str:
        """Export trades to CSV."""
        if not result.trades:
            self.logger.info("No trades to export")
            return ""

        trades_file = self.output_dir / f"{base_filename}_trades.csv"

        # Create DataFrame with trade details
        trades_data = []
        for trade in result.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'price': trade.price,
                'quantity': trade.quantity,
                'value': trade.value,
                # Optional on the Trade dataclass - read defensively.
                'commission': getattr(trade, 'commission', 0.0),
                'backtest_id': backtest_id
            })

        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(trades_file, index=False)
        return str(trades_file)

    def set_output_directory(self, output_dir: str) -> None:
        """Set the output directory for CSV files."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
