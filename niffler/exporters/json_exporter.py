"""
JSON Exporter

Writes a run's result document to a JSON file.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from .base_exporter import BaseExporter
from .run_record import RunRecord
from ..backtesting.backtest_result import BacktestResult
from ..utils.json_utils import safe_json_dump
from ..utils.run_identity import RUN_KINDS


class JsonExporter(BaseExporter):
    """Exporter that writes one JSON document per run.

    This is the durable record of an optimization or an analysis, and the file a
    later ``--params-file`` step reads: its ``run`` block is what lets that step
    record this run as its parent.
    """

    SUPPORTED_KINDS = RUN_KINDS

    def __init__(self, output_path: Optional[str] = None, indent: int = 2):
        """
        Initialize the JSON exporter.

        Args:
            output_path: File to write. None derives a name from the run's kind,
                strategy and id, in the working directory
            indent: Indentation of the written JSON
        """
        super().__init__({'output_path': output_path, 'indent': indent})
        self.output_path = output_path
        self.indent = indent
        #: Path of the last file written, so a caller can tell the user where it went.
        self.last_path: Optional[Path] = None

    def export_run(self, record: RunRecord) -> None:
        """
        Write a run's document to disk.

        Args:
            record: The run to export

        Raises:
            OSError: If the file cannot be written
            TypeError: If the document cannot be serialised
        """
        self._write(record.document, self._path_for(
            record.identity.kind, record.strategy_key, record.identity.run_id))

    def export_backtest_result(self, result: BacktestResult, run_id: str,
                               metadata: Dict[str, Any]) -> None:
        """
        Write a backtest's metadata document to disk.

        The equity curve and the trades are not included: they are tabular, and
        the CSV exporter is the one that writes them.

        Args:
            result: BacktestResult object containing all backtest data
            run_id: Unique identifier for this backtest run
            metadata: The metadata document built by ExporterManager

        Raises:
            ExportError: If the result does not contain exportable data
            OSError: If the file cannot be written
        """
        self.require_valid_result(result, "JSON")
        self._write(metadata, self._path_for(
            metadata.get('kind') or 'backtest', metadata.get('strategy_key'), run_id))

    def _path_for(self, kind: str, strategy_key: Optional[str], run_id: str) -> Path:
        """Return the configured path, or a name derived from the run."""
        if self.output_path:
            return Path(self.output_path)
        return Path(f"{kind}_{strategy_key or 'run'}_{run_id[:8]}.json")

    def _write(self, document: Dict[str, Any], path: Path) -> None:
        """Serialise a document, sanitising inf/NaN to null."""
        try:
            with open(path, 'w') as handle:
                safe_json_dump(document, handle, indent=self.indent, default=str)
        except (OSError, TypeError, ValueError) as e:
            # Never swallow this: the caller reports the run as failed instead
            # of claiming success while no file was written.
            self.logger.error(f"Error saving results to {path}: {e}")
            raise
        self.last_path = path
        # Printed, not only logged: where the durable record went is the one
        # thing a user needs from this exporter.
        print(f"Full results saved to: {path}")
