"""
Unit tests for run provenance flowing through the export path.

The provenance record is collected once per run at the CLI boundary and threaded into
every exporter through the shared metadata dictionary. These tests pin that contract at
each hop: metadata construction, the manager's fan-out, the three exporters, and the
Elasticsearch mapping that has to be right *before* documents are indexed against it.
"""

import json
import shutil
import tempfile
import unittest
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from niffler.backtesting.backtest_result import BacktestResult
from niffler.backtesting.trade import Trade, TradeSide
from niffler.exporters.base_exporter import BaseExporter
from niffler.exporters.console_exporter import ConsoleExporter
from niffler.exporters.csv_exporter import CSVExporter
from niffler.exporters.exporter_manager import ExporterManager
from niffler.utils.json_utils import safe_json_dumps

# A representative record - the shape collect_provenance() produces, with fixed values
# so the tests do not depend on the git state of the machine running them.
SAMPLE_PROVENANCE = {
    'run_timestamp_utc': '2026-08-05T12:34:56+00:00',
    'code': {
        'git_sha': 'a' * 40,
        'git_sha_short': 'a' * 12,
        'branch': 'feat/provenance',
        'dirty': False,
        'niffler_version': '0.1.0',
    },
    'data': {
        'path': '/data/BTCUSDT_binance_1d.csv',
        'sha256': 'b' * 64,
        'size_bytes': 4096,
        'modified_utc': '2026-08-01T00:00:00+00:00',
    },
    'environment': {
        'python_version': '3.13.14',
        'platform': 'Windows-11',
        'packages': {'pandas': '2.3.1', 'numpy': '2.3.1', 'ccxt': None},
    },
}


def _make_result() -> Mock:
    """Build a BacktestResult stand-in with every field the metadata builders read."""
    result = Mock(spec=BacktestResult)
    result.strategy_name = "Simple MA Strategy"
    result.symbol = "BTC/USDT"
    result.start_date = datetime(2024, 1, 1)
    result.end_date = datetime(2024, 1, 3)
    result.initial_capital = 10000.0
    result.final_capital = 10200.0
    result.total_return = 200.0
    result.total_return_pct = 2.0
    result.max_drawdown = -1.5
    result.sharpe_ratio = 1.25
    result.win_rate = 100.0
    result.total_trades = 1
    result.profit_factor = 2.5
    result.average_win = 200.0
    result.average_loss = 0.0
    result.largest_win = 200.0
    result.largest_loss = 0.0
    result.num_winning_trades = 1
    result.num_losing_trades = 0
    # Transaction costs are always rendered, so they must be real numbers here.
    result.total_commission = 10.0
    result.total_slippage = 5.0
    # No benchmark and no significance verdict: these tests are about the
    # provenance line, and both blocks then take their "nothing to report" path
    # instead of needing a full set of comparison fields.
    result.benchmark_name = None
    result.benchmark_error = None
    result.significance_verdict = ''
    result.portfolio_values = pd.Series(
        [10000.0, 10100.0, 10200.0],
        index=[datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    )
    result.trades = [
        Trade(
            timestamp=datetime(2024, 1, 2),
            symbol="BTC/USDT",
            side=TradeSide.BUY,
            quantity=0.5,
            price=20000.0,
            value=10000.0
        )
    ]
    return result


class _RecordingExporter(BaseExporter):
    """Exporter that only remembers the metadata it was handed."""

    def __init__(self):
        super().__init__()
        self.metadata = None

    def export_backtest_result(self, result, backtest_id, metadata):
        self.metadata = metadata


class TestMetadataCarriesProvenance(unittest.TestCase):
    """The one metadata builder accepts and embeds a provenance record."""

    def setUp(self):
        self.result = _make_result()
        self.manager = ExporterManager()

    def test_exporters_do_not_build_metadata_themselves(self):
        """One builder, so no exporter can hand its sink a poorer document."""
        self.assertFalse(hasattr(BaseExporter, 'create_metadata'))

    def test_manager_metadata_includes_provenance(self):
        metadata = self.manager.create_metadata(
            self.result, {'short_window': 10}, 'BTC/USDT', 10000.0, 0.001,
            SAMPLE_PROVENANCE
        )

        self.assertEqual(metadata['provenance'], SAMPLE_PROVENANCE)

    def test_manager_metadata_omits_key_without_provenance(self):
        metadata = self.manager.create_metadata(
            self.result, {'short_window': 10}, 'BTC/USDT', 10000.0, 0.001
        )

        self.assertNotIn('provenance', metadata)

    def test_metadata_with_provenance_round_trips_through_safe_json_dumps(self):
        """allow_nan=False must not choke on the record the exporters carry."""
        metadata = self.manager.create_metadata(
            self.result, {'short_window': 10}, 'BTC/USDT', 10000.0, 0.001,
            SAMPLE_PROVENANCE
        )

        restored = json.loads(safe_json_dumps(metadata, default=str))

        self.assertEqual(restored['provenance']['code']['git_sha'], 'a' * 40)
        # An uninstalled package stays an explicit null rather than vanishing.
        self.assertIsNone(restored['provenance']['environment']['packages']['ccxt'])


class TestManagerSharesOneRecord(unittest.TestCase):
    """The record is collected once by the caller and shared by every exporter."""

    def test_every_exporter_receives_the_same_record(self):
        manager = ExporterManager()
        first, second = _RecordingExporter(), _RecordingExporter()
        manager.add_exporter(first)
        manager.add_exporter(second)

        summary = manager.export_backtest_result(
            result=_make_result(),
            strategy_params={'short_window': 10},
            symbol='BTC/USDT',
            initial_capital=10000.0,
            commission=0.001,
            provenance=SAMPLE_PROVENANCE
        )

        self.assertTrue(summary.ok)
        self.assertEqual(first.metadata['provenance'], SAMPLE_PROVENANCE)
        # Identity, not just equality: the data file is hashed once for the whole run.
        self.assertIs(first.metadata['provenance'], second.metadata['provenance'])


class TestConsoleExporterProvenanceLine(unittest.TestCase):
    """A short line, and a loud marker when the working tree was dirty."""

    def setUp(self):
        self.exporter = ConsoleExporter()
        self.result = _make_result()

    def _export(self, provenance):
        metadata = {'provenance': provenance} if provenance is not None else {}
        with patch('sys.stdout', new_callable=StringIO) as stdout:
            self.exporter.export_backtest_result(self.result, 'id-123', metadata)
        return stdout.getvalue()

    def test_prints_short_sha_branch_and_data_hash(self):
        output = self._export(SAMPLE_PROVENANCE)

        self.assertIn('Provenance:', output)
        self.assertIn('a' * 12, output)
        self.assertIn('feat/provenance', output)
        self.assertIn('b' * 12, output)
        # Concise: one line, not a dump of the whole record.
        self.assertNotIn('python_version', output)

    def test_dirty_run_is_marked(self):
        dirty = {**SAMPLE_PROVENANCE, 'code': {**SAMPLE_PROVENANCE['code'], 'dirty': True}}

        self.assertIn('DIRTY', self._export(dirty))

    def test_no_provenance_prints_no_line(self):
        output = self._export(None)

        self.assertNotIn('Provenance:', output)
        self.assertIn('BACKTEST RESULTS', output)


class TestCSVExporterProvenanceFile(unittest.TestCase):
    """A sidecar JSON file next to the CSVs, named with the shared sanitiser."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        self.exporter = CSVExporter(self.temp_dir)
        self.result = _make_result()

    def _files(self):
        return sorted(p.name for p in Path(self.temp_dir).iterdir())

    def test_writes_provenance_sidecar(self):
        self.exporter.export_backtest_result(
            self.result, 'abcdef12-3456', {'provenance': SAMPLE_PROVENANCE}
        )

        provenance_files = [name for name in self._files() if name.endswith('_provenance.json')]
        self.assertEqual(len(provenance_files), 1)

        written = json.loads((Path(self.temp_dir) / provenance_files[0]).read_text())
        self.assertEqual(written['code']['git_sha'], 'a' * 40)
        self.assertEqual(written['backtest_id'], 'abcdef12-3456')

    def test_sidecar_filename_is_sanitised(self):
        """BTC/USDT must not become a directory separator - reuse the audit's sanitiser."""
        self.exporter.export_backtest_result(
            self.result, 'abcdef12-3456', {'provenance': SAMPLE_PROVENANCE}
        )

        provenance_files = [name for name in self._files() if name.endswith('_provenance.json')]
        self.assertTrue(provenance_files[0].startswith('BTC_USDT_'))
        # Nothing escaped into a nested directory.
        self.assertEqual([p.is_file() for p in Path(self.temp_dir).iterdir()], [True] * 4)

    def test_no_sidecar_without_provenance(self):
        self.exporter.export_backtest_result(self.result, 'abcdef12-3456', {})

        self.assertFalse(any(name.endswith('_provenance.json') for name in self._files()))


class TestElasticsearchBacktestsMapping(unittest.TestCase):
    """
    The mapping is the one thing that cannot be fixed after the fact.

    Once documents are indexed, a field mapped as ``text`` (analysed, not aggregatable)
    or a SHA mapped as anything but ``keyword`` requires a reindex to correct. These
    assertions are cheap insurance against that.
    """

    @classmethod
    def setUpClass(cls):
        mapping_file = (
            Path(__file__).resolve().parents[2]
            / 'config' / 'elasticsearch' / 'mappings' / 'backtests.json'
        )
        cls.mapping = json.loads(mapping_file.read_text())
        cls.provenance = cls.mapping['mappings']['properties']['provenance']['properties']

    def test_shas_versions_and_paths_are_keywords(self):
        code = self.provenance['code']['properties']
        data = self.provenance['data']['properties']
        environment = self.provenance['environment']['properties']

        for field in (code['git_sha'], code['git_sha_short'], code['branch'],
                      code['niffler_version'], data['path'], data['sha256'],
                      environment['python_version'], environment['platform']):
            self.assertEqual(field['type'], 'keyword')

    def test_dirty_is_boolean(self):
        self.assertEqual(self.provenance['code']['properties']['dirty']['type'], 'boolean')

    def test_timestamps_are_dates(self):
        self.assertEqual(self.provenance['run_timestamp_utc']['type'], 'date')
        self.assertEqual(self.provenance['data']['properties']['modified_utc']['type'], 'date')

    def test_size_is_a_long(self):
        self.assertEqual(self.provenance['data']['properties']['size_bytes']['type'], 'long')

    def test_package_versions_are_keywords(self):
        packages = self.provenance['environment']['properties']['packages']['properties']

        self.assertIn('pandas', packages)
        for field in packages.values():
            self.assertEqual(field['type'], 'keyword')

    def test_new_packages_map_to_keyword_dynamically(self):
        """A package added to TRACKED_PACKAGES later must not land as analysed text."""
        templates = self.mapping['mappings']['dynamic_templates']
        matching = [
            template for entry in templates for template in entry.values()
            if template['path_match'] == 'provenance.environment.packages.*'
        ]

        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]['mapping']['type'], 'keyword')

    def test_mapping_covers_every_field_the_collector_emits(self):
        """The mapping and the collector must not drift apart."""
        from niffler.utils.provenance import TRACKED_PACKAGES

        self.assertEqual(
            set(self.provenance['code']['properties']),
            set(SAMPLE_PROVENANCE['code'])
        )
        self.assertEqual(
            set(self.provenance['data']['properties']),
            set(SAMPLE_PROVENANCE['data'])
        )
        self.assertEqual(
            set(self.provenance['environment']['properties']['packages']['properties']),
            set(TRACKED_PACKAGES)
        )


if __name__ == '__main__':
    unittest.main()
