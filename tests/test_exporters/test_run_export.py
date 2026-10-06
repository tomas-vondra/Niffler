"""Exporting a run that is not a single backtest.

Three properties are pinned here. An exporter declares which kinds of run it
can export, and one that cannot is refused when it is created - before the
computation. The JSON exporter's file is the durable record, so it must be
valid JSON and must name its run. And every saved document gets its ``run`` and
``provenance`` blocks from one builder.
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from niffler.backtesting.backtest_result import BacktestResult
from niffler.exporters import (
    EXPORTER_CLASSES,
    BaseExporter,
    ConsoleExporter,
    CSVExporter,
    ExportError,
    ExporterManager,
    JsonExporter,
    get_exporters_supporting,
)
from niffler.utils.run_identity import RUN_KINDS, RunIdentity, new_run_identity


def record_for(kind='optimize', body=None, provenance=None, strategy_key='rsi',
               identity=None):
    return ExporterManager.create_run_record(
        identity or new_run_identity(kind), strategy_key,
        body if body is not None else {'results': []}, provenance=provenance)


class TempDirTestCase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name='out.json'):
        return os.path.join(self.temp_dir, name)

    def read(self, name='out.json'):
        with open(self.path(name), 'r') as handle:
            return json.load(handle)


class TestSupportedKinds(unittest.TestCase):

    def test_every_exporter_declares_only_real_kinds(self):
        for name, exporter_class in EXPORTER_CLASSES.items():
            with self.subTest(exporter=name):
                self.assertTrue(exporter_class.SUPPORTED_KINDS)
                self.assertLessEqual(set(exporter_class.SUPPORTED_KINDS), set(RUN_KINDS))

    def test_every_exporter_supports_a_backtest(self):
        self.assertEqual(get_exporters_supporting('backtest'), list(EXPORTER_CLASSES))

    def test_every_kind_has_an_exporter_that_keeps_a_durable_record(self):
        for kind in RUN_KINDS:
            with self.subTest(kind=kind):
                self.assertIn('json', get_exporters_supporting(kind))

    def test_the_default_export_run_refuses_rather_than_doing_nothing(self):
        """Returning normally would be reported as a successful export."""
        with self.assertRaises(ExportError) as ctx:
            CSVExporter(output_dir=tempfile.gettempdir()).export_run(record_for('optimize'))
        self.assertIn('optimize', str(ctx.exception))


class TestKindIsCheckedAtCreation(unittest.TestCase):

    def test_an_exporter_that_cannot_export_the_kind_is_rejected(self):
        manager = ExporterManager()
        with self.assertRaises(ValueError) as ctx:
            manager.create_exporters_from_list(['console', 'csv'], kind='optimize')
        message = str(ctx.exception)
        self.assertIn('csv', message)
        self.assertIn('optimize', message)
        # The message says what would work instead.
        self.assertIn('json', message)
        # Nothing was created: the run must not start half-configured.
        self.assertEqual(manager.get_exporter_count(), 0)

    def test_the_same_list_is_accepted_for_a_backtest(self):
        manager = ExporterManager()
        self.assertEqual(
            manager.create_exporters_from_list(['console', 'csv'], kind='backtest'), [])
        self.assertEqual(manager.get_exporter_count(), 2)

    def test_no_kind_keeps_the_old_behaviour(self):
        manager = ExporterManager()
        manager.create_exporters_from_list(['console', 'csv'])
        self.assertEqual(manager.get_exporter_count(), 2)


class TestCreateRunRecord(unittest.TestCase):

    def test_the_run_block_carries_identity_and_registry_key(self):
        identity = RunIdentity(run_id='r1', kind='optimize', experiment='exp',
                               parent_run_id=None, profile='prof')
        document = record_for(identity=identity, strategy_key='rsi').document
        self.assertEqual(document['run'], {
            'run_id': 'r1', 'kind': 'optimize', 'experiment': 'exp',
            'parent_run_id': None, 'profile': 'prof', 'strategy_key': 'rsi',
        })

    def test_provenance_is_attached_only_when_supplied(self):
        self.assertNotIn('provenance', record_for().document)
        provenance = {'code': {'git_sha': 'a' * 40}}
        self.assertEqual(record_for(provenance=provenance).document['provenance'],
                         provenance)

    def test_the_body_is_not_mutated(self):
        body = {'results': []}
        record_for(body=body)
        self.assertEqual(body, {'results': []})


class TestJsonExporter(TempDirTestCase):

    def test_the_document_is_written_with_its_run_block(self):
        record = record_for(body={'results': [{'parameters': {'rsi_period': 9}}]})
        with patch('sys.stdout', new=io.StringIO()):
            JsonExporter(output_path=self.path()).export_run(record)

        saved = self.read()
        self.assertEqual(saved['results'][0]['parameters'], {'rsi_period': 9})
        self.assertEqual(saved['run']['run_id'], record.identity.run_id)

    def test_non_finite_numbers_become_null(self):
        """A bare NaN literal is not JSON and no strict parser reads it back."""
        record = record_for(body={'sharpe': float('inf'), 'win_rate': float('nan')})
        with patch('sys.stdout', new=io.StringIO()):
            JsonExporter(output_path=self.path()).export_run(record)

        with open(self.path(), 'r') as handle:
            text = handle.read()
        self.assertNotIn('NaN', text)
        self.assertNotIn('Infinity', text)
        self.assertEqual(json.loads(text)['sharpe'], None)

    def test_the_default_name_identifies_the_run(self):
        record = record_for('walk_forward', strategy_key='breakout')
        exporter = JsonExporter()
        original = os.getcwd()
        os.chdir(self.temp_dir)
        try:
            with patch('sys.stdout', new=io.StringIO()):
                exporter.export_run(record)
        finally:
            os.chdir(original)

        expected = f"walk_forward_breakout_{record.identity.run_id[:8]}.json"
        self.assertEqual(str(exporter.last_path), expected)
        self.assertTrue(os.path.exists(self.path(expected)))

    def test_an_unwritable_path_raises_instead_of_reporting_success(self):
        exporter = JsonExporter(output_path=os.path.join(self.temp_dir, 'missing', 'x.json'))
        with self.assertRaises(OSError):
            exporter.export_run(record_for())
        self.assertIsNone(exporter.last_path)

    def test_a_backtest_is_exported_as_its_metadata_document(self):
        result = Mock(spec=BacktestResult)
        result.portfolio_values = pd.Series([1.0, 2.0])
        result.trades = []
        metadata = {'run_id': 'r1', 'kind': 'backtest', 'strategy_key': 'rsi',
                    'total_return_pct': 12.5}
        with patch('sys.stdout', new=io.StringIO()):
            JsonExporter(output_path=self.path()).export_backtest_result(
                result, 'r1', metadata)
        self.assertEqual(self.read(), metadata)


class TestManagerExportRun(TempDirTestCase):

    def test_every_exporter_receives_the_record(self):
        manager = ExporterManager()
        manager.create_exporters_from_list(
            ['console', 'json'], kind='optimize', output_path=self.path())
        record = record_for()

        with patch('sys.stdout', new=io.StringIO()) as stdout:
            summary = manager.export_run(record)

        self.assertTrue(summary.ok)
        self.assertEqual(summary.run_id, record.identity.run_id)
        self.assertEqual(summary.successes, ['ConsoleExporter', 'JsonExporter'])
        self.assertIn(record.identity.run_id, stdout.getvalue())
        self.assertEqual(self.read()['run']['run_id'], record.identity.run_id)

    def test_one_failing_exporter_does_not_stop_the_others(self):
        failing = Mock(spec=BaseExporter)
        failing.export_run.side_effect = OSError('disk full')
        failing.logger = Mock()
        manager = ExporterManager()
        manager.add_exporter(failing)
        manager.add_exporter(JsonExporter(output_path=self.path()))

        with patch('sys.stdout', new=io.StringIO()):
            summary = manager.export_run(record_for())

        self.assertFalse(summary.ok)
        self.assertEqual(summary.successes, ['JsonExporter'])
        self.assertEqual(summary.failures[0][1], 'disk full')
        self.assertTrue(os.path.exists(self.path()))

    def test_console_prints_the_identity(self):
        record = record_for(identity=RunIdentity(run_id='r1', kind='screen',
                                                 experiment='exp'))
        with patch('sys.stdout', new=io.StringIO()) as stdout:
            ConsoleExporter().export_run(record)
        self.assertIn('r1', stdout.getvalue())
        self.assertIn('experiment: exp', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
