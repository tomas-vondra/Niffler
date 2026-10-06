"""Exporting optimizations, analyses, comparisons and screens to Elasticsearch.

Elasticsearch has no joins, so what is pinned here is what a dashboard relies
on to stand in for one: every document of a run carries the same header, ids
are deterministic so a re-export overwrites, and a value that could not be
known is null rather than a plausible default.
"""

import json
import math
import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from niffler.exporters import ExportError, ExporterManager
from niffler.exporters.elasticsearch_exporter import ElasticsearchExporter, _document_safe
from niffler.exporters.run_record import (
    DETAIL_COMPARISON,
    DETAIL_FOLD,
    DETAIL_SIMULATION,
    DETAIL_TRIAL,
    DETAIL_TYPES,
)
from niffler.utils.run_identity import RUN_KINDS, RunIdentity

MAPPINGS = Path(__file__).resolve().parents[2] / 'config' / 'elasticsearch' / 'mappings'

PROVENANCE = {
    'run_timestamp_utc': '2026-10-06T10:00:00+00:00',
    'code': {'git_sha': 'a' * 40, 'dirty': True},
    'data': {'path': 'data/BTCUSDT_1d.csv', 'sha256': 'b' * 64},
    'environment': {'python_version': '3.13.0'},
}

SETTINGS = {'initial_capital': 10000.0, 'commission': 0.001, 'cost_model': 'none',
            'risk_manager': None}


def identity(kind='optimize'):
    return RunIdentity(run_id='run-1', kind=kind, experiment='breakout-btc',
                       parent_run_id=None, profile='breakout-btc')


def optimization_record(**overrides):
    arguments = {
        'provenance': PROVENANCE,
        'settings': SETTINGS,
        'symbol': 'BTCUSDT',
        'summary': {'n_trials': 2, 'best_parameters': {'entry_window': 20},
                    'grid_median': 4.2},
        'details': {DETAIL_TRIAL: [
            {'rank': 1, 'parameters': {'entry_window': 20}, 'total_return_pct': 12.0,
             'status': 'ok'},
            {'rank': 2, 'parameters': {'entry_window': 30}, 'total_return_pct': 3.0,
             'status': 'ok'},
        ]},
    }
    arguments.update(overrides)
    return ExporterManager.create_run_record(identity(), 'breakout', {'results': []},
                                             **arguments)


class TestRunHeader(unittest.TestCase):

    def test_the_header_carries_identity_settings_and_fingerprint(self):
        header = optimization_record().header
        self.assertEqual(header['run_id'], 'run-1')
        self.assertEqual(header['experiment'], 'breakout-btc')
        self.assertEqual(header['strategy_key'], 'breakout')
        self.assertEqual(header['symbol'], 'BTCUSDT')
        self.assertEqual(header['commission'], 0.001)
        self.assertEqual(header['git_sha'], 'a' * 40)
        self.assertIs(header['git_dirty'], True)
        self.assertEqual(header['data_sha256'], 'b' * 64)

    def test_an_undetermined_dirty_flag_stays_null(self):
        """False would assert a cleanliness nobody checked."""
        unknown = {**PROVENANCE, 'code': {'git_sha': None, 'dirty': None}}
        self.assertIsNone(optimization_record(provenance=unknown).header['git_dirty'])
        self.assertIsNone(optimization_record(provenance=None).header['git_dirty'])

    def test_a_multi_dataset_run_claims_no_single_data_fingerprint(self):
        per_file = {'a.csv': PROVENANCE, 'b.csv': PROVENANCE}
        record = optimization_record(provenance=per_file)
        self.assertIsNone(record.header['data_sha256'])
        # The code is the same for every file, so it is still recorded.
        self.assertEqual(record.header['git_sha'], 'a' * 40)
        # No single record describes the run, so none is attached to the summary.
        self.assertIsNone(record.provenance)
        self.assertEqual(record.document['provenance'], per_file)

    def test_an_unknown_detail_type_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            optimization_record(details={'trail': []})
        self.assertIn('trail', str(ctx.exception))


class TestElasticsearchExportRun(unittest.TestCase):

    def setUp(self):
        patcher = patch.dict(os.environ, {
            'ELASTICSEARCH_HOST': 'test-host', 'ELASTICSEARCH_PORT': '9200',
            'ELASTICSEARCH_INDEX_PREFIX': 'test-prefix', 'ELASTICSEARCH_SCHEME': 'http',
            'ELASTICSEARCH_API_KEY': '', 'ELASTICSEARCH_USERNAME': '',
            'ELASTICSEARCH_PASSWORD': '',
        })
        patcher.start()
        self.addCleanup(patcher.stop)

        self.exporter = ElasticsearchExporter()
        self.client = Mock()
        self.client.indices.exists.return_value = False
        self.exporter.es_client = self.client
        self.bulked = []

        for target, replacement in (
                ('_connect', lambda: True),
                ('_bulk_index', self.bulked.extend)):
            attribute = patch.object(self.exporter, target, side_effect=replacement)
            attribute.start()
            self.addCleanup(attribute.stop)

    def summary_call(self):
        return self.client.index.call_args.kwargs

    def test_every_kind_of_run_is_supported(self):
        self.assertEqual(ElasticsearchExporter.SUPPORTED_KINDS, RUN_KINDS)

    def test_the_summary_goes_to_the_runs_index_keyed_by_run_id(self):
        self.exporter.export_run(optimization_record())

        call = self.summary_call()
        self.assertEqual(call['index'], 'test-prefix-runs')
        self.assertEqual(call['id'], 'run-1')
        self.assertEqual(call['body']['kind'], 'optimize')
        self.assertEqual(call['body']['experiment'], 'breakout-btc')
        self.assertEqual(call['body']['grid_median'], 4.2)
        self.assertEqual(call['body']['provenance'], PROVENANCE)
        self.assertIn('created_at', call['body'])

    def test_each_trial_is_one_document_with_the_shared_header(self):
        self.exporter.export_run(optimization_record())

        self.assertEqual(len(self.bulked), 2)
        first = self.bulked[0]
        self.assertEqual(first['_index'], 'test-prefix-trials')
        source = first['_source']
        self.assertEqual(source['doc_type'], 'trial')
        self.assertEqual(source['row_index'], 0)
        self.assertEqual(source['parameters'], {'entry_window': 20})
        # The join a document store cannot do, done by copying.
        for field in ('run_id', 'experiment', 'strategy_key', 'symbol', 'git_sha',
                      'cost_model'):
            self.assertEqual(source[field], self.summary_call()['body'][field], field)

    def test_document_ids_are_deterministic_so_a_re_export_overwrites(self):
        self.exporter.export_run(optimization_record())
        first_ids = [action['_id'] for action in self.bulked]
        self.bulked.clear()
        self.exporter.export_run(optimization_record())

        self.assertEqual(first_ids, ['run-1:trial:0', 'run-1:trial:1'])
        self.assertEqual([action['_id'] for action in self.bulked], first_ids)

    def test_each_detail_type_has_its_own_index(self):
        expected = {
            DETAIL_TRIAL: 'test-prefix-trials',
            DETAIL_FOLD: 'test-prefix-folds',
            DETAIL_SIMULATION: 'test-prefix-simulations',
            DETAIL_COMPARISON: 'test-prefix-comparisons',
        }
        self.assertEqual(set(expected), set(DETAIL_TYPES))
        for detail_type, index_name in expected.items():
            with self.subTest(detail_type=detail_type):
                self.assertEqual(self.exporter.detail_index(detail_type), index_name)

    def test_indices_are_created_from_their_mapping_files(self):
        self.exporter.export_run(optimization_record())
        created = [call.kwargs['index'] for call in self.client.indices.create.call_args_list]
        self.assertEqual(created, ['test-prefix-runs', 'test-prefix-trials'])

    def test_a_run_with_no_detail_rows_writes_only_its_summary(self):
        self.exporter.export_run(optimization_record(details={DETAIL_TRIAL: []}))
        self.client.index.assert_called_once()
        self.assertEqual(self.bulked, [])

    def test_a_row_value_overrides_the_header(self):
        """A comparison row names its own strategy and its own data fingerprint."""
        record = ExporterManager.create_run_record(
            identity('compare'), None, {},
            provenance={'a.csv': PROVENANCE},
            details={DETAIL_COMPARISON: [
                {'symbol': 'AAA', 'strategy_key': 'rsi', 'data_sha256': 'c' * 64}]})
        self.exporter.export_run(record)

        source = self.bulked[0]['_source']
        self.assertEqual(source['strategy_key'], 'rsi')
        self.assertEqual(source['symbol'], 'AAA')
        self.assertEqual(source['data_sha256'], 'c' * 64)

    def test_non_finite_numbers_and_timestamps_are_made_indexable(self):
        record = optimization_record(details={DETAIL_FOLD: [{
            'test_start': pd.Timestamp('2024-01-01'),
            'efficiency_ratio': float('nan'),
            'train_return_pct': float('inf'),
        }]})
        self.exporter.export_run(record)

        source = self.bulked[0]['_source']
        self.assertEqual(source['test_start'], '2024-01-01T00:00:00')
        self.assertIsNone(source['efficiency_ratio'])
        self.assertIsNone(source['train_return_pct'])

    def test_an_unreachable_cluster_raises(self):
        with patch.object(self.exporter, '_connect', return_value=False):
            with self.assertRaises(ExportError):
                self.exporter.export_run(optimization_record())
        self.client.index.assert_not_called()


class TestDocumentSafe(unittest.TestCase):

    def test_nested_values_are_converted(self):
        safe = _document_safe({
            'when': datetime(2024, 5, 1, 12, 30),
            'nested': {'values': [1.5, float('nan')]},
            'missing': pd.NaT,
        })
        self.assertEqual(safe['when'], '2024-05-01T12:30:00')
        self.assertEqual(safe['nested']['values'][0], 1.5)
        self.assertIsNone(safe['nested']['values'][1])
        self.assertIsNone(safe['missing'])
        # Nothing non-finite survives, at any depth.
        self.assertFalse(any(isinstance(v, float) and not math.isfinite(v)
                             for v in safe['nested']['values'] if v is not None))


class TestMappingFiles(unittest.TestCase):

    def mapping(self, name):
        with open(MAPPINGS / f'{name}.json', 'r') as handle:
            return json.load(handle)['mappings']

    def test_every_detail_index_has_a_mapping_file(self):
        for detail_type, (_, mapping_name) in ElasticsearchExporter._DETAIL_INDICES.items():
            with self.subTest(detail_type=detail_type):
                self.assertIn('properties', self.mapping(mapping_name))

    def test_parameters_are_flattened_so_no_document_locks_a_type(self):
        """RSI's oversold is an int in the search space and a float as a default."""
        self.assertEqual(
            self.mapping('runs')['properties']['strategy_params']['type'], 'flattened')
        self.assertEqual(
            self.mapping('runs')['properties']['best_parameters']['type'], 'flattened')
        for name in ('trials', 'run_details'):
            with self.subTest(mapping=name):
                self.assertEqual(
                    self.mapping(name)['properties']['parameters']['type'], 'flattened')

    def test_unmapped_integers_become_doubles(self):
        """Otherwise the first document decides, and 10 then silently truncates 10.5."""
        for name in ('runs', 'trials', 'run_details'):
            with self.subTest(mapping=name):
                templates = {
                    key: value
                    for template in self.mapping(name)['dynamic_templates']
                    for key, value in template.items()
                }
                self.assertEqual(
                    templates['unmapped_integers_as_double']['mapping']['type'], 'double')

    def test_the_identity_fields_are_keywords_in_every_index(self):
        for name in ('runs', 'trials', 'run_details'):
            properties = self.mapping(name)['properties']
            for field in ('run_id', 'kind', 'experiment', 'parent_run_id', 'profile',
                          'strategy_key'):
                with self.subTest(mapping=name, field=field):
                    self.assertEqual(properties[field]['type'], 'keyword')


if __name__ == '__main__':
    unittest.main()
