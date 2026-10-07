"""Names that are written in more than one place must agree.

A string spelled twice fails silently when the copies drift: a filter matches
nothing, a field never maps, a dashboard panel comes up empty. Each test here
puts two such copies side by side - the two summary builders, the exporter and
the dashboard, the writer of a result file and its readers - so a rename on one
side fails here instead of in Grafana three weeks later.
"""

import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from niffler.backtesting.backtest_result import BacktestResult
from niffler.backtesting.run_config import RunConfig
from niffler.backtesting.trade import Trade, TradeSide
from niffler.exporters import ExporterManager, JsonExporter
from niffler.exporters.elasticsearch_exporter import ElasticsearchExporter
from niffler.exporters.run_record import DETAIL_TRIAL, HEADER_FIELDS
from niffler.optimization.base_optimizer import BaseOptimizer
from niffler.utils.provenance import (
    collect_provenance,
    is_provenance_record,
    provenance_fingerprint,
)
from niffler.utils.run_identity import RUN_KINDS, RunIdentity
from scripts import analyze, optimize
from scripts.common import read_params_file
from tests.test_scripts.test_flag_spelling import parser_of

MAPPINGS = REPO / 'config' / 'elasticsearch' / 'mappings'
DASHBOARDS = REPO / 'config' / 'grafana' / 'dashboards'

PROVENANCE = {
    'code': {'git_sha': 'a' * 40, 'dirty': False},
    'data': {'path': 'data/BTCUSDT_1d.csv', 'sha256': 'b' * 64},
    'environment': {'python_version': '3.13.0'},
}
IDENTITY = RunIdentity(run_id='run-1', kind='backtest', experiment='exp',
                       parent_run_id='opt-1', profile='prof')
SETTINGS = RunConfig().to_metadata()


def backtest_result():
    index = pd.date_range('2024-01-01', periods=3, freq='D')
    result = Mock(spec=BacktestResult)
    result.strategy_name = 'Simple MA Strategy'
    result.symbol = 'BTCUSDT'
    result.start_date = datetime(2024, 1, 1)
    result.end_date = datetime(2024, 1, 3)
    result.portfolio_values = pd.Series([10000.0, 10100.0, 10050.0], index=index)
    result.trades = [
        Trade(index[0], 'BTCUSDT', TradeSide.BUY, 100.0, 1.0, 100.0),
        Trade(index[2], 'BTCUSDT', TradeSide.SELL, 110.0, 1.0, 110.0),
    ]
    for name in ('final_capital', 'total_return', 'total_return_pct', 'max_drawdown',
                 'sharpe_ratio', 'win_rate', 'profit_factor', 'average_win',
                 'average_loss', 'largest_win', 'largest_loss'):
        setattr(result, name, 1.0)
    result.total_trades = 2
    result.num_winning_trades = 1
    result.num_losing_trades = 0
    return result


def backtest_metadata():
    return ExporterManager().create_metadata(
        backtest_result(), {'short_window': 10}, 'BTCUSDT', 10000.0, 0.001,
        provenance=PROVENANCE, cost_model=SETTINGS['cost_model'],
        risk_manager=SETTINGS['risk_manager'], identity=IDENTITY,
        strategy_key='simple_ma', settings=SETTINGS)


def mapping(name):
    with open(MAPPINGS / f'{name}.json', 'r') as handle:
        return json.load(handle)


class TestOneHeaderForEveryKindOfRun(unittest.TestCase):
    """A backtest summary and any other summary share an index, so they share a header."""

    def run_header(self):
        return ExporterManager.create_run_record(
            IDENTITY, 'simple_ma', {}, provenance=PROVENANCE, settings=SETTINGS,
            symbol='BTCUSDT').header

    def test_a_backtest_summary_carries_the_whole_run_header(self):
        header = self.run_header()
        metadata = backtest_metadata()
        self.assertLessEqual(set(header), set(metadata))
        for name, value in header.items():
            with self.subTest(field=name):
                self.assertEqual(metadata[name], value)

    def test_the_fingerprint_is_top_level_on_a_backtest_too(self):
        """A git_sha filter on the runs index used to miss every backtest."""
        metadata = backtest_metadata()
        self.assertEqual(metadata['git_sha'], 'a' * 40)
        self.assertIs(metadata['git_dirty'], False)
        self.assertEqual(metadata['data_sha256'], 'b' * 64)

    def test_the_header_names_every_declared_header_field(self):
        self.assertLessEqual(set(HEADER_FIELDS), set(self.run_header()))
        self.assertLessEqual(set(HEADER_FIELDS), set(backtest_metadata()))


class TestBacktestDetailDocuments(unittest.TestCase):
    """Equity points, trades and positions are part of the run like its summary is."""

    def setUp(self):
        patcher = patch.dict(os.environ, {
            'ELASTICSEARCH_INDEX_PREFIX': 'niffler', 'ELASTICSEARCH_API_KEY': '',
            'ELASTICSEARCH_USERNAME': '', 'ELASTICSEARCH_PASSWORD': ''})
        patcher.start()
        self.addCleanup(patcher.stop)

    def export(self):
        exporter = ElasticsearchExporter()
        exporter.es_client = Mock()
        exporter.es_client.indices.exists.return_value = True
        actions = []
        with patch.object(exporter, '_connect', return_value=True), \
                patch.object(exporter, '_bulk_index', side_effect=actions.extend):
            exporter.export_backtest_result(backtest_result(), 'run-1', backtest_metadata())
        return actions

    def test_every_detail_document_carries_the_header(self):
        actions = self.export()
        self.assertEqual({action['_index'] for action in actions},
                         {'niffler-portfolio-values', 'niffler-trades', 'niffler-positions'})
        metadata = backtest_metadata()
        for action in actions:
            for name in HEADER_FIELDS:
                with self.subTest(index=action['_index'], field=name):
                    self.assertEqual(action['_source'][name], metadata[name])

    def test_ids_are_deterministic_so_a_re_export_overwrites(self):
        first = [action['_id'] for action in self.export()]
        self.assertEqual(first, [action['_id'] for action in self.export()])
        self.assertEqual(len(set(first)), len(first))
        self.assertIn('run-1:portfolio:0', first)
        self.assertIn('run-1:trade:1', first)


class TestIndexNamesHaveOneSource(unittest.TestCase):

    def setUp(self):
        patcher = patch.dict(os.environ, {'ELASTICSEARCH_INDEX_PREFIX': 'niffler'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.catalog = ElasticsearchExporter().index_catalog()
        self.names = {index['name'] for index in self.catalog}

    def test_every_index_has_a_mapping_file_with_single_node_settings(self):
        for index in self.catalog:
            with self.subTest(index=index['name']):
                document = mapping(index['mapping'])
                self.assertIn('properties', document['mappings'])
                # One replica on one node leaves every index yellow forever.
                self.assertEqual(document['settings']['number_of_replicas'], 0)

    def test_every_index_maps_the_header_and_its_time_field(self):
        for index in self.catalog:
            properties = mapping(index['mapping'])['mappings']['properties']
            for name in HEADER_FIELDS + (index['time_field'],):
                with self.subTest(index=index['name'], field=name):
                    self.assertIn(name, properties)

    def test_kibana_gets_a_data_view_for_every_index(self):
        """The hand-kept list had lost the positions index."""
        sys.path.insert(0, str(REPO / 'visualization'))
        import setup_kibana

        views = setup_kibana.niffler_data_views()
        self.assertEqual({view['pattern'] for view in views}, self.names)
        by_pattern = {view['pattern']: view['time_field'] for view in views}
        for index in self.catalog:
            self.assertEqual(by_pattern[index['name']], index['time_field'])

    def test_kibana_follows_the_configured_prefix(self):
        sys.path.insert(0, str(REPO / 'visualization'))
        import setup_kibana

        with patch.dict(os.environ, {'ELASTICSEARCH_INDEX_PREFIX': 'research'}):
            patterns = {view['pattern'] for view in setup_kibana.niffler_data_views()}
        self.assertIn('research-runs', patterns)
        self.assertFalse(any(pattern.startswith('niffler-') for pattern in patterns))


class TestDashboardsNameRealThings(unittest.TestCase):
    """A dashboard is a copy of the schema by nature; this is what keeps it honest."""

    def setUp(self):
        patcher = patch.dict(os.environ, {'ELASTICSEARCH_INDEX_PREFIX': 'niffler'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.catalog = ElasticsearchExporter().index_catalog()
        self.dashboards = {
            path.name: path.read_text(encoding='utf-8')
            for path in sorted(DASHBOARDS.glob('*.json'))
        }
        self.assertTrue(self.dashboards)

    def test_every_index_a_panel_queries_is_one_the_exporter_writes(self):
        names = {index['name'] for index in self.catalog}
        for name, text in self.dashboards.items():
            queried = set(re.findall(r'_index:([A-Za-z0-9_-]+)', text))
            with self.subTest(dashboard=name):
                self.assertTrue(queried)
                self.assertLessEqual(queried, names)

    def test_every_kind_a_panel_filters_on_is_a_run_kind(self):
        for name, text in self.dashboards.items():
            with self.subTest(dashboard=name):
                self.assertLessEqual(set(re.findall(r'\bkind:([a-z_]+)', text)),
                                     set(RUN_KINDS))

    def test_every_field_a_query_filters_on_is_mapped(self):
        mapped = set()
        for index in self.catalog:
            mapped |= set(mapping(index['mapping'])['mappings']['properties'])
        for name, text in self.dashboards.items():
            filtered = set(re.findall(r'\bAND ([a-z_]+):', text))
            with self.subTest(dashboard=name):
                self.assertLessEqual(filtered, mapped)


class TestOptimizationFileRoundTrip(unittest.TestCase):
    """The real writer against its two readers, with nothing mocked in between."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def document(self):
        optimizer = Mock()
        optimizer.strategy_class = type('Strategy', (), {})
        optimizer.sort_by = 'total_return'
        optimizer.run_config = RunConfig()

        def trial(parameters, total_return_pct, total_trades):
            result = Mock()
            result.parameters = parameters
            backtest = result.backtest_result
            backtest.total_return = total_return_pct * 100
            backtest.total_return_pct = total_return_pct
            backtest.sharpe_ratio = 1.0
            backtest.max_drawdown = -5.0
            backtest.total_trades = total_trades
            backtest.win_rate = 50.0
            backtest.benchmark_return_pct = 3.0
            backtest.excess_return_pct = total_return_pct - 3.0
            backtest.round_trip_count = total_trades // 2
            backtest.p_value = 0.2
            return result

        return BaseOptimizer.results_document(optimizer, [
            trial({'short_window': 7, 'long_window': 33}, 12.0, 14),
            trial({'short_window': 9, 'long_window': 40}, 0.0, 0),
        ])

    def test_the_export_views_read_what_the_optimizer_wrote(self):
        args = Mock(method='grid', sort_by='total_return')
        summary, details = optimize.build_export_views(
            self.document(), args, 'exhaustive', False, dict(optimize._NO_PLATEAU_SUMMARY))

        self.assertEqual(summary['n_trials'], 2)
        self.assertEqual(summary['best_parameters'], {'short_window': 7, 'long_window': 33})
        self.assertEqual(summary['total_return_pct'], 12.0)
        rows = details[DETAIL_TRIAL]
        self.assertEqual([row['status'] for row in rows], ['ok', 'no_trades'])
        self.assertEqual(rows[0]['parameters'], {'short_window': 7, 'long_window': 33})

    def test_the_params_file_reader_reads_what_the_optimizer_wrote(self):
        identity = RunIdentity(run_id='opt-1', kind='optimize', experiment='exp')
        record = ExporterManager.create_run_record(identity, 'simple_ma', self.document())
        path = os.path.join(self.temp_dir, 'opt.json')
        with patch('sys.stdout', new=io.StringIO()):
            JsonExporter(output_path=path).export_run(record)

        parameters, parent = read_params_file(path)
        self.assertEqual(parameters, {'short_window': 7, 'long_window': 33})
        self.assertEqual((parent.run_id, parent.experiment), ('opt-1', 'exp'))

    def test_every_trial_metric_the_optimizer_writes_is_mapped(self):
        properties = mapping('trials')['mappings']['properties']
        for name in self.document()['results'][0]['metrics']:
            with self.subTest(metric=name):
                self.assertIn(name, properties)


class TestProvenanceHasOneReader(unittest.TestCase):

    def test_the_fingerprint_reads_a_real_record(self):
        """Hand-written fixtures would keep passing after the producer renamed a key."""
        record = collect_provenance(str(REPO / 'pyproject.toml'))
        fingerprint = provenance_fingerprint(record)

        self.assertTrue(is_provenance_record(record))
        self.assertEqual(set(fingerprint), {'git_sha', 'git_dirty', 'data_sha256'})
        self.assertEqual(fingerprint['git_sha'], record['code']['git_sha'])
        self.assertEqual(fingerprint['git_dirty'], record['code']['dirty'])
        self.assertEqual(fingerprint['data_sha256'], record['data']['sha256'])
        self.assertTrue(fingerprint['data_sha256'])

    def test_no_record_is_unknown_not_clean(self):
        self.assertEqual(provenance_fingerprint(None),
                         {'git_sha': None, 'git_dirty': None, 'data_sha256': None})

    def test_a_dict_of_records_is_not_a_record(self):
        self.assertFalse(is_provenance_record({'a.csv': PROVENANCE}))
        self.assertFalse(is_provenance_record(None))


class TestChoicesComeFromTheirSource(unittest.TestCase):

    def choices(self, module, dest):
        parser = parser_of(module)
        return next(action.choices for action in parser._actions if action.dest == dest)

    def test_optimize_metric_flags_offer_exactly_the_optimizer_metrics(self):
        metrics = list(BaseOptimizer.METRICS_CONFIG)
        self.assertEqual(self.choices(optimize, 'sort_by'), metrics)
        self.assertEqual(self.choices(optimize, 'plateau_metric'), metrics)

    def test_the_analysis_choices_are_run_kinds(self):
        """--analysis is used as the run's kind, so it may only offer kinds."""
        self.assertLessEqual(set(self.choices(analyze, 'analysis')), set(RUN_KINDS))


if __name__ == '__main__':
    unittest.main()
