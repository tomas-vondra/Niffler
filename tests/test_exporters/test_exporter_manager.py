"""
Unit tests for ExporterManager.
"""

import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
from datetime import datetime
import pandas as pd

from niffler.exporters.exporter_manager import ExporterManager, ExportSummary
from niffler.exporters.base_exporter import BaseExporter
from niffler.exporters.console_exporter import ConsoleExporter
from niffler.exporters.csv_exporter import CSVExporter
from niffler.exporters.elasticsearch_exporter import ElasticsearchExporter
from niffler.exporters.registry import EXPORTER_CLASSES
from tests.test_exporters.test_registry import _ProbeExporter
from niffler.backtesting.backtest_result import BacktestResult


class TestExporterManager(unittest.TestCase):
    """Test cases for ExporterManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ExporterManager()
        
        # Create mock BacktestResult
        self.mock_result = Mock(spec=BacktestResult)
        self.mock_result.strategy_name = "Simple MA Strategy"
        self.mock_result.start_date = datetime(2024, 1, 1)
        self.mock_result.end_date = datetime(2024, 12, 31)
        self.mock_result.final_capital = 11500.0
        self.mock_result.total_return = 1500.0
        self.mock_result.total_return_pct = 15.0
        self.mock_result.max_drawdown = -5.2
        self.mock_result.sharpe_ratio = 1.234
        self.mock_result.win_rate = 65.5
        self.mock_result.total_trades = 25
        self.mock_result.profit_factor = 2.5
        self.mock_result.average_win = 150.0
        self.mock_result.average_loss = 80.0
        self.mock_result.largest_win = 500.0
        self.mock_result.largest_loss = 200.0
        self.mock_result.num_winning_trades = 16
        self.mock_result.num_losing_trades = 9
        self.mock_result.total_commission = 30.0
        self.mock_result.total_slippage = 12.0
        self.mock_result.benchmark_name = "buy_and_hold"
        self.mock_result.benchmark_return_pct = 20.0
        self.mock_result.benchmark_sharpe_ratio = 0.9
        self.mock_result.benchmark_max_drawdown = -8.0
        self.mock_result.benchmark_total_cost = 10.0
        self.mock_result.benchmark_error = None
        self.mock_result.excess_return_pct = -5.0
        self.mock_result.information_ratio = -0.25
        self.mock_result.round_trip_count = 12
        self.mock_result.mean_trade_return_pct = 0.4
        self.mock_result.t_statistic = 1.1
        self.mock_result.p_value = 0.29
        self.mock_result.sharpe_ci_low = -0.3
        self.mock_result.sharpe_ci_high = 2.1
        self.mock_result.sharpe_ci_confidence = 0.95
        self.mock_result.significance_min_trades = 30
        self.mock_result.is_sample_sufficient = False
        self.mock_result.is_significant = None
        self.mock_result.significance_verdict = "SAMPLE TOO SMALL: 12 round trip(s)."
    
    def test_init(self):
        """Test ExporterManager initialization."""
        manager = ExporterManager()
        self.assertEqual(len(manager.exporters), 0)
        self.assertEqual(manager.creation_failures, [])

    def test_the_manager_holds_no_second_name_to_class_map(self):
        """The registry is the only one; a manager-local copy could shadow it."""
        self.assertFalse(hasattr(ExporterManager, 'EXPORTER_TYPES'))

    def test_exporter_types_registry(self):
        """Test that the registry contains the correct mappings."""
        expected_types = {
            'console': ConsoleExporter,
            'csv': CSVExporter,
            'elasticsearch': ElasticsearchExporter
        }
        self.assertEqual(EXPORTER_CLASSES, expected_types)

    def test_get_available_exporter_names(self):
        """Test getting available exporter names."""
        names = ExporterManager.get_available_exporter_names()
        expected_names = ['console', 'csv', 'elasticsearch']
        self.assertEqual(sorted(names), sorted(expected_names))
    
    def test_add_exporter(self):
        """Test adding an exporter."""
        mock_exporter = Mock(spec=BaseExporter)
        
        self.manager.add_exporter(mock_exporter)
        
        self.assertEqual(len(self.manager.exporters), 1)
        self.assertEqual(self.manager.exporters[0], mock_exporter)
    
    def test_create_exporter_by_name_console(self):
        """Test creating console exporter by name."""
        exporter = self.manager.create_exporter_by_name('console')
        
        self.assertIsInstance(exporter, ConsoleExporter)
        self.assertEqual(len(self.manager.exporters), 1)
        self.assertEqual(self.manager.exporters[0], exporter)
    
    def test_create_exporter_by_name_csv(self):
        """Test creating CSV exporter by name."""
        output_dir = '/tmp/test'
        exporter = self.manager.create_exporter_by_name('csv', output_dir=output_dir)
        
        self.assertIsInstance(exporter, CSVExporter)
        self.assertEqual(len(self.manager.exporters), 1)
        self.assertEqual(self.manager.exporters[0], exporter)
    
    @patch.dict('os.environ', {
        'ELASTICSEARCH_HOST': 'test-host',
        'ELASTICSEARCH_PORT': '9200',
        'ELASTICSEARCH_INDEX_PREFIX': 'test-prefix'
    })
    def test_create_exporter_by_name_elasticsearch(self):
        """Test creating Elasticsearch exporter by name."""
        exporter = self.manager.create_exporter_by_name('elasticsearch')
        
        self.assertIsInstance(exporter, ElasticsearchExporter)
        self.assertEqual(len(self.manager.exporters), 1)
        self.assertEqual(self.manager.exporters[0], exporter)
    
    def test_create_exporter_by_name_elasticsearch_with_params(self):
        """Test creating Elasticsearch exporter with custom parameters."""
        exporter = self.manager.create_exporter_by_name(
            'elasticsearch',
            host='custom-host',
            port=9300,
            index_prefix='custom-prefix'
        )
        
        self.assertIsInstance(exporter, ElasticsearchExporter)
        self.assertEqual(exporter.host, 'custom-host')
        self.assertEqual(exporter.port, 9300)
        self.assertEqual(exporter.index_prefix, 'custom-prefix')
    
    def test_create_exporter_by_name_case_insensitive(self):
        """Test creating exporter with different case."""
        exporter1 = self.manager.create_exporter_by_name('CONSOLE')
        exporter2 = self.manager.create_exporter_by_name('Console')
        exporter3 = self.manager.create_exporter_by_name('  csv  ')
        
        self.assertIsInstance(exporter1, ConsoleExporter)
        self.assertIsInstance(exporter2, ConsoleExporter)
        self.assertIsInstance(exporter3, CSVExporter)
    
    def test_create_exporter_by_name_unknown_type(self):
        """Test creating exporter with unknown type."""
        with self.assertRaises(ValueError) as context:
            self.manager.create_exporter_by_name('unknown_type')
        
        error_message = str(context.exception)
        self.assertIn('Unknown exporter type: unknown_type', error_message)
        self.assertIn('Available types:', error_message)
    
    def test_create_exporters_from_list(self):
        """Test creating multiple exporters from list."""
        exporter_names = ['console', 'csv']
        
        self.manager.create_exporters_from_list(exporter_names, output_dir='/tmp/test')
        
        self.assertEqual(len(self.manager.exporters), 2)
        self.assertIsInstance(self.manager.exporters[0], ConsoleExporter)
        self.assertIsInstance(self.manager.exporters[1], CSVExporter)
    
    def test_create_exporters_from_list_with_unknown(self):
        """An unknown exporter is recorded as a failure, not silently skipped."""
        exporter_names = ['console', 'unknown', 'csv']

        with patch('niffler.exporters.exporter_manager.logger') as mock_logger:
            failures = self.manager.create_exporters_from_list(
                exporter_names, output_dir='/tmp/test'
            )

        # Should create console and csv exporters, skip unknown
        self.assertEqual(len(self.manager.exporters), 2)
        self.assertIsInstance(self.manager.exporters[0], ConsoleExporter)
        self.assertIsInstance(self.manager.exporters[1], CSVExporter)

        # The unknown exporter must be reported, not merely printed and forgotten
        self.assertEqual([name for name, _ in failures], ['unknown'])
        self.assertEqual(self.manager.creation_failures, failures)
        mock_logger.error.assert_called_once()
        self.assertIn('unknown', str(mock_logger.error.call_args))
    
    def test_export_backtest_result(self):
        """Test exporting backtest result with multiple exporters."""
        # Add mock exporters
        mock_exporter1 = Mock(spec=BaseExporter)
        mock_exporter2 = Mock(spec=BaseExporter)
        self.manager.add_exporter(mock_exporter1)
        self.manager.add_exporter(mock_exporter2)
        
        strategy_params = {'param1': 'value1'}
        symbol = 'BTC-USD'
        initial_capital = 10000.0
        commission = 0.001
        
        summary = self.manager.export_backtest_result(
            self.mock_result, strategy_params, symbol, initial_capital, commission
        )

        # Check that a summary with a generated backtest_id is returned
        self.assertIsInstance(summary, ExportSummary)
        backtest_id = summary.backtest_id
        self.assertIsInstance(backtest_id, str)
        self.assertEqual(len(backtest_id), 36)  # UUID length
        self.assertTrue(summary.ok)
        self.assertEqual(summary.failures, [])
        self.assertEqual(len(summary.successes), 2)

        # Check that both exporters were called
        mock_exporter1.export_backtest_result.assert_called_once()
        mock_exporter2.export_backtest_result.assert_called_once()
        
        # Check the arguments passed to exporters
        args1 = mock_exporter1.export_backtest_result.call_args
        args2 = mock_exporter2.export_backtest_result.call_args
        
        self.assertEqual(args1[0][0], self.mock_result)  # result
        self.assertEqual(args1[0][1], backtest_id)      # backtest_id
        self.assertIsInstance(args1[0][2], dict)        # metadata
        
        self.assertEqual(args2[0][0], self.mock_result)
        self.assertEqual(args2[0][1], backtest_id)
        self.assertIsInstance(args2[0][2], dict)
    
    def test_export_backtest_result_with_custom_id(self):
        """Test exporting with custom backtest ID."""
        mock_exporter = Mock(spec=BaseExporter)
        self.manager.add_exporter(mock_exporter)
        
        custom_id = 'custom-backtest-id'
        strategy_params = {'param1': 'value1'}
        symbol = 'BTC-USD'
        initial_capital = 10000.0
        commission = 0.001
        
        summary = self.manager.export_backtest_result(
            self.mock_result, strategy_params, symbol, initial_capital, commission,
            backtest_id=custom_id
        )

        self.assertEqual(summary.backtest_id, custom_id)
        self.assertTrue(summary.ok)


        # Check that exporter was called with custom ID
        args = mock_exporter.export_backtest_result.call_args
        self.assertEqual(args[0][1], custom_id)
    
    def test_export_backtest_result_exporter_error(self):
        """Test export when one exporter fails."""
        # Add mock exporters - one fails, one succeeds
        mock_exporter1 = Mock(spec=BaseExporter)
        mock_exporter1.export_backtest_result.side_effect = Exception("Export failed")
        mock_exporter1.logger = Mock()

        mock_exporter2 = Mock(spec=BaseExporter)

        self.manager.add_exporter(mock_exporter1)
        self.manager.add_exporter(mock_exporter2)

        strategy_params = {'param1': 'value1'}
        symbol = 'BTC-USD'
        initial_capital = 10000.0
        commission = 0.001

        # Should not raise exception, but continue with other exporters
        summary = self.manager.export_backtest_result(
            self.mock_result, strategy_params, symbol, initial_capital, commission
        )

        # Both exporters should have been called
        mock_exporter1.export_backtest_result.assert_called_once()
        mock_exporter2.export_backtest_result.assert_called_once()

        # Error should have been logged
        mock_exporter1.logger.error.assert_called_once()

        # The failure must be visible to the caller, not silently swallowed
        self.assertFalse(summary.ok)
        self.assertEqual(summary.failures, [('BaseExporter', 'Export failed')])
        self.assertEqual(summary.successes, ['BaseExporter'])

    def test_export_backtest_result_summary_reports_exporter_names(self):
        """Summary uses the exporter class names for successes and failures."""
        class FailingExporter(BaseExporter):
            def export_backtest_result(self, result, backtest_id, metadata):
                raise RuntimeError("disk full")

        class WorkingExporter(BaseExporter):
            def export_backtest_result(self, result, backtest_id, metadata):
                return None

        self.manager.add_exporter(WorkingExporter())
        self.manager.add_exporter(FailingExporter())

        summary = self.manager.export_backtest_result(
            self.mock_result, {}, 'BTC-USD', 10000.0, 0.001
        )

        self.assertEqual(summary.successes, ['WorkingExporter'])
        self.assertEqual(summary.failures, [('FailingExporter', 'disk full')])
        self.assertFalse(summary.ok)

    def test_export_backtest_result_no_exporters(self):
        """An empty exporter list still yields a successful, empty summary."""
        summary = self.manager.export_backtest_result(
            self.mock_result, {}, 'BTC-USD', 10000.0, 0.001
        )

        self.assertTrue(summary.ok)
        self.assertEqual(summary.successes, [])
        self.assertEqual(summary.failures, [])

    def test_export_backtest_result_exporter_without_logger(self):
        """A failing exporter that exposes no logger falls back to the module logger."""
        broken_exporter = Mock(spec=BaseExporter)
        broken_exporter.export_backtest_result.side_effect = Exception("boom")
        self.manager.add_exporter(broken_exporter)

        summary = self.manager.export_backtest_result(
            self.mock_result, {}, 'BTC-USD', 10000.0, 0.001
        )

        self.assertFalse(summary.ok)
        self.assertEqual(summary.failures, [('BaseExporter', 'boom')])

    def test_unreachable_elasticsearch_is_reported_as_a_failure(self):
        """A cluster that cannot be reached must not be reported as a successful export.

        Regression test: the exporter used to log "Cannot connect ... skipping export"
        and return normally, so the manager counted it as a success and the CLI exited 0
        while nothing had been indexed.
        """
        self.mock_result.portfolio_values = pd.Series(
            [10000.0, 10100.0],
            index=[datetime(2024, 1, 1), datetime(2024, 1, 2)]
        )
        self.mock_result.trades = []

        exporter = ElasticsearchExporter(host='127.0.0.1', port=9)
        self.manager.add_exporter(exporter)

        with patch.object(ElasticsearchExporter, '_connect', return_value=False):
            summary = self.manager.export_backtest_result(
                self.mock_result, {}, 'BTC-USD', 10000.0, 0.001
            )

        self.assertFalse(summary.ok)
        self.assertEqual(summary.successes, [])
        self.assertEqual(len(summary.failures), 1)
        name, error = summary.failures[0]
        self.assertEqual(name, 'ElasticsearchExporter')
        self.assertIn('http://127.0.0.1:9', error)

    def test_export_summary_ok_property(self):
        """ExportSummary.ok is driven purely by the failure list."""
        self.assertTrue(ExportSummary(successes=['A'], failures=[], backtest_id='id').ok)
        self.assertFalse(
            ExportSummary(successes=[], failures=[('A', 'err')], backtest_id='id').ok
        )

    def test_create_exporter_by_name_elasticsearch_auth_params(self):
        """Authentication and TLS options are forwarded to the Elasticsearch exporter."""
        exporter = self.manager.create_exporter_by_name(
            'elasticsearch',
            host='secure-host',
            scheme='https',
            api_key='secret-key',
            timeout=5,
            verify_certs=False
        )

        self.assertEqual(exporter.scheme, 'https')
        self.assertEqual(exporter.url, 'https://secure-host:9200')
        self.assertEqual(exporter.timeout, 5)
        self.assertFalse(exporter.verify_certs)
        self.assertEqual(exporter._auth_mode(), 'api_key')


    def test_generate_backtest_id(self):
        """Test backtest ID generation."""
        id1 = self.manager._generate_backtest_id()
        id2 = self.manager._generate_backtest_id()
        
        # IDs should be strings and unique
        self.assertIsInstance(id1, str)
        self.assertIsInstance(id2, str)
        self.assertNotEqual(id1, id2)
        
        # IDs should be valid UUIDs
        self.assertEqual(len(id1), 36)
        self.assertEqual(len(id2), 36)
    
    def test_create_metadata(self):
        """Test metadata creation."""
        strategy_params = {'param1': 'value1'}
        symbol = 'BTC-USD'
        initial_capital = 10000.0
        commission = 0.001
        
        metadata = self.manager.create_metadata(
            self.mock_result, strategy_params, symbol, initial_capital, commission
        )
        
        expected_metadata = {
            'cost_model': None,
            'risk_manager': None,
            'total_commission': 30.0,
            'total_slippage': 12.0,
            'strategy_name': 'Simple MA Strategy',
            'strategy_params': strategy_params,
            'symbol': symbol,
            'start_date': '2024-01-01T00:00:00',
            'end_date': '2024-12-31T00:00:00',
            'initial_capital': initial_capital,
            'final_capital': 11500.0,
            'commission': commission,
            'total_return': 1500.0,
            'total_return_pct': 15.0,
            'max_drawdown': -5.2,
            'sharpe_ratio': 1.234,
            'win_rate': 65.5,
            'total_trades': 25,
            'profit_factor': 2.5,
            'average_win': 150.0,
            'average_loss': 80.0,
            'largest_win': 500.0,
            'largest_loss': 200.0,
            'num_winning_trades': 16,
            'num_losing_trades': 9,
            'benchmark_name': 'buy_and_hold',
            'benchmark_return_pct': 20.0,
            'benchmark_sharpe_ratio': 0.9,
            'benchmark_max_drawdown': -8.0,
            'benchmark_total_cost': 10.0,
            'benchmark_error': None,
            'excess_return_pct': -5.0,
            'information_ratio': -0.25,
            'round_trip_count': 12,
            'mean_trade_return_pct': 0.4,
            't_statistic': 1.1,
            'p_value': 0.29,
            'sharpe_ci_low': -0.3,
            'sharpe_ci_high': 2.1,
            'sharpe_ci_confidence': 0.95,
            'significance_min_trades': 30,
            'is_sample_sufficient': False,
            # None, not False: the sample was below the gate, so no verdict was
            # rendered. Exporting False here would claim a negative result.
            'is_significant': None,
            'significance_verdict': 'SAMPLE TOO SMALL: 12 round trip(s).'
        }

        self.assertEqual(metadata, expected_metadata)

    def test_the_document_handed_to_exporters_is_the_complete_one(self):
        """The deleted BaseExporter.create_metadata built a far poorer document.

        It carried neither profit_factor, nor the six trade statistics, nor any
        benchmark or significance field - a new exporter that called it would have
        indexed a silently incomplete record.
        """
        recorder = Mock(spec=BaseExporter)
        self.manager.add_exporter(recorder)

        self.manager.export_backtest_result(
            self.mock_result, {}, 'BTC-USD', 10000.0, 0.001
        )

        metadata = recorder.export_backtest_result.call_args[0][2]
        self.assertEqual(
            set(metadata),
            set(self.manager.create_metadata(
                self.mock_result, {}, 'BTC-USD', 10000.0, 0.001
            ))
        )
        for field in ('profit_factor', 'average_win', 'average_loss', 'largest_win',
                      'largest_loss', 'num_winning_trades', 'num_losing_trades',
                      'benchmark_name', 'benchmark_return_pct', 'benchmark_sharpe_ratio',
                      'benchmark_max_drawdown', 'benchmark_total_cost', 'benchmark_error',
                      'excess_return_pct', 'information_ratio', 'round_trip_count',
                      'mean_trade_return_pct', 't_statistic', 'p_value', 'sharpe_ci_low',
                      'sharpe_ci_high', 'sharpe_ci_confidence', 'significance_min_trades',
                      'is_sample_sufficient', 'is_significant', 'significance_verdict'):
            with self.subTest(field=field):
                self.assertIn(field, metadata)

    def test_get_exporter_count(self):
        """Test getting exporter count."""
        self.assertEqual(self.manager.get_exporter_count(), 0)
        
        self.manager.create_exporter_by_name('console')
        self.assertEqual(self.manager.get_exporter_count(), 1)
        
        self.manager.create_exporter_by_name('csv')
        self.assertEqual(self.manager.get_exporter_count(), 2)
    
    def test_clear_exporters(self):
        """Test clearing all exporters."""
        self.manager.create_exporter_by_name('console')
        self.manager.create_exporter_by_name('csv')
        self.assertEqual(self.manager.get_exporter_count(), 2)
        
        self.manager.clear_exporters()
        self.assertEqual(self.manager.get_exporter_count(), 0)
    
    def test_get_exporter_names(self):
        """Test getting names of configured exporters."""
        self.assertEqual(self.manager.get_exporter_names(), [])
        
        self.manager.create_exporter_by_name('console')
        self.manager.create_exporter_by_name('csv')
        
        names = self.manager.get_exporter_names()
        expected_names = ['ConsoleExporter', 'CSVExporter']
        self.assertEqual(sorted(names), sorted(expected_names))


class TestExporterCreationFailuresAreReported(unittest.TestCase):
    """An exporter that never got built must not be reported as a success."""

    def setUp(self):
        self.manager = ExporterManager()

        self.result = Mock(spec=BacktestResult)
        self.result.strategy_name = "Simple MA Strategy"
        self.result.start_date = datetime(2024, 1, 1)
        self.result.end_date = datetime(2024, 12, 31)
        self.result.final_capital = 11500.0
        self.result.total_return = 1500.0
        self.result.total_return_pct = 15.0
        self.result.max_drawdown = -5.0
        self.result.sharpe_ratio = 1.2
        self.result.win_rate = 60.0
        self.result.total_trades = 10
        self.result.profit_factor = 1.5
        self.result.average_win = 250.0
        self.result.average_loss = 100.0
        self.result.largest_win = 500.0
        self.result.largest_loss = 200.0
        self.result.num_winning_trades = 6
        self.result.num_losing_trades = 4
        self.result.total_commission = 10.0
        self.result.total_slippage = 4.0
        self.result.benchmark_name = "buy_and_hold"
        self.result.benchmark_return_pct = 20.0
        self.result.benchmark_sharpe_ratio = 0.9
        self.result.benchmark_max_drawdown = -8.0
        self.result.benchmark_total_cost = 10.0
        self.result.benchmark_error = None
        self.result.excess_return_pct = -5.0
        self.result.information_ratio = -0.25
        self.result.round_trip_count = 4
        self.result.mean_trade_return_pct = 0.4
        self.result.t_statistic = 1.1
        self.result.p_value = 0.29
        self.result.sharpe_ci_low = None
        self.result.sharpe_ci_high = None
        self.result.sharpe_ci_confidence = 0.95
        self.result.significance_min_trades = 30
        self.result.is_sample_sufficient = False
        self.result.is_significant = None
        self.result.significance_verdict = "SAMPLE TOO SMALL: 4 round trip(s)."
        self.result.symbol = 'BTC-USD'
        self.result.initial_capital = 10000.0
        self.result.trades = []
        self.result.portfolio_values = pd.Series(
            [10000.0, 10500.0, 11500.0],
            index=pd.date_range('2024-01-01', periods=3, freq='D')
        )

    def _export(self):
        return self.manager.export_backtest_result(
            result=self.result, strategy_params={}, symbol='BTC-USD',
            initial_capital=10000.0, commission=0.001
        )

    def test_constructor_failure_lands_in_the_export_summary(self):
        """An Elasticsearch exporter rejected by its own validation must be visible."""
        with patch('niffler.exporters.exporter_manager.logger'):
            self.manager.create_exporters_from_list(['console', 'elasticsearch'],
                                                    scheme='ftp')

        self.assertEqual(self.manager.get_exporter_names(), ['ConsoleExporter'])

        summary = self._export()

        self.assertFalse(summary.ok)
        self.assertEqual(summary.successes, ['ConsoleExporter'])
        self.assertEqual([name for name, _ in summary.failures], ['elasticsearch'])

    def test_all_exporters_created_yields_an_ok_summary(self):
        self.manager.create_exporters_from_list(['console'])

        summary = self._export()

        self.assertTrue(summary.ok)
        self.assertEqual(summary.failures, [])

    def test_clear_exporters_forgets_creation_failures(self):
        with patch('niffler.exporters.exporter_manager.logger'):
            self.manager.create_exporters_from_list(['console', 'unknown'])

        self.manager.clear_exporters()
        self.manager.create_exporters_from_list(['console'])

        self.assertTrue(self._export().ok)


class TestOptionBroadcast(unittest.TestCase):
    """Options are matched to constructors by signature, not by an if/elif chain."""

    def setUp(self):
        self.manager = ExporterManager()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def test_each_exporter_receives_only_what_it_accepts(self):
        self.manager.create_exporters_from_list(
            ['console', 'csv', 'elasticsearch'],
            output_dir=self.temp_dir, host='es.example', port=9201
        )

        console, csv_exporter, elasticsearch = self.manager.exporters
        self.assertIsInstance(console, ConsoleExporter)
        self.assertEqual(str(csv_exporter.output_dir), self.temp_dir)
        self.assertEqual(elasticsearch.host, 'es.example')
        self.assertEqual(elasticsearch.port, 9201)

    def test_a_newly_registered_exporter_needs_no_kwargs_branch(self):
        """One registry entry, and the option reaches the constructor."""
        with patch.dict(EXPORTER_CLASSES, {'probe': _ProbeExporter}):
            self.manager.create_exporters_from_list(
                ['probe', 'csv'], probe_token='niffler-probe', output_dir=self.temp_dir
            )

        probe, csv_exporter = self.manager.exporters
        self.assertEqual(probe.probe_token, 'niffler-probe')
        # The probe's option must not leak into the exporter that does not accept it.
        self.assertEqual(str(csv_exporter.output_dir), self.temp_dir)

    def test_an_option_no_requested_exporter_accepts_raises(self):
        """Silently dropping it is how --csv-output-dir used to be ignored."""
        with self.assertRaises(ValueError) as context:
            self.manager.create_exporters_from_list(['console'], output_dir=self.temp_dir)

        message = str(context.exception)
        self.assertIn('No requested exporter accepts: output_dir', message)
        self.assertIn('console accepts config', message)
        self.assertEqual(self.manager.exporters, [])

    def test_an_option_one_of_several_accepts_is_fine(self):
        self.manager.create_exporters_from_list(
            ['console', 'csv'], output_dir=self.temp_dir
        )

        self.assertEqual(len(self.manager.exporters), 2)

    def test_an_unknown_exporter_name_is_still_only_a_recorded_failure(self):
        with patch('niffler.exporters.exporter_manager.logger'):
            failures = self.manager.create_exporters_from_list(
                ['csv', 'nope'], output_dir=self.temp_dir
            )

        self.assertEqual([name for name, _ in failures], ['nope'])
        self.assertEqual(len(self.manager.exporters), 1)


if __name__ == '__main__':
    unittest.main()
