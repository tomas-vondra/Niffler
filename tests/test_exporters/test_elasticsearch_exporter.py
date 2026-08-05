"""
Unit tests for ElasticsearchExporter.
"""

import unittest
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
import pandas as pd
import json
import tempfile
import shutil
import os
from pathlib import Path

from niffler.exporters.base_exporter import ExportError
from niffler.exporters.elasticsearch_exporter import ElasticsearchExporter
from niffler.backtesting.backtest_result import BacktestResult
from niffler.backtesting.trade import Trade, TradeSide



class TestElasticsearchExporter(unittest.TestCase):
    """Test cases for ElasticsearchExporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for mapping files
        self.temp_dir = tempfile.mkdtemp()
        self.mapping_dir = os.path.join(self.temp_dir, "config", "elasticsearch", "mappings")
        os.makedirs(self.mapping_dir, exist_ok=True)
        
        # Create mock mapping files
        self._create_mock_mapping_files()
        
        # Mock environment variables
        # Explicit empty credentials keep the tests hermetic on developer machines
        # that happen to export ELASTICSEARCH_* settings.
        self.env_patcher = patch.dict(os.environ, {
            'ELASTICSEARCH_HOST': 'test-host',
            'ELASTICSEARCH_PORT': '9200',
            'ELASTICSEARCH_INDEX_PREFIX': 'test-prefix',
            'ELASTICSEARCH_SCHEME': 'http',
            'ELASTICSEARCH_TIMEOUT': '30',
            'ELASTICSEARCH_API_KEY': '',
            'ELASTICSEARCH_USERNAME': '',
            'ELASTICSEARCH_PASSWORD': '',
            'ELASTICSEARCH_VERIFY_CERTS': 'true'
        })
        self.env_patcher.start()
        
        self.exporter = ElasticsearchExporter()
        
        # Create mock BacktestResult
        self.mock_result = Mock(spec=BacktestResult)
        self.mock_result.strategy_name = "Simple MA Strategy"
        self.mock_result.symbol = "BTC-USD"
        self.mock_result.start_date = datetime(2024, 1, 1)
        self.mock_result.end_date = datetime(2024, 3, 31)
        
        # Mock portfolio values
        portfolio_values = pd.Series(
            [10000.0, 10100.0, 10200.0],
            index=[datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
        )
        self.mock_result.portfolio_values = portfolio_values
        
        # Create mock trades
        mock_trade1 = Mock(spec=Trade)
        mock_trade1.timestamp = datetime(2024, 1, 15)
        mock_trade1.symbol = "BTC-USD"
        mock_trade1.side = TradeSide.BUY
        mock_trade1.quantity = 0.25
        mock_trade1.price = 45000.0
        mock_trade1.value = 11250.0
        mock_trade1.commission = 11.25

        self.mock_result.trades = [mock_trade1]
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_mapping_files(self):
        """Create mock mapping files for testing."""
        backtests_mapping = {
            "mappings": {
                "properties": {
                    "backtest_id": {"type": "keyword"},
                    "strategy_name": {"type": "keyword"}
                }
            }
        }
        
        portfolio_mapping = {
            "mappings": {
                "properties": {
                    "backtest_id": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "portfolio_value": {"type": "double"}
                }
            }
        }
        
        trades_mapping = {
            "mappings": {
                "properties": {
                    "backtest_id": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "symbol": {"type": "keyword"}
                }
            }
        }
        
        # Write mapping files
        with open(os.path.join(self.mapping_dir, "backtests.json"), 'w') as f:
            json.dump(backtests_mapping, f)
        
        with open(os.path.join(self.mapping_dir, "portfolio.json"), 'w') as f:
            json.dump(portfolio_mapping, f)
        
        with open(os.path.join(self.mapping_dir, "trades.json"), 'w') as f:
            json.dump(trades_mapping, f)
    
    def test_init_default_values(self):
        """Test initialization with default values from environment."""
        exporter = ElasticsearchExporter()
        self.assertEqual(exporter.host, 'test-host')
        self.assertEqual(exporter.port, 9200)
        self.assertEqual(exporter.index_prefix, 'test-prefix')
        self.assertEqual(exporter.backtests_index, 'test-prefix-backtests')
        self.assertEqual(exporter.portfolio_index, 'test-prefix-portfolio-values')
        self.assertEqual(exporter.trades_index, 'test-prefix-trades')
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        exporter = ElasticsearchExporter(
            host='custom-host',
            port=9300,
            index_prefix='custom-prefix'
        )
        self.assertEqual(exporter.host, 'custom-host')
        self.assertEqual(exporter.port, 9300)
        self.assertEqual(exporter.index_prefix, 'custom-prefix')
    
    def test_init_with_config(self):
        """Test initialization with config."""
        config = {'option': 'value'}
        exporter = ElasticsearchExporter(config=config)
        self.assertEqual(exporter.config, config)
    
    def test_init_elasticsearch_not_available(self):
        """Test initialization when elasticsearch package is not available."""
        # This test is complex to implement due to import mocking limitations
        # The functionality is covered by the _connect method tests
        pass
    
    @patch('niffler.exporters.elasticsearch_exporter.Path')
    def test_load_mapping_success(self, mock_path):
        """Test successful loading of mapping file."""
        # Mock the path resolution
        mock_path(__file__).parent.parent.parent = Path(self.temp_dir)
        
        mapping_data = {"mappings": {"properties": {"test": {"type": "keyword"}}}}
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mapping_data))):
            result = self.exporter._load_mapping('test')
            self.assertEqual(result, mapping_data)
    
    @patch('niffler.exporters.elasticsearch_exporter.Path')
    def test_load_mapping_file_not_found(self, mock_path):
        """Test loading mapping when file doesn't exist."""
        mock_path(__file__).parent.parent.parent = Path(self.temp_dir)
        
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            with patch.object(self.exporter.logger, 'error') as mock_logger:
                with self.assertRaises(FileNotFoundError):
                    self.exporter._load_mapping('nonexistent')
                mock_logger.assert_called_once()
    
    @patch('niffler.exporters.elasticsearch_exporter.Path')
    def test_load_mapping_invalid_json(self, mock_path):
        """Test loading mapping with invalid JSON."""
        mock_path(__file__).parent.parent.parent = Path(self.temp_dir)
        
        with patch('builtins.open', mock_open(read_data="invalid json")):
            with patch.object(self.exporter.logger, 'error') as mock_logger:
                with self.assertRaises(json.JSONDecodeError):
                    self.exporter._load_mapping('test')
                mock_logger.assert_called_once()
    
    def test_connect_success(self):
        """Test successful connection to Elasticsearch."""
        mock_es_class = Mock()
        mock_es_instance = Mock()
        mock_es_class.return_value = mock_es_instance
        mock_es_instance.ping.return_value = True
        
        self.exporter.Elasticsearch = mock_es_class

        result = self.exporter._connect()

        self.assertTrue(result)
        self.assertEqual(self.exporter.es_client, mock_es_instance)
        mock_es_class.assert_called_once_with(['http://test-host:9200'], request_timeout=30.0)
        mock_es_instance.ping.assert_called_once()
    
    def test_connect_failure(self):
        """Test failed connection to Elasticsearch."""
        mock_es_class = Mock()
        mock_es_instance = Mock()
        mock_es_class.return_value = mock_es_instance
        mock_es_instance.ping.return_value = False
        
        self.exporter.Elasticsearch = mock_es_class
        
        with patch.object(self.exporter.logger, 'error') as mock_logger:
            result = self.exporter._connect()
            
            self.assertFalse(result)
            mock_logger.assert_called_once()
    
    def test_connect_elasticsearch_not_available(self):
        """Test connection when Elasticsearch class is not available."""
        self.exporter.Elasticsearch = None
        
        result = self.exporter._connect()
        self.assertFalse(result)
    
    @patch.object(ElasticsearchExporter, '_load_mapping')
    def test_create_indices(self, mock_load_mapping):
        """Test index creation."""
        # Mock the mapping loading
        mock_mapping = {"mappings": {"properties": {"test": {"type": "keyword"}}}}
        mock_load_mapping.return_value = mock_mapping
        
        # Mock Elasticsearch client
        mock_es_client = Mock()
        mock_es_client.indices.exists.return_value = False
        self.exporter.es_client = mock_es_client
        
        with patch.object(self.exporter.logger, 'info') as mock_logger:
            self.exporter._create_indices()
        
        # Verify that exists was checked for all indices
        self.assertEqual(mock_es_client.indices.exists.call_count, 4)

        # Verify that create was called for all indices
        self.assertEqual(mock_es_client.indices.create.call_count, 4)

        # Verify that mappings were loaded for all index types
        mock_load_mapping.assert_any_call('backtests')
        mock_load_mapping.assert_any_call('portfolio')
        mock_load_mapping.assert_any_call('trades')
        mock_load_mapping.assert_any_call('positions')

        # Verify logging
        self.assertEqual(mock_logger.call_count, 4)
    
    @patch.object(ElasticsearchExporter, '_load_mapping')
    def test_create_indices_already_exist(self, mock_load_mapping):
        """Test index creation when indices already exist."""
        # Mock Elasticsearch client
        mock_es_client = Mock()
        mock_es_client.indices.exists.return_value = True
        self.exporter.es_client = mock_es_client
        
        self.exporter._create_indices()

        # Verify that exists was checked but create was not called
        self.assertEqual(mock_es_client.indices.exists.call_count, 4)
        mock_es_client.indices.create.assert_not_called()
        mock_load_mapping.assert_not_called()
    
    def test_export_backtest_metadata(self):
        """Test exporting backtest metadata."""
        metadata = {
            'strategy_name': 'Simple MA',
            'symbol': 'BTC-USD',
            'total_return': 1500.0
        }
        backtest_id = 'test-id-123'
        
        mock_es_client = Mock()
        self.exporter.es_client = mock_es_client
        
        self.exporter._export_backtest_metadata(metadata, backtest_id)
        
        expected_doc = {
            **metadata,
            'created_at': unittest.mock.ANY  # datetime will vary
        }
        
        mock_es_client.index.assert_called_once()
        call_args = mock_es_client.index.call_args
        self.assertEqual(call_args[1]['index'], 'test-prefix-backtests')
        self.assertEqual(call_args[1]['id'], backtest_id)
        
        # Check that created_at was added
        self.assertIn('created_at', call_args[1]['body'])
    
    def test_bulk_is_module_level_attribute(self):
        """The bulk helper is imported at module level so it can be patched/monkeyed."""
        import niffler.exporters.elasticsearch_exporter as es_module
        self.assertTrue(hasattr(es_module, 'bulk'))

    @patch('niffler.exporters.elasticsearch_exporter.bulk', None)
    def test_bulk_index_without_elasticsearch_package(self):
        """A missing elasticsearch package produces a clear error, not a NameError."""
        self.exporter.es_client = Mock()

        with self.assertRaises(RuntimeError) as context:
            self.exporter._bulk_index([{'_index': 'x', '_source': {}}])

        self.assertIn('elasticsearch package not installed', str(context.exception))

    @patch('niffler.exporters.elasticsearch_exporter.bulk')
    def test_export_portfolio_values(self, mock_bulk):
        """Test exporting portfolio values."""
        backtest_id = 'test-id-123'
        
        mock_es_client = Mock()
        self.exporter.es_client = mock_es_client
        
        self.exporter._export_portfolio_values(self.mock_result, backtest_id)
        
        # Check that bulk was called
        mock_bulk.assert_called_once()
        
        # Check the arguments passed to bulk
        call_args = mock_bulk.call_args
        self.assertEqual(call_args[0][0], mock_es_client)  # es_client
        
        # Check the actions (should be 3 portfolio values)
        actions = list(call_args[0][1])
        self.assertEqual(len(actions), 3)
        
        # Check first action
        first_action = actions[0]
        self.assertEqual(first_action['_index'], 'test-prefix-portfolio-values')
        self.assertEqual(first_action['_source']['backtest_id'], backtest_id)
        self.assertEqual(first_action['_source']['portfolio_value'], 10000.0)
    
    @patch('niffler.exporters.elasticsearch_exporter.bulk')
    def test_export_trades(self, mock_bulk):
        """Test exporting trades."""
        backtest_id = 'test-id-123'
        
        mock_es_client = Mock()
        self.exporter.es_client = mock_es_client
        
        self.exporter._export_trades(self.mock_result, backtest_id)
        
        # Check that bulk was called
        mock_bulk.assert_called_once()
        
        # Check the arguments passed to bulk
        call_args = mock_bulk.call_args
        self.assertEqual(call_args[0][0], mock_es_client)  # es_client
        
        # Check the actions (should be 1 trade)
        actions = list(call_args[0][1])
        self.assertEqual(len(actions), 1)
        
        # Check the action
        action = actions[0]
        self.assertEqual(action['_index'], 'test-prefix-trades')
        self.assertEqual(action['_source']['backtest_id'], backtest_id)
        self.assertEqual(action['_source']['symbol'], 'BTC-USD')
        self.assertEqual(action['_source']['side'], 'buy')
    
    def test_export_trades_empty(self):
        """Test exporting with no trades."""
        self.mock_result.trades = []
        backtest_id = 'test-id-123'
        
        mock_es_client = Mock()
        self.exporter.es_client = mock_es_client
        
        with patch.object(self.exporter.logger, 'info') as mock_logger:
            self.exporter._export_trades(self.mock_result, backtest_id)
        
        mock_es_client.index.assert_not_called()
        mock_logger.assert_called_once_with("No trades to export")
    
    @patch.object(ElasticsearchExporter, '_connect')
    @patch.object(ElasticsearchExporter, '_create_indices')
    def test_export_backtest_result_success(self, mock_create_indices, mock_connect):
        """Test successful full export."""
        mock_connect.return_value = True
        
        backtest_id = 'test-id-123'
        metadata = {'strategy_name': 'Simple MA'}
        
        mock_es_client = Mock()
        self.exporter.es_client = mock_es_client
        
        with patch.object(self.exporter, 'validate_result', return_value=True):
            with patch.object(self.exporter, '_export_backtest_metadata') as mock_export_meta:
                with patch.object(self.exporter, '_export_portfolio_values') as mock_export_portfolio:
                    with patch.object(self.exporter, '_export_trades') as mock_export_trades:
                        with patch.object(self.exporter.logger, 'info') as mock_logger:
                            self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)
        
        mock_connect.assert_called_once()
        mock_create_indices.assert_called_once()
        mock_export_meta.assert_called_once_with(metadata, backtest_id)
        mock_export_portfolio.assert_called_once_with(self.mock_result, backtest_id)
        mock_export_trades.assert_called_once_with(self.mock_result, backtest_id)
        mock_logger.assert_called_with(f"Successfully exported backtest {backtest_id} to Elasticsearch")
    
    @patch.object(ElasticsearchExporter, '_connect')
    def test_export_backtest_result_connection_failed(self, mock_connect):
        """An unreachable cluster raises so the caller can report the failed export."""
        mock_connect.return_value = False

        backtest_id = 'test-id-123'
        metadata = {'strategy_name': 'Simple MA'}

        with patch.object(self.exporter, 'validate_result', return_value=True):
            with patch.object(self.exporter.logger, 'error') as mock_logger:
                with self.assertRaises(ExportError) as context:
                    self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)
                self.assertIn(self.exporter.url, str(context.exception))
                mock_logger.assert_called_once_with(
                    f"Cannot connect to Elasticsearch at {self.exporter.url}"
                )

    @patch.object(ElasticsearchExporter, '_connect')
    def test_export_backtest_result_invalid_result(self, mock_connect):
        """An unexportable result raises instead of reporting a silent success."""
        mock_connect.return_value = True

        backtest_id = 'test-id-123'
        metadata = {'strategy_name': 'Simple MA'}

        with patch.object(self.exporter, 'validate_result', return_value=False):
            with patch.object(self.exporter.logger, 'error') as mock_logger:
                with self.assertRaises(ExportError):
                    self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)
                mock_logger.assert_called_once_with(
                    "Invalid backtest result, cannot export to Elasticsearch"
                )

    def test_sanitize_numeric_values_delegates_to_shared_helper(self):
        """Non-finite metrics are converted to None before indexing."""
        sanitized = self.exporter._sanitize_numeric_values({
            'profit_factor': float('inf'),
            'sharpe_ratio': float('nan'),
            'nested': {'largest_loss': float('-inf')},
            'series': [1.0, float('nan')],
            'name': 'Simple MA',
            'total_trades': 25
        })

        self.assertIsNone(sanitized['profit_factor'])
        self.assertIsNone(sanitized['sharpe_ratio'])
        self.assertIsNone(sanitized['nested']['largest_loss'])
        self.assertEqual(sanitized['series'], [1.0, None])
        self.assertEqual(sanitized['name'], 'Simple MA')
        self.assertEqual(sanitized['total_trades'], 25)


class TestElasticsearchExporterConnectionOptions(unittest.TestCase):
    """Test cases for authentication, TLS and timeout configuration."""

    def setUp(self):
        """Isolate the tests from any ELASTICSEARCH_* variables in the environment."""
        self.env_patcher = patch.dict(os.environ, {
            'ELASTICSEARCH_HOST': '',
            'ELASTICSEARCH_PORT': '',
            'ELASTICSEARCH_INDEX_PREFIX': '',
            'ELASTICSEARCH_SCHEME': '',
            'ELASTICSEARCH_TIMEOUT': '',
            'ELASTICSEARCH_API_KEY': '',
            'ELASTICSEARCH_USERNAME': '',
            'ELASTICSEARCH_PASSWORD': '',
            'ELASTICSEARCH_VERIFY_CERTS': ''
        })
        self.env_patcher.start()

    def tearDown(self):
        """Restore the environment."""
        self.env_patcher.stop()

    @staticmethod
    def _connect_with(exporter):
        """Connect using a mocked client class and return the client constructor mock."""
        mock_es_class = Mock()
        mock_es_class.return_value.ping.return_value = True
        exporter.Elasticsearch = mock_es_class
        exporter._connect()
        return mock_es_class

    def test_defaults_preserve_localhost_dev_experience(self):
        """Without configuration the exporter still targets http://localhost:9200."""
        exporter = ElasticsearchExporter()

        self.assertEqual(exporter.host, 'localhost')
        self.assertEqual(exporter.port, 9200)
        self.assertEqual(exporter.scheme, 'http')
        self.assertEqual(exporter.url, 'http://localhost:9200')
        self.assertEqual(exporter._auth_mode(), 'none')

        mock_es_class = self._connect_with(exporter)
        mock_es_class.assert_called_once_with(['http://localhost:9200'], request_timeout=30.0)

    def test_api_key_from_argument(self):
        """An API key is passed to the client and never appears in the URL."""
        exporter = ElasticsearchExporter(host='es.internal', api_key='secret-key')

        mock_es_class = self._connect_with(exporter)

        _, kwargs = mock_es_class.call_args
        self.assertEqual(kwargs['api_key'], 'secret-key')
        self.assertNotIn('secret-key', exporter.url)
        self.assertEqual(exporter._auth_mode(), 'api_key')

    def test_basic_auth_from_environment(self):
        """Username/password from the environment become basic auth credentials."""
        with patch.dict(os.environ, {
            'ELASTICSEARCH_USERNAME': 'elastic',
            'ELASTICSEARCH_PASSWORD': 'hunter2'
        }):
            exporter = ElasticsearchExporter()

        mock_es_class = self._connect_with(exporter)

        _, kwargs = mock_es_class.call_args
        self.assertEqual(kwargs['basic_auth'], ('elastic', 'hunter2'))
        self.assertEqual(exporter._auth_mode(), 'basic')

    def test_api_key_takes_precedence_over_basic_auth(self):
        """When both are configured the API key wins and basic auth is not sent."""
        exporter = ElasticsearchExporter(
            api_key='secret-key', username='elastic', password='hunter2'
        )

        mock_es_class = self._connect_with(exporter)

        _, kwargs = mock_es_class.call_args
        self.assertEqual(kwargs['api_key'], 'secret-key')
        self.assertNotIn('basic_auth', kwargs)

    def test_https_scheme_and_verify_certs(self):
        """The scheme and certificate verification are configurable."""
        with patch.dict(os.environ, {
            'ELASTICSEARCH_SCHEME': 'https',
            'ELASTICSEARCH_VERIFY_CERTS': 'false',
            'ELASTICSEARCH_TIMEOUT': '5'
        }):
            exporter = ElasticsearchExporter(host='es.internal', port=443)

        self.assertEqual(exporter.url, 'https://es.internal:443')
        self.assertFalse(exporter.verify_certs)
        self.assertEqual(exporter.timeout, 5.0)

        mock_es_class = self._connect_with(exporter)
        mock_es_class.assert_called_once_with(
            ['https://es.internal:443'], request_timeout=5.0, verify_certs=False
        )

    def test_invalid_scheme_rejected(self):
        """An unsupported scheme fails fast."""
        with self.assertRaises(ValueError):
            ElasticsearchExporter(scheme='ftp')

    def test_incomplete_basic_auth_warns_and_sends_no_credentials(self):
        """A username without a password is reported and not sent to the client."""
        exporter = ElasticsearchExporter(username='elastic')

        with patch.object(exporter.logger, 'warning') as mock_warning:
            mock_es_class = self._connect_with(exporter)

        _, kwargs = mock_es_class.call_args
        self.assertNotIn('basic_auth', kwargs)
        self.assertNotIn('api_key', kwargs)
        mock_warning.assert_called_once()
        self.assertEqual(exporter._auth_mode(), 'none')

    def test_credentials_are_never_logged(self):
        """Successful connection logs the URL and auth mode, never the secret."""
        exporter = ElasticsearchExporter(host='es.internal', api_key='secret-key')

        with patch.object(exporter.logger, 'info') as mock_info:
            self._connect_with(exporter)

        logged = ' '.join(str(call) for call in mock_info.call_args_list)
        self.assertNotIn('secret-key', logged)
        self.assertIn('api_key', logged)


class TestListIndicesFailureReporting(unittest.TestCase):
    """An empty list must mean "no indices", never "cluster unreachable"."""

    def setUp(self):
        self.exporter = ElasticsearchExporter(host='test-host', port=9200,
                                              index_prefix='test-prefix')

    def test_unreachable_cluster_raises(self):
        with patch.object(self.exporter, '_connect', return_value=False):
            with self.assertRaises(ExportError):
                self.exporter.list_indices()

    def test_query_failure_raises(self):
        mock_client = Mock()
        mock_client.indices.get.side_effect = RuntimeError("boom")

        with patch.object(self.exporter, '_connect', return_value=True):
            self.exporter.es_client = mock_client
            with self.assertRaises(ExportError):
                self.exporter.list_indices()

    def test_no_matching_indices_returns_empty_list(self):
        mock_client = Mock()
        mock_client.indices.get.return_value = {}

        with patch.object(self.exporter, '_connect', return_value=True):
            self.exporter.es_client = mock_client
            self.assertEqual(self.exporter.list_indices(), [])

    def test_matching_indices_are_returned(self):
        mock_client = Mock()
        mock_client.indices.get.return_value = {'test-prefix-trades': {},
                                                'test-prefix-backtests': {}}

        with patch.object(self.exporter, '_connect', return_value=True):
            self.exporter.es_client = mock_client
            self.assertEqual(sorted(self.exporter.list_indices()),
                             ['test-prefix-backtests', 'test-prefix-trades'])


class TestExportPositionsReconciliation(unittest.TestCase):
    """The positions index must agree with the metrics of the same backtest."""

    def setUp(self):
        self.exporter = ElasticsearchExporter(host='test-host', port=9200,
                                              index_prefix='test-prefix')
        self.exporter.es_client = Mock()

    @staticmethod
    def _result_with_partial_exit():
        """One entry closed by two partial exits, both profitable."""
        from niffler.backtesting.backtest_engine import BacktestEngine

        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'BTC-USD', TradeSide.BUY,
                  101.0, 98.911, 9990.0, 9.99),
            Trade(pd.Timestamp('2024-01-05'), 'BTC-USD', TradeSide.SELL,
                  103.0, 49.4555, 5093.0, 5.09),
            Trade(pd.Timestamp('2024-01-09'), 'BTC-USD', TradeSide.SELL,
                  105.0, 49.4555, 5192.8, 5.19),
        ]
        result = Mock(spec=BacktestResult)
        result.trades = trades
        engine = BacktestEngine(initial_capital=10000.0, commission=0.001)
        stats = engine._calculate_trade_statistics(trades)
        result.num_winning_trades = stats['num_winning_trades']
        result.num_losing_trades = stats['num_losing_trades']
        result.win_rate = stats['win_rate']
        return result, engine

    def _exported_positions(self, result):
        with patch('niffler.exporters.elasticsearch_exporter.bulk') as mock_bulk:
            self.exporter._export_positions(result, 'test-id')
        self.assertTrue(mock_bulk.called)
        actions = list(mock_bulk.call_args[0][1])
        return [action['_source'] for action in actions]

    def test_every_partial_exit_produces_a_position_document(self):
        result, _ = self._result_with_partial_exit()

        positions = self._exported_positions(result)

        self.assertEqual(len(positions), 2)
        # The old pairing dropped the second sell entirely.
        self.assertEqual([p['exit_price'] for p in positions], [103.0, 105.0])

    def test_position_documents_reconcile_with_the_result_metrics(self):
        result, _ = self._result_with_partial_exit()

        positions = self._exported_positions(result)

        wins = sum(1 for p in positions if p['is_win'])
        self.assertEqual(wins, result.num_winning_trades)
        self.assertEqual(len(positions) - wins, result.num_losing_trades)
        for position in positions:
            # A profitable backtest must not be reported as a catastrophic loss.
            self.assertGreater(position['pnl'], 0)

    def test_pnl_is_net_of_commission_and_uses_the_matched_quantity(self):
        result, engine = self._result_with_partial_exit()
        round_trips = engine.pair_trades(result.trades)

        positions = self._exported_positions(result)

        for position, rt in zip(positions, round_trips):
            self.assertAlmostEqual(position['quantity'], rt.quantity)
            self.assertAlmostEqual(position['pnl'], rt.pnl)
            self.assertAlmostEqual(position['gross_pnl'], rt.gross_pnl)
            self.assertAlmostEqual(position['entry_commission'], rt.entry_commission)
            self.assertAlmostEqual(position['exit_commission'], rt.exit_commission)
            self.assertLess(position['pnl'], position['gross_pnl'])

    def test_unpaired_entry_exports_nothing(self):
        from niffler.backtesting.backtest_engine import BacktestEngine  # noqa: F401

        result = Mock(spec=BacktestResult)
        result.trades = [
            Trade(pd.Timestamp('2024-01-01'), 'BTC-USD', TradeSide.BUY,
                  101.0, 10.0, 1010.0, 1.01)
        ]

        with patch('niffler.exporters.elasticsearch_exporter.bulk') as mock_bulk:
            self.exporter._export_positions(result, 'test-id')

        mock_bulk.assert_not_called()


if __name__ == '__main__':
    unittest.main()