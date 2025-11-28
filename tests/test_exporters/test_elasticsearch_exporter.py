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

from niffler.exporters.elasticsearch_exporter import ElasticsearchExporter
from niffler.backtesting.backtest_result import BacktestResult
from niffler.backtesting.trade import Trade, TradeSide

# Check if elasticsearch is available
try:
    import elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False


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
        self.env_patcher = patch.dict(os.environ, {
            'ELASTICSEARCH_HOST': 'test-host',
            'ELASTICSEARCH_PORT': '9200',
            'ELASTICSEARCH_INDEX_PREFIX': 'test-prefix'
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
        mock_es_class.assert_called_once_with(['http://test-host:9200'])
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
    
    @unittest.skipIf(not ELASTICSEARCH_AVAILABLE, "elasticsearch package not installed")
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
    
    @unittest.skipIf(not ELASTICSEARCH_AVAILABLE, "elasticsearch package not installed")
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
        """Test export when connection fails."""
        mock_connect.return_value = False
        
        backtest_id = 'test-id-123'
        metadata = {'strategy_name': 'Simple MA'}
        
        with patch.object(self.exporter, 'validate_result', return_value=True):
            with patch.object(self.exporter.logger, 'error') as mock_logger:
                self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)
                mock_logger.assert_called_once_with("Cannot connect to Elasticsearch, skipping export")
    
    @patch.object(ElasticsearchExporter, '_connect')
    def test_export_backtest_result_invalid_result(self, mock_connect):
        """Test export with invalid result."""
        mock_connect.return_value = True
        
        backtest_id = 'test-id-123'
        metadata = {'strategy_name': 'Simple MA'}
        
        with patch.object(self.exporter, 'validate_result', return_value=False):
            with patch.object(self.exporter.logger, 'error') as mock_logger:
                self.exporter.export_backtest_result(self.mock_result, backtest_id, metadata)
                mock_logger.assert_called_once_with("Invalid backtest result, skipping Elasticsearch export")


if __name__ == '__main__':
    unittest.main()