"""
Elasticsearch Exporter

Exports backtest results to Elasticsearch for visualization with Grafana.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import os
import json
from pathlib import Path

from .base_exporter import BaseExporter
from ..backtesting.backtest_result import BacktestResult


class ElasticsearchExporter(BaseExporter):
    """Exporter that saves backtest results to Elasticsearch."""
    
    def __init__(self, host: str = None, port: int = None, 
                 index_prefix: str = None, config: Dict[str, Any] = None):
        """
        Initialize Elasticsearch exporter.
        
        Args:
            host: Elasticsearch host (uses ELASTICSEARCH_HOST env var if not provided)
            port: Elasticsearch port (uses ELASTICSEARCH_PORT env var if not provided)
            index_prefix: Prefix for index names (uses ELASTICSEARCH_INDEX_PREFIX env var if not provided)
            config: Additional configuration options
        """
        super().__init__(config)
        self.host = host or os.getenv('ELASTICSEARCH_HOST', 'localhost')
        self.port = port or int(os.getenv('ELASTICSEARCH_PORT', '9200'))
        self.index_prefix = index_prefix or os.getenv('ELASTICSEARCH_INDEX_PREFIX', 'niffler')
        self.es_client = None
        
        # Index names
        self.backtests_index = f"{self.index_prefix}-backtests"
        self.portfolio_index = f"{self.index_prefix}-portfolio-values"
        self.trades_index = f"{self.index_prefix}-trades"
        
        # Try to import elasticsearch
        try:
            from elasticsearch import Elasticsearch
            self.Elasticsearch = Elasticsearch
        except ImportError:
            self.logger.error("elasticsearch package not installed. Run: pip install elasticsearch")
            self.Elasticsearch = None
    
    def _connect(self) -> bool:
        """Connect to Elasticsearch cluster."""
        if self.Elasticsearch is None:
            return False
            
        try:
            self.es_client = self.Elasticsearch([f"http://{self.host}:{self.port}"])
            # Test connection
            if self.es_client.ping():
                self.logger.info(f"Connected to Elasticsearch at {self.host}:{self.port}")
                return True
            else:
                self.logger.error(f"Cannot connect to Elasticsearch at {self.host}:{self.port}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False
    
    def export_backtest_result(self, result: BacktestResult, backtest_id: str, 
                              metadata: Dict[str, Any]) -> None:
        """
        Export backtest results to Elasticsearch.
        
        Args:
            result: BacktestResult object containing all backtest data
            backtest_id: Unique identifier for this backtest run
            metadata: Additional metadata about the backtest
        """
        if not self.validate_result(result):
            self.logger.error("Invalid backtest result, skipping Elasticsearch export")
            return
        
        if not self._connect():
            self.logger.error("Cannot connect to Elasticsearch, skipping export")
            return
        
        try:
            # Create indices if they don't exist
            self._create_indices()
            
            # Export backtest metadata
            self._export_backtest_metadata(metadata, backtest_id)
            
            # Export portfolio values
            self._export_portfolio_values(result, backtest_id)
            
            # Export trades
            self._export_trades(result, backtest_id)
            
            self.logger.info(f"Successfully exported backtest {backtest_id} to Elasticsearch")
            
        except Exception as e:
            self.logger.error(f"Failed to export to Elasticsearch: {e}")
            raise
    
    def _load_mapping(self, mapping_name: str) -> Dict[str, Any]:
        """Load Elasticsearch mapping from JSON file."""
        # Get the project root directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        mapping_file = project_root / "config" / "elasticsearch" / "mappings" / f"{mapping_name}.json"
        
        try:
            with open(mapping_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Mapping file not found: {mapping_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in mapping file {mapping_file}: {e}")
            raise

    def _create_indices(self) -> None:
        """Create Elasticsearch indices with mappings loaded from configuration files."""
        # Define index mappings
        indices_config = [
            (self.backtests_index, "backtests"),
            (self.portfolio_index, "portfolio"),
            (self.trades_index, "trades")
        ]
        
        # Create indices with mappings loaded from files
        for index_name, mapping_name in indices_config:
            if not self.es_client.indices.exists(index=index_name):
                mapping = self._load_mapping(mapping_name)
                self.es_client.indices.create(index=index_name, body=mapping)
                self.logger.info(f"Created Elasticsearch index: {index_name} using mapping: {mapping_name}.json")
    
    def _export_backtest_metadata(self, metadata: Dict[str, Any], backtest_id: str) -> None:
        """Export backtest metadata to Elasticsearch."""
        doc = {
            **metadata,
            "backtest_id": backtest_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.es_client.index(
            index=self.backtests_index,
            id=backtest_id,
            body=doc
        )
        self.logger.debug(f"Exported backtest metadata for {backtest_id}")
    
    def _export_portfolio_values(self, result: BacktestResult, backtest_id: str) -> None:
        """Export portfolio values to Elasticsearch using bulk API."""
        if result.portfolio_values.empty:
            self.logger.warning("No portfolio values to export")
            return
        
        # Prepare bulk data
        actions = []
        created_at = datetime.utcnow().isoformat()
        for timestamp, value in result.portfolio_values.items():
            action = {
                "_index": self.portfolio_index,
                "_source": {
                    "backtest_id": backtest_id,
                    "timestamp": timestamp.isoformat(),
                    "portfolio_value": float(value),
                    "created_at": created_at
                }
            }
            actions.append(action)
        
        # Bulk insert
        from elasticsearch.helpers import bulk
        bulk(self.es_client, actions)
        self.logger.debug(f"Exported {len(actions)} portfolio values for {backtest_id}")
    
    def _export_trades(self, result: BacktestResult, backtest_id: str) -> None:
        """Export trades to Elasticsearch using bulk API."""
        if not result.trades:
            self.logger.info("No trades to export")
            return
        
        # Prepare bulk data
        actions = []
        created_at = datetime.utcnow().isoformat()
        for trade in result.trades:
            action = {
                "_index": self.trades_index,
                "_source": {
                    "backtest_id": backtest_id,
                    "timestamp": trade.timestamp.isoformat(),
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "value": trade.value,
                    "created_at": created_at
                }
            }
            actions.append(action)
        
        # Bulk insert
        from elasticsearch.helpers import bulk
        bulk(self.es_client, actions)
        self.logger.debug(f"Exported {len(actions)} trades for {backtest_id}")
    
    def test_connection(self) -> bool:
        """Test connection to Elasticsearch."""
        return self._connect()
    
    def list_indices(self) -> List[str]:
        """List all indices with the configured prefix."""
        if not self._connect():
            return []
        
        try:
            indices = self.es_client.indices.get(index=f"{self.index_prefix}-*")
            return list(indices.keys())
        except Exception as e:
            self.logger.error(f"Failed to list indices: {e}")
            return []