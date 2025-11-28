"""
Elasticsearch Exporter

Exports backtest results to Elasticsearch for visualization with Grafana.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
import pandas as pd
import numpy as np
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
        self.positions_index = f"{self.index_prefix}-positions"
        
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

            # Export positions (paired trades with P&L)
            self._export_positions(result, backtest_id)

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
            (self.trades_index, "trades"),
            (self.positions_index, "positions")
        ]
        
        # Create indices with mappings loaded from files
        for index_name, mapping_name in indices_config:
            if not self.es_client.indices.exists(index=index_name):
                mapping = self._load_mapping(mapping_name)
                self.es_client.indices.create(index=index_name, body=mapping)
                self.logger.info(f"Created Elasticsearch index: {index_name} using mapping: {mapping_name}.json")
    
    def _sanitize_numeric_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize numeric values in a dictionary for Elasticsearch.
        Converts Infinity and NaN to None.
        """
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_numeric_values(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [
                    self._sanitize_numeric_values(v) if isinstance(v, dict) else
                    None if isinstance(v, (float, int)) and (np.isinf(v) or np.isnan(v)) else v
                    for v in value
                ]
            elif isinstance(value, (float, int)):
                if np.isinf(value) or np.isnan(value):
                    sanitized[key] = None
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        return sanitized

    def _export_backtest_metadata(self, metadata: Dict[str, Any], backtest_id: str) -> None:
        """Export backtest metadata to Elasticsearch."""
        doc = {
            **metadata,
            "backtest_id": backtest_id,
            "created_at": datetime.now(UTC).isoformat()
        }

        # Sanitize numeric values (convert Infinity/NaN to None)
        doc = self._sanitize_numeric_values(doc)

        self.es_client.index(
            index=self.backtests_index,
            id=backtest_id,
            body=doc
        )
        self.logger.debug(f"Exported backtest metadata for {backtest_id}")
    
    def _export_portfolio_values(self, result: BacktestResult, backtest_id: str) -> None:
        """Export portfolio values with drawdown, rolling Sharpe ratio, and volatility to Elasticsearch using bulk API."""
        if result.portfolio_values.empty:
            self.logger.warning("No portfolio values to export")
            return

        # Convert to DataFrame for easier calculations
        df = pd.DataFrame({
            'portfolio_value': result.portfolio_values.values
        }, index=result.portfolio_values.index)

        # Calculate drawdown percentage
        running_peak = np.maximum.accumulate(df['portfolio_value'].values)
        df['drawdown_pct'] = (df['portfolio_value'].values - running_peak) / running_peak * 100

        # Calculate rolling metrics (30-day window)
        window = 30
        returns = df['portfolio_value'].pct_change()
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        # Annualize: assume 252 trading days per year
        # Sharpe = (mean_return * 252) / (std_return * sqrt(252))
        df['rolling_sharpe_30d'] = np.where(
            rolling_std > 0,
            (rolling_mean * 252) / (rolling_std * np.sqrt(252)),
            np.nan
        )

        # Rolling volatility (annualized standard deviation of returns)
        # Volatility = std * sqrt(252)
        df['rolling_volatility_30d'] = rolling_std * np.sqrt(252) * 100  # in percentage

        # Prepare bulk data
        actions = []
        created_at = datetime.now(UTC).isoformat()
        for timestamp, row in df.iterrows():
            action = {
                "_index": self.portfolio_index,
                "_source": {
                    "backtest_id": backtest_id,
                    "timestamp": timestamp.isoformat(),
                    "portfolio_value": float(row['portfolio_value']),
                    "drawdown_pct": float(row['drawdown_pct']),
                    "rolling_sharpe_30d": float(row['rolling_sharpe_30d']) if not pd.isna(row['rolling_sharpe_30d']) else None,
                    "rolling_volatility_30d": float(row['rolling_volatility_30d']) if not pd.isna(row['rolling_volatility_30d']) else None,
                    "created_at": created_at
                }
            }
            actions.append(action)

        # Bulk insert
        from elasticsearch.helpers import bulk
        bulk(self.es_client, actions)
        self.logger.debug(f"Exported {len(actions)} portfolio values with metrics for {backtest_id}")
    
    def _export_trades(self, result: BacktestResult, backtest_id: str) -> None:
        """Export trades to Elasticsearch using bulk API."""
        if not result.trades:
            self.logger.info("No trades to export")
            return
        
        # Prepare bulk data
        actions = []
        created_at = datetime.now(UTC).isoformat()
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

    def _export_positions(self, result: BacktestResult, backtest_id: str) -> None:
        """
        Export paired trades (positions) with P&L calculations to Elasticsearch.

        Pairs buy and sell trades to create complete positions with:
        - Entry/exit prices and timestamps
        - P&L (absolute and percentage)
        - Duration
        - Win/loss indicator
        """
        if not result.trades:
            self.logger.info("No trades to pair into positions")
            return

        # Pair buy/sell trades
        positions = []
        open_position = None
        position_counter = 0
        created_at = datetime.now(UTC).isoformat()

        for trade in result.trades:
            if trade.side.value == "buy":
                # Entry trade
                if open_position is not None:
                    self.logger.warning(f"Opening new position while one is already open at {trade.timestamp}")
                open_position = {
                    "entry_trade": trade,
                    "symbol": trade.symbol
                }
            elif trade.side.value == "sell" and open_position is not None:
                # Exit trade - complete the position
                entry_trade = open_position["entry_trade"]

                # Calculate P&L
                pnl = trade.value - entry_trade.value
                pnl_pct = (pnl / entry_trade.value) * 100 if entry_trade.value != 0 else 0

                # Calculate duration
                duration = trade.timestamp - entry_trade.timestamp
                duration_days = duration.total_seconds() / (24 * 3600)
                duration_hours = duration.total_seconds() / 3600

                # Create position document
                position_counter += 1
                position = {
                    "backtest_id": backtest_id,
                    "position_id": f"{backtest_id}-pos-{position_counter}",
                    "symbol": trade.symbol,
                    "entry_timestamp": entry_trade.timestamp.isoformat(),
                    "exit_timestamp": trade.timestamp.isoformat(),
                    "entry_price": entry_trade.price,
                    "exit_price": trade.price,
                    "quantity": entry_trade.quantity,
                    "entry_value": entry_trade.value,
                    "exit_value": trade.value,
                    "pnl": float(pnl),
                    "pnl_pct": float(pnl_pct),
                    "duration_days": float(duration_days),
                    "duration_hours": float(duration_hours),
                    "is_win": pnl > 0,
                    "created_at": created_at
                }
                positions.append(position)
                open_position = None

        if not positions:
            self.logger.info("No complete positions to export (unpaired trades)")
            return

        # Prepare bulk data
        actions = []
        for position in positions:
            action = {
                "_index": self.positions_index,
                "_source": position
            }
            actions.append(action)

        # Bulk insert
        from elasticsearch.helpers import bulk
        bulk(self.es_client, actions)
        self.logger.info(f"Exported {len(actions)} positions for {backtest_id}")

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