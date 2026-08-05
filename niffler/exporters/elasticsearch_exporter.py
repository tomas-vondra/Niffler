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

from .base_exporter import BaseExporter, ExportError
from ..utils.json_utils import sanitize_numeric_values
from ..backtesting.backtest_result import BacktestResult
from ..backtesting.round_trip import pair_trades

# Optional dependency: the exporter degrades gracefully when it is not installed.
try:
    from elasticsearch import Elasticsearch
except ImportError:  # pragma: no cover - depends on the environment
    Elasticsearch = None

try:
    from elasticsearch.helpers import bulk
except ImportError:  # pragma: no cover - depends on the environment
    bulk = None

# Values accepted by the boolean environment variables.
_TRUTHY = frozenset({'1', 'true', 'yes', 'on'})


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read an environment variable, treating an empty value as unset.

    Args:
        name: Environment variable name
        default: Value used when the variable is unset or empty

    Returns:
        The variable value, or the default
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_flag(name: str, default: bool) -> bool:
    """
    Read a boolean environment variable.

    Args:
        name: Environment variable name
        default: Value used when the variable is unset or empty

    Returns:
        The parsed boolean value
    """
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in _TRUTHY


class ElasticsearchExporter(BaseExporter):
    """Exporter that saves backtest results to Elasticsearch."""

    def __init__(self, host: str = None, port: int = None,
                 index_prefix: str = None, config: Dict[str, Any] = None,
                 scheme: str = None, api_key: str = None,
                 username: str = None, password: str = None,
                 timeout: float = None, verify_certs: bool = None):
        """
        Initialize Elasticsearch exporter.

        All connection settings fall back to environment variables so deployments can be
        configured without code changes. Defaults keep the local development experience
        unchanged (http://localhost:9200, no authentication).

        Args:
            host: Elasticsearch host (uses ELASTICSEARCH_HOST env var if not provided)
            port: Elasticsearch port (uses ELASTICSEARCH_PORT env var if not provided)
            index_prefix: Prefix for index names (uses ELASTICSEARCH_INDEX_PREFIX env var if not provided)
            config: Additional configuration options
            scheme: URL scheme, 'http' or 'https' (uses ELASTICSEARCH_SCHEME env var, default 'http')
            api_key: Elasticsearch API key (uses ELASTICSEARCH_API_KEY env var); takes
                precedence over basic authentication
            username: Basic-auth user (uses ELASTICSEARCH_USERNAME env var)
            password: Basic-auth password (uses ELASTICSEARCH_PASSWORD env var)
            timeout: Request timeout in seconds (uses ELASTICSEARCH_TIMEOUT env var, default 30)
            verify_certs: Whether to verify TLS certificates for https connections
                (uses ELASTICSEARCH_VERIFY_CERTS env var, default True)
        """
        super().__init__(config)
        self.host = host or _env('ELASTICSEARCH_HOST', 'localhost')
        self.port = port or int(_env('ELASTICSEARCH_PORT', '9200'))
        self.index_prefix = index_prefix or _env('ELASTICSEARCH_INDEX_PREFIX', 'niffler')
        self.scheme = (scheme or _env('ELASTICSEARCH_SCHEME', 'http')).lower()
        self.timeout = float(
            timeout if timeout is not None else _env('ELASTICSEARCH_TIMEOUT', '30')
        )
        self.verify_certs = (
            verify_certs if verify_certs is not None
            else _env_flag('ELASTICSEARCH_VERIFY_CERTS', True)
        )

        # Credentials - never logged.
        self._api_key = api_key or _env('ELASTICSEARCH_API_KEY')
        self._username = username or _env('ELASTICSEARCH_USERNAME')
        self._password = password or _env('ELASTICSEARCH_PASSWORD')

        if self.scheme not in ('http', 'https'):
            raise ValueError(f"Invalid Elasticsearch scheme: {self.scheme}. Use 'http' or 'https'.")

        self.es_client = None

        # Index names
        self.backtests_index = f"{self.index_prefix}-backtests"
        self.portfolio_index = f"{self.index_prefix}-portfolio-values"
        self.trades_index = f"{self.index_prefix}-trades"
        self.positions_index = f"{self.index_prefix}-positions"

        # Optional elasticsearch dependency, resolved at import time
        if Elasticsearch is None:
            self.logger.error("elasticsearch package not installed. Run: pip install elasticsearch")
        self.Elasticsearch = Elasticsearch

    @property
    def url(self) -> str:
        """Connection URL for the configured cluster (never contains credentials)."""
        return f"{self.scheme}://{self.host}:{self.port}"

    def _auth_mode(self) -> str:
        """Describe the configured authentication mode without revealing secrets."""
        if self._api_key:
            return "api_key"
        if self._username and self._password:
            return "basic"
        return "none"

    def _build_client_kwargs(self) -> Dict[str, Any]:
        """
        Build the keyword arguments for the Elasticsearch client.

        Returns:
            Dictionary of client options including authentication, TLS and timeout
        """
        client_kwargs: Dict[str, Any] = {'request_timeout': self.timeout}

        if self._api_key:
            client_kwargs['api_key'] = self._api_key
        elif self._username and self._password:
            client_kwargs['basic_auth'] = (self._username, self._password)
        elif self._username or self._password:
            self.logger.warning(
                "Incomplete Elasticsearch basic-auth configuration: both "
                "ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD are required"
            )

        if self.scheme == 'https':
            client_kwargs['verify_certs'] = self.verify_certs

        return client_kwargs

    def _connect(self) -> bool:
        """Connect to Elasticsearch cluster."""
        if self.Elasticsearch is None:
            return False

        try:
            self.es_client = self.Elasticsearch([self.url], **self._build_client_kwargs())
            # Test connection
            if self.es_client.ping():
                self.logger.info(f"Connected to Elasticsearch at {self.url} (auth: {self._auth_mode()})")
                return True
            else:
                self.logger.error(f"Cannot connect to Elasticsearch at {self.url}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False

    def _bulk_index(self, actions: List[Dict[str, Any]]) -> None:
        """
        Send prepared bulk actions to Elasticsearch.

        Args:
            actions: Bulk actions to index

        Raises:
            RuntimeError: If the elasticsearch package is not installed
        """
        if bulk is None:
            raise RuntimeError(
                "elasticsearch package not installed. Run: pip install elasticsearch"
            )
        bulk(self.es_client, actions)

    def export_backtest_result(self, result: BacktestResult, backtest_id: str, 
                              metadata: Dict[str, Any]) -> None:
        """
        Export backtest results to Elasticsearch.
        
        Args:
            result: BacktestResult object containing all backtest data
            backtest_id: Unique identifier for this backtest run
            metadata: Additional metadata about the backtest

        Raises:
            ExportError: If the result is not exportable or the cluster is unreachable
            Exception: If indexing fails
        """
        self.require_valid_result(result, "Elasticsearch")

        if not self._connect():
            message = f"Cannot connect to Elasticsearch at {self.url}"
            self.logger.error(message)
            raise ExportError(message)


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

        Thin wrapper around the shared
        :func:`niffler.utils.json_utils.sanitize_numeric_values` helper, which
        converts Infinity and NaN to None.

        Args:
            data: Document to sanitize

        Returns:
            Document with all non-finite numbers replaced by None
        """
        return sanitize_numeric_values(data)

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
        self._bulk_index(actions)
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
                    # Optional on the Trade dataclass - read defensively.
                    "commission": getattr(trade, 'commission', 0.0),
                    "created_at": created_at
                }
            }
            actions.append(action)

        # Bulk insert
        self._bulk_index(actions)
        self.logger.debug(f"Exported {len(actions)} trades for {backtest_id}")

    def _export_positions(self, result: BacktestResult, backtest_id: str) -> None:
        """
        Export realised round trips (positions) with P&L calculations to Elasticsearch.

        Pairing goes through :func:`niffler.backtesting.round_trip.pair_trades`, the
        same FIFO routine the engine derives ``win_rate``, ``profit_factor`` and the
        win/loss counts from. Using a second, hand-rolled pairing here made the
        ``niffler-positions`` index contradict the metrics written for the same
        ``backtest_id``: it dropped every sell after the first, compared a partial
        exit's notional against the whole entry's, and ignored commission entirely.

        Each document carries entry/exit prices and timestamps, the matched quantity,
        P&L net of both commissions (absolute and as a percentage of the entry
        notional), the holding duration and a win/loss indicator.

        Args:
            result: BacktestResult whose trades are paired
            backtest_id: Unique identifier for this backtest run
        """
        if not result.trades:
            self.logger.info("No trades to pair into positions")
            return

        round_trips = pair_trades(result.trades)

        if not round_trips:
            self.logger.info("No complete positions to export (unpaired trades)")
            return

        created_at = datetime.now(UTC).isoformat()
        positions = []

        for counter, rt in enumerate(round_trips, start=1):
            entry_value = rt.entry_price * rt.quantity
            pnl = rt.pnl
            pnl_pct = (pnl / entry_value) * 100 if entry_value != 0 else 0.0

            duration = rt.exit_timestamp - rt.entry_timestamp
            duration_days = duration.total_seconds() / (24 * 3600)
            duration_hours = duration.total_seconds() / 3600

            positions.append({
                "backtest_id": backtest_id,
                "position_id": f"{backtest_id}-pos-{counter}",
                "symbol": rt.symbol,
                "entry_timestamp": rt.entry_timestamp.isoformat(),
                "exit_timestamp": rt.exit_timestamp.isoformat(),
                "entry_price": rt.entry_price,
                "exit_price": rt.exit_price,
                "quantity": rt.quantity,
                "entry_value": float(entry_value),
                "exit_value": float(rt.exit_price * rt.quantity),
                "entry_commission": float(rt.entry_commission),
                "exit_commission": float(rt.exit_commission),
                "gross_pnl": float(rt.gross_pnl),
                "pnl": float(pnl),
                "pnl_pct": float(pnl_pct),
                "duration_days": float(duration_days),
                "duration_hours": float(duration_hours),
                "is_win": rt.is_win,
                "created_at": created_at
            })

        # Prepare bulk data
        actions = []
        for position in positions:
            action = {
                "_index": self.positions_index,
                "_source": position
            }
            actions.append(action)

        # Bulk insert
        self._bulk_index(actions)
        self.logger.info(f"Exported {len(actions)} positions for {backtest_id}")

    def test_connection(self) -> bool:
        """Test connection to Elasticsearch."""
        return self._connect()
    
    def list_indices(self) -> List[str]:
        """
        List all indices with the configured prefix.

        An empty list means the cluster genuinely holds no matching index. A failure
        to reach or query the cluster raises instead, because returning ``[]`` for
        both makes "Elasticsearch is down" indistinguishable from "nothing indexed".

        Returns:
            Names of the existing indices carrying the configured prefix

        Raises:
            ExportError: If the cluster is unreachable or the query fails
        """
        if not self._connect():
            message = f"Cannot connect to Elasticsearch at {self.url}"
            self.logger.error(message)
            raise ExportError(message)

        try:
            indices = self.es_client.indices.get(index=f"{self.index_prefix}-*")
            return list(indices.keys())
        except Exception as e:
            self.logger.error(f"Failed to list indices: {e}")
            raise ExportError(f"Failed to list Elasticsearch indices: {e}") from e