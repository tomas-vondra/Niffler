from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd

#: Optional exit columns a strategy may add next to ``signal``. Each is read on
#: the bar of a buy signal, NaN meaning "not set"; the engine's handling is
#: documented in :mod:`niffler.backtesting.exits`.
STOP_PRICE_COLUMN = 'stop_price'
TAKE_PROFIT_COLUMN = 'take_profit_price'
TRAILING_STOP_COLUMN = 'trailing_stop_pct'
MAX_BARS_HELD_COLUMN = 'max_bars_held'

EXIT_COLUMNS = (STOP_PRICE_COLUMN, TAKE_PROFIT_COLUMN, TRAILING_STOP_COLUMN,
                MAX_BARS_HELD_COLUMN)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    All trading strategies should inherit from this class.
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None, 
                 risk_manager=None):
        self.name = name
        self.parameters = parameters or {}
        self.risk_manager = risk_manager
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the input data.
        
        Args:
            data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                 with timestamp as index
                 
        Returns:
            DataFrame with same index as input data and additional columns:
            - 'signal': 1 for buy, -1 for sell, 0 for hold
            - 'position_size': fraction of portfolio to allocate (0.0 to 1.0)

            and, optionally, any of the exit columns in ``EXIT_COLUMNS``:
            - 'stop_price': absolute stop for the position a buy opens; on a
              later bar while it is open, a value moves the stop up (never down)
            - 'take_profit_price': absolute target for that position
            - 'trailing_stop_pct': stop trailing the highest high since entry
              by this fraction (0 < pct < 1)
            - 'max_bars_held': leave at the open this many bars after entry
        """
        pass
        
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of the strategy."""
        pass
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has the required format.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            return False
            
        if data.empty:
            return False
            
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
            
        return True