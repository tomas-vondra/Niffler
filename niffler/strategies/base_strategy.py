from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd


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