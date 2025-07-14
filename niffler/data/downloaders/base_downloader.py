import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod


class BaseDownloader(ABC):
    """
    Abstract base class for market data downloaders.
    All data downloaders should inherit from this class.
    """
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def download(self, **kwargs) -> Optional[pd.DataFrame]:
        """Download market data."""
        pass
        
    @abstractmethod
    def get_supported_timeframes(self) -> list:
        """Get list of supported timeframes."""
        pass
        
    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate that the timeframe is supported."""
        return timeframe in self.get_supported_timeframes()