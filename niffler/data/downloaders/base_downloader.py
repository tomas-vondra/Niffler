import pandas as pd
from abc import ABC, abstractmethod


class BaseDownloader(ABC):
    """
    Abstract base class for market data downloaders.
    All data downloaders should inherit from this class.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def download(self, **kwargs) -> pd.DataFrame:
        """Download market data.

        Implementations never signal failure with a falsy return value: they
        raise :class:`~niffler.data.exceptions.NoDataAvailableError` when the
        source genuinely has nothing, and
        :class:`~niffler.data.exceptions.DownloadError` when the download broke.

        Returns:
            DataFrame with the downloaded market data.
        """
        pass
        
    @abstractmethod
    def get_supported_timeframes(self) -> list:
        """Get list of supported timeframes."""
        pass
        
    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate that the timeframe is supported."""
        return timeframe in self.get_supported_timeframes()