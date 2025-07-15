import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors.
    All data preprocessors should inherit from this class.
    """
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the DataFrame and return the cleaned/validated version."""
        pass
        
    def can_process(self, df: pd.DataFrame) -> bool:
        """Check if this preprocessor can process the given DataFrame."""
        return not df.empty
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"