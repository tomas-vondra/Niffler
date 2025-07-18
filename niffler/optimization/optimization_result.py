from dataclasses import dataclass
from typing import Dict, Any
from niffler.backtesting.backtest_result import BacktestResult


@dataclass
class OptimizationResult:
    """Result of a single parameter combination optimization."""
    parameters: Dict[str, Any]
    backtest_result: BacktestResult
    
    def __post_init__(self) -> None:
        """Validate the optimization result after initialization."""
        if not isinstance(self.parameters, dict):
            raise TypeError("parameters must be a dictionary")
        if self.backtest_result is None:
            raise ValueError("backtest_result cannot be None")