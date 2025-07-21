from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from niffler.backtesting.backtest_result import BacktestResult


@dataclass
class AnalysisResult:
    """
    Container for analysis results from Walk-forward or Monte Carlo analysis.
    """
    
    analysis_type: str  # 'walk_forward' or 'monte_carlo'
    strategy_name: str
    symbol: str
    analysis_start_date: datetime
    analysis_end_date: datetime
    
    # Core results
    individual_results: List[BacktestResult]
    combined_metrics: Dict[str, float]
    
    # Analysis-specific data
    analysis_parameters: Dict[str, Any]
    
    # Statistical measures
    stability_metrics: Dict[str, float]
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if not self.individual_results:
            raise ValueError("individual_results cannot be empty")
    
    @property
    def n_periods(self) -> int:
        """Number of analysis periods."""
        return len(self.individual_results)
    
    @property
    def total_returns(self) -> List[float]:
        """List of total returns from each period."""
        return [result.total_return for result in self.individual_results]
    
    @property
    def sharpe_ratios(self) -> List[float]:
        """List of Sharpe ratios from each period."""
        return [result.sharpe_ratio or 0.0 for result in self.individual_results]
    
    @property
    def max_drawdowns(self) -> List[float]:
        """List of max drawdowns from each period."""
        return [result.max_drawdown for result in self.individual_results]
    
    @property
    def win_rates(self) -> List[float]:
        """List of win rates from each period."""
        return [result.win_rate for result in self.individual_results]
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics across periods.
        
        Returns:
            Dictionary with statistics for each metric
        """
        metrics = {
            'total_return': self.total_returns,
            'sharpe_ratio': self.sharpe_ratios,
            'max_drawdown': self.max_drawdowns,
            'win_rate': self.win_rates
        }
        
        summary = {}
        for metric_name, values in metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': pd.Series(values).mean(),
                    'std': pd.Series(values).std(),
                    'min': min(values),
                    'max': max(values),
                    'median': pd.Series(values).median(),
                    'count': len(values)
                }
            else:
                summary[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 
                    'max': 0.0, 'median': 0.0, 'count': 0
                }
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame for analysis.
        
        Returns:
            DataFrame with one row per analysis period
        """
        data = []
        for i, result in enumerate(self.individual_results):
            row = {
                'period': i + 1,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'total_return': result.total_return,
                'total_return_pct': result.total_return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_performance_consistency(self) -> Dict[str, float]:
        """
        Calculate performance consistency metrics.
        
        Returns:
            Dictionary with consistency measures
        """
        returns = self.total_returns
        if not returns:
            return {'consistency_ratio': 0.0, 'positive_periods_pct': 0.0}
        
        positive_periods = sum(1 for r in returns if r > 0)
        positive_periods_pct = (positive_periods / len(returns)) * 100
        
        # Consistency ratio: mean return / std return (higher is better)
        mean_return = pd.Series(returns).mean()
        std_return = pd.Series(returns).std()
        consistency_ratio = mean_return / std_return if std_return > 0 else 0.0
        
        return {
            'consistency_ratio': consistency_ratio,
            'positive_periods_pct': positive_periods_pct
        }