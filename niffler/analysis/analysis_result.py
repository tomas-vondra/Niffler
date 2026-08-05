from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
from datetime import datetime

from niffler.backtesting.backtest_result import BacktestResult


# Fraction of failed runs above which analyzers emit a loud WARNING. A run where a
# noticeable share of simulations/folds silently died is survivorship-biased and its
# distribution must not be read as if every attempt had succeeded.
FAILURE_RATE_WARNING_THRESHOLD = 0.05


def log_failure_rate(context: str,
                     attempted: int,
                     failed: int,
                     threshold: float = FAILURE_RATE_WARNING_THRESHOLD) -> float:
    """
    Compute the failure rate of an analysis run and log it at an appropriate level.

    Args:
        context: Human readable description of what was run (used in the log message)
        attempted: Number of runs that were attempted
        failed: Number of runs that failed or were discarded
        threshold: Failure rate (0-1) above which a WARNING is emitted

    Returns:
        Failure rate as a fraction between 0.0 and 1.0 (0.0 when nothing was attempted)
    """
    if attempted <= 0:
        return 0.0

    failure_rate = failed / attempted

    if failure_rate > threshold:
        logging.warning(
            f"{context}: {failed}/{attempted} runs failed or were discarded "
            f"({failure_rate * 100:.1f}%). Reported metrics cover only the "
            f"{attempted - failed} surviving runs and are survivorship-biased."
        )
    elif failed > 0:
        logging.info(
            f"{context}: {failed}/{attempted} runs failed or were discarded "
            f"({failure_rate * 100:.1f}%)."
        )

    return failure_rate


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

    # Survivorship accounting: how many runs were attempted and how many were lost.
    # Without these, a run where most simulations failed still reports a confident
    # looking distribution over the handful of survivors.
    attempted_runs: int = 0
    failed_runs: int = 0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if not self.individual_results:
            raise ValueError("individual_results cannot be empty")

        if self.attempted_runs <= 0:
            # Backwards compatible default: assume every result came from its own attempt.
            self.attempted_runs = len(self.individual_results) + self.failed_runs

    @property
    def successful_runs(self) -> int:
        """Number of runs that produced a usable result."""
        return len(self.individual_results)

    @property
    def failure_rate(self) -> float:
        """Fraction (0-1) of attempted runs that failed or were discarded."""
        if self.attempted_runs <= 0:
            return 0.0
        return self.failed_runs / self.attempted_runs

    @property
    def is_survivorship_biased(self) -> bool:
        """True when enough runs were lost that the distribution cannot be trusted."""
        return self.failure_rate > FAILURE_RATE_WARNING_THRESHOLD

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