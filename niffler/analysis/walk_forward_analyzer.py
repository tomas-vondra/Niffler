import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from niffler.strategies.base_strategy import BaseStrategy
from niffler.backtesting.backtest_engine import BacktestEngine
from .analysis_result import AnalysisResult


class WalkForwardAnalyzer:
    """
    Walk-forward analysis implementation for strategy validation.
    
    Walk-forward analysis tests strategy robustness by testing pre-optimized
    parameters on rolling time windows to validate out-of-sample performance
    across different market conditions and time periods.
    
    This analyzer assumes you have already run optimization (optimize.py) to find
    optimal parameters, and now want to test those parameters across different
    time periods to assess temporal robustness.
    """
    
    def __init__(self, 
                 strategy_class: Type[BaseStrategy],
                 optimal_parameters: Dict[str, Any],
                 test_window_months: int = 6,
                 step_months: int = 3,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 n_jobs: Optional[int] = None,
                 max_results_in_memory: int = 1000):
        """
        Initialize Walk-Forward Analyzer.
        
        Args:
            strategy_class: Strategy class to analyze
            optimal_parameters: Pre-optimized parameters from optimize.py
            test_window_months: Months of data for each test window
            step_months: Months to step forward between test windows
            initial_capital: Starting capital for backtests
            commission: Commission rate for trades
            n_jobs: Number of parallel jobs (None = auto-detect)
            max_results_in_memory: Maximum number of results to keep in memory
        """
        self.strategy_class = strategy_class
        self.optimal_parameters = optimal_parameters
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.initial_capital = initial_capital
        self.commission = commission
        self.n_jobs = n_jobs or min(mp.cpu_count(), 4)  # Default to max 4 processes
        self.max_results_in_memory = max_results_in_memory
        
        self._validate_parameters()
        self._validate_strategy_parameters()
        
        # Create reusable instances for better performance
        self._strategy = self.strategy_class(**self.optimal_parameters)
        self._backtest_engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission
        )
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not self.optimal_parameters:
            raise ValueError("optimal_parameters cannot be empty")
        if self.test_window_months <= 0:
            raise ValueError("test_window_months must be positive")
        if self.step_months <= 0:
            raise ValueError("step_months must be positive")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.commission < 0:
            raise ValueError("commission cannot be negative")
    
    def _validate_strategy_parameters(self) -> None:
        """Validate that optimal parameters are compatible with the strategy class."""
        try:
            # Try to create a strategy instance to validate parameters
            test_strategy = self.strategy_class(**self.optimal_parameters)
            logging.info("Strategy parameter validation successful")
        except Exception as e:
            raise ValueError(f"Invalid parameters for {self.strategy_class.__name__}: {e}")
    
    def analyze(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> AnalysisResult:
        """
        Perform walk-forward analysis on the given data.
        
        Args:
            data: Historical price data with OHLCV columns
            symbol: Symbol identifier
            
        Returns:
            AnalysisResult containing all walk-forward test results
        """
        if data.empty or len(data) < 100:
            raise ValueError("Insufficient data for walk-forward analysis")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Validate minimum data for meaningful analysis
        min_required_periods = max(2, (self.test_window_months * 30) // 7)  # Rough estimate
        if len(data) < min_required_periods:
            raise ValueError(f"Need at least {min_required_periods} data points for {self.test_window_months}-month windows")
        
        logging.info(f"Starting walk-forward analysis for {symbol}")
        logging.info(f"Using parameters: {self.optimal_parameters}")
        logging.info(f"Test window: {self.test_window_months} months")
        logging.info(f"Step size: {self.step_months} months")
        
        # Generate walk-forward periods
        periods = self._generate_walk_forward_periods(data)
        logging.info(f"Generated {len(periods)} walk-forward periods")
        
        if not periods:
            raise ValueError("No valid walk-forward periods found")
        
        # Run analysis for each period
        if self.n_jobs == 1:
            results = self._run_periods_sequential(data, symbol, periods)
        else:
            results = self._run_periods_parallel(data, symbol, periods)
        
        if not results:
            raise ValueError("No successful walk-forward periods")
        
        # Calculate combined metrics
        combined_metrics = self._calculate_combined_metrics(results)
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(results)
        
        return AnalysisResult(
            analysis_type='walk_forward',
            strategy_name=self.strategy_class.__name__,
            symbol=symbol,
            analysis_start_date=data.index[0],
            analysis_end_date=data.index[-1],
            individual_results=results,
            combined_metrics=combined_metrics,
            analysis_parameters={
                'optimal_parameters': self.optimal_parameters,
                'test_window_months': self.test_window_months,
                'step_months': self.step_months,
                'n_periods': len(results)
            },
            stability_metrics=stability_metrics,
            metadata={
                'test_periods': [(r.start_date, r.end_date) for r in results]
            }
        )
    
    def _run_periods_sequential(self, data: pd.DataFrame, symbol: str, periods: List[tuple]) -> List[Any]:
        """Run walk-forward periods sequentially with memory management."""
        results = []
        
        for i, (test_start, test_end) in enumerate(periods):
            if i % 10 == 0:
                logging.info(f"Testing period {i+1}/{len(periods)}: {test_start.date()} to {test_end.date()}")
                
                # Memory management: keep only the best results if we have too many
                if len(results) > self.max_results_in_memory:
                    results = self._manage_memory_efficient_results(results)
            
            try:
                period_result = self._run_single_period(data, test_start, test_end, symbol, i+1)
                
                if period_result:
                    results.append(period_result)
                
            except Exception as e:
                logging.warning(f"Error in period {i+1}: {e}")
                continue
        
        return results
    
    def _run_periods_parallel(self, data: pd.DataFrame, symbol: str, periods: List[tuple]) -> List[Any]:
        """Run walk-forward periods in parallel."""
        results = []
        failed_count = 0
        
        logging.info(f"Running {len(periods)} periods using {self.n_jobs} parallel jobs")
        
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all jobs
                future_to_period = {}
                for i, (test_start, test_end) in enumerate(periods):
                    try:
                        future = executor.submit(
                            self._run_single_period_static, 
                            data, test_start, test_end, symbol, i+1,
                            self.strategy_class, self.optimal_parameters,
                            self.initial_capital, self.commission
                        )
                        future_to_period[future] = (i+1, test_start, test_end)
                    except Exception as e:
                        logging.warning(f"Failed to submit period {i+1}: {e}")
                        failed_count += 1
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_period):
                    period_num, test_start, test_end = future_to_period[future]
                    completed += 1
                    
                    if completed % 10 == 0:
                        logging.info(f"Completed {completed}/{len(periods)} periods")
                    
                    try:
                        result = future.result(timeout=60)  # 1 minute timeout per period
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logging.warning(f"Period {period_num} ({test_start.date()}-{test_end.date()}) failed: {e}")
                        failed_count += 1
                        
        except Exception as e:
            logging.error(f"Critical error in parallel walk-forward: {e}")
            raise
        
        if failed_count > 0:
            success_rate = (len(results) / len(periods)) * 100
            logging.warning(f"Parallel walk-forward completed with {failed_count} failures ({success_rate:.1f}% success rate)")
        
        return results
    
    def _manage_memory_efficient_results(self, results: List[Any]) -> List[Any]:
        """Manage results list to prevent excessive memory usage by keeping only the best results."""
        if len(results) <= self.max_results_in_memory:
            return results
        
        logging.info(f"Memory management: trimming {len(results)} results to {self.max_results_in_memory // 2}")
        
        # Sort by total return (descending - best first)
        results.sort(key=lambda r: r.total_return, reverse=True)
        
        # Keep only the best half
        keep_count = self.max_results_in_memory // 2
        return results[:keep_count]
    
    @staticmethod
    def _run_single_period_static(data: pd.DataFrame, test_start: datetime, test_end: datetime,
                                 symbol: str, period_number: int, strategy_class: Type[BaseStrategy],
                                 optimal_parameters: Dict[str, Any], initial_capital: float,
                                 commission: float) -> Optional[Any]:
        """Static method for parallel processing (must be picklable)."""
        try:
            # Create analyzer instance for this process
            analyzer = WalkForwardAnalyzer(
                strategy_class=strategy_class,
                optimal_parameters=optimal_parameters,
                test_window_months=1,  # Not used in static method
                step_months=1,  # Not used in static method
                initial_capital=initial_capital,
                commission=commission,
                n_jobs=1  # Single job for static method
            )
            
            # Extract test data
            test_data = data[(data.index >= test_start) & (data.index < test_end)].copy()
            
            if len(test_data) < 30:
                return None
            
            # Run backtest
            result = analyzer._backtest_engine.run_backtest(analyzer._strategy, test_data, symbol)
            
            # Add period metadata
            result.metadata = {
                'period_number': period_number,
                'parameters_used': optimal_parameters,
                'test_start': test_start,
                'test_end': test_end,
                'test_data_points': len(test_data)
            }
            
            return result
            
        except Exception as e:
            logging.debug(f"Static period {period_number} failed: {e}")
            return None
    
    def _generate_walk_forward_periods(self, data: pd.DataFrame) -> List[tuple]:
        """Generate test periods for walk-forward analysis."""
        periods = []
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_date = start_date
        
        while True:
            # Calculate test window
            test_start = current_date
            test_end = test_start + relativedelta(months=self.test_window_months)
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            # Ensure we have actual data in the test window
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            if len(test_data) >= 30:  # Minimum data requirement
                periods.append((test_start, test_end))
            
            # Step forward
            current_date += relativedelta(months=self.step_months)
        
        return periods
    
    def _run_single_period(self, data: pd.DataFrame, 
                          test_start: datetime, test_end: datetime,
                          symbol: str, period_number: int) -> Optional[Any]:
        """Run testing for a single walk-forward period."""
        
        # Extract test data  
        test_data = data[(data.index >= test_start) & (data.index < test_end)].copy()
        if len(test_data) < 30:
            logging.warning(f"Insufficient test data: {len(test_data)} rows")
            return None
        
        try:
            # Use pre-created strategy and backtest engine instances for better performance
            result = self._backtest_engine.run_backtest(self._strategy, test_data, symbol)
            
            # Add metadata about the test period
            result.metadata = {
                'period_number': period_number,
                'parameters_used': self.optimal_parameters,
                'test_start': test_start,
                'test_end': test_end,
                'test_data_points': len(test_data)
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error in walk-forward period {period_number}: {e}")
            return None
    
    def _calculate_combined_metrics(self, results: List[Any]) -> Dict[str, float]:
        """Calculate combined metrics across all walk-forward periods."""
        if not results:
            return {}
        
        # Aggregate returns and performance metrics
        returns = [r.total_return for r in results]
        return_pcts = [r.total_return_pct for r in results]
        sharpe_ratios = [r.sharpe_ratio or 0.0 for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]
        total_trades = [r.total_trades for r in results]
        
        # Calculate combined portfolio performance
        combined_returns = []
        for result in results:
            if hasattr(result, 'portfolio_values') and result.portfolio_values is not None:
                period_returns = result.portfolio_values.pct_change().dropna()
                combined_returns.extend(period_returns.tolist())
        
        combined_sharpe = 0.0
        if combined_returns:
            combined_returns_series = pd.Series(combined_returns)
            if combined_returns_series.std() > 0:
                combined_sharpe = np.sqrt(252) * combined_returns_series.mean() / combined_returns_series.std()
        
        metrics = {
            'total_periods': len(results),
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'avg_return_pct': np.mean(return_pcts),
            'median_return_pct': np.median(return_pcts),
            'std_return_pct': np.std(return_pcts),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'combined_sharpe_ratio': combined_sharpe,
            'avg_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': max(max_drawdowns) if max_drawdowns else 0.0,
            'best_max_drawdown': min(max_drawdowns) if max_drawdowns else 0.0,
            'avg_win_rate': np.mean(win_rates),
            'avg_trades_per_period': np.mean(total_trades),
            'positive_return_periods': sum(1 for r in returns if r > 0),
            'positive_return_pct': (sum(1 for r in returns if r > 0) / len(returns)) * 100,
            'profitable_periods': sum(1 for r in return_pcts if r > 0),
            'profitable_periods_pct': (sum(1 for r in return_pcts if r > 0) / len(return_pcts)) * 100
        }
        
        return metrics
    
    def _calculate_stability_metrics(self, results: List[Any]) -> Dict[str, float]:
        """Calculate metrics measuring performance stability across periods."""
        if not results:
            return {}
        
        # Performance stability metrics
        returns = [r.total_return for r in results]
        return_pcts = [r.total_return_pct for r in results]
        sharpe_ratios = [r.sharpe_ratio or 0.0 for r in results]
        
        stability_metrics = {
            'return_volatility': np.std(returns),
            'return_pct_volatility': np.std(return_pcts),
            'sharpe_volatility': np.std(sharpe_ratios),
            'return_consistency': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0,
            'return_pct_consistency': np.mean(return_pcts) / np.std(return_pcts) if np.std(return_pcts) > 0 else 0.0,
            'temporal_stability': self._calculate_temporal_stability(results)
        }
        
        # Calculate rolling correlations if we have enough periods
        if len(results) >= 4:
            stability_metrics.update(self._calculate_rolling_stability(results))
        
        return stability_metrics
    
    def _calculate_temporal_stability(self, results: List[Any]) -> float:
        """Calculate temporal stability - how consistent performance is over time."""
        if len(results) < 3:
            return 1.0
        
        returns = [r.total_return for r in results]
        
        # Calculate period-to-period changes
        changes = [returns[i+1] - returns[i] for i in range(len(returns) - 1)]
        
        # Count direction changes (positive to negative or vice versa)
        direction_changes = 0
        for i in range(len(changes) - 1):
            if (changes[i] > 0) != (changes[i+1] > 0):
                direction_changes += 1
        
        # Calculate temporal stability (lower direction changes = higher stability)
        max_possible_changes = len(changes) - 1
        if max_possible_changes > 0:
            stability = 1.0 - (direction_changes / max_possible_changes)
        else:
            stability = 1.0
        
        return stability
    
    def _calculate_rolling_stability(self, results: List[Any]) -> Dict[str, float]:
        """Calculate rolling performance stability metrics."""
        returns = [r.total_return for r in results]
        
        # Calculate rolling average stability
        window_size = min(4, len(returns) // 2)
        rolling_means = []
        
        for i in range(len(returns) - window_size + 1):
            window_returns = returns[i:i + window_size]
            rolling_means.append(np.mean(window_returns))
        
        rolling_stability = {
            'rolling_mean_stability': 1.0 / (1.0 + np.std(rolling_means)) if len(rolling_means) > 1 else 1.0,
            'trend_consistency': self._calculate_trend_consistency(returns)
        }
        
        return rolling_stability
    
    def _calculate_trend_consistency(self, returns: List[float]) -> float:
        """Calculate how consistent the performance trend is across periods."""
        if len(returns) < 3:
            return 1.0
        
        # Calculate period-to-period changes
        changes = [returns[i+1] - returns[i] for i in range(len(returns) - 1)]
        
        # Count direction changes (positive to negative or vice versa)
        direction_changes = 0
        for i in range(len(changes) - 1):
            if (changes[i] > 0) != (changes[i+1] > 0):
                direction_changes += 1
        
        # Calculate trend consistency (lower direction changes = higher consistency)
        max_possible_changes = len(changes) - 1
        if max_possible_changes > 0:
            consistency = 1.0 - (direction_changes / max_possible_changes)
        else:
            consistency = 1.0
        
        return consistency