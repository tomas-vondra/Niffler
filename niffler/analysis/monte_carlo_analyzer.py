import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from niffler.strategies.base_strategy import BaseStrategy
from niffler.backtesting.backtest_engine import BacktestEngine
from .analysis_result import AnalysisResult


class MonteCarloAnalyzer:
    """
    Monte Carlo analysis implementation for strategy robustness testing.
    
    Monte Carlo analysis tests strategy robustness by:
    1. Using fixed optimal parameters (no parameter uncertainty)
    2. Running backtests with block bootstrap sampling of historical data
    3. Analyzing the distribution of results to assess strategy reliability
    """
    
    def __init__(self,
                 strategy_class: Type[BaseStrategy],
                 optimal_parameters: Dict[str, Any],
                 n_simulations: int = 1000,
                 bootstrap_sample_pct: float = 0.8,
                 block_size_days: int = 30,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 n_jobs: Optional[int] = None,
                 max_results_in_memory: int = 10000,
                 random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo Analyzer.
        
        Args:
            strategy_class: Strategy class to analyze
            optimal_parameters: Pre-optimized parameters from optimize.py
            n_simulations: Number of Monte Carlo simulations to run
            bootstrap_sample_pct: Percentage of data to sample in each simulation
            block_size_days: Size of blocks for block bootstrap (days)
            initial_capital: Starting capital for backtests
            commission: Commission rate for trades
            n_jobs: Number of parallel jobs (None = auto-detect)
            max_results_in_memory: Maximum number of results to keep in memory
            random_seed: Random seed for reproducibility
        """
        self.strategy_class = strategy_class
        self.optimal_parameters = optimal_parameters
        self.n_simulations = n_simulations
        self.bootstrap_sample_pct = bootstrap_sample_pct
        self.block_size_days = block_size_days
        self.initial_capital = initial_capital
        self.commission = commission
        self.n_jobs = n_jobs or min(mp.cpu_count(), 4)  # Default to max 4 processes
        self.max_results_in_memory = max_results_in_memory
        
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
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
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if not 0.1 <= self.bootstrap_sample_pct <= 1.0:
            raise ValueError("bootstrap_sample_pct must be between 0.1 and 1.0")
        if self.block_size_days <= 0:
            raise ValueError("block_size_days must be positive")
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
        Perform Monte Carlo analysis on the given data.
        
        Args:
            data: Historical price data with OHLCV columns
            symbol: Symbol identifier
            
        Returns:
            AnalysisResult containing all Monte Carlo simulation results
        """
        if data.empty or len(data) < 100:
            raise ValueError("Insufficient data for Monte Carlo analysis")
        
        # Validate block size relative to data length
        if self.block_size_days >= len(data):
            logging.warning(f"Block size ({self.block_size_days}) >= data length ({len(data)}). Using smaller block size.")
            # Use a reasonable fraction of data length
            self.block_size_days = max(1, len(data) // 4)
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        logging.info(f"Starting Monte Carlo analysis for {symbol}")
        logging.info(f"Using parameters: {self.optimal_parameters}")
        logging.info(f"Simulations: {self.n_simulations}")
        logging.info(f"Bootstrap sample: {self.bootstrap_sample_pct*100:.1f}%")
        logging.info(f"Block size: {self.block_size_days} days")
        
        # Run Monte Carlo simulations
        if self.n_jobs == 1:
            results = self._run_simulations_sequential(data, symbol)
        else:
            results = self._run_simulations_parallel(data, symbol)
        
        if not results:
            raise ValueError("No successful Monte Carlo simulations")
        
        logging.info(f"Completed {len(results)}/{self.n_simulations} successful simulations")
        
        # Calculate combined metrics
        combined_metrics = self._calculate_combined_metrics(results)
        
        # Calculate distribution statistics
        distribution_stats = self._calculate_distribution_statistics(results)
        
        return AnalysisResult(
            analysis_type='monte_carlo',
            strategy_name=self.strategy_class.__name__,
            symbol=symbol,
            analysis_start_date=data.index[0],
            analysis_end_date=data.index[-1],
            individual_results=results,
            combined_metrics=combined_metrics,
            analysis_parameters={
                'optimal_parameters': self.optimal_parameters,
                'n_simulations': len(results),
                'bootstrap_sample_pct': self.bootstrap_sample_pct,
                'block_size_days': self.block_size_days,
                'success_rate': len(results) / self.n_simulations
            },
            stability_metrics=distribution_stats,
            metadata={
                'simulation_details': [r.metadata for r in results if hasattr(r, 'metadata')]
            }
        )
    
    def _run_simulations_sequential(self, data: pd.DataFrame, symbol: str) -> List[Any]:
        """Run simulations sequentially with memory management."""
        results = []
        
        for i in range(self.n_simulations):
            if i % 100 == 0:
                logging.info(f"Running simulation {i+1}/{self.n_simulations}")
                
                # Memory management: keep only the best results if we have too many
                if len(results) > self.max_results_in_memory:
                    results = self._manage_memory_efficient_results(results)
            
            try:
                simulation_result = self._run_single_simulation(data, symbol, i)
                
                if simulation_result:
                    results.append(simulation_result)
                
            except Exception as e:
                logging.warning(f"Error in simulation {i+1}: {e}")
                continue
        
        return results
    
    def _run_simulations_parallel(self, data: pd.DataFrame, symbol: str) -> List[Any]:
        """Run simulations in parallel."""
        results = []
        failed_count = 0
        
        logging.info(f"Running {self.n_simulations} simulations using {self.n_jobs} parallel jobs")
        
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all jobs
                future_to_sim_id = {}
                for i in range(self.n_simulations):
                    try:
                        future = executor.submit(
                            self._run_single_simulation_static, 
                            data, symbol, i, self.strategy_class, self.optimal_parameters,
                            self.bootstrap_sample_pct, self.block_size_days,
                            self.initial_capital, self.commission
                        )
                        future_to_sim_id[future] = i
                    except Exception as e:
                        logging.warning(f"Failed to submit simulation {i}: {e}")
                        failed_count += 1
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_sim_id):
                    sim_id = future_to_sim_id[future]
                    completed += 1
                    
                    if completed % 100 == 0:
                        logging.info(f"Completed {completed}/{self.n_simulations} simulations")
                    
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per simulation
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logging.warning(f"Simulation {sim_id} failed: {e}")
                        failed_count += 1
                        
        except Exception as e:
            logging.error(f"Critical error in parallel simulation: {e}")
            raise
        
        if failed_count > 0:
            success_rate = (len(results) / self.n_simulations) * 100
            logging.warning(f"Parallel simulation completed with {failed_count} failures ({success_rate:.1f}% success rate)")
        
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
    def _run_single_simulation_static(data: pd.DataFrame, symbol: str, sim_id: int,
                                     strategy_class: Type[BaseStrategy], optimal_parameters: Dict[str, Any],
                                     bootstrap_sample_pct: float, block_size_days: int,
                                     initial_capital: float, commission: float) -> Optional[Any]:
        """Static method for parallel processing (must be picklable)."""
        try:
            # Create analyzer instance for this process
            analyzer = MonteCarloAnalyzer(
                strategy_class=strategy_class,
                optimal_parameters=optimal_parameters,
                n_simulations=1,  # Not used in static method
                bootstrap_sample_pct=bootstrap_sample_pct,
                block_size_days=block_size_days,
                initial_capital=initial_capital,
                commission=commission,
                n_jobs=1  # Single job for static method
            )
            
            # Run the simulation
            sampled_data = analyzer._block_bootstrap_sample(data)
            
            if len(sampled_data) < 50:
                return None
            
            result = analyzer._backtest_engine.run_backtest(analyzer._strategy, sampled_data, symbol)
            
            # Add simulation metadata
            result.metadata = {
                'simulation_id': sim_id,
                'parameters_used': optimal_parameters,
                'sample_size': len(sampled_data),
                'sample_start': sampled_data.index[0],
                'sample_end': sampled_data.index[-1],
                'original_data_coverage': len(sampled_data) / len(data)
            }
            
            return result
            
        except Exception as e:
            logging.debug(f"Static simulation {sim_id} failed: {e}")
            return None
    
    def _run_single_simulation(self, data: pd.DataFrame, symbol: str, sim_id: int) -> Optional[Any]:
        """Run a single Monte Carlo simulation."""
        
        # Block bootstrap sample the data
        sampled_data = self._block_bootstrap_sample(data)
        
        if len(sampled_data) < 50:  # Minimum data requirement
            return None
        
        try:
            # Use pre-created strategy and backtest engine instances for better performance
            result = self._backtest_engine.run_backtest(self._strategy, sampled_data, symbol)
            
            # Add simulation metadata
            result.metadata = {
                'simulation_id': sim_id,
                'parameters_used': self.optimal_parameters,
                'sample_size': len(sampled_data),
                'sample_start': sampled_data.index[0],
                'sample_end': sampled_data.index[-1],
                'original_data_coverage': len(sampled_data) / len(data)
            }
            
            return result
            
        except Exception as e:
            logging.debug(f"Simulation {sim_id} failed: {e}")
            return None
    
    def _block_bootstrap_sample(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform block bootstrap sampling to preserve time series structure.
        
        Block bootstrap maintains temporal dependencies by sampling consecutive blocks
        of data rather than individual observations.
        """
        n_samples = int(len(data) * self.bootstrap_sample_pct)
        
        if len(data) < self.block_size_days:
            # If data is shorter than block size, return simple sample
            return data.sample(n=min(n_samples, len(data)))
        
        blocks = []
        n_blocks_needed = max(1, n_samples // self.block_size_days)
        
        for _ in range(n_blocks_needed):
            # Choose random starting point
            max_start = len(data) - self.block_size_days
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + self.block_size_days
            
            block = data.iloc[start_idx:end_idx].copy()
            blocks.append(block)
        
        # Concatenate blocks
        sampled_data = pd.concat(blocks, ignore_index=False)
        
        # Trim to exact sample size
        if len(sampled_data) > n_samples:
            sampled_data = sampled_data.iloc[:n_samples]
        
        # Sort by index to maintain chronological order
        sampled_data = sampled_data.sort_index()
        
        return sampled_data
    
    def _calculate_combined_metrics(self, results: List[Any]) -> Dict[str, float]:
        """Calculate combined metrics across all Monte Carlo simulations."""
        if not results:
            return {}
        
        returns = [r.total_return for r in results]
        return_pcts = [r.total_return_pct for r in results]
        sharpe_ratios = [r.sharpe_ratio or 0.0 for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]
        total_trades = [r.total_trades for r in results]
        
        return {
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'mean_return_pct': np.mean(return_pcts),
            'median_return_pct': np.median(return_pcts),
            'std_return_pct': np.std(return_pcts),
            'mean_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': max(max_drawdowns) if max_drawdowns else 0.0,
            'best_max_drawdown': min(max_drawdowns) if max_drawdowns else 0.0,
            'mean_win_rate': np.mean(win_rates),
            'mean_trades_per_simulation': np.mean(total_trades),
            'positive_return_simulations': sum(1 for r in returns if r > 0),
            'positive_return_pct': (sum(1 for r in returns if r > 0) / len(returns)) * 100,
            'profitable_simulations': sum(1 for r in return_pcts if r > 0),
            'profitable_simulations_pct': (sum(1 for r in return_pcts if r > 0) / len(return_pcts)) * 100,
            'total_simulations': len(results)
        }
    
    def _calculate_distribution_statistics(self, results: List[Any]) -> Dict[str, float]:
        """Calculate distribution statistics for performance metrics."""
        if not results:
            return {}
        
        returns = np.array([r.total_return for r in results])
        return_pcts = np.array([r.total_return_pct for r in results])
        sharpe_ratios = np.array([r.sharpe_ratio or 0.0 for r in results])
        
        stats = {}
        
        # Value at Risk (VaR) and Conditional VaR for absolute returns
        if len(returns) > 0:
            stats.update({
                'return_var_5pct': np.percentile(returns, 5),
                'return_var_1pct': np.percentile(returns, 1),
                'return_cvar_5pct': np.mean(returns[returns <= np.percentile(returns, 5)]),
                'return_cvar_1pct': np.mean(returns[returns <= np.percentile(returns, 1)]),
                'return_skewness': self._calculate_skewness(returns),
                'return_kurtosis': self._calculate_kurtosis(returns)
            })
        
        # Value at Risk (VaR) and Conditional VaR for percentage returns
        if len(return_pcts) > 0:
            stats.update({
                'return_pct_var_5pct': np.percentile(return_pcts, 5),
                'return_pct_var_1pct': np.percentile(return_pcts, 1),
                'return_pct_cvar_5pct': np.mean(return_pcts[return_pcts <= np.percentile(return_pcts, 5)]),
                'return_pct_cvar_1pct': np.mean(return_pcts[return_pcts <= np.percentile(return_pcts, 1)]),
                'return_pct_skewness': self._calculate_skewness(return_pcts),
                'return_pct_kurtosis': self._calculate_kurtosis(return_pcts)
            })
        
        # Sharpe ratio distribution
        if len(sharpe_ratios) > 0:
            stats.update({
                'sharpe_skewness': self._calculate_skewness(sharpe_ratios),
                'sharpe_kurtosis': self._calculate_kurtosis(sharpe_ratios)
            })
        
        # Confidence intervals for returns
        confidence_levels = [0.90, 0.95, 0.99]
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_pct = (alpha / 2) * 100
            upper_pct = (1 - alpha / 2) * 100
            
            stats[f'return_ci_{int(conf_level*100)}_lower'] = np.percentile(returns, lower_pct)
            stats[f'return_ci_{int(conf_level*100)}_upper'] = np.percentile(returns, upper_pct)
            
            stats[f'return_pct_ci_{int(conf_level*100)}_lower'] = np.percentile(return_pcts, lower_pct)
            stats[f'return_pct_ci_{int(conf_level*100)}_upper'] = np.percentile(return_pcts, upper_pct)
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        n = len(data)
        skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        n = len(data)
        kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4)
        kurtosis -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))  # Excess kurtosis
        return kurtosis
    
    def get_percentile_results(self, results: List[Any], percentiles: List[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Get percentile statistics for key metrics.
        
        Args:
            results: List of backtest results
            percentiles: List of percentiles to calculate (default: [5, 25, 50, 75, 95])
            
        Returns:
            Dictionary with percentile statistics for each metric
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        
        if not results:
            return {}
        
        metrics = {
            'total_return': [r.total_return for r in results],
            'total_return_pct': [r.total_return_pct for r in results],
            'sharpe_ratio': [r.sharpe_ratio or 0.0 for r in results],
            'max_drawdown': [r.max_drawdown for r in results],
            'win_rate': [r.win_rate for r in results]
        }
        
        percentile_stats = {}
        
        for metric_name, values in metrics.items():
            percentile_stats[metric_name] = {}
            for p in percentiles:
                percentile_stats[metric_name][f'p{p}'] = np.percentile(values, p)
        
        return percentile_stats