import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Type
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from niffler.strategies.base_strategy import BaseStrategy
from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.cost_model import CostModel
from .analysis_result import AnalysisResult, log_failure_rate


# Minimum number of bars a bootstrapped path must contain to be worth backtesting.
MIN_SAMPLE_BARS = 50


class MonteCarloAnalyzer:
    """
    Monte Carlo analysis implementation for strategy robustness testing.

    Monte Carlo analysis tests strategy robustness by:
    1. Using fixed optimal parameters (no parameter uncertainty)
    2. Building synthetic price paths by block-bootstrapping the historical *return*
       series and compounding the resampled returns from the real starting price
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
                 random_seed: Optional[int] = None,
                 cost_model: Optional[CostModel] = None):
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
            random_seed: Base random seed. Each simulation derives its own seed from it
                (``random_seed + simulation_id``) so sequential and parallel runs - and
                therefore Windows spawn-based workers - reproduce identical results.
            cost_model: Transaction cost model applied to every fill of every
                simulated path, so the distribution describes the market the
                strategy would actually trade in.
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
        self.random_seed = random_seed
        self.cost_model = cost_model

        # Seed the global generators too, so any strategy code that reaches for them is
        # reproducible as well. The bootstrap itself uses the explicit generator below.
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed % (2 ** 32))
        self._rng = np.random.default_rng(random_seed)

        self._validate_parameters()
        self._validate_strategy_parameters()
        
        # Create reusable instances for better performance
        self._strategy = self.strategy_class(**self.optimal_parameters)
        self._backtest_engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
            cost_model=self.cost_model
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
            self.strategy_class(**self.optimal_parameters)
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
        
        # Validate block size relative to data length. A block that covers most of
        # the sample leaves almost no distinct start positions, so every simulation
        # replays the same contiguous history and the "distribution" collapses to a
        # single point while still being reported as n_simulations draws.
        max_usable_block = max(1, (len(data) - 1) // 2)
        if self.block_size_days > max_usable_block:
            logging.warning(
                f"Block size ({self.block_size_days}) leaves too few distinct block start "
                f"positions for {len(data)} bars: every simulation would replay nearly the "
                f"same path. Reducing it to {max_usable_block}."
            )
            self.block_size_days = max_usable_block
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        logging.info(f"Starting Monte Carlo analysis for {symbol}")
        logging.info(f"Using parameters: {self.optimal_parameters}")
        logging.info(f"Simulations: {self.n_simulations}")
        logging.info(f"Bootstrap sample: {self.bootstrap_sample_pct*100:.1f}%")
        logging.info(f"Block size: {self.block_size_days} days")
        
        # Run Monte Carlo simulations
        if self.n_jobs == 1:
            results, failed = self._run_simulations_sequential(data, symbol)
        else:
            results, failed = self._run_simulations_parallel(data, symbol)

        attempted = self.n_simulations
        failure_rate = log_failure_rate(
            f"Monte Carlo analysis for {symbol}", attempted, failed
        )

        if not results:
            raise ValueError(
                f"No successful Monte Carlo simulations ({failed}/{attempted} failed)"
            )

        logging.info(f"Completed {len(results)}/{attempted} successful simulations")

        # Calculate combined metrics
        combined_metrics = self._calculate_combined_metrics(results)
        combined_metrics.update({
            'attempted_simulations': float(attempted),
            'failed_simulations': float(failed),
            'failure_rate_pct': failure_rate * 100.0,
        })

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
                'attempted_simulations': attempted,
                'failed_simulations': failed,
                'failure_rate': failure_rate,
                'bootstrap_sample_pct': self.bootstrap_sample_pct,
                'block_size_days': self.block_size_days,
                'random_seed': self.random_seed,
                'cost_model': (self.cost_model.description
                               if self.cost_model is not None else None),
                'success_rate': len(results) / attempted
            },
            stability_metrics=distribution_stats,
            metadata={
                'simulation_details': [r.metadata for r in results if hasattr(r, 'metadata')]
            },
            attempted_runs=attempted,
            failed_runs=failed,
        )

    def _run_simulations_sequential(self, data: pd.DataFrame, symbol: str) -> Tuple[List[Any], int]:
        """
        Run simulations sequentially with memory management.

        Args:
            data: Historical price data
            symbol: Symbol identifier

        Returns:
            Tuple of (successful results, number of failed or discarded simulations)
        """
        results = []
        failed_count = 0

        for i in range(self.n_simulations):
            if i % 100 == 0:
                logging.info(f"Running simulation {i+1}/{self.n_simulations}")

                # Memory management: bound the number of retained results
                if len(results) > self.max_results_in_memory:
                    results = self._manage_memory_efficient_results(results)

            try:
                simulation_result = self._run_single_simulation(data, symbol, i)
            except Exception as e:
                logging.warning(f"Error in simulation {i+1}: {e}")
                failed_count += 1
                continue

            if simulation_result is None:
                failed_count += 1
                continue

            results.append(simulation_result)

        return results, failed_count

    def _run_simulations_parallel(self, data: pd.DataFrame, symbol: str) -> Tuple[List[Any], int]:
        """
        Run simulations in parallel.

        Args:
            data: Historical price data
            symbol: Symbol identifier

        Results are re-ordered by simulation id before being returned. ``as_completed``
        yields futures in completion order, which varies between runs, so collecting
        straight into a list would make a seeded parallel run non-reproducible: the
        reported per-simulation table would be shuffled, floating-point aggregates would
        differ in their last digits, and ``_manage_memory_efficient_results`` would retain
        an arbitrary subset on large runs.

        Returns:
            Tuple of (successful results ordered by simulation id, number of failed or
            discarded simulations)
        """
        completed_by_sim_id: Dict[int, Any] = {}
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
                            self.initial_capital, self.commission,
                            self._simulation_seed(i), self.cost_model
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
                            completed_by_sim_id[sim_id] = result
                        else:
                            failed_count += 1
                    except Exception as e:
                        logging.warning(f"Simulation {sim_id} failed: {e}")
                        failed_count += 1

        except Exception as e:
            logging.error(f"Critical error in parallel simulation: {e}")
            raise

        # Restore submission order so a seeded run is byte-for-byte reproducible.
        results = [completed_by_sim_id[sim_id] for sim_id in sorted(completed_by_sim_id)]

        if failed_count > 0:
            success_rate = (len(results) / self.n_simulations) * 100
            logging.warning(f"Parallel simulation completed with {failed_count} failures ({success_rate:.1f}% success rate)")

        return results, failed_count

    def _manage_memory_efficient_results(self, results: List[Any]) -> List[Any]:
        """
        Trim the result list to bound memory usage.

        Args:
            results: Simulation results accumulated so far

        Returns:
            The most recently produced half of the results
        """
        if len(results) <= self.max_results_in_memory:
            return results

        keep_count = self.max_results_in_memory // 2
        logging.warning(
            f"Memory management: trimming {len(results)} simulation results to "
            f"{keep_count}. The reported distribution covers only the retained subset."
        )

        # Keep the most recent simulations. Keeping the *best* ones (as an earlier
        # implementation did) would silently truncate the left tail of the distribution
        # and turn every downstream percentile, VaR and CVaR into an optimistic fiction.
        return results[-keep_count:]

    def _simulation_seed(self, sim_id: int) -> Optional[int]:
        """
        Derive the deterministic seed for one simulation.

        Args:
            sim_id: Zero-based simulation index

        Returns:
            ``random_seed + sim_id``, or None when no base seed was configured
        """
        if self.random_seed is None:
            return None
        return self.random_seed + sim_id

    def _rng_for_simulation(self, sim_id: int) -> np.random.Generator:
        """
        Build the random generator for one simulation.

        Args:
            sim_id: Zero-based simulation index

        Returns:
            A generator seeded deterministically when a base seed was configured, and
            seeded from OS entropy otherwise
        """
        return np.random.default_rng(self._simulation_seed(sim_id))

    @staticmethod
    def _run_single_simulation_static(data: pd.DataFrame, symbol: str, sim_id: int,
                                     strategy_class: Type[BaseStrategy], optimal_parameters: Dict[str, Any],
                                     bootstrap_sample_pct: float, block_size_days: int,
                                     initial_capital: float, commission: float,
                                     random_seed: Optional[int] = None,
                                     cost_model: Optional[CostModel] = None) -> Optional[Any]:
        """
        Static method for parallel processing (must be picklable).

        Args:
            data: Historical price data
            symbol: Symbol identifier
            sim_id: Zero-based simulation index
            strategy_class: Strategy class to backtest
            optimal_parameters: Fixed strategy parameters
            bootstrap_sample_pct: Fraction of the history length to synthesise
            block_size_days: Bootstrap block size
            initial_capital: Starting capital for backtests
            commission: Commission rate for trades
            random_seed: Seed for THIS simulation. Must be passed explicitly: on spawn
                based platforms (Windows) the worker process does not inherit the
                parent's seeded global generators.
            cost_model: Transaction cost model applied to every fill

        Returns:
            The backtest result, or None if the simulation could not be run
        """
        try:
            # Create analyzer instance for this process, seeded for this simulation.
            analyzer = MonteCarloAnalyzer(
                strategy_class=strategy_class,
                optimal_parameters=optimal_parameters,
                n_simulations=1,  # Not used in static method
                bootstrap_sample_pct=bootstrap_sample_pct,
                block_size_days=block_size_days,
                initial_capital=initial_capital,
                commission=commission,
                n_jobs=1,  # Single job for static method
                random_seed=random_seed,
                cost_model=cost_model
            )

            # Run the simulation with the seed derived for it, so the worker reproduces
            # exactly what a sequential run with the same base seed would produce.
            sampled_data = analyzer._block_bootstrap_sample(
                data, rng=np.random.default_rng(random_seed)
            )

            if len(sampled_data) < MIN_SAMPLE_BARS:
                logging.warning(
                    f"Simulation {sim_id} discarded: bootstrapped path has only "
                    f"{len(sampled_data)} bars (minimum {MIN_SAMPLE_BARS})"
                )
                return None

            result = analyzer._backtest_engine.run_backtest(analyzer._strategy, sampled_data, symbol)

            # Add simulation metadata
            result.metadata = {
                'simulation_id': sim_id,
                'random_seed': random_seed,
                'parameters_used': optimal_parameters,
                'sample_size': len(sampled_data),
                'sample_start': sampled_data.index[0],
                'sample_end': sampled_data.index[-1],
                'original_data_coverage': len(sampled_data) / len(data)
            }

            return result

        except Exception as e:
            logging.warning(f"Static simulation {sim_id} failed: {e}")
            return None

    def _run_single_simulation(self, data: pd.DataFrame, symbol: str, sim_id: int) -> Optional[Any]:
        """
        Run a single Monte Carlo simulation.

        Args:
            data: Historical price data
            symbol: Symbol identifier
            sim_id: Zero-based simulation index

        Returns:
            The backtest result, or None if the simulation could not be run
        """
        # Build a synthetic price path from block-bootstrapped returns
        sampled_data = self._block_bootstrap_sample(data, rng=self._rng_for_simulation(sim_id))

        if len(sampled_data) < MIN_SAMPLE_BARS:
            logging.warning(
                f"Simulation {sim_id} discarded: bootstrapped path has only "
                f"{len(sampled_data)} bars (minimum {MIN_SAMPLE_BARS})"
            )
            return None

        try:
            # Use pre-created strategy and backtest engine instances for better performance
            result = self._backtest_engine.run_backtest(self._strategy, sampled_data, symbol)

            # Add simulation metadata
            result.metadata = {
                'simulation_id': sim_id,
                'random_seed': self._simulation_seed(sim_id),
                'parameters_used': self.optimal_parameters,
                'sample_size': len(sampled_data),
                'sample_start': sampled_data.index[0],
                'sample_end': sampled_data.index[-1],
                'original_data_coverage': len(sampled_data) / len(data)
            }

            return result

        except Exception as e:
            logging.warning(f"Simulation {sim_id} failed: {e}")
            return None

    def _block_bootstrap_sample(self, data: pd.DataFrame,
                                rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """
        Build a synthetic OHLCV path by block-bootstrapping the historical return series.

        Blocks of consecutive *returns* are drawn with replacement (preserving the
        intra-block autocorrelation) and concatenated in the order they were drawn - that
        random order IS the resampling, so the blocks are deliberately never re-sorted back
        into chronological order. The resampled returns are compounded from the real
        starting price to reconstruct a price path, which avoids the huge artificial gap
        returns that gluing raw price *levels* together would create.

        Each synthetic bar keeps the open/high/low proportions of the historical bar its
        return came from, rescaled around the reconstructed close, and the result carries a
        fresh monotonic DatetimeIndex.

        Args:
            data: Historical OHLCV data with a DatetimeIndex
            rng: Generator to draw block start positions from (defaults to the analyzer's)

        Returns:
            A synthetic OHLCV DataFrame with ``high >= max(open, close)`` and
            ``low <= min(open, close)`` on every bar
        """
        generator = rng if rng is not None else self._rng

        n_samples = max(2, int(len(data) * self.bootstrap_sample_pct))
        n_samples = min(n_samples, len(data))

        close = data['close'].to_numpy(dtype=float)
        if len(close) < 2:
            return data.copy()

        # Returns aligned with bars 1..n-1 of the original data.
        returns = close[1:] / close[:-1] - 1.0
        n_returns = len(returns)

        block_size = max(1, min(self.block_size_days, n_returns))
        n_needed = n_samples - 1  # bar 0 is the seed bar

        # Draw block start positions until we have enough returns.
        positions: List[np.ndarray] = []
        drawn = 0
        max_start = n_returns - block_size
        if max_start <= 0:
            logging.warning(
                f"Block size ({block_size}) leaves a single possible start position for "
                f"{n_returns} returns: every simulation reproduces the same path and the "
                f"resulting distribution carries no information."
            )
        while drawn < n_needed:
            start_idx = int(generator.integers(0, max_start + 1))
            positions.append(np.arange(start_idx, start_idx + block_size))
            drawn += block_size

        # NOTE: no sort_index() here - the drawn order is the resample.
        idx = np.concatenate(positions)[:n_needed]

        resampled_returns = returns[idx]

        # Compound the resampled returns from the real starting price.
        start_price = close[0]
        synthetic_close = start_price * np.cumprod(1.0 + resampled_returns)
        synthetic_close = np.concatenate(([start_price], synthetic_close))

        # Source bars for the resampled returns are the bars those returns closed.
        source_rows = idx + 1
        source_rows = np.concatenate(([0], source_rows))

        source_close = close[source_rows]
        scale = synthetic_close / source_close

        synthetic = pd.DataFrame({
            'open': data['open'].to_numpy(dtype=float)[source_rows] * scale,
            'high': data['high'].to_numpy(dtype=float)[source_rows] * scale,
            'low': data['low'].to_numpy(dtype=float)[source_rows] * scale,
            'close': synthetic_close,
        })

        # Scaling is uniform per bar so ordering is preserved, but guard against
        # floating point drift so we can never emit high < low.
        synthetic['high'] = synthetic[['open', 'high', 'close']].max(axis=1)
        synthetic['low'] = synthetic[['open', 'low', 'close']].min(axis=1)

        for column in data.columns:
            if column not in synthetic.columns:
                synthetic[column] = data[column].to_numpy()[source_rows]

        synthetic = synthetic[list(data.columns)]
        synthetic.index = self._synthetic_index(data, len(synthetic))

        return synthetic

    @staticmethod
    def _synthetic_index(data: pd.DataFrame, n_bars: int) -> pd.DatetimeIndex:
        """
        Build a fresh, strictly increasing DatetimeIndex for a synthetic path.

        Args:
            data: Original data, used for the start timestamp and bar spacing
            n_bars: Number of bars the synthetic path contains

        Returns:
            A DatetimeIndex of length ``n_bars`` with no duplicates or gaps
        """
        start = data.index[0]

        if len(data.index) > 1:
            deltas = np.diff(data.index.to_numpy())
            step = pd.Timedelta(np.median(deltas))
        else:
            step = pd.Timedelta(days=1)

        if step <= pd.Timedelta(0):
            step = pd.Timedelta(days=1)

        return pd.DatetimeIndex([start + i * step for i in range(n_bars)])

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
            # max_drawdown is reported as a negative percentage, so the *worst*
            # drawdown across simulations is the minimum, not the maximum. The tail
            # this analysis exists to expose must not be reported as the best case.
            'worst_max_drawdown': min(max_drawdowns) if max_drawdowns else 0.0,
            'best_max_drawdown': max(max_drawdowns) if max_drawdowns else 0.0,
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