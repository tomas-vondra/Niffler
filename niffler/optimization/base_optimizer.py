from abc import ABC, abstractmethod
import pandas as pd
import logging
import signal
from typing import Dict, Any, List, Optional, Type, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
from datetime import datetime
from decimal import Decimal, getcontext
import threading
import random

from niffler.strategies.base_strategy import BaseStrategy
from niffler.backtesting.backtest_engine import BacktestEngine
from .parameter_space import ParameterSpace
from .optimization_result import OptimizationResult


class BaseOptimizer(ABC):
    """Abstract base class for parameter optimizers."""
    
    # Configuration constants
    DEFAULT_MAX_WORKERS = 4
    DEFAULT_INITIAL_CAPITAL = 10000.0
    DEFAULT_COMMISSION = 0.001
    DEFAULT_SORT_BY = 'total_return'
    BACKTEST_TIMEOUT_SECONDS = 300  # 5 minutes per backtest
    REQUIRED_DATA_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    MAX_RESULTS_IN_MEMORY = 1000  # Limit results in memory for large optimizations
    DECIMAL_PRECISION = 28  # Decimal precision for float calculations
    
    # Define metrics configuration (metric_name: (higher_is_better, accessor_function))
    METRICS_CONFIG = {
        'total_return': (True, lambda r: r.backtest_result.total_return),
        'sharpe_ratio': (True, lambda r: r.backtest_result.sharpe_ratio if r.backtest_result.sharpe_ratio is not None else float('-inf')),
        'max_drawdown': (False, lambda r: r.backtest_result.max_drawdown),  # Lower is better
        'win_rate': (True, lambda r: r.backtest_result.win_rate),
        'total_trades': (True, lambda r: r.backtest_result.total_trades)
    }
    
    def __init__(self, 
                 strategy_class: Type[BaseStrategy],
                 parameter_space: ParameterSpace,
                 data: pd.DataFrame,
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 commission: float = DEFAULT_COMMISSION,
                 sort_by: str = DEFAULT_SORT_BY,
                 n_jobs: Optional[int] = None):
        """
        Initialize base optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space: Parameter search space
            data: Historical price data for backtesting
            initial_capital: Starting capital for backtests
            commission: Commission rate for trades
            sort_by: Metric to sort results by for display ('total_return', 'sharpe_ratio', etc.)
            n_jobs: Number of parallel jobs (None = auto-detect)
        """
        self.strategy_class = strategy_class
        self.parameter_space = parameter_space
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.sort_by = sort_by
        self.n_jobs = n_jobs or min(mp.cpu_count(), self.DEFAULT_MAX_WORKERS)
        
        # Validate inputs
        self._validate_inputs()
        
        # Create reusable backtest engine for better performance
        self._backtest_engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission
        )
        
        # Initialize shutdown flag for graceful termination with thread safety
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._setup_signal_handlers()
    
    def _validate_inputs(self) -> None:
        """Validate optimizer input parameters."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        
        if self.commission < 0:
            raise ValueError("commission cannot be negative")
        
        if self.sort_by not in self.METRICS_CONFIG:
            available_metrics = ', '.join(self.METRICS_CONFIG.keys())
            raise ValueError(f"sort_by must be one of: {available_metrics}")
        
        if self.data.empty:
            raise ValueError("data cannot be empty")
        
        missing_columns = [col for col in self.REQUIRED_DATA_COLUMNS if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"data is missing required columns: {missing_columns}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logging.info("Shutdown signal received. Finishing current evaluations...")
            with self._shutdown_lock:
                self._shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _check_shutdown(self) -> bool:
        """Check if shutdown was requested and log status."""
        with self._shutdown_lock:
            if self._shutdown_requested:
                logging.info("Shutdown requested - stopping optimization")
                return True
        return False
    
    @abstractmethod
    def optimize(self) -> List[OptimizationResult]:
        """
        Run optimization and return results.
        
        Returns:
            List of optimization results sorted by objective value (best first)
        """
        pass
    
    def _evaluate_combinations(self, combinations: List[Dict[str, Any]]) -> List[OptimizationResult]:
        """Evaluate parameter combinations using parallel processing."""
        logging.info(f"Evaluating {len(combinations)} parameter combinations using {self.n_jobs} jobs")
        
        # Choose evaluation method based on job count
        if self.n_jobs == 1:
            results = self._evaluate_sequential(combinations)
        else:
            results = self._evaluate_parallel(combinations)
        
        # Sort and log results
        sorted_results = self._sort_and_log_results(results)
        return sorted_results
    
    def _evaluate_sequential(self, combinations: List[Dict[str, Any]]) -> List[OptimizationResult]:
        """Evaluate combinations sequentially (single-threaded)."""
        results = []
        for i, params in enumerate(combinations):
            if self._check_shutdown():
                break
                
            logging.debug(f"Evaluating combination {i+1}/{len(combinations)}: {params}")
            result = self._evaluate_single_combination(params)
            if result is not None:
                results = self._manage_memory_efficient_results(results, result)
        return results
    
    def _evaluate_parallel(self, combinations: List[Dict[str, Any]]) -> List[OptimizationResult]:
        """Evaluate combinations in parallel using ProcessPoolExecutor."""
        results = []
        failed_count = 0
        
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all jobs
                future_to_params = {}
                for params in combinations:
                    try:
                        future = executor.submit(self._evaluate_single_combination_static, 
                                               params, self.strategy_class, self.data, 
                                               self.initial_capital, self.commission)
                        future_to_params[future] = params
                    except Exception as e:
                        logging.warning(f"Failed to submit job for {params}: {e}")
                        failed_count += 1
                
                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_params)):
                    if self._check_shutdown():
                        # Cancel remaining futures
                        for remaining_future in future_to_params:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    
                    params = future_to_params[future]
                    try:
                        result = future.result(timeout=self.BACKTEST_TIMEOUT_SECONDS)
                        if result is not None:
                            results = self._manage_memory_efficient_results(results, result)
                        logging.debug(f"Completed {i+1}/{len(combinations)}: {params}")
                    except TimeoutError:
                        logging.warning(f"Timeout evaluating {params} after {self.BACKTEST_TIMEOUT_SECONDS}s")
                        failed_count += 1
                    except (EOFError, BrokenPipeError) as e:
                        logging.warning(f"Process communication error for {params}: {e}")
                        failed_count += 1
                    except Exception as e:
                        logging.warning(f"Error evaluating {params}: {e}")
                        failed_count += 1
                        
        except Exception as e:
            logging.error(f"Critical error in parallel evaluation: {e}")
            raise
        
        if failed_count > 0:
            success_rate = (len(combinations) - failed_count) / len(combinations) * 100
            logging.warning(f"Parallel evaluation completed with {failed_count} failures ({success_rate:.1f}% success rate)")
        
        return results
    
    def _sort_and_log_results(self, results: List[OptimizationResult]) -> List[OptimizationResult]:
        """Sort results by the specified metric and log completion."""
        if self.sort_by in self.METRICS_CONFIG:
            higher_is_better, accessor_func = self.METRICS_CONFIG[self.sort_by]
            results.sort(key=accessor_func, reverse=higher_is_better)
        else:
            # Default to total return if unknown sort metric
            _, default_accessor = self.METRICS_CONFIG['total_return']
            results.sort(key=default_accessor, reverse=True)
            logging.warning(f"Unknown sort metric '{self.sort_by}', using 'total_return'")
        
        if results:
            _, accessor_func = self.METRICS_CONFIG.get(self.sort_by, self.METRICS_CONFIG['total_return'])
            sort_value = accessor_func(results[0])
            logging.info(f"Optimization completed. Best {self.sort_by}: {sort_value:.4f}")
        else:
            logging.warning("No valid results found")
        
        return results
    
    def _manage_memory_efficient_results(self, results: List[OptimizationResult], 
                                       new_result: OptimizationResult) -> List[OptimizationResult]:
        """Manage results list to prevent excessive memory usage by keeping only the best results."""
        results.append(new_result)
        
        # If we have too many results, keep only the best ones
        if len(results) > self.MAX_RESULTS_IN_MEMORY:
            # Sort by metric to keep the best results
            if self.sort_by in self.METRICS_CONFIG:
                higher_is_better, accessor_func = self.METRICS_CONFIG[self.sort_by]
                results.sort(key=accessor_func, reverse=higher_is_better)
            else:
                # Default to total return
                _, accessor_func = self.METRICS_CONFIG['total_return']
                results.sort(key=accessor_func, reverse=True)
            
            # Keep only the best half
            keep_count = self.MAX_RESULTS_IN_MEMORY // 2
            results = results[:keep_count]
            logging.debug(f"Memory management: kept top {keep_count} results")
        
        return results
    
    def _evaluate_single_combination(self, parameters: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Evaluate a single parameter combination."""
        try:
            # Create strategy instance
            strategy = self.strategy_class(**parameters)
            
            # Run backtest using reusable engine
            backtest_result = self._backtest_engine.run_backtest(strategy, self.data)
            
            return OptimizationResult(
                parameters=parameters,
                backtest_result=backtest_result
            )
            
        except Exception as e:
            logging.warning(f"Error evaluating parameters {parameters}: {e}")
            return None
    
    @staticmethod
    def _evaluate_single_combination_static(parameters: Dict[str, Any],
                                          strategy_class: Type[BaseStrategy],
                                          data: pd.DataFrame,
                                          initial_capital: float,
                                          commission: float) -> Optional[OptimizationResult]:
        """Static method for parallel processing (must be picklable)."""
        try:
            # Create strategy instance
            strategy = strategy_class(**parameters)
            
            # Run backtest
            engine = BacktestEngine(
                initial_capital=initial_capital,
                commission=commission
            )
            
            backtest_result = engine.run_backtest(strategy, data)
            
            return OptimizationResult(
                parameters=parameters,
                backtest_result=backtest_result
            )
            
        except Exception as e:
            strategy_name = strategy_class.__name__ if strategy_class else "Unknown"
            logging.warning(f"Error evaluating {strategy_name} with parameters {parameters}: {e}")
            return None
    
    def save_results(self, results: List[OptimizationResult], filename: str) -> None:
        """Save optimization results to JSON file."""
        output_data = {
            'metadata': {
                'optimizer_class': self.__class__.__name__,
                'strategy_class': self.strategy_class.__name__,
                'sort_by': self.sort_by,
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'n_combinations': len(results),
                'timestamp': datetime.now().isoformat()
            },
            'results': []
        }
        
        for result in results:
            result_data = {
                'parameters': result.parameters,
                'metrics': {
                    'total_return': result.backtest_result.total_return,
                    'sharpe_ratio': result.backtest_result.sharpe_ratio,
                    'max_drawdown': result.backtest_result.max_drawdown,
                    'total_trades': result.backtest_result.total_trades,
                    'win_rate': result.backtest_result.win_rate,
                    'profit_factor': (result.backtest_result.total_profits / abs(result.backtest_result.total_losses) 
                                    if result.backtest_result.total_losses != 0 else None)
                }
            }
            output_data['results'].append(result_data)
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logging.info(f"Optimization results saved to {filename}")
    
    def analyze_best_metrics(self, results: List[OptimizationResult]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze results to find best parameters for each metric.
        
        Args:
            results: List of optimization results
            
        Returns:
            Dictionary mapping metric names to best parameter combinations and values
        """
        if not results:
            return {}
        
        best_metrics = {}
        
        # Use the class-level metrics configuration
        for metric_name, (higher_is_better, accessor) in self.METRICS_CONFIG.items():
            try:
                if higher_is_better:
                    best_result = max(results, key=accessor)
                else:
                    best_result = min(results, key=accessor)
                
                best_metrics[metric_name] = {
                    'parameters': best_result.parameters,
                    'value': accessor(best_result),
                    'higher_is_better': higher_is_better
                }
            except (ValueError, TypeError) as e:
                logging.warning(f"Error analyzing metric {metric_name}: {e}")
                continue
        
        return best_metrics
    
    # Common parameter generation utilities
    def _generate_float_range(self, min_val: float, max_val: float, step: float) -> List[float]:
        """Generate float range with proper precision handling using Decimal."""
        # Set precision for decimal calculations
        getcontext().prec = self.DECIMAL_PRECISION
        
        values = []
        min_decimal = Decimal(str(min_val))
        max_decimal = Decimal(str(max_val))
        step_decimal = Decimal(str(step))
        
        current = min_decimal
        while current <= max_decimal:
            values.append(float(current))
            current += step_decimal
        
        return values
    
    def _generate_int_range(self, min_val: int, max_val: int, step: int = 1) -> List[int]:
        """Generate integer range with step support."""
        return list(range(min_val, max_val + 1, step))
    
    def _calculate_float_steps(self, min_val: float, max_val: float, step: float) -> Tuple[int, int]:
        """Calculate min and max steps for float parameter with step."""
        getcontext().prec = self.DECIMAL_PRECISION
        
        min_decimal = Decimal(str(min_val))
        max_decimal = Decimal(str(max_val))
        step_decimal = Decimal(str(step))
        
        min_steps = int(min_decimal / step_decimal)
        max_steps = int(max_decimal / step_decimal)
        
        return min_steps, max_steps
    
    def _steps_to_float(self, steps: int, step: float) -> float:
        """Convert steps back to float value with precision."""
        getcontext().prec = self.DECIMAL_PRECISION
        
        step_decimal = Decimal(str(step))
        result_decimal = Decimal(steps) * step_decimal
        
        return float(result_decimal)
    
    def _count_parameter_combinations(self, param_name: str, config: Dict[str, Any]) -> int:
        """Count the number of combinations for a single parameter."""
        if config['type'] == 'int':
            step = config.get('step', 1)
            return len(range(config['min'], config['max'] + 1, step))
        elif config['type'] == 'float':
            step = config.get('step')
            if step is not None:
                min_steps, max_steps = self._calculate_float_steps(config['min'], config['max'], step)
                return max_steps - min_steps + 1
            else:
                return float('inf')  # Continuous parameter
        elif config['type'] == 'choice':
            return len(config['choices'])
        else:
            raise ValueError(f"Unknown parameter type: {config['type']}")
    
    def _generate_parameter_values(self, param_name: str, config: Dict[str, Any]) -> List[Any]:
        """Generate all possible values for a parameter (for grid search)."""
        if config['type'] == 'int':
            step = config.get('step', 1)
            return self._generate_int_range(config['min'], config['max'], step)
        elif config['type'] == 'float':
            step = config.get('step', 0.1)
            return self._generate_float_range(config['min'], config['max'], step)
        elif config['type'] == 'choice':
            return list(config['choices'])
        else:
            raise ValueError(f"Unknown parameter type: {config['type']}")
    
    def _generate_random_parameter_value(self, param_name: str, config: Dict[str, Any]) -> Any:
        """Generate a random value for a parameter (for random search)."""
        
        if config['type'] == 'int':
            return random.randint(config['min'], config['max'])
        elif config['type'] == 'float':
            step = config.get('step')
            if step is not None:
                min_steps, max_steps = self._calculate_float_steps(config['min'], config['max'], step)
                random_steps = random.randint(min_steps, max_steps)
                return self._steps_to_float(random_steps, step)
            else:
                return random.uniform(config['min'], config['max'])
        elif config['type'] == 'choice':
            return random.choice(config['choices'])
        else:
            raise ValueError(f"Unknown parameter type: {config['type']}")