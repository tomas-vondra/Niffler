from abc import ABC, abstractmethod
import pandas as pd
import logging
import signal
from typing import Dict, Any, List, Optional, Type
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
from decimal import Decimal, getcontext
import threading
import random

from niffler.strategies.base_strategy import BaseStrategy
from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.run_config import RunConfig, resolve_run_config
from niffler.utils.json_utils import safe_json_dump
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
    # Limit results in memory for large optimizations. When the cap is hit the
    # worst-scoring half is discarded, which never changes the winner (the
    # running best always survives a purge) but does leave a SCORE-BIASED
    # subset: the surviving combinations beat any baseline far more often than
    # the grid does. Callers that describe the whole grid rather than its top
    # rows must raise the cap (max_results_in_memory) and/or check
    # results_truncated before reporting a distribution.
    MAX_RESULTS_IN_MEMORY = 1000
    DECIMAL_PRECISION = 28  # Decimal precision for float calculations
    
    # Define metrics configuration (metric_name: (higher_is_better, accessor_function))
    METRICS_CONFIG = {
        'total_return': (True, lambda r: r.backtest_result.total_return_pct),
        'sharpe_ratio': (True, lambda r: r.backtest_result.sharpe_ratio if r.backtest_result.sharpe_ratio is not None else float('-inf')),
        # max_drawdown is a negative percentage (-40 is worse than -5), so the best
        # result is the LARGEST value. Sorting ascending here would rank the deepest
        # drawdown first and hand walk-forward analysis an adversarial parameter set.
        'max_drawdown': (True, lambda r: r.backtest_result.max_drawdown),
        'win_rate': (True, lambda r: r.backtest_result.win_rate),
        'total_trades': (True, lambda r: r.backtest_result.total_trades),
        # Return over buy-and-hold on the same bars, charged the same costs.
        # Over one dataset the benchmark is a constant, so this ORDERS results
        # identically to total_return - what it changes is what the number
        # means: a "best" of -12 says the winning parameter set lost to doing
        # nothing, which sorting on total_return alone never tells you. A result
        # with no benchmark sorts last rather than being silently treated as
        # zero excess.
        'excess_return_pct': (True, lambda r: (
            r.backtest_result.excess_return_pct
            if getattr(r.backtest_result, 'excess_return_pct', None) is not None
            else float('-inf')
        )),
    }

    # Metrics that describe a run rather than rank it. A trade count has no
    # "best": neither more nor fewer trades is better, and the direction flag
    # METRICS_CONFIG carries for it exists only so results can be *ordered* by
    # it on request. analyze_best_metrics skips these rather than announcing a
    # "Best Total Trades" that means nothing.
    DESCRIPTIVE_METRICS = frozenset({'total_trades'})
    
    def __init__(self,
                 strategy_class: Type[BaseStrategy],
                 parameter_space: ParameterSpace,
                 data: pd.DataFrame,
                 sort_by: str = DEFAULT_SORT_BY,
                 n_jobs: Optional[int] = None,
                 run_config: Optional[RunConfig] = None,
                 max_results_in_memory: Optional[int] = None):
        """
        Initialize base optimizer.

        Args:
            strategy_class: Strategy class to optimize
            parameter_space: Parameter search space
            data: Historical price data for backtesting
            sort_by: Metric to sort results by for display ('total_return', 'sharpe_ratio', etc.)
            n_jobs: Number of parallel jobs (None = auto-detect)
            run_config: Every engine setting every candidate backtest runs
                under, as one value (see
                :class:`niffler.backtesting.run_config.RunConfig`). None uses
                the engine's own defaults. It replaced loose
                ``initial_capital``/``commission``/``cost_model`` arguments:
                those three were all the optimizer ever forwarded, so a caller
                who set a benchmark or an annualisation factor silently got the
                default inside every grid cell.
            max_results_in_memory: How many results to retain before the
                worst-scoring half is discarded. None (default) uses
                MAX_RESULTS_IN_MEMORY. Raising it keeps the full grid, which is
                what any whole-grid statistic needs: the discarded results are
                the losing ones, so a truncated run's surviving sample is
                biased upwards. Whether a purge happened is reported by
                results_truncated.
        """
        self.strategy_class = strategy_class
        self.parameter_space = parameter_space
        self.data = data
        self.run_config = resolve_run_config(run_config)
        self.sort_by = sort_by
        self.n_jobs = n_jobs or min(mp.cpu_count(), self.DEFAULT_MAX_WORKERS)
        self.max_results_in_memory = (max_results_in_memory
                                      if max_results_in_memory is not None
                                      else self.MAX_RESULTS_IN_MEMORY)
        # Set the first time results are discarded, so a caller reporting on
        # the whole grid can tell a complete sample from a score-biased one.
        self._results_truncated = False

        # Validate inputs
        self._validate_inputs()

        # Create reusable backtest engine for better performance
        self._backtest_engine = BacktestEngine.from_config(self.run_config)

        # Initialize shutdown flag for graceful termination with thread safety
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._setup_signal_handlers()
    
    def _validate_inputs(self) -> None:
        """Validate the optimizer's own inputs.

        The engine settings are not re-checked here: constructing the
        ``RunConfig`` already validated every one of them, in one place, with
        one message.
        """
        if self.max_results_in_memory < 2:
            raise ValueError("max_results_in_memory must be at least 2")

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
    
    @property
    def results_truncated(self) -> bool:
        """
        Whether results were discarded to stay under the memory cap.

        Returns:
            True when at least one purge happened, which means the returned
            results are the best-scoring survivors rather than every
            combination evaluated. The winner is unaffected; any statistic over
            the *whole* grid is not, and must say so instead of reporting a
            biased sample.
        """
        return self._results_truncated

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
        """Evaluate combinations in parallel using ProcessPoolExecutor.

        Results are retained in **submission order**, not completion order. A
        pool hands work back in whatever order the workers finish, so collecting
        in that order would make two things depend on the machine and the worker
        count: which results survive a memory purge, and how equal-scoring
        results are ordered by the stable sort that follows. A seeded run at
        ``n_jobs=1`` and the same run at ``n_jobs=8`` have to produce identical
        output, so completed results are held in a small reorder buffer and
        drained as soon as the next expected index is available. A failed
        evaluation releases its index too, or the drain would stall behind it.
        """
        results = []
        failed_count = 0

        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all jobs, remembering each one's position
                future_to_index = {}
                for index, params in enumerate(combinations):
                    try:
                        future = executor.submit(self._evaluate_single_combination_static,
                                               params, self.strategy_class, self.data,
                                               self.run_config)
                        future_to_index[future] = index
                    except Exception as e:
                        logging.warning(f"Failed to submit job for {params}: {e}")
                        failed_count += 1

                # Collect as they complete, but retain in submission order
                completed: Dict[int, Optional[OptimizationResult]] = {}
                next_index = 0
                for i, future in enumerate(as_completed(future_to_index)):
                    if self._check_shutdown():
                        # Cancel remaining futures
                        for remaining_future in future_to_index:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break

                    index = future_to_index[future]
                    params = combinations[index]
                    try:
                        completed[index] = future.result(timeout=self.BACKTEST_TIMEOUT_SECONDS)
                        logging.debug(f"Completed {i+1}/{len(combinations)}: {params}")
                    except TimeoutError:
                        logging.warning(f"Timeout evaluating {params} after {self.BACKTEST_TIMEOUT_SECONDS}s")
                        completed[index] = None
                        failed_count += 1
                    except (EOFError, BrokenPipeError) as e:
                        logging.warning(f"Process communication error for {params}: {e}")
                        completed[index] = None
                        failed_count += 1
                    except Exception as e:
                        logging.warning(f"Error evaluating {params}: {e}")
                        completed[index] = None
                        failed_count += 1

                    # Drain every position that is now contiguous with the last
                    # one retained, so retention order never depends on timing.
                    while next_index in completed:
                        result = completed.pop(next_index)
                        next_index += 1
                        if result is not None:
                            results = self._manage_memory_efficient_results(results, result)

                # Anything still buffered (a shutdown broke the loop early)
                for index in sorted(completed):
                    result = completed[index]
                    if result is not None:
                        results = self._manage_memory_efficient_results(results, result)


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
        """Manage results list to prevent excessive memory usage by keeping only the best results.

        Discarding the worst half never changes the winner, but it does leave a
        score-biased sample; ``results_truncated`` records that it happened so
        whole-grid statistics can refuse to describe it.
        """
        results.append(new_result)

        # If we have too many results, keep only the best ones
        if len(results) > self.max_results_in_memory:
            # Sort by metric to keep the best results
            if self.sort_by in self.METRICS_CONFIG:
                higher_is_better, accessor_func = self.METRICS_CONFIG[self.sort_by]
                results.sort(key=accessor_func, reverse=higher_is_better)
            else:
                # Default to total return
                _, accessor_func = self.METRICS_CONFIG['total_return']
                results.sort(key=accessor_func, reverse=True)
            
            # Keep only the best half
            keep_count = self.max_results_in_memory // 2
            results = results[:keep_count]
            self._results_truncated = True
            logging.warning(
                f"Memory management: discarded the worst results and kept the top "
                f"{keep_count}. The surviving sample is biased towards high "
                f"{self.sort_by}; statistics over the whole grid must not be "
                f"computed from it."
            )
        
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
                                          run_config: Optional[RunConfig] = None
                                          ) -> Optional[OptimizationResult]:
        """Static method for parallel processing (must be picklable).

        The worker builds its own engine, so it needs the *whole* run
        configuration: a worker that received three loose floats ran every
        candidate under default settings while the parent thought otherwise.
        """
        try:
            # Create strategy instance
            strategy = strategy_class(**parameters)

            # Run backtest
            engine = BacktestEngine.from_config(resolve_run_config(run_config))

            backtest_result = engine.run_backtest(strategy, data)
            
            return OptimizationResult(
                parameters=parameters,
                backtest_result=backtest_result
            )
            
        except Exception as e:
            strategy_name = strategy_class.__name__ if strategy_class else "Unknown"
            logging.warning(f"Error evaluating {strategy_name} with parameters {parameters}: {e}")
            return None
    
    def save_results(self, results: List[OptimizationResult], filename: str,
                     provenance: Optional[Dict[str, Any]] = None) -> None:
        """
        Save optimization results to a JSON file.

        Metrics such as ``sharpe_ratio`` or ``win_rate`` can legitimately be ``inf`` or
        ``NaN`` for degenerate parameter combinations. Those values are sanitised to
        ``null`` via :func:`niffler.utils.json_utils.safe_json_dump`, which also
        forces ``allow_nan=False``, so the file is always valid RFC 8259 JSON.

        Args:
            results: Optimization results to serialise
            filename: Destination path for the JSON file
            provenance: Optional run provenance record (see
                :func:`niffler.utils.provenance.collect_provenance`), written under a
                top-level ``provenance`` key. An optimisation run whose code and input
                data cannot be identified is exactly as unreproducible as a backtest's
        """
        output_data = {
            'metadata': {
                'optimizer_class': self.__class__.__name__,
                'strategy_class': self.strategy_class.__name__,
                'sort_by': self.sort_by,
                # Every engine setting the grid ran under, not just the three
                # that used to be forwarded. A saved result whose annualisation
                # factor or benchmark is unrecorded is not reproducible.
                **self.run_config.to_metadata(),
                'n_combinations': len(results),
                'timestamp': datetime.now().isoformat()
            },
            'results': []
        }

        if provenance is not None:
            output_data['provenance'] = provenance
        
        for result in results:
            result_data = {
                'parameters': result.parameters,
                'metrics': {
                    'total_return': result.backtest_result.total_return,
                    'total_return_pct': result.backtest_result.total_return_pct,
                    'sharpe_ratio': result.backtest_result.sharpe_ratio,
                    'max_drawdown': result.backtest_result.max_drawdown,
                    'total_trades': result.backtest_result.total_trades,
                    'win_rate': result.backtest_result.win_rate,
                    # None when no benchmark ran, which is not the same as 0.
                    'benchmark_return_pct': getattr(
                        result.backtest_result, 'benchmark_return_pct', None),
                    'excess_return_pct': getattr(
                        result.backtest_result, 'excess_return_pct', None),
                    'round_trip_count': getattr(
                        result.backtest_result, 'round_trip_count', 0),
                    'p_value': getattr(result.backtest_result, 'p_value', None)
                }
            }
            output_data['results'].append(result_data)
        
        with open(filename, 'w') as f:
            safe_json_dump(output_data, f, indent=2)
        
        logging.info(f"Optimization results saved to {filename}")
    
    def analyze_best_metrics(self, results: List[OptimizationResult]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze results to find best parameters for each metric.

        Metrics in DESCRIPTIVE_METRICS are skipped: they describe a run rather
        than rank it, and reporting a "best" one invites a reader to select on
        it. Sorting by such a metric is still available - what is not available
        is calling the result best.

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
            if metric_name in self.DESCRIPTIVE_METRICS:
                continue
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
    
    def _count_parameter_combinations(self, param_name: str, config: Dict[str, Any]) -> int:
        """Count the number of combinations for a single parameter.

        Counted off the same lattice both search methods sample from
        (``min + k * step``), so an estimate never disagrees with the number of
        values that actually get generated.
        """
        if config['type'] == 'int':
            step = config.get('step', 1)
            return len(range(config['min'], config['max'] + 1, step))
        elif config['type'] == 'float':
            step = config.get('step')
            if step is not None:
                return self._lattice_size(config['min'], config['max'], step) + 1
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
    
    def _lattice_size(self, min_val: float, max_val: float, step: float) -> int:
        """Count the steps of ``step`` that fit between two bounds.

        This is the single definition of the lattice a stepped parameter lives
        on: the reachable values are ``min_val + k * step`` for
        ``k`` in ``[0, _lattice_size(...)]``, which is exactly what
        :meth:`_generate_float_range` and :meth:`_generate_int_range` produce.
        Grid search and random search both derive their values from it, so the
        two methods search the same space.

        Args:
            min_val: Lower bound (always reachable)
            max_val: Upper bound (reachable only when it lands on the lattice)
            step: Spacing between values

        Returns:
            The largest ``k`` for which ``min_val + k * step <= max_val``.
        """
        getcontext().prec = self.DECIMAL_PRECISION

        span = Decimal(str(max_val)) - Decimal(str(min_val))
        return int(span / Decimal(str(step)))

    def _generate_random_parameter_value(self, param_name: str, config: Dict[str, Any]) -> Any:
        """Generate a random value for a parameter (for random search).

        Stepped parameters are sampled from the **same lattice grid search
        enumerates**: ``min + k * step``. Drawing a bare ``randint(min, max)``
        for an integer parameter made ``--method random`` and ``--method grid``
        search different spaces - a ``long_window`` declared as 20..100 step 5
        has 17 legal values, and random search was drawing from all 81 - so the
        two methods could not be compared, and the grid a surface analysis
        reconstructs from random-search results was five times finer than the
        one the parameter space declares.

        The float branch had the same fault in a subtler form: it sampled
        ``k * step`` rather than ``min + k * step``, which drifts off the grid's
        values - and can return a value **below** ``min`` - whenever ``min`` is
        not itself a multiple of ``step``.
        """

        if config['type'] == 'int':
            step = config.get('step', 1)
            steps = self._lattice_size(config['min'], config['max'], step)
            return config['min'] + random.randint(0, steps) * step
        elif config['type'] == 'float':
            step = config.get('step')
            if step is not None:
                getcontext().prec = self.DECIMAL_PRECISION
                steps = self._lattice_size(config['min'], config['max'], step)
                offset = Decimal(random.randint(0, steps)) * Decimal(str(step))
                return float(Decimal(str(config['min'])) + offset)
            else:
                return random.uniform(config['min'], config['max'])
        elif config['type'] == 'choice':
            return random.choice(config['choices'])
        else:
            raise ValueError(f"Unknown parameter type: {config['type']}")