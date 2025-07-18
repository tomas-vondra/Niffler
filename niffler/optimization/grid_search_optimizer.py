from typing import List, Dict, Any, Iterator
import itertools
import logging
from .base_optimizer import BaseOptimizer
from .optimization_result import OptimizationResult


class GridSearchOptimizer(BaseOptimizer):
    """Grid search parameter optimizer."""
    
    # Configuration constants
    MAX_COMBINATIONS_WARNING = 1000000  # Warn if grid search will generate more than this
    
    def optimize(self) -> List[OptimizationResult]:
        """
        Perform grid search optimization.
        
        Returns:
            List of optimization results sorted by objective value (best first)
        """
        # Estimate size and warn if too large
        estimated_size = self._estimate_combinations_count()
        
        # Generate combinations lazily and evaluate them
        combinations_generator = self._generate_grid_combinations_lazy()
        logging.info(f"Starting grid search with {estimated_size} parameter combinations")
        
        return self._evaluate_combinations_lazy(combinations_generator, estimated_size)
    
    def _evaluate_combinations_lazy(self, combinations_generator: Iterator[Dict[str, Any]], 
                                   estimated_size: int) -> List[OptimizationResult]:
        """Evaluate combinations from generator without loading all into memory."""
        results = []
        
        for i, params in enumerate(combinations_generator):
            if self._check_shutdown():
                break
                
            logging.debug(f"Evaluating combination {i+1}/{estimated_size}: {params}")
            result = self._evaluate_single_combination(params)
            if result is not None:
                results = self._manage_memory_efficient_results(results, result)
        
        # Sort and log results
        return self._sort_and_log_results(results)
    
    def _estimate_combinations_count(self) -> int:
        """Estimate total combinations before generation."""
        total = 1
        for name, config in self.parameter_space.parameters.items():
            count = self._count_parameter_combinations(name, config)
            if count == float('inf'):
                logging.error(f"Grid search cannot handle continuous parameter '{name}' without step size. "
                             f"Please specify a step size or use random search instead.")
                raise ValueError(f"Grid search requires step size for float parameter '{name}'")
            total *= count
            
            # Early warning for very large searches
            if total > self.MAX_COMBINATIONS_WARNING:
                logging.warning(
                    f"Grid search will generate {total:,}+ combinations. "
                    f"Consider reducing parameter ranges or using random search."
                )
                break
        
        return total
    
    def _generate_grid_combinations_lazy(self) -> Iterator[Dict[str, Any]]:
        """Generate combinations lazily to save memory."""
        param_values = self._build_param_values()
        param_names = list(param_values.keys())
        
        for combination in itertools.product(*[param_values[name] for name in param_names]):
            yield dict(zip(param_names, combination))
    
    def _build_param_values(self) -> Dict[str, List[Any]]:
        """Build parameter value lists for combination generation."""
        param_values = {}
        
        for name, config in self.parameter_space.parameters.items():
            param_values[name] = self._generate_parameter_values(name, config)
        
        return param_values