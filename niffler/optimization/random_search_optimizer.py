from typing import List, Optional, Dict, Any, Set
import random
import logging
from .base_optimizer import BaseOptimizer
from .optimization_result import OptimizationResult


class RandomSearchOptimizer(BaseOptimizer):
    """Random search parameter optimizer."""
    
    # Configuration constants
    MAX_ATTEMPTS_MULTIPLIER = 10  # Maximum attempts = n_samples * this multiplier
    DEFAULT_SAMPLE_RATIO = 0.1  # Sample 10% of total space by default
    MIN_SAMPLES = 10  # Minimum samples even for small spaces
    MAX_SAMPLES = 10000  # Maximum samples for very large spaces
    DUPLICATE_RATE_THRESHOLD = 0.8  # Stop early if duplicate rate exceeds this
    
    def optimize(self, n_trials: Optional[int] = None, seed: Optional[int] = None) -> List[OptimizationResult]:
        """
        Perform random search optimization.
        
        Args:
            n_trials: Number of random parameter combinations to try (None = auto-estimate)
            seed: Random seed for reproducibility
            
        Returns:
            List of optimization results sorted by objective value (best first)
        """
        # Use smart estimation if n_trials not specified
        if n_trials is None:
            n_trials = self.estimate_optimal_samples()
            logging.info(f"Using estimated optimal sample size: {n_trials}")
        else:
            # Provide suggestion if user's choice seems suboptimal
            estimated_optimal = self.estimate_optimal_samples()
            if abs(n_trials - estimated_optimal) / estimated_optimal > 0.5:  # 50% difference
                logging.info(f"Suggestion: Consider using {estimated_optimal} samples instead of {n_trials} for better coverage")
        
        combinations = self._generate_random_combinations(n_trials, seed)
        logging.info(f"Starting random search with {n_trials} parameter combinations")
        
        return self._evaluate_combinations(combinations)
    
    def _generate_random_combinations(self, n_samples: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate unique random parameter combinations."""
        if seed is not None:
            random.seed(seed)
        
        combinations = []
        seen_combinations: Set[int] = set()
        
        max_attempts = n_samples * self.MAX_ATTEMPTS_MULTIPLIER  # Prevent infinite loops
        attempts = 0
        duplicate_count = 0
        
        while len(combinations) < n_samples and attempts < max_attempts:
            combination = self._generate_single_combination()
            
            # Use hash for much faster duplicate checking
            combination_hash = self._hash_combination(combination)
            
            if combination_hash not in seen_combinations:
                seen_combinations.add(combination_hash)
                combinations.append(combination)
            else:
                duplicate_count += 1
            
            attempts += 1
            
            # Early termination if duplicate rate is too high
            if attempts > 100:  # Only check after some attempts
                duplicate_rate = duplicate_count / attempts
                if duplicate_rate > self.DUPLICATE_RATE_THRESHOLD:
                    logging.warning(
                        f"Stopping early due to high duplicate rate ({duplicate_rate:.1%}). "
                        f"Generated {len(combinations)} unique combinations out of {n_samples} requested. "
                        f"Consider expanding parameter ranges or reducing n_trials."
                    )
                    break
        
        if len(combinations) < n_samples and attempts >= max_attempts:
            logging.warning(
                f"Reached maximum attempts ({max_attempts}). "
                f"Generated {len(combinations)} unique combinations out of {n_samples} requested. "
                f"Consider reducing n_trials or expanding parameter ranges."
            )
        
        return combinations
    
    def _hash_combination(self, combination: Dict[str, Any]) -> int:
        """Generate a hash for a parameter combination for fast duplicate checking."""
        # Use frozenset for more efficient hashing (no sorting needed)
        return hash(frozenset(combination.items()))
    
    def _generate_single_combination(self) -> Dict[str, Any]:
        """Generate a single random parameter combination."""
        combination = {}
        for name, config in self.parameter_space.parameters.items():
            combination[name] = self._generate_random_parameter_value(name, config)
        
        return combination
    
    def estimate_optimal_samples(self) -> int:
        """Estimate optimal number of samples based on parameter space size."""
        total_combinations = self._estimate_total_combinations()
        
        if total_combinations == float('inf'):
            # Continuous parameters present, use default max
            return self.MAX_SAMPLES
        
        # Calculate sample size as percentage of total space
        estimated_samples = int(total_combinations * self.DEFAULT_SAMPLE_RATIO)
        
        # Apply bounds
        estimated_samples = max(self.MIN_SAMPLES, estimated_samples)
        estimated_samples = min(self.MAX_SAMPLES, estimated_samples)
        
        logging.info(f"Estimated parameter space size: {total_combinations:,}")
        logging.info(f"Recommended sample size: {estimated_samples}")
        
        return estimated_samples
    
    def _estimate_total_combinations(self) -> float:
        """Estimate total number of parameter combinations."""
        total = 1.0
        
        for name, config in self.parameter_space.parameters.items():
            count = self._count_parameter_combinations(name, config)
            if count == float('inf'):
                return float('inf')
            total *= count
        
        return total