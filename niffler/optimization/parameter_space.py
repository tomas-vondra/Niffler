from typing import Dict, Any, List, Tuple


class ParameterSpace:
    """Defines parameter search space for optimization."""
    
    def __init__(self, parameters: Dict[str, Dict[str, Any]]):
        """
        Initialize parameter space.
        
        Args:
            parameters: Dict mapping parameter names to their configuration.
                       Each parameter config should have:
                       - 'type': 'int', 'float', or 'choice'
                       - 'min'/'max' for int/float, or 'choices' for choice
                       - 'step' (optional) for int/float
        
        Example:
            {
                'short_window': {'type': 'int', 'min': 5, 'max': 20, 'step': 1},
                'long_window': {'type': 'int', 'min': 20, 'max': 100, 'step': 5},
                'position_size': {'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.1}
            }
        """
        self.parameters = parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate parameter configuration."""
        for name, config in self.parameters.items():
            if 'type' not in config:
                raise ValueError(f"Parameter '{name}' missing 'type'")
            
            param_type = config['type']
            if param_type in ['int', 'float']:
                if 'min' not in config or 'max' not in config:
                    raise ValueError(f"Parameter '{name}' missing 'min' or 'max'")
                if config['min'] >= config['max']:
                    raise ValueError(f"Parameter '{name}' min must be less than max")
            elif param_type == 'choice':
                if 'choices' not in config:
                    raise ValueError(f"Parameter '{name}' missing 'choices'")
                if not config['choices']:
                    raise ValueError(f"Parameter '{name}' choices cannot be empty")
            else:
                raise ValueError(f"Parameter '{name}' has unknown type '{param_type}'")

# Predefined parameter spaces for common strategies
SIMPLE_MA_PARAMETER_SPACE = ParameterSpace({
    'short_window': {'type': 'int', 'min': 5, 'max': 20, 'step': 1},
    'long_window': {'type': 'int', 'min': 20, 'max': 100, 'step': 5},
    'position_size': {'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.1}
})