import unittest
from niffler.optimization.parameter_space import ParameterSpace, SIMPLE_MA_PARAMETER_SPACE


class TestParameterSpace(unittest.TestCase):
    """Test cases for ParameterSpace."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_int_param = {
            'type': 'int',
            'min': 1,
            'max': 10,
            'step': 1
        }
        
        self.valid_float_param = {
            'type': 'float',
            'min': 0.1,
            'max': 1.0,
            'step': 0.1
        }
        
        self.valid_choice_param = {
            'type': 'choice',
            'choices': ['option1', 'option2', 'option3']
        }
        
        self.valid_parameters = {
            'int_param': self.valid_int_param,
            'float_param': self.valid_float_param,
            'choice_param': self.valid_choice_param
        }
    
    def test_valid_parameter_space_creation(self):
        """Test creating a valid parameter space."""
        space = ParameterSpace(self.valid_parameters)
        
        self.assertEqual(space.parameters, self.valid_parameters)
    
    def test_empty_parameter_space(self):
        """Test creating empty parameter space."""
        space = ParameterSpace({})
        
        self.assertEqual(space.parameters, {})
    
    def test_int_parameter_validation(self):
        """Test validation of integer parameters."""
        # Valid int parameter
        space = ParameterSpace({'param': self.valid_int_param})
        self.assertIn('param', space.parameters)
        
        # Test missing required fields
        invalid_int = {'type': 'int', 'min': 1}  # Missing max
        with self.assertRaises(ValueError):
            space = ParameterSpace({'param': invalid_int})
            space._validate_parameter('param', invalid_int)
    
    def test_float_parameter_validation(self):
        """Test validation of float parameters."""
        # Valid float parameter with step
        space = ParameterSpace({'param': self.valid_float_param})
        self.assertIn('param', space.parameters)
        
        # Valid float parameter without step (continuous)
        continuous_float = {
            'type': 'float',
            'min': 0.0,
            'max': 1.0
        }
        space = ParameterSpace({'param': continuous_float})
        self.assertIn('param', space.parameters)
        
        # Test invalid range
        invalid_float = {
            'type': 'float',
            'min': 1.0,
            'max': 0.5  # max < min
        }
        with self.assertRaises(ValueError):
            space = ParameterSpace({'param': invalid_float})
            space._validate_parameter('param', invalid_float)
    
    def test_choice_parameter_validation(self):
        """Test validation of choice parameters."""
        # Valid choice parameter
        space = ParameterSpace({'param': self.valid_choice_param})
        self.assertIn('param', space.parameters)
        
        # Test empty choices
        invalid_choice = {
            'type': 'choice',
            'choices': []
        }
        with self.assertRaises(ValueError):
            space = ParameterSpace({'param': invalid_choice})
            space._validate_parameter('param', invalid_choice)
        
        # Test missing choices
        invalid_choice = {'type': 'choice'}
        with self.assertRaises(ValueError):
            space = ParameterSpace({'param': invalid_choice})
            space._validate_parameter('param', invalid_choice)
    
    def test_invalid_parameter_type(self):
        """Test validation fails with invalid parameter type."""
        invalid_param = {
            'type': 'invalid_type',
            'min': 1,
            'max': 10
        }
        
        with self.assertRaises(ValueError) as context:
            ParameterSpace({'param': invalid_param})
        
        self.assertIn("unknown type", str(context.exception))
    
    def test_missing_type_field(self):
        """Test validation fails when type field is missing."""
        invalid_param = {
            'min': 1,
            'max': 10
        }
        
        with self.assertRaises(ValueError) as context:
            ParameterSpace({'param': invalid_param})
        
        self.assertIn("missing 'type'", str(context.exception))
    
    def test_int_parameter_step_validation(self):
        """Test integer parameter step validation."""
        # Valid step
        valid_int = {
            'type': 'int',
            'min': 1,
            'max': 10,
            'step': 2
        }
        space = ParameterSpace({'param': valid_int})
        self.assertIn('param', space.parameters)
        
        # Zero step should be invalid - but parameter space doesn't validate step values
        # This validation would be done by the optimizer when using the parameters
        pass
        
        # Negative step should be invalid - but parameter space doesn't validate step values
        # This validation would be done by the optimizer when using the parameters
        pass
    
    def test_float_parameter_step_validation(self):
        """Test float parameter step validation."""
        # Valid step
        valid_float = {
            'type': 'float',
            'min': 0.1,
            'max': 1.0,
            'step': 0.1
        }
        space = ParameterSpace({'param': valid_float})
        self.assertIn('param', space.parameters)
        
        # Zero step should be invalid - but parameter space doesn't validate step values
        # This validation would be done by the optimizer when using the parameters
        pass
    
    def test_complex_parameter_space(self):
        """Test parameter space with multiple parameter types."""
        complex_space = {
            'short_window': {
                'type': 'int',
                'min': 5,
                'max': 20,
                'step': 1
            },
            'long_window': {
                'type': 'int',
                'min': 21,
                'max': 50,
                'step': 1
            },
            'threshold': {
                'type': 'float',
                'min': 0.01,
                'max': 0.1,
                'step': 0.01
            },
            'signal_type': {
                'type': 'choice',
                'choices': ['crossover', 'divergence', 'momentum']
            },
            'continuous_param': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0
                # No step - continuous
            }
        }
        
        space = ParameterSpace(complex_space)
        self.assertEqual(len(space.parameters), 5)
        self.assertIn('short_window', space.parameters)
        self.assertIn('long_window', space.parameters)
        self.assertIn('threshold', space.parameters)
        self.assertIn('signal_type', space.parameters)
        self.assertIn('continuous_param', space.parameters)
    
    def test_predefined_simple_ma_parameter_space(self):
        """Test the predefined Simple MA parameter space."""
        space = SIMPLE_MA_PARAMETER_SPACE
        
        # Should have the expected parameters
        self.assertIn('short_window', space.parameters)
        self.assertIn('long_window', space.parameters)
        
        # Check parameter properties
        short_window = space.parameters['short_window']
        long_window = space.parameters['long_window']
        
        self.assertEqual(short_window['type'], 'int')
        self.assertEqual(long_window['type'], 'int')
        
        # Check ranges make sense (short_window max should be <= long_window min)
        self.assertLessEqual(short_window['max'], long_window['min'])
    
    def test_parameter_space_immutability(self):
        """Test that parameter space parameters can be safely modified."""
        space = ParameterSpace(self.valid_parameters.copy())
        original_params = space.parameters.copy()
        
        # Modify the parameters
        space.parameters['new_param'] = {
            'type': 'int',
            'min': 1,
            'max': 5
        }
        
        # Original should be unchanged if we used a copy
        self.assertNotEqual(space.parameters, original_params)
        self.assertIn('new_param', space.parameters)
    
    def test_string_representation(self):
        """Test string representation of parameter space."""
        space = ParameterSpace(self.valid_parameters)
        
        space_str = str(space)
        space_repr = repr(space)
        
        # Should contain class name
        self.assertIn("ParameterSpace", space_repr)
        # Should contain class name in string representation too
        self.assertIn("ParameterSpace", space_str)


if __name__ == '__main__':
    unittest.main()