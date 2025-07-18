#!/usr/bin/env python3
"""
Test runner for optimization package tests.

This script can be used to run all optimization tests with proper setup.

Usage:
    python test_runner.py
    python test_runner.py --module test_optimization_result
    python test_runner.py --integration-only
"""

import sys
import unittest
import argparse


def main():
    """Run optimization package tests."""
    parser = argparse.ArgumentParser(description='Run optimization package tests')
    parser.add_argument('--module', help='Run specific test module')
    parser.add_argument('--integration-only', action='store_true', 
                       help='Run only integration tests')
    parser.add_argument('--unit-only', action='store_true',
                       help='Run only unit tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure test discovery
    loader = unittest.TestLoader()
    
    if args.module:
        # Run specific module
        suite = loader.loadTestsFromName(f'tests.test_optimization.{args.module}')
    elif args.integration_only:
        # Run only integration tests
        suite = loader.loadTestsFromName('tests.test_optimization.test_integration')
    elif args.unit_only:
        # Run all unit tests (exclude integration)
        test_modules = [
            'test_optimization_result',
            'test_parameter_space', 
            'test_base_optimizer',
            'test_grid_search_optimizer',
            'test_random_search_optimizer',
            'test_optimizer'
        ]
        suite = unittest.TestSuite()
        for module in test_modules:
            suite.addTests(loader.loadTestsFromName(f'tests.test_optimization.{module}'))
    else:
        # Run all tests
        suite = loader.discover('tests.test_optimization', pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()