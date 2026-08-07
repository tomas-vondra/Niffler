"""Strategy parameter handling in backtest.py.

Since the strategy registry made construction generic, ``backtest.py`` no longer
knows which parameters belong to which strategy. These tests pin the rule that
replaced that knowledge: a parameter the chosen strategy does not accept is an
error, never silently dropped - the same discipline the cost-model flags follow.
"""

import argparse
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.backtest import build_strategy_parameters, main


def args_for(strategy: str, **overrides) -> argparse.Namespace:
    """Build a Namespace matching the parser's strategy-parameter defaults."""
    values = {
        'strategy': strategy,
        'params': None,
        'short_window': None,
        'long_window': None,
        'position_size': None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestBuildStrategyParameters(unittest.TestCase):

    def test_no_flags_yields_no_parameters(self):
        """An untouched CLI must fall through to the strategy's own defaults."""
        self.assertEqual({}, build_strategy_parameters(args_for('simple_ma')))

    def test_named_flags_are_collected(self):
        parameters = build_strategy_parameters(
            args_for('simple_ma', short_window=5, long_window=40)
        )

        self.assertEqual({'short_window': 5, 'long_window': 40}, parameters)

    def test_params_json_is_parsed(self):
        parameters = build_strategy_parameters(
            args_for('rsi', params='{"rsi_period": 9, "oversold": 25}')
        )

        self.assertEqual({'rsi_period': 9, 'oversold': 25}, parameters)

    def test_an_explicit_flag_overrides_params_json(self):
        parameters = build_strategy_parameters(
            args_for('simple_ma', params='{"short_window": 3}', short_window=11)
        )

        self.assertEqual(11, parameters['short_window'])

    def test_position_size_is_shared_by_every_strategy(self):
        for strategy in ('simple_ma', 'rsi', 'breakout'):
            with self.subTest(strategy=strategy):
                parameters = build_strategy_parameters(
                    args_for(strategy, position_size=0.3)
                )
                self.assertEqual(0.3, parameters['position_size'])

    def test_a_flag_from_another_strategy_is_an_error(self):
        """--strategy rsi --short-window 5 must fail, not run RSI with defaults."""
        with self.assertRaises(ValueError) as ctx:
            build_strategy_parameters(args_for('rsi', short_window=5))

        message = str(ctx.exception)
        self.assertIn('--short-window', message)
        self.assertIn('rsi', message)
        # The message must say what the strategy does accept.
        self.assertIn('rsi_period', message)

    def test_an_unknown_params_key_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            build_strategy_parameters(
                args_for('breakout', params='{"rsi_period": 9}')
            )

        self.assertIn('rsi_period', str(ctx.exception))

    def test_invalid_params_json_is_reported(self):
        with self.assertRaises(ValueError) as ctx:
            build_strategy_parameters(args_for('rsi', params='{not json}'))

        self.assertIn('--params', str(ctx.exception))

    def test_params_must_be_a_json_object(self):
        with self.assertRaises(ValueError) as ctx:
            build_strategy_parameters(args_for('rsi', params='[1, 2, 3]'))

        self.assertIn('JSON object', str(ctx.exception))

    def test_each_strategys_own_parameters_are_accepted(self):
        cases = {
            'simple_ma': '{"short_window": 8, "long_window": 25}',
            'rsi': '{"rsi_period": 10, "oversold": 20, "overbought": 80}',
            'breakout': '{"entry_window": 30, "exit_window": 12}',
        }
        for strategy, params in cases.items():
            with self.subTest(strategy=strategy):
                self.assertTrue(build_strategy_parameters(args_for(strategy, params=params)))


class TestMainRejectsForeignFlags(unittest.TestCase):
    """The error must reach the exit code, not just the helper."""

    def _run(self, argv):
        """Run main() with a valid data file mocked in, capturing its output.

        The data has to load successfully or the run would exit 1 for the wrong
        reason and the assertion below would prove nothing.
        """
        data = pd.DataFrame(
            {
                'open': [100.0] * 60,
                'high': [105.0] * 60,
                'low': [95.0] * 60,
                'close': [102.0] * 60,
                'volume': [1000.0] * 60,
            },
            index=pd.date_range('2024-01-01', periods=60, freq='D'),
        )
        stderr = io.StringIO()

        with patch('sys.argv', argv), \
                patch('scripts.backtest.load_data', return_value=data), \
                patch('scripts.backtest.setup_logging'), \
                patch('sys.stdout', new=io.StringIO()), \
                patch('sys.stderr', new=stderr):
            exit_code = main()

        return exit_code, stderr.getvalue()

    def test_main_exits_non_zero_and_names_the_foreign_flag(self):
        exit_code, stderr = self._run(
            ['backtest.py', '--data', 'ok.csv', '--strategy', 'rsi',
             '--short-window', '5']
        )

        self.assertEqual(1, exit_code)
        self.assertIn('--short-window', stderr)

    def test_a_valid_run_of_a_new_strategy_is_not_broken_by_the_check(self):
        """Positive control: the same path succeeds with parameters rsi accepts."""
        exit_code, stderr = self._run(
            ['backtest.py', '--data', 'ok.csv', '--strategy', 'rsi',
             '--params', '{"rsi_period": 9}']
        )

        self.assertEqual(0, exit_code, f"unexpected failure: {stderr}")


if __name__ == '__main__':
    unittest.main()
