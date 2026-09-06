"""
Unit tests for the shared risk-management command line.

Two things must hold. A flag name is not a constructor keyword -
``--max-position-size`` configures ``position_size_pct`` - so the translation
has to happen or ``create_risk_manager`` rejects the run. And a flag the chosen
manager does not read is an error, because a silently dropped
``--stop-loss-pct`` leaves the user believing stops are armed on a run that
trades without any.
"""

import argparse
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.run_config import RunConfig
from niffler.risk import NO_RISK_MANAGER, FixedRiskManager
from scripts.common import (
    add_risk_manager_arguments,
    build_risk_manager,
    build_run_config,
    describe_risk_configuration,
    report_run_config,
)


def parse(*argv):
    """Parse a risk-management command line the way the scripts do."""
    parser = argparse.ArgumentParser()
    add_risk_manager_arguments(parser)
    return parser.parse_args(list(argv))


class TestRiskManagerArguments(unittest.TestCase):
    """The flags themselves."""

    def test_default_is_no_risk_management(self):
        self.assertEqual(parse().risk_manager, NO_RISK_MANAGER)

    def test_tuning_flags_default_to_unset(self):
        args = parse()

        self.assertIsNone(args.max_position_size)
        self.assertIsNone(args.stop_loss_pct)
        self.assertIsNone(args.max_positions)
        self.assertIsNone(args.max_risk_per_trade)


class TestBuildRiskManager(unittest.TestCase):
    """Turning parsed flags into a manager."""

    def test_none_builds_nothing(self):
        self.assertIsNone(build_risk_manager(parse()))

    def test_a_script_without_the_flags_gets_nothing(self):
        self.assertIsNone(build_risk_manager(argparse.Namespace()))

    def test_fixed_builds_a_fixed_manager(self):
        self.assertIsInstance(build_risk_manager(parse('--risk-manager', 'fixed')),
                              FixedRiskManager)

    def test_the_flag_name_is_translated_to_the_constructor_keyword(self):
        manager = build_risk_manager(
            parse('--risk-manager', 'fixed', '--max-position-size', '0.35'))

        self.assertEqual(manager.position_size_pct, 0.35)

    def test_every_flag_reaches_its_parameter(self):
        manager = build_risk_manager(parse(
            '--risk-manager', 'fixed',
            '--max-position-size', '0.3',
            '--stop-loss-pct', '0.07',
            '--max-positions', '2',
            '--max-risk-per-trade', '0.04',
        ))

        self.assertEqual(manager.position_size_pct, 0.3)
        self.assertEqual(manager.stop_loss_pct, 0.07)
        self.assertEqual(manager.max_positions, 2)
        self.assertEqual(manager.max_risk_per_trade, 0.04)

    def test_unsupplied_flags_keep_the_documented_defaults(self):
        """These are backtest.py's historical numbers, not the class's."""
        manager = build_risk_manager(parse('--risk-manager', 'fixed'))

        self.assertEqual(manager.position_size_pct, 0.2)
        self.assertEqual(manager.stop_loss_pct, 0.05)
        self.assertEqual(manager.max_positions, 5)
        self.assertEqual(manager.max_risk_per_trade, 0.02)

    def test_a_flag_without_a_manager_is_an_error(self):
        with self.assertRaises(ValueError) as context:
            build_risk_manager(parse('--stop-loss-pct', '0.07'))

        self.assertIn('--stop-loss-pct', str(context.exception))

    def test_the_error_names_every_offending_flag(self):
        with self.assertRaises(ValueError) as context:
            build_risk_manager(parse('--stop-loss-pct', '0.07',
                                     '--max-positions', '3'))

        message = str(context.exception)
        self.assertIn('--max-positions', message)
        self.assertIn('--stop-loss-pct', message)

    def test_an_unknown_manager_is_an_error(self):
        with self.assertRaises(ValueError):
            build_risk_manager(argparse.Namespace(risk_manager='kelly'))


class TestRunConfigCarriesIt(unittest.TestCase):
    """build_run_config is the one place a CLI becomes engine settings."""

    def test_the_manager_lands_on_the_config(self):
        config = build_run_config(parse('--risk-manager', 'fixed',
                                        '--max-position-size', '0.15'))

        self.assertIsInstance(config.risk_manager, FixedRiskManager)
        self.assertEqual(config.risk_manager.position_size_pct, 0.15)

    def test_the_default_config_has_none(self):
        self.assertIsNone(build_run_config(parse()).risk_manager)

    def test_an_inconsistent_command_line_fails_before_any_work(self):
        with self.assertRaises(ValueError):
            build_run_config(parse('--max-positions', '3'))


class TestReporting(unittest.TestCase):
    """A run always states the risk configuration it used."""

    def test_no_manager_reads_as_none(self):
        self.assertEqual(describe_risk_configuration(None), NO_RISK_MANAGER)

    def test_a_manager_reads_as_its_name_and_parameters(self):
        rendered = describe_risk_configuration(FixedRiskManager(position_size_pct=0.25))

        self.assertIn('fixed', rendered)
        self.assertIn('position_size_pct=0.25', rendered)

    def test_report_run_config_prints_it(self):
        buffer = io.StringIO()
        config = RunConfig(risk_manager=FixedRiskManager(position_size_pct=0.25))

        with redirect_stdout(buffer):
            report_run_config(config, stream=io.StringIO())

        self.assertIn('Risk management: fixed', buffer.getvalue())

    def test_report_run_config_says_none_when_there_is_none(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            report_run_config(RunConfig(), stream=io.StringIO())

        self.assertIn(f"Risk management: {NO_RISK_MANAGER}", buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
