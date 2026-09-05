"""Tests for the TOML configuration file that supplies CLI defaults.

The property under test throughout is precedence: a value must be overridable
by everything above it and must never override the command line.
"""

import argparse
import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.strategies.registry import get_parameter_spec
from scripts import analyze, optimize, screen
from scripts.common import add_cost_model_arguments, build_cost_model
from scripts.config_file import (
    CONFIG_ORIGINS_ATTR,
    CONFIG_PATH_ENV_VAR,
    ConfigError,
    LoadedConfig,
    add_config_arguments,
    apply_config,
    load_config,
    script_sections,
)


def build_parser() -> argparse.ArgumentParser:
    """A parser carrying every argument shape the real scripts use."""
    parser = argparse.ArgumentParser(prog='backtest.py', add_help=False)
    parser.add_argument('--data', required=True)
    parser.add_argument('--strategy', default='simple_ma',
                        choices=['simple_ma', 'rsi', 'breakout'])
    parser.add_argument('--capital', '--initial-capital', dest='initial_capital',
                        type=float, default=10000.0)
    parser.add_argument('--commission', type=float, default=0.001)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=None)
    parser.add_argument('--simulations', type=int, default=1000)
    parser.add_argument('--log-level', dest='log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--datasets', dest='datasets', nargs='+', default=None)
    add_config_arguments(parser)
    return parser


class ConfigFileTestCase(unittest.TestCase):
    """Shared temporary-directory plumbing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def write(self, text: str, name: str = 'niffler.toml') -> str:
        """Write a config file and return its path."""
        path = os.path.join(self.temp_dir, name)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        return path

    def parse(self, text: str, argv, name: str = 'niffler.toml'):
        """Parse argv with a config file applied, returning (args, loaded)."""
        path = self.write(text, name)
        parser = build_parser()
        argv = ['--config', path] + list(argv)
        loaded = apply_config(parser, 'backtest', argv=argv)
        return parser.parse_args(argv), loaded


class TestPrecedence(ConfigFileTestCase):
    """argparse default, shared section, script section, profile, command line."""

    LAYERED = """
[common]
data = "common.csv"
commission = 0.002
simulations = 10

[backtest]
data = "backtest.csv"
commission = 0.003

[profile.quick]
commission = 0.004
"""

    def test_argparse_default_survives_when_nothing_sets_it(self):
        args, _ = self.parse(self.LAYERED, [])
        self.assertEqual(args.initial_capital, 10000.0)

    def test_common_section_applies(self):
        args, _ = self.parse(self.LAYERED, [])
        self.assertEqual(args.simulations, 10)

    def test_script_section_beats_common(self):
        args, _ = self.parse(self.LAYERED, [])
        self.assertEqual(args.data, 'backtest.csv')
        self.assertEqual(args.commission, 0.003)

    def test_profile_beats_script_section(self):
        args, _ = self.parse(self.LAYERED, ['--profile', 'quick'])
        self.assertEqual(args.commission, 0.004)
        self.assertEqual(args.data, 'backtest.csv')

    def test_command_line_beats_everything(self):
        args, _ = self.parse(
            self.LAYERED, ['--profile', 'quick', '--commission', '0.005',
                           '--data', 'typed.csv'])
        self.assertEqual(args.commission, 0.005)
        self.assertEqual(args.data, 'typed.csv')

    def test_sections_are_reported_in_order(self):
        _, loaded = self.parse(self.LAYERED, ['--profile', 'quick'])
        self.assertEqual(loaded.sections, ['common', 'backtest', 'profile.quick'])
        self.assertIn('niffler.toml', loaded.describe())

    def test_origins_name_the_section_a_value_came_from(self):
        _, loaded = self.parse(self.LAYERED, [])
        self.assertIn('[backtest]', loaded.origins['data'])
        self.assertIn('[common]', loaded.origins['simulations'])


def env_with(value):
    """A copy of the environment with the config path set to value."""
    environment = dict(os.environ)
    environment[CONFIG_PATH_ENV_VAR] = value
    return environment


def env_without():
    """A copy of the environment with the config path variable removed."""
    environment = dict(os.environ)
    environment.pop(CONFIG_PATH_ENV_VAR, None)
    return environment


class TestRequiredArguments(ConfigFileTestCase):
    """A file may satisfy a required flag; nothing else may."""

    def test_file_satisfies_required_data(self):
        args, _ = self.parse('[backtest]\ndata = "from-file.csv"\n', [])
        self.assertEqual(args.data, 'from-file.csv')

    def test_missing_everywhere_still_errors(self):
        parser = build_parser()
        path = self.write('[backtest]\ncommission = 0.01\n')
        apply_config(parser, 'backtest', argv=['--config', path])

        with patch('sys.stderr', new_callable=StringIO) as stderr:
            with self.assertRaises(SystemExit):
                parser.parse_args(['--config', path])

        self.assertIn('--data', stderr.getvalue())

    def test_no_config_file_leaves_required_alone(self):
        parser = build_parser()
        with patch.dict(os.environ, env_with(''), clear=True):
            self.assertIsNone(apply_config(parser, 'backtest', argv=[]))
        with patch('sys.stderr', new_callable=StringIO):
            with self.assertRaises(SystemExit):
                parser.parse_args([])


class TestFileDiscovery(ConfigFileTestCase):
    """Which file gets read, and when a missing one is an error."""

    def test_missing_default_file_is_not_an_error(self):
        parser = build_parser()
        with patch.dict(os.environ, env_with(''), clear=True):
            self.assertIsNone(apply_config(parser, 'backtest', argv=[]))

    def test_implicit_file_in_the_working_directory_is_read(self):
        self.write('[backtest]\ndata = "implicit.csv"\n')
        parser = build_parser()
        cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            with patch.dict(os.environ, env_without(), clear=True):
                apply_config(parser, 'backtest', argv=[])
        finally:
            os.chdir(cwd)
        self.assertEqual(parser.parse_args([]).data, 'implicit.csv')

    def test_environment_variable_names_the_file(self):
        path = self.write('[backtest]\ndata = "env.csv"\n', name='other.toml')
        parser = build_parser()
        with patch.dict(os.environ, env_with(path), clear=True):
            apply_config(parser, 'backtest', argv=[])
        self.assertEqual(parser.parse_args([]).data, 'env.csv')

    def test_missing_explicit_file_is_an_error(self):
        missing = os.path.join(self.temp_dir, 'nope.toml')
        with self.assertRaises(ConfigError) as raised:
            load_config(build_parser(), 'backtest', argv=['--config', missing])
        self.assertIn('not found', str(raised.exception))

    def test_missing_environment_file_is_an_error(self):
        missing = os.path.join(self.temp_dir, 'nope.toml')
        with patch.dict(os.environ, env_with(missing), clear=True):
            with self.assertRaises(ConfigError) as raised:
                load_config(build_parser(), 'backtest', argv=[])
        self.assertIn(CONFIG_PATH_ENV_VAR, str(raised.exception))

    def test_malformed_file_is_an_error(self):
        path = self.write('[backtest\ndata = ')
        with self.assertRaises(ConfigError) as raised:
            load_config(build_parser(), 'backtest', argv=['--config', path])
        self.assertIn('Could not parse', str(raised.exception))

    def test_configuration_error_exits_through_the_parser(self):
        path = self.write('[backtest]\nnot_a_flag = 1\n')
        with patch('sys.stderr', new_callable=StringIO) as stderr:
            with self.assertRaises(SystemExit):
                apply_config(build_parser(), 'backtest', argv=['--config', path])
        self.assertIn('not_a_flag', stderr.getvalue())


class TestKeyValidation(ConfigFileTestCase):
    """Unknown keys, unknown sections, and values argparse would have rejected."""

    def load(self, text, argv=()):
        path = self.write(text)
        return load_config(build_parser(), 'backtest',
                           argv=['--config', path] + list(argv))

    def test_unknown_key_in_script_section_names_the_valid_ones(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('[backtest]\ncomision = 0.01\n')
        message = str(raised.exception)
        self.assertIn('comision', message)
        self.assertIn('commission', message)

    def test_unknown_key_in_shared_section_is_skipped(self):
        loaded = self.load('[common]\nnot_a_flag = 1\ncommission = 0.01\n')
        self.assertEqual(loaded.values['commission'], 0.01)
        self.assertNotIn('not_a_flag', loaded.values)

    def test_unknown_section_is_an_error(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('[backtests]\ndata = "x.csv"\n')
        self.assertIn('backtests', str(raised.exception))

    def test_top_level_value_is_an_error(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('commission = 0.01\n')
        self.assertIn('must live in a section', str(raised.exception))

    def test_unknown_profile_is_an_error(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('[profile.slow]\ncommission = 0.01\n', ['--profile', 'quick'])
        self.assertIn('slow', str(raised.exception))

    def test_value_outside_choices_is_rejected(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('[backtest]\nstrategy = "does_not_exist"\n')
        self.assertIn('simple_ma', str(raised.exception))

    def test_value_is_converted_the_way_the_flag_would(self):
        loaded = self.load('[backtest]\ninitial_capital = 5000\n')
        self.assertIsInstance(loaded.values['initial_capital'], float)

    def test_unconvertible_value_is_rejected(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('[backtest]\nn_jobs = "many"\n')
        self.assertIn('n_jobs', str(raised.exception))

    def test_flag_requires_a_boolean(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('[backtest]\nclean = "yes"\n')
        self.assertIn('true or false', str(raised.exception))

    def test_boolean_for_a_value_flag_is_rejected(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('[backtest]\ncommission = true\n')
        self.assertIn('not true/false', str(raised.exception))

    def test_store_true_flag_is_applied(self):
        loaded = self.load('[backtest]\nclean = true\n')
        self.assertIs(loaded.values['clean'], True)

    def test_list_reaches_a_multi_value_flag(self):
        loaded = self.load('[backtest]\ndatasets = ["a.csv", "b.csv"]\n')
        self.assertEqual(loaded.values['datasets'], ['a.csv', 'b.csv'])

    def test_single_value_is_wrapped_for_a_multi_value_flag(self):
        loaded = self.load('[backtest]\ndatasets = "a.csv"\n')
        self.assertEqual(loaded.values['datasets'], ['a.csv'])

    def test_list_for_a_single_value_flag_is_rejected(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('[backtest]\ndata = ["a.csv", "b.csv"]\n')
        self.assertIn('single value', str(raised.exception))

    def test_unknown_sub_table_is_an_error(self):
        with self.assertRaises(ConfigError) as raised:
            self.load('[backtest.nested]\nkey = "value"\n')
        self.assertIn('nested', str(raised.exception))

    def test_script_sections_cover_every_cli(self):
        self.assertEqual(
            sorted(script_sections()),
            ['analyze', 'backtest', 'compare', 'download_data', 'optimize',
             'preprocessor', 'screen'])


class TestCostModelOrigin(ConfigFileTestCase):
    """A rejected cost-model flag must say where the value came from."""

    def parser_with_costs(self):
        parser = argparse.ArgumentParser(prog='optimize.py', add_help=False)
        add_cost_model_arguments(parser)
        add_config_arguments(parser)
        return parser

    def test_rejected_flag_names_the_config_file(self):
        parser = self.parser_with_costs()
        path = self.write('[costs]\nimpact_coefficient = 0.2\n')
        argv = ['--config', path, '--cost-model', 'fixed']
        apply_config(parser, 'optimize', argv=argv)

        with self.assertRaises(ValueError) as raised:
            build_cost_model(parser.parse_args(argv))

        message = str(raised.exception)
        self.assertIn('--impact-coefficient', message)
        self.assertIn('set in', message)
        self.assertIn('[costs]', message)

    def test_typed_flag_is_not_attributed_to_a_file(self):
        parser = self.parser_with_costs()
        argv = ['--cost-model', 'fixed', '--impact-coefficient', '0.2']

        with self.assertRaises(ValueError) as raised:
            build_cost_model(parser.parse_args(argv))

        message = str(raised.exception)
        self.assertIn('--impact-coefficient', message)
        self.assertNotIn('set in', message)

    def test_origins_reach_the_parsed_namespace(self):
        parser = self.parser_with_costs()
        path = self.write('[costs]\nhalf_spread_bps = 2.0\n')
        argv = ['--config', path]
        apply_config(parser, 'optimize', argv=argv)
        args = parser.parse_args(argv)
        self.assertIn('half_spread_bps', getattr(args, CONFIG_ORIGINS_ATTR))


class TestParameterSpaceOverride(unittest.TestCase):
    """[optimize.parameter_space.STRATEGY] widens or moves a strategy's space."""

    def config(self, table):
        loaded = LoadedConfig(path=Path('niffler.toml'))
        loaded.tables['parameter_space'] = table
        return loaded

    def test_no_config_uses_the_strategy_spec(self):
        space = optimize.build_parameter_space('simple_ma', None)
        self.assertEqual(space.parameters,
                         get_parameter_spec('simple_ma'))

    def test_override_replaces_only_the_named_parameter(self):
        wider = {'short_window': {'type': 'int', 'min': 5, 'max': 40, 'step': 1}}
        space = optimize.build_parameter_space(
            'simple_ma', self.config({'simple_ma': wider}))

        self.assertEqual(space.parameters['short_window']['max'], 40)
        self.assertEqual(space.parameters['long_window'],
                         get_parameter_spec('simple_ma')['long_window'])

    def test_override_for_another_strategy_is_ignored(self):
        wider = {'short_window': {'type': 'int', 'min': 5, 'max': 40, 'step': 1}}
        space = optimize.build_parameter_space(
            'rsi', self.config({'simple_ma': wider}))
        self.assertEqual(space.parameters, get_parameter_spec('rsi'))

    def test_parameter_the_strategy_rejects_is_an_error(self):
        bogus = {'not_a_parameter': {'type': 'int', 'min': 1, 'max': 2}}
        with self.assertRaises(ValueError) as raised:
            optimize.build_parameter_space(
                'simple_ma', self.config({'simple_ma': bogus}))
        self.assertIn('not_a_parameter', str(raised.exception))

    def test_invalid_space_is_rejected_by_parameter_space(self):
        inverted = {'short_window': {'type': 'int', 'min': 40, 'max': 5}}
        with self.assertRaises(ValueError) as raised:
            optimize.build_parameter_space(
                'simple_ma', self.config({'simple_ma': inverted}))
        self.assertIn('short_window', str(raised.exception))

    def test_non_table_entry_is_an_error(self):
        with self.assertRaises(ValueError) as raised:
            optimize.build_parameter_space(
                'simple_ma', self.config({'simple_ma': {'short_window': 5}}))
        self.assertIn('short_window', str(raised.exception))


class TestFlagAliases(unittest.TestCase):
    """The spelling drift between scripts is now aliases, not renames."""

    ANALYZE_BASE = ['--data', 'x.csv', '--analysis', 'monte_carlo',
                    '--strategy', 'simple_ma']

    def test_analyze_accepts_both_seed_spellings(self):
        parser = analyze.create_parser()
        self.assertEqual(
            parser.parse_args(self.ANALYZE_BASE + ['--seed', '7']).seed, 7)
        self.assertEqual(
            parser.parse_args(self.ANALYZE_BASE + ['--random_seed', '7']).seed, 7)

    def test_analyze_accepts_every_capital_spelling(self):
        parser = analyze.create_parser()
        for flag in ('--capital', '--initial-capital', '--initial_capital'):
            args = parser.parse_args(self.ANALYZE_BASE + [flag, '250'])
            self.assertEqual(args.initial_capital, 250.0)

    def test_analyze_has_a_log_level_flag(self):
        parser = analyze.create_parser()
        args = parser.parse_args(self.ANALYZE_BASE + ['--log-level', 'WARNING'])
        self.assertEqual(args.log_level, 'WARNING')

    def test_screen_accepts_both_jobs_spellings(self):
        parser = screen.build_parser()
        base = ['--data', 'x.csv', '--strategy', 'simple_ma']
        self.assertEqual(parser.parse_args(base + ['--jobs', '3']).n_jobs, 3)
        self.assertEqual(parser.parse_args(base + ['--n_jobs', '3']).n_jobs, 3)

    def test_config_flags_are_declared_on_the_scripts(self):
        base = ['--data', 'x.csv', '--strategy', 'simple_ma']
        args = screen.build_parser().parse_args(base + ['--profile', 'quick'])
        self.assertEqual(args.profile, 'quick')


if __name__ == '__main__':
    unittest.main()
