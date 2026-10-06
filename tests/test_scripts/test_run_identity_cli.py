"""Run identity and strategy parameters at the command line.

Covers the seams where a run could be filed wrong without anyone noticing: the
params file that carries a parent link, the order parameter sources override
one another, and the configuration file rules for where an experiment may be
named.
"""

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.exporters import ExporterManager
from niffler.utils.run_identity import ExperimentMismatchError
from scripts import backtest
from scripts.common import (
    PARAMS_TABLE,
    ParentRun,
    add_experiment_arguments,
    add_strategy_parameter_arguments,
    build_run_identity,
    read_params_file,
    resolve_strategy_parameters,
)
from scripts.config_file import (
    ConfigError,
    add_config_arguments,
    apply_config,
    load_config,
    typed_on_command_line,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='backtest.py', add_help=False)
    parser.add_argument('--data', required=True)
    parser.add_argument('--strategy', default='simple_ma')
    parser.add_argument('--short-window', type=int, default=None)
    add_strategy_parameter_arguments(parser)
    add_experiment_arguments(parser)
    add_config_arguments(parser)
    return parser


class TempDirTestCase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def write(self, name: str, text: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        return path

    def write_json(self, name: str, document) -> str:
        return self.write(name, json.dumps(document))

    def optimization_file(self, experiment='breakout-btc', run_id='opt-01',
                          parameters=None) -> str:
        return self.write_json('opt.json', {
            'run': {'run_id': run_id, 'kind': 'optimize', 'experiment': experiment},
            'results': [
                {'parameters': parameters or {'short_window': 7, 'long_window': 33}},
                {'parameters': {'short_window': 1, 'long_window': 2}},
            ],
        })

    def parse(self, toml: str, argv, tables=(PARAMS_TABLE,)):
        """Parse argv with a config file applied: (args, loaded, parser, argv)."""
        path = self.write('niffler.toml', toml)
        parser = build_parser()
        argv = ['--config', path, '--data', 'x.csv'] + list(argv)
        loaded = apply_config(parser, 'backtest', argv=argv, tables=tables)
        return parser.parse_args(argv), loaded, parser, argv


class TestReadParamsFile(TempDirTestCase):

    def test_an_optimization_file_yields_the_winner_and_its_run(self):
        parameters, parent = read_params_file(self.optimization_file())
        self.assertEqual(parameters, {'short_window': 7, 'long_window': 33})
        self.assertEqual(parent.run_id, 'opt-01')
        self.assertEqual(parent.experiment, 'breakout-btc')

    def test_a_file_without_a_run_block_has_no_parent(self):
        """Hand-written, or saved before runs had ids: unknown is None."""
        path = self.write_json('p.json', {'parameters': {'short_window': 5}})
        parameters, parent = read_params_file(path)
        self.assertEqual(parameters, {'short_window': 5})
        self.assertIsNone(parent)

    def test_a_bare_object_is_read_as_the_parameters(self):
        path = self.write_json('p.json', {'short_window': 5})
        self.assertEqual(read_params_file(path), ({'short_window': 5}, None))

    def test_an_empty_result_list_is_an_error_not_an_empty_parameter_set(self):
        path = self.write_json('p.json', {'results': []})
        with self.assertRaises(ValueError) as ctx:
            read_params_file(path)
        self.assertIn('no optimization results', str(ctx.exception))

    def test_a_missing_file_names_where_the_path_was_configured(self):
        with self.assertRaises(ValueError) as ctx:
            read_params_file('nope.json', origin='niffler.toml [backtest]')
        self.assertIn('niffler.toml [backtest]', str(ctx.exception))

    def test_a_non_object_document_is_rejected(self):
        with self.assertRaises(ValueError):
            read_params_file(self.write_json('p.json', [1, 2]))


class TestParameterPrecedence(TempDirTestCase):
    """strategy defaults < toml table < params file < --params < flags."""

    TOML = """
[backtest.params]
short_window = 3
long_window = 50
position_size = 0.5
"""
    FLAGS = {'short_window': '--short-window'}

    def resolve(self, argv, toml=None):
        args, loaded, _, _ = self.parse(self.TOML if toml is None else toml, argv)
        return resolve_strategy_parameters(args, args.strategy, config=loaded,
                                           flags=self.FLAGS)

    def test_the_toml_table_applies_when_nothing_else_does(self):
        resolved = self.resolve([])
        self.assertEqual(resolved.values,
                         {'short_window': 3, 'long_window': 50, 'position_size': 0.5})
        self.assertTrue(resolved.supplied)
        self.assertIsNone(resolved.parent)

    def test_the_params_file_overrides_the_toml_table(self):
        """A default saved in the file must not override an optimized winner."""
        resolved = self.resolve(['--params-file', self.optimization_file()])
        self.assertEqual(resolved.values['short_window'], 7)
        self.assertEqual(resolved.values['long_window'], 33)

    def test_a_parameter_the_file_does_not_hold_still_comes_from_the_table(self):
        resolved = self.resolve(['--params-file', self.optimization_file()])
        self.assertEqual(resolved.values['position_size'], 0.5)

    def test_params_overrides_the_file(self):
        resolved = self.resolve(['--params-file', self.optimization_file(),
                                 '--params', '{"short_window": 9}'])
        self.assertEqual(resolved.values['short_window'], 9)
        self.assertEqual(resolved.values['long_window'], 33)

    def test_a_convenience_flag_overrides_everything(self):
        resolved = self.resolve(['--params-file', self.optimization_file(),
                                 '--params', '{"short_window": 9}',
                                 '--short-window', '11'])
        self.assertEqual(resolved.values['short_window'], 11)

    def test_the_file_is_recorded_as_the_parent(self):
        resolved = self.resolve(['--params-file', self.optimization_file()])
        self.assertEqual(resolved.parent.run_id, 'opt-01')

    def test_both_spellings_of_the_file_flag_work(self):
        resolved = self.resolve(['--params_file', self.optimization_file()])
        self.assertEqual(resolved.values['short_window'], 7)

    def test_nothing_supplied_is_distinguishable_from_defaults(self):
        resolved = self.resolve([], toml='')
        self.assertEqual(resolved.values, {})
        self.assertFalse(resolved.supplied)

    def test_an_unknown_table_key_is_an_error_naming_the_table(self):
        with self.assertRaises(ValueError) as ctx:
            self.resolve([], toml='[backtest.params]\nrsi_period = 14\n')
        message = str(ctx.exception)
        self.assertIn('rsi_period', message)
        self.assertIn('[backtest.params]', message)

    def test_an_unknown_file_key_is_an_error_naming_the_file(self):
        path = self.optimization_file(parameters={'rsi_period': 14})
        with self.assertRaises(ValueError) as ctx:
            self.resolve(['--params-file', path], toml='')
        self.assertIn('opt.json', str(ctx.exception))

    def test_a_profile_table_replaces_the_script_table(self):
        toml = self.TOML + '\n[profile.fast.params]\nshort_window = 2\n'
        resolved = self.resolve(['--profile', 'fast'], toml=toml)
        # Replaced, not merged: long_window and position_size are gone.
        self.assertEqual(resolved.values, {'short_window': 2})


class TestTypedOnCommandLine(TempDirTestCase):

    PROFILE = '[profile.a]\nexperiment = "a"\n'

    def test_a_typed_flag_is_detected(self):
        _, _, parser, argv = self.parse('', ['--experiment', 'a'])
        self.assertTrue(typed_on_command_line(parser, 'experiment', argv))

    def test_a_file_supplied_value_is_not_typed(self):
        args, _, parser, argv = self.parse(self.PROFILE, ['--profile', 'a'])
        self.assertEqual(args.experiment, 'a')
        self.assertFalse(typed_on_command_line(parser, 'experiment', argv))

    def test_typing_the_value_the_file_holds_still_counts_as_typed(self):
        """Comparing values could not tell these two apart."""
        _, _, parser, argv = self.parse(self.PROFILE,
                                        ['--profile', 'a', '--experiment', 'a'])
        self.assertTrue(typed_on_command_line(parser, 'experiment', argv))

    def test_the_parser_is_left_as_it_was_found(self):
        _, _, parser, argv = self.parse(self.PROFILE, ['--profile', 'a'])
        typed_on_command_line(parser, 'experiment', argv)
        again = parser.parse_args(argv)
        self.assertEqual(again.experiment, 'a')
        with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()):
            parser.parse_args([])  # --data is still required


class TestExperimentInConfig(TempDirTestCase):

    def test_a_profile_may_name_the_experiment(self):
        args, loaded, _, _ = self.parse('[profile.a]\nexperiment = "a"\n',
                                        ['--profile', 'a'])
        self.assertEqual(args.experiment, 'a')
        self.assertEqual(loaded.profile, 'a')

    def test_a_shared_section_may_not(self):
        path = self.write('niffler.toml', '[common]\nexperiment = "a"\n')
        with self.assertRaises(ConfigError) as ctx:
            load_config(build_parser(), 'backtest', argv=['--config', path])
        self.assertIn('[profile.<name>]', str(ctx.exception))

    def test_a_script_section_may_not(self):
        path = self.write('niffler.toml', '[backtest]\nexperiment = "a"\n')
        with self.assertRaises(ConfigError):
            load_config(build_parser(), 'backtest', argv=['--config', path])

    def test_a_profile_key_the_script_does_not_read_is_reported(self):
        """A misspelt 'experiment' must not vanish without a word."""
        _, loaded, _, _ = self.parse(
            '[profile.a]\nexperimnt = "a"\ntrials = 50\n', ['--profile', 'a'])
        self.assertEqual(sorted(loaded.unread_profile_keys), ['experimnt', 'trials'])
        self.assertIn('experimnt', loaded.describe())

    def test_another_scripts_profile_table_is_reported_not_rejected(self):
        _, loaded, _, _ = self.parse(
            '[profile.a.parameter_space.simple_ma]\nshort_window = {min = 1, max = 3}\n',
            ['--profile', 'a'])
        self.assertEqual(loaded.unread_profile_keys, ['parameter_space'])


class TestBuildRunIdentity(TempDirTestCase):

    PARENT = ParentRun(run_id='opt-01', experiment='a', source='opt.json')

    def identity(self, toml, argv, parent=None):
        args, loaded, parser, argv = self.parse(toml, argv)
        return build_run_identity(
            args, 'backtest', config=loaded, parent=parent,
            experiment_typed=typed_on_command_line(parser, 'experiment', argv))

    def test_an_unnamed_run_with_no_parent_has_no_experiment(self):
        identity, note = self.identity('', [])
        self.assertIsNone(identity.experiment)
        self.assertIsNone(identity.parent_run_id)
        self.assertIsNone(note)

    def test_an_unnamed_run_inherits_and_records_its_parent(self):
        identity, note = self.identity('', [], parent=self.PARENT)
        self.assertEqual(identity.experiment, 'a')
        self.assertEqual(identity.parent_run_id, 'opt-01')
        self.assertIn('inherited', note)

    def test_a_profile_experiment_that_differs_is_an_error(self):
        with self.assertRaises(ExperimentMismatchError) as ctx:
            self.identity('[profile.b]\nexperiment = "b"\n', ['--profile', 'b'],
                          parent=self.PARENT)
        self.assertIn('[profile.b]', str(ctx.exception))

    def test_a_typed_experiment_that_differs_is_allowed_and_noted(self):
        identity, note = self.identity('', ['--experiment', 'b'], parent=self.PARENT)
        self.assertEqual(identity.experiment, 'b')
        self.assertEqual(identity.parent_run_id, 'opt-01')
        self.assertIn('"a"', note)

    def test_the_profile_name_is_recorded(self):
        identity, _ = self.identity('[profile.a]\nexperiment = "a"\n', ['--profile', 'a'])
        self.assertEqual(identity.profile, 'a')
        self.assertEqual(identity.experiment, 'a')

    def test_a_blank_experiment_is_rejected(self):
        with self.assertRaises(ValueError):
            self.identity('', ['--experiment', '  '])

    def test_the_run_block_carries_the_registry_key(self):
        identity, _ = self.identity('', [])
        block = ExporterManager.create_run_record(identity, 'rsi', {}).document['run']
        self.assertEqual(block['strategy_key'], 'rsi')
        self.assertEqual(block['run_id'], identity.run_id)


class TestBacktestMain(TempDirTestCase):
    """The rules have to reach the exit code and the printed identity."""

    def run_main(self, argv):
        data = pd.DataFrame(
            {'open': [100.0] * 60, 'high': [105.0] * 60, 'low': [95.0] * 60,
             'close': [102.0] * 60, 'volume': [1000.0] * 60},
            index=pd.date_range('2024-01-01', periods=60, freq='D'),
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('sys.argv', ['backtest.py', '--data', 'ok.csv'] + argv), \
                patch('scripts.backtest.load_data', return_value=data), \
                patch('scripts.backtest.setup_logging'), \
                patch('sys.stdout', new=stdout), patch('sys.stderr', new=stderr):
            exit_code = backtest.main()
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_a_params_file_links_the_backtest_to_its_optimization(self):
        exit_code, stdout, stderr = self.run_main(
            ['--params-file', self.optimization_file()])
        self.assertEqual(exit_code, 0, stderr)
        self.assertIn('experiment: breakout-btc', stdout)
        self.assertIn('parent: opt-01', stdout)
        self.assertIn('inherited', stdout)

    def test_a_mismatched_profile_stops_the_run(self):
        config = self.write('niffler.toml', '[profile.other]\nexperiment = "other"\n')
        exit_code, _, stderr = self.run_main(
            ['--config', config, '--profile', 'other',
             '--params-file', self.optimization_file()])
        self.assertEqual(exit_code, 1)
        self.assertIn('Experiment mismatch', stderr)

    def test_typing_the_experiment_lets_the_same_run_through(self):
        config = self.write('niffler.toml', '[profile.other]\nexperiment = "other"\n')
        exit_code, stdout, stderr = self.run_main(
            ['--config', config, '--profile', 'other', '--experiment', 'other',
             '--params-file', self.optimization_file()])
        self.assertEqual(exit_code, 0, stderr)
        self.assertIn('experiment: other', stdout)
        self.assertIn('parent: opt-01', stdout)

    def test_a_run_with_no_experiment_says_so(self):
        exit_code, stdout, stderr = self.run_main([])
        self.assertEqual(exit_code, 0, stderr)
        self.assertIn('experiment: (none)', stdout)


if __name__ == '__main__':
    unittest.main()
