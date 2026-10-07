"""Every flag has a hyphen spelling, and no spelling that worked was removed.

The scripts grew up separately: some flags were ``--train_window``, others
``--initial-capital``, and which was which had to be remembered per script.
The rule pinned here is that an option carrying an underscore is always an
alias of one that does not - checked against each script's real parser, so a
flag added tomorrow is covered without editing this file.
"""

import argparse
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts import (
    analyze,
    backtest,
    compare,
    download_data,
    optimize,
    preprocessor,
    screen,
)

SCRIPTS = (analyze, backtest, compare, download_data, optimize, preprocessor, screen)


class _ParserCaptured(Exception):
    """Raised in place of parsing, carrying the parser main() built."""

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__('parser captured')
        self.parser = parser


def parser_of(module) -> argparse.ArgumentParser:
    """Return the parser a script's main() builds, without running the script."""
    def capture(self, *args, **kwargs):
        raise _ParserCaptured(self)

    with patch('sys.argv', [f'{module.__name__}.py']), \
            patch.object(argparse.ArgumentParser, 'parse_args', capture), \
            patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
        try:
            module.main()
        except _ParserCaptured as captured:
            return captured.parser
    raise AssertionError(f"{module.__name__}.main() never parsed its arguments")


def long_options(action: argparse.Action):
    return [option for option in action.option_strings if option.startswith('--')]


class TestEveryFlagHasAHyphenSpelling(unittest.TestCase):

    def test_every_underscore_spelling_has_its_hyphen_twin(self):
        """--a_b always also works as --a-b, so the rule needs no per-flag memory."""
        for module in SCRIPTS:
            parser = parser_of(module)
            for action in parser._actions:
                options = long_options(action)
                for option in options:
                    if '_' not in option:
                        continue
                    with self.subTest(script=module.__name__, flag=option):
                        self.assertIn(option.replace('_', '-'), options)

    def test_the_hyphen_spelling_is_the_one_help_shows_first(self):
        for module in SCRIPTS:
            parser = parser_of(module)
            for action in parser._actions:
                options = long_options(action)
                if len(options) < 2 or not any('_' in option for option in options):
                    continue
                with self.subTest(script=module.__name__, flag=options[0]):
                    self.assertNotIn('_', min(
                        (option for option in options if '_' not in option),
                        key=options.index))


class TestOldSpellingsStillParse(unittest.TestCase):
    """Nothing typed before this change may stop working."""

    CASES = (
        (analyze, ['--data', 'x.csv', '--analysis', 'walk_forward', '--strategy', 'rsi',
                   '--train_window', '18', '--test_window', '4',
                   '--optimization_method', 'random',
                   '--optimization_metric', 'sharpe_ratio',
                   '--bootstrap_pct', '0.5', '--block_size', '10',
                   '--random_seed', '7', '--n_jobs', '2', '--initial_capital', '5000',
                   '--params_file', 'p.json'],
         {'train_window': 18, 'test_window': 4, 'optimization_method': 'random',
          'optimization_metric': 'sharpe_ratio', 'bootstrap_pct': 0.5, 'block_size': 10,
          'seed': 7, 'n_jobs': 2, 'initial_capital': 5000.0, 'params_file': 'p.json'}),
        (compare, ['--data', 'a.csv', 'b.csv', '--train_window', '18',
                   '--test_window', '4', '--optimization_method', 'random',
                   '--optimization_metric', 'sharpe_ratio', '--n_jobs', '2'],
         {'train_window': 18, 'test_window': 4, 'optimization_method': 'random',
          'optimization_metric': 'sharpe_ratio', 'n_jobs': 2}),
        (screen, ['--data', 'a.csv', '--strategy', 'rsi', '--train_window', '18',
                  '--test_window', '4', '--optimization_method', 'random',
                  '--optimization_metric', 'sharpe_ratio', '--n_jobs', '2'],
         {'train_window': 18, 'test_window': 4, 'optimization_method': 'random',
          'optimization_metric': 'sharpe_ratio', 'n_jobs': 2}),
        (download_data, ['--source', 'yahoo', '--symbol', 'SPY', '--timeframe', '1d',
                         '--start_date', '2024-01-01', '--end_date', '2024-02-01'],
         {'start_date': '2024-01-01', 'end_date': '2024-02-01'}),
        (optimize, ['--data', 'x.csv', '--strategy', 'rsi', '--n_jobs', '2'],
         {'n_jobs': 2}),
    )

    def test_old_and_new_spellings_give_the_same_result(self):
        for module, old_argv, expected in self.CASES:
            parser = parser_of(module)
            # Same command line, underscores turned into hyphens in the flag names.
            new_argv = [token.replace('_', '-') if token.startswith('--') else token
                        for token in old_argv]
            with self.subTest(script=module.__name__):
                old = vars(parser.parse_args(old_argv))
                new = vars(parser.parse_args(new_argv))
                self.assertEqual(old, new)
                for dest, value in expected.items():
                    self.assertEqual(old[dest], value, dest)

    def test_the_dest_a_config_file_keys_on_did_not_change(self):
        """niffler.toml keys are argparse dests; renaming one would orphan a user's file."""
        for module, _, expected in self.CASES:
            parser = parser_of(module)
            dests = {action.dest for action in parser._actions}
            with self.subTest(script=module.__name__):
                self.assertLessEqual(set(expected), dests)


if __name__ == '__main__':
    unittest.main()
