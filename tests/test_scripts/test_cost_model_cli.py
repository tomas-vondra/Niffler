"""
Unit tests for the shared transaction-cost command line.

The point of these is that a cost flag is never silently ignored, and that a run
with no cost model says so loudly instead of passing itself off as realistic.
"""

import argparse
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.cost_model import (
    FixedSlippageModel,
    VolumeShareSlippageModel,
    ZeroCostModel,
)
from scripts.common import (
    COST_MODEL_CHOICES,
    add_cost_model_arguments,
    build_cost_model,
    report_cost_model,
)


def parse(*argv):
    """Parse a cost-model command line the way the scripts do."""
    parser = argparse.ArgumentParser()
    add_cost_model_arguments(parser)
    return parser.parse_args(list(argv))


class TestCostModelArguments(unittest.TestCase):
    """The flags themselves."""

    def test_default_is_no_cost_model(self):
        self.assertEqual(parse().cost_model, 'none')

    def test_tuning_flags_default_to_unset(self):
        args = parse()

        self.assertIsNone(args.slippage_bps)
        self.assertIsNone(args.half_spread_bps)
        self.assertIsNone(args.impact_coefficient)
        self.assertIsNone(args.max_participation)

    def test_every_choice_builds(self):
        for choice in COST_MODEL_CHOICES:
            with self.subTest(choice=choice):
                self.assertIsNotNone(build_cost_model(parse('--cost-model', choice)))


class TestBuildCostModel(unittest.TestCase):
    """Turning parsed flags into a model."""

    def test_none_builds_a_zero_cost_model(self):
        self.assertIsInstance(build_cost_model(parse()), ZeroCostModel)

    def test_fixed_uses_the_supplied_basis_points(self):
        model = build_cost_model(
            parse('--cost-model', 'fixed', '--slippage-bps', '12', '--half-spread-bps', '3')
        )

        self.assertIsInstance(model, FixedSlippageModel)
        self.assertEqual(model.slippage_bps, 12.0)
        self.assertEqual(model.half_spread_bps, 3.0)

    def test_volume_uses_the_supplied_parameters(self):
        model = build_cost_model(
            parse('--cost-model', 'volume', '--half-spread-bps', '2',
                  '--impact-coefficient', '0.05', '--max-participation', '0.25')
        )

        self.assertIsInstance(model, VolumeShareSlippageModel)
        self.assertEqual(model.half_spread_bps, 2.0)
        self.assertEqual(model.impact_coefficient, 0.05)
        self.assertEqual(model.max_participation, 0.25)

    def test_selected_model_falls_back_to_documented_defaults(self):
        model = build_cost_model(parse('--cost-model', 'fixed'))

        self.assertGreater(model.slippage_bps, 0.0)
        self.assertGreater(model.half_spread_bps, 0.0)

    def test_invalid_values_are_rejected_by_the_model(self):
        with self.assertRaises(ValueError):
            build_cost_model(parse('--cost-model', 'fixed', '--slippage-bps', '-1'))
        with self.assertRaises(ValueError):
            build_cost_model(parse('--cost-model', 'volume', '--max-participation', '0'))


class TestIncompatibleFlagsAreRejected(unittest.TestCase):
    """
    A flag the selected model does not read is an error.

    Accepting it would let someone believe they are paying market impact while
    the run charges none - the exact class of quiet lie this feature exists to
    remove.
    """

    def test_none_rejects_every_tuning_flag(self):
        for flag, value in (('--slippage-bps', '5'), ('--half-spread-bps', '1'),
                            ('--impact-coefficient', '0.1'),
                            ('--max-participation', '0.1')):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError) as raised:
                    build_cost_model(parse(flag, value))
                self.assertIn(flag, str(raised.exception))

    def test_fixed_rejects_the_volume_only_flags(self):
        for flag, value in (('--impact-coefficient', '0.1'),
                            ('--max-participation', '0.1')):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError):
                    build_cost_model(parse('--cost-model', 'fixed', flag, value))

    def test_volume_rejects_the_fixed_only_flag(self):
        with self.assertRaises(ValueError):
            build_cost_model(parse('--cost-model', 'volume', '--slippage-bps', '5'))

    def test_the_error_names_every_offending_flag(self):
        with self.assertRaises(ValueError) as raised:
            build_cost_model(parse('--cost-model', 'fixed',
                                   '--impact-coefficient', '0.1',
                                   '--max-participation', '0.2'))

        message = str(raised.exception)
        self.assertIn('--impact-coefficient', message)
        self.assertIn('--max-participation', message)

    def test_an_unknown_model_name_is_rejected(self):
        args = parse()
        args.cost_model = 'telepathy'

        with self.assertRaises(ValueError):
            build_cost_model(args)


class TestFrictionlessRunsAreLabelled(unittest.TestCase):
    """A zero-cost run must never be presented as realistic."""

    def _report(self, model):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('sys.stdout', stdout):
            warned = report_cost_model(model, stream=stderr)
        return warned, stdout.getvalue(), stderr.getvalue()

    def test_no_cost_model_warns_prominently(self):
        warned, stdout, stderr = self._report(ZeroCostModel())

        self.assertTrue(warned)
        self.assertIn('FRICTIONLESS', stderr)
        self.assertIn('--cost-model', stderr)
        self.assertIn('none', stdout)

    def test_a_zeroed_out_fixed_model_warns_too(self):
        """Selecting a model and then zeroing it is still a frictionless run."""
        model = build_cost_model(parse('--cost-model', 'fixed',
                                       '--slippage-bps', '0', '--half-spread-bps', '0'))

        warned, _, stderr = self._report(model)

        self.assertTrue(warned)
        self.assertIn('FRICTIONLESS', stderr)

    def test_a_real_cost_model_is_described_without_a_warning(self):
        model = build_cost_model(parse('--cost-model', 'fixed', '--slippage-bps', '5'))

        warned, stdout, stderr = self._report(model)

        self.assertFalse(warned)
        self.assertEqual(stderr, '')
        self.assertIn('fixed', stdout)

    def test_a_participation_cap_alone_is_not_frictionless(self):
        model = build_cost_model(parse('--cost-model', 'volume',
                                       '--half-spread-bps', '0',
                                       '--impact-coefficient', '0',
                                       '--max-participation', '0.1'))

        warned, _, _ = self._report(model)

        self.assertFalse(warned)


if __name__ == '__main__':
    unittest.main()
