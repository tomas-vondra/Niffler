"""Contract tests for the risk-manager registry.

These are what turn :mod:`niffler.risk.registry` from a convention into a seam.
``backtest.py`` used to hardcode ``choices=['none', 'fixed']`` and construct the
manager from an ``if`` chain, so ``KellyRiskManager`` - a class that has existed
the whole time - was unselectable no matter how finished it became. The tests
below pin the three properties that stop that recurring: registering a manager is
one edit, the CLI's choices come from the registry, and a parameter the chosen
manager does not accept is a loud error.
"""

import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.risk.base_risk_manager import BaseRiskManager
from niffler.risk.fixed_risk_manager import FixedRiskManager
from niffler.risk.registry import (
    NO_RISK_MANAGER,
    RISK_MANAGER_CLASSES,
    create_risk_manager,
    describe_risk_manager,
    get_available_risk_managers,
    get_risk_manager_class,
    get_risk_manager_name,
    get_risk_manager_parameter_names,
)


def argparse_accepts(main, argv):
    """True if argv survived argument parsing, whatever happened afterwards."""
    with patch('sys.argv', argv), \
            patch('sys.stderr', new=io.StringIO()), \
            patch('sys.stdout', new=io.StringIO()):
        try:
            main()
        except SystemExit as e:
            # argparse exits 2 on an invalid choice; anything else means the
            # arguments parsed and the run failed later, which is fine here.
            return e.code != 2
        except Exception:
            return True
    return True


class DummyRiskManager(BaseRiskManager):
    """A minimal manager, used to prove registration costs exactly one edit."""

    def __init__(self, appetite: float = 0.1, max_positions: int = 2):
        self.appetite = appetite
        self.max_positions = max_positions
        super().__init__({
            'max_position_size': appetite,
            'max_risk_per_trade': appetite,
            'max_positions': max_positions,
        })

    def calculate_position_size(self, signal, current_price, portfolio_value,
                                historical_data, current_position=0.0):
        return self.appetite if signal == 1 else abs(current_position)

    def calculate_stop_loss(self, entry_price, signal, historical_data):
        return None

    def should_close_position(self, current_price, entry_price, stop_loss_price,
                              signal, unrealized_pnl):
        return False, "dummy"

    def _validate_config(self):
        if not 0 < self.appetite <= 1.0:
            raise ValueError("appetite must be between 0 and 1")


class TestRegistration(unittest.TestCase):
    """One registry line is the whole cost of adding a risk manager."""

    def setUp(self):
        self.addCleanup(RISK_MANAGER_CLASSES.pop, 'dummy', None)
        RISK_MANAGER_CLASSES['dummy'] = DummyRiskManager

    def test_a_registered_manager_is_immediately_selectable(self):
        self.assertIn('dummy', get_available_risk_managers())
        self.assertIs(get_risk_manager_class('dummy'), DummyRiskManager)

    def test_a_registered_manager_is_constructed_generically(self):
        manager = create_risk_manager('dummy', {'appetite': 0.25})

        self.assertIsInstance(manager, DummyRiskManager)
        self.assertEqual(manager.appetite, 0.25)

    def test_a_registered_manager_can_be_announced_without_overriding_metrics(self):
        """backtest.py prints this key right after construction."""
        manager = create_risk_manager('dummy')

        self.assertEqual(manager.get_risk_metrics()['risk_management_type'],
                         'DummyRiskManager')

    def test_the_backtest_cli_accepts_every_registered_manager(self):
        """The CLI's choices derive from the registry, not from a literal list.

        A manager registered after ``backtest.py`` was imported is still accepted,
        which is only true because ``--risk-manager`` builds its choices from
        :func:`get_available_risk_managers`.
        """
        from scripts.backtest import main

        for name in get_available_risk_managers():
            with self.subTest(risk_manager=name):
                self.assertTrue(
                    argparse_accepts(
                        main,
                        ['backtest.py', '--data', 'no_such_file.csv',
                         '--strategy', 'simple_ma', '--risk-manager', name],
                    ),
                    f"backtest.py rejected the registered risk manager '{name}'"
                )


class TestRegistryLookups(unittest.TestCase):

    def test_none_is_offered_first_and_maps_to_no_manager(self):
        self.assertEqual(get_available_risk_managers()[0], NO_RISK_MANAGER)
        self.assertIsNone(create_risk_manager(NO_RISK_MANAGER))

    def test_none_has_no_class_because_it_is_the_absence_of_one(self):
        with self.assertRaises(ValueError):
            get_risk_manager_class(NO_RISK_MANAGER)

    def test_unknown_name_lists_the_available_ones(self):
        with self.assertRaises(ValueError) as ctx:
            get_risk_manager_class('kelly')

        self.assertIn('kelly', str(ctx.exception))
        self.assertIn('fixed', str(ctx.exception))

    def test_fixed_is_registered_and_constructs(self):
        manager = create_risk_manager('fixed', {'position_size_pct': 0.15})

        self.assertIsInstance(manager, FixedRiskManager)
        self.assertEqual(manager.position_size_pct, 0.15)

    def test_parameter_names_exclude_self(self):
        names = get_risk_manager_parameter_names('fixed')

        self.assertNotIn('self', names)
        self.assertIn('position_size_pct', names)
        self.assertIn('max_positions', names)


class TestUnacceptedParameters(unittest.TestCase):
    """A setting the chosen manager cannot use is an error, never a silent drop."""

    def test_unaccepted_parameter_names_what_is_accepted(self):
        with self.assertRaises(ValueError) as ctx:
            create_risk_manager('fixed', {'lookback_periods': 50})

        message = str(ctx.exception)
        self.assertIn('lookback_periods', message)
        self.assertIn('FixedRiskManager', message)
        # The message has to say what the manager *does* take.
        self.assertIn('position_size_pct', message)
        self.assertIn('stop_loss_pct', message)

    def test_an_invalid_value_still_raises_from_the_manager(self):
        """Registry validation checks names; the manager still validates values."""
        with self.assertRaises(ValueError):
            create_risk_manager('fixed', {'position_size_pct': 5.0})


class TestLayering(unittest.TestCase):
    """niffler.risk must not depend on niffler.backtesting or niffler.strategies."""

    def test_importing_the_risk_package_pulls_in_no_engine_module(self):
        """The one-way dependency is what keeps the contract importable.

        ``niffler/backtesting/portfolio.py`` imports ``PortfolioSnapshot`` and the
        engine imports the ``RiskManager`` protocol. If anything under
        ``niffler/risk`` imported back into the backtesting or strategy layer that
        would close the loop into a circular import. Run in a subprocess because
        the rest of the suite has already imported those layers in this process.
        """
        probe = (
            "import sys;"
            f"sys.path.insert(0, r'{project_root}');"
            "import niffler.risk;"
            "print([m for m in sys.modules"
            " if m.startswith('niffler.backtesting') or m.startswith('niffler.strategies')])"
        )
        result = subprocess.run(
            [sys.executable, '-c', probe], capture_output=True, text=True
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            '[]', result.stdout.strip(),
            "niffler.risk must not import from niffler.backtesting or "
            f"niffler.strategies; it pulled in {result.stdout.strip()}"
        )


class TestDescribeRiskManager(unittest.TestCase):
    """A recorded run must say how to rebuild the risk manager, not just name it."""

    def test_none_describes_as_the_reserved_name(self):
        self.assertEqual(describe_risk_manager(None),
                         {'name': NO_RISK_MANAGER, 'class': None, 'parameters': {}})

    def test_a_registered_manager_names_its_registry_key(self):
        self.assertEqual(describe_risk_manager(FixedRiskManager())['name'], 'fixed')

    def test_the_description_round_trips_through_create(self):
        original = FixedRiskManager(position_size_pct=0.3, stop_loss_pct=0.07,
                                    max_positions=2, max_risk_per_trade=0.04)

        description = describe_risk_manager(original)
        rebuilt = create_risk_manager(description['name'], description['parameters'])

        self.assertEqual(describe_risk_manager(rebuilt), description)

    def test_an_unregistered_class_has_no_name_and_no_parameters(self):
        class Unregistered(FixedRiskManager):
            pass

        description = describe_risk_manager(Unregistered())

        self.assertIsNone(description['name'])
        self.assertEqual(description['class'], 'Unregistered')
        self.assertEqual(description['parameters'], {})

    def test_get_risk_manager_name_matches(self):
        self.assertEqual(get_risk_manager_name(FixedRiskManager()), 'fixed')
        self.assertEqual(get_risk_manager_name(None), NO_RISK_MANAGER)


if __name__ == '__main__':
    unittest.main()
