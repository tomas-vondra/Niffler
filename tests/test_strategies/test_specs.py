"""Every registered strategy has a spec, and the spec agrees with the class.

A spec is documentation that a test can hold to account. These checks run over
the registry and the spec directory, so a strategy added tomorrow is covered
without editing this file: registering a class without a spec fails here, and so
does a spec whose parameters, defaults or search ranges have drifted from the
class it describes.
"""

import inspect
import re
import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.strategies.capabilities import CAPABILITIES
from niffler.strategies.registry import STRATEGY_CLASSES, create_strategy
from niffler.strategies.spec import (
    SPEC_DIR,
    STATUS_IMPLEMENTED,
    load_all_specs,
)


def constructor_defaults(strategy_class) -> dict:
    """Keyword defaults of a strategy's __init__, risk_manager excluded."""
    signature = inspect.signature(strategy_class.__init__)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name not in ('self', 'risk_manager')
    }


class TestSpecDirectory(unittest.TestCase):

    def setUp(self):
        self.specs = load_all_specs()

    def test_every_spec_file_validates(self):
        # load_all_specs raises on the first invalid file; reaching here is the test.
        self.assertEqual(
            sorted(path.stem for path in SPEC_DIR.glob('*.toml')), sorted(self.specs))

    def test_the_directory_holds_only_specs(self):
        """A stray .tml or .TOML would be skipped by the loader without a word."""
        stray = [p.name for p in SPEC_DIR.iterdir() if p.suffix != '.toml']
        self.assertEqual([], stray)

    def test_every_registered_strategy_has_a_spec(self):
        missing = sorted(set(STRATEGY_CLASSES) - set(self.specs))
        self.assertEqual(
            [], missing,
            f"registered without a spec in {SPEC_DIR}: {missing}. "
            f"Write one - see docs/strategy-specs.md")

    def test_a_spec_is_implemented_exactly_when_it_is_registered(self):
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                if key in STRATEGY_CLASSES:
                    self.assertEqual(
                        STATUS_IMPLEMENTED, spec.status,
                        f"{key} is registered, so its spec needs dates.implemented")
                else:
                    self.assertIsNone(
                        spec.implemented,
                        f"{key} has dates.implemented but is not in STRATEGY_CLASSES")


class TestSpecMatchesItsClass(unittest.TestCase):
    """The spec of a registered strategy describes that class, field by field."""

    def setUp(self):
        self.specs = {k: s for k, s in load_all_specs().items() if k in STRATEGY_CLASSES}

    def test_class_and_module_names(self):
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                strategy_class = STRATEGY_CLASSES[key]
                self.assertEqual(spec.class_name, strategy_class.__name__)
                self.assertEqual(f"niffler.strategies.{spec.module}", strategy_class.__module__)

    def test_display_name(self):
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                self.assertEqual(spec.name, create_strategy(key).name)

    def test_parameters_and_defaults(self):
        """The spec's parameters plus position_size are the constructor's, defaults equal."""
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                self.assertEqual(constructor_defaults(STRATEGY_CLASSES[key]), spec.defaults())

    def test_parameter_spec(self):
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                self.assertEqual(STRATEGY_CLASSES[key].PARAMETER_SPEC, spec.parameter_spec())

    def test_parameter_order_matches_the_constructor(self):
        """The scaffold writes the constructor in spec order; keep the specs in that order too."""
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                self.assertEqual(
                    list(constructor_defaults(STRATEGY_CLASSES[key])),
                    list(spec.defaults()))

    def test_a_published_value_equal_to_the_default_is_the_usual_case(self):
        """Not a rule, but a default that departs from the source should say so in the spec."""
        for key, spec in self.specs.items():
            for parameter in spec.parameters:
                if parameter.published is None or parameter.published == parameter.default:
                    continue
                with self.subTest(spec=key, parameter=parameter.name):
                    self.assertIsNotNone(
                        spec.rules.deviations,
                        f"{key}.{parameter.name} defaults to {parameter.default} but the "
                        f"source published {parameter.published}; explain it in rules.deviations")


class TestCapabilities(unittest.TestCase):

    def test_names_are_snake_case(self):
        for name in CAPABILITIES:
            with self.subTest(capability=name):
                self.assertRegex(name, re.compile(r'^[a-z][a-z0-9_]*$'))

    def test_every_capability_explains_itself(self):
        """An unsupported capability's note is the reason a spec is recorded as unsupported."""
        for name, capability in CAPABILITIES.items():
            with self.subTest(capability=name):
                self.assertTrue(capability.description.strip())
                self.assertTrue(capability.note.strip())

    def test_look_ahead_fills_stay_unsupported(self):
        """Filling at the signal bar's close is look-ahead; it must never become 'supported'."""
        self.assertFalse(CAPABILITIES['same_bar_close_fill'].supported)

    def test_the_shipped_strategies_need_only_supported_capabilities(self):
        for key, spec in load_all_specs().items():
            if key in STRATEGY_CLASSES:
                with self.subTest(spec=key):
                    self.assertEqual([], spec.unsupported_reasons())


if __name__ == '__main__':
    unittest.main()
