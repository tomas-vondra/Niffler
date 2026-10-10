"""The scaffold: what it generates from a spec, and what it refuses to do.

The strongest check here regenerates each shipped strategy from its spec and
compares the result with the hand-written class: same constructor, same
defaults, same ``PARAMETER_SPEC``, same display name. If the scaffold and the
shipped classes ever disagree, one of them is wrong.
"""

import dataclasses
import io
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Every CLI reads ./niffler.toml; a developer's own file must not reach main().
import tests.test_scripts  # noqa: F401  (sets NIFFLER_CONFIG='')

from niffler.strategies.registry import STRATEGY_CLASSES
from niffler.strategies.scaffold import (
    REGISTRY_PATH,
    ScaffoldError,
    add_registry_entry,
    plan_scaffold,
    render_strategy_module,
    render_test_module,
    strategy_module_path,
    test_module_path,
    write_scaffold,
)
from niffler.strategies.spec import SPEC_DIR, load_all_specs, load_spec
from tests.test_strategies.test_registry import make_ohlcv
from tests.test_strategies.test_specs import constructor_defaults

PACKAGE = 'niffler.strategies'


def unimplemented(spec, key=None):
    """The same spec as if it had just been written: not implemented, optionally renamed."""
    changes = {'implemented': None}
    if key is not None:
        camel = ''.join(part.capitalize() for part in key.split('_'))
        changes.update(key=key, class_name=f"{camel}Strategy")
    return dataclasses.replace(spec, **changes)


def load_generated(spec, source):
    """Execute a generated strategy module inside the niffler.strategies package."""
    name = f"{PACKAGE}.{spec.module}"
    module = types.ModuleType(name)
    module.__package__ = PACKAGE
    exec(compile(source, f"<{name}>", 'exec'), module.__dict__)
    return module


class TestRegeneratingTheShippedStrategies(unittest.TestCase):
    """Rendering a shipped spec reproduces the shipped class's interface."""

    def setUp(self):
        self.specs = {k: s for k, s in load_all_specs().items() if k in STRATEGY_CLASSES}
        self.assertTrue(self.specs)

    def generated_class(self, spec):
        module = load_generated(spec, render_strategy_module(unimplemented(spec)))
        return getattr(module, spec.class_name)

    def test_constructor_and_defaults(self):
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                self.assertEqual(constructor_defaults(STRATEGY_CLASSES[key]),
                                 constructor_defaults(self.generated_class(spec)))

    def test_parameter_spec(self):
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                self.assertEqual(STRATEGY_CLASSES[key].PARAMETER_SPEC,
                                 self.generated_class(spec).PARAMETER_SPEC)

    def test_display_name_parameters_and_description(self):
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                shipped = STRATEGY_CLASSES[key]()
                generated = self.generated_class(spec)()
                self.assertEqual(shipped.name, generated.name)
                self.assertEqual(shipped.parameters, generated.parameters)
                self.assertIn(spec.name, generated.get_description())
                for name in spec.defaults():
                    self.assertEqual(getattr(shipped, name), getattr(generated, name))

    def test_the_generated_algorithm_refuses_to_run(self):
        """A scaffold must not pass for a strategy that silently never trades."""
        data = make_ohlcv(60)
        for key, spec in self.specs.items():
            with self.subTest(spec=key):
                with self.assertRaises(NotImplementedError) as raised:
                    self.generated_class(spec)().generate_signals(data)
                self.assertIn(f"specs/{key}.toml", str(raised.exception))

    def test_the_generated_class_still_validates_its_input(self):
        spec = next(iter(self.specs.values()))
        with self.assertRaises(ValueError):
            self.generated_class(spec)().generate_signals(pd.DataFrame())


class TestRenderedTests(unittest.TestCase):
    """The rule-test template runs, and fails until someone writes it."""

    def run_generated_tests(self, spec):
        strategy_module = load_generated(spec, render_strategy_module(spec))
        test_source = render_test_module(spec)
        test_module = types.ModuleType(f"tests.test_strategies.test_{spec.module}")
        test_module.__file__ = str(project_root / test_module_path(spec))
        with patch.dict(sys.modules, {strategy_module.__name__: strategy_module}):
            exec(compile(test_source, test_module.__file__, 'exec'), test_module.__dict__)
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        return result, test_source

    def test_every_rule_test_fails_until_written(self):
        spec = unimplemented(load_spec(SPEC_DIR / 'simple_ma.toml'), key='crossover_probe')
        result, _ = self.run_generated_tests(spec)
        self.assertEqual(2, result.testsRun)
        self.assertEqual(2, len(result.failures))
        self.assertEqual([], result.errors)

    def test_a_spec_reading_high_or_low_gets_the_intra_bar_stub(self):
        spec = unimplemented(load_spec(SPEC_DIR / 'breakout.toml'), key='channel_probe')
        result, source = self.run_generated_tests(spec)
        self.assertEqual(3, result.testsRun)
        self.assertEqual(3, len(result.failures))
        self.assertIn('test_a_bar_does_not_judge_its_own_close_by_its_own_range', source)

    def test_a_close_only_spec_does_not(self):
        spec = unimplemented(load_spec(SPEC_DIR / 'rsi.toml'), key='rsi_probe')
        self.assertNotIn('own_range', render_test_module(spec))

    def test_the_rules_are_copied_into_both_files(self):
        """Each writer reads the rules without opening the other's file."""
        spec = unimplemented(load_spec(SPEC_DIR / 'rsi.toml'), key='rsi_probe')
        fragment = 'alpha = 1 / rsi_period'
        self.assertIn(fragment, render_strategy_module(spec))
        self.assertIn(fragment, render_test_module(spec))

    def test_prose_cannot_break_out_of_the_docstring(self):
        spec = unimplemented(load_spec(SPEC_DIR / 'rsi.toml'), key='rsi_probe')
        spec = dataclasses.replace(spec, rules=dataclasses.replace(
            spec.rules, entry='Buy when """ appears, or a \\ backslash.'))
        for source in (render_strategy_module(spec), render_test_module(spec)):
            compile(source, '<rendered>', 'exec')


class TestRegistryEntry(unittest.TestCase):

    def setUp(self):
        self.source = (project_root / REGISTRY_PATH).read_text(encoding='utf-8')
        self.spec = unimplemented(load_spec(SPEC_DIR / 'breakout.toml'), key='donchian_probe')

    def test_the_entry_and_the_import_are_added(self):
        edited = add_registry_entry(self.source, self.spec)
        self.assertIn("    'donchian_probe': DonchianProbeStrategy,\n}", edited)
        self.assertIn(
            "from .breakout_strategy import BreakoutStrategy\n"
            "from .donchian_probe_strategy import DonchianProbeStrategy\n"
            "from .rsi_strategy import RSIStrategy\n",
            edited)

    def test_an_import_sorting_last_goes_after_the_others(self):
        spec = unimplemented(self.spec, key='zigzag')
        edited = add_registry_entry(self.source, spec)
        self.assertIn("from .simple_ma_strategy import SimpleMAStrategy\n"
                      "from .zigzag_strategy import ZigzagStrategy\n", edited)

    def test_the_edited_registry_imports_and_registers_the_class(self):
        generated = load_generated(self.spec, render_strategy_module(self.spec))
        edited = add_registry_entry(self.source, self.spec)

        registry = types.ModuleType(f"{PACKAGE}.registry")
        registry.__package__ = PACKAGE
        with patch.dict(sys.modules, {generated.__name__: generated}):
            exec(compile(edited, '<registry>', 'exec'), registry.__dict__)

        self.assertIs(generated.DonchianProbeStrategy,
                      registry.STRATEGY_CLASSES['donchian_probe'])
        self.assertEqual(list(STRATEGY_CLASSES) + ['donchian_probe'],
                         list(registry.STRATEGY_CLASSES))

    def test_an_already_registered_key_is_refused(self):
        with self.assertRaises(ScaffoldError) as raised:
            add_registry_entry(self.source, unimplemented(self.spec, key='rsi'))
        self.assertIn("'rsi' is already in STRATEGY_CLASSES", str(raised.exception))

    def test_a_registry_of_an_unexpected_shape_is_refused(self):
        with self.assertRaises(ScaffoldError):
            add_registry_entry("STRATEGIES = dict()\n", self.spec)


class ScaffoldTreeTestCase(unittest.TestCase):
    """A throwaway repository tree holding the real registry and spec directory."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / REGISTRY_PATH).parent.mkdir(parents=True)
        shutil.copy(project_root / REGISTRY_PATH, self.root / REGISTRY_PATH)
        self.specs = self.root / 'niffler' / 'strategies' / 'specs'
        shutil.copytree(SPEC_DIR, self.specs)

    def add_spec(self, key: str, template: str = 'breakout', implemented: bool = False,
                 extra: str = '') -> Path:
        """Write a spec under a new key, copied from a shipped one."""
        camel = ''.join(part.capitalize() for part in key.split('_'))
        text = (SPEC_DIR / f"{template}.toml").read_text(encoding='utf-8')
        text = text.replace(f'key = "{template}"', f'key = "{key}"')
        text = text.replace(f'class_name = "{load_spec(SPEC_DIR / f"{template}.toml").class_name}"',
                            f'class_name = "{camel}Strategy"')
        if not implemented:
            text = '\n'.join(line for line in text.splitlines()
                             if not line.startswith('implemented ='))
        path = self.specs / f"{key}.toml"
        path.write_text(extra + text, encoding='utf-8')
        return path


class TestPlanAndWrite(ScaffoldTreeTestCase):

    def test_writes_the_module_the_test_and_the_registry(self):
        spec = load_spec(self.add_spec('channel_probe'))
        written = write_scaffold(plan_scaffold(spec, self.root), self.root)

        self.assertEqual({strategy_module_path(spec): True, test_module_path(spec): True,
                          REGISTRY_PATH: False}, written)
        self.assertIn('class ChannelProbeStrategy(BaseStrategy)',
                      (self.root / strategy_module_path(spec)).read_text(encoding='utf-8'))
        self.assertIn("'channel_probe': ChannelProbeStrategy",
                      (self.root / REGISTRY_PATH).read_text(encoding='utf-8'))

    def test_an_existing_file_is_never_overwritten(self):
        spec = load_spec(self.add_spec('channel_probe'))
        target = self.root / strategy_module_path(spec)
        target.write_text('# hand-written\n', encoding='utf-8')

        with self.assertRaises(ScaffoldError) as raised:
            plan_scaffold(spec, self.root)
        self.assertIn('never overwrites', str(raised.exception))
        self.assertEqual('# hand-written\n', target.read_text(encoding='utf-8'))

    def test_an_implemented_spec_is_refused(self):
        with self.assertRaises(ScaffoldError) as raised:
            plan_scaffold(load_spec(self.specs / 'rsi.toml'), self.root)
        self.assertIn('nothing to scaffold', str(raised.exception))

    def test_an_unsupported_spec_is_refused_with_its_reasons(self):
        path = self.add_spec('short_probe')
        path.write_text(path.read_text(encoding='utf-8').replace(
            'capabilities = ["long_entry"', 'capabilities = ["short_entry", "long_entry"'),
            encoding='utf-8')
        with self.assertRaises(ScaffoldError) as raised:
            plan_scaffold(load_spec(path), self.root)
        self.assertIn('unsupported rather than approximated', str(raised.exception))
        self.assertIn('short_entry: ', str(raised.exception))


class TestCommandLine(ScaffoldTreeTestCase):

    def run_main(self, *argv):
        from scripts.scaffold_strategy import main

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('sys.argv', ['scaffold_strategy.py', *argv, '--root', str(self.root)]), \
                patch('sys.stdout', stdout), patch('sys.stderr', stderr):
            code = main()
        return code, stdout.getvalue(), stderr.getvalue()

    def test_scaffolds_a_spec_by_key(self):
        spec = load_spec(self.add_spec('channel_probe'))
        code, out, _ = self.run_main('channel_probe')

        self.assertEqual(0, code)
        self.assertTrue((self.root / strategy_module_path(spec)).exists())
        self.assertTrue((self.root / test_module_path(spec)).exists())
        self.assertIn('dates.implemented', out)

    def test_scaffolds_a_spec_by_path(self):
        path = self.add_spec('channel_probe')
        code, _, _ = self.run_main(str(path))
        self.assertEqual(0, code)

    def test_dry_run_writes_nothing(self):
        spec = load_spec(self.add_spec('channel_probe'))
        registry_before = (self.root / REGISTRY_PATH).read_text(encoding='utf-8')

        code, out, _ = self.run_main('channel_probe', '--dry-run')

        self.assertEqual(0, code)
        self.assertIn(f"would create {strategy_module_path(spec)}", out)
        self.assertIn(f"would edit   {REGISTRY_PATH}", out)
        self.assertFalse((self.root / strategy_module_path(spec)).exists())
        self.assertEqual(registry_before, (self.root / REGISTRY_PATH).read_text(encoding='utf-8'))

    def test_an_invalid_spec_exits_1_and_names_the_problem(self):
        self.add_spec('channel_probe', extra='author = "someone"\n')
        code, _, err = self.run_main('channel_probe')
        self.assertEqual(1, code)
        self.assertIn("unknown key 'author'", err)

    def test_a_missing_spec_exits_1(self):
        code, _, err = self.run_main('no_such_idea')
        self.assertEqual(1, code)
        self.assertIn('no_such_idea.toml', err)

    def test_list_shows_status_and_reasons(self):
        path = self.add_spec('short_probe')
        path.write_text(path.read_text(encoding='utf-8').replace(
            'capabilities = ["long_entry"', 'capabilities = ["short_entry", "long_entry"'),
            encoding='utf-8')
        self.add_spec('channel_probe')

        code, out, _ = self.run_main('--list')

        self.assertEqual(0, code)
        self.assertRegex(out, r'rsi\s+implemented\s+RSI Mean Reversion')
        self.assertRegex(out, r'channel_probe\s+specified')
        self.assertRegex(out, r'short_probe\s+unsupported')
        self.assertIn('short_entry: ', out)

    def test_no_spec_and_no_list_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_main()
        self.assertEqual(2, raised.exception.code)


if __name__ == '__main__':
    unittest.main()
