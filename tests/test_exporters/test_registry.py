"""
The shared exporter contract.

These tests iterate the registry, so a newly registered exporter is covered
automatically - the same arrangement as tests/test_strategies/test_registry.py.
They pin the two properties that let ExporterManager derive an exporter's options
from its signature instead of a hand-written kwargs branch.
"""

import inspect
import shutil
import tempfile
import unittest
from unittest import mock
from typing import Any, Dict, Optional

from niffler.exporters.base_exporter import BaseExporter
from niffler.exporters.csv_exporter import CSVExporter
from niffler.exporters.registry import (
    EXPORTER_CLASSES,
    create_exporter,
    get_available_exporters,
    get_exporter_class,
    get_exporter_option_names,
)


class _ProbeExporter(BaseExporter):
    """A throwaway exporter with a distinctive option, registered in one edit."""

    def __init__(self, probe_token: str = 'unset', config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.probe_token = probe_token

    def export_backtest_result(self, result, backtest_id, metadata) -> None:
        pass


class TestRegistryContract(unittest.TestCase):
    """Properties every registered exporter must have."""

    def test_every_registered_class_is_an_exporter(self):
        for name, exporter_class in EXPORTER_CLASSES.items():
            with self.subTest(exporter=name):
                self.assertTrue(issubclass(exporter_class, BaseExporter))

    def test_no_exporter_swallows_options_with_var_keyword(self):
        """**kwargs would make the derived option set 'everything' again."""
        for name, exporter_class in EXPORTER_CLASSES.items():
            with self.subTest(exporter=name):
                kinds = [
                    parameter.kind
                    for parameter in inspect.signature(exporter_class.__init__).parameters.values()
                ]
                self.assertNotIn(inspect.Parameter.VAR_KEYWORD, kinds)
                self.assertNotIn(inspect.Parameter.VAR_POSITIONAL, kinds)

    def test_every_option_has_a_default(self):
        """The library builds exporters as exporter_class(**options)."""
        for name, exporter_class in EXPORTER_CLASSES.items():
            with self.subTest(exporter=name):
                for parameter_name, parameter in inspect.signature(
                    exporter_class.__init__
                ).parameters.items():
                    if parameter_name == 'self':
                        continue
                    self.assertIsNot(
                        parameter.default, inspect.Parameter.empty,
                        f"{exporter_class.__name__}.{parameter_name} has no default"
                    )

    def test_every_exporter_constructs_with_no_options(self):
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)

        for name in get_available_exporters():
            with self.subTest(exporter=name):
                # The CSV exporter creates its output directory on construction; keep
                # that out of the repository root.
                options = {'output_dir': temp_dir} if name == 'csv' else {}
                self.assertIsInstance(create_exporter(name, options), BaseExporter)

    def test_available_names_match_the_registry(self):
        self.assertEqual(get_available_exporters(), list(EXPORTER_CLASSES))


class TestLookup(unittest.TestCase):
    """Name resolution, including the CLI's whitespace and casing."""

    def test_names_are_normalised(self):
        self.assertIs(get_exporter_class('  CSV  '), CSVExporter)

    def test_unknown_name_lists_the_known_ones(self):
        with self.assertRaises(ValueError) as context:
            get_exporter_class('parquet')

        message = str(context.exception)
        self.assertIn('Unknown exporter type: parquet', message)
        for name in get_available_exporters():
            self.assertIn(name, message)


class TestOptionsAreDerivedFromTheSignature(unittest.TestCase):
    """The fix for the kwargs-filter seam: options come from inspect.signature."""

    def test_csv_accepts_output_dir(self):
        self.assertIn('output_dir', get_exporter_option_names('csv'))

    def test_console_does_not_accept_output_dir(self):
        self.assertNotIn('output_dir', get_exporter_option_names('console'))

    def test_elasticsearch_options_are_not_a_hand_written_list(self):
        options = get_exporter_option_names('elasticsearch')

        self.assertLessEqual(
            {'host', 'port', 'index_prefix', 'scheme', 'api_key', 'username',
             'password', 'timeout', 'verify_certs', 'config'},
            options
        )

    def test_csv_exporter_receives_its_output_dir(self):
        """The bug this seam produced: the option was dropped and defaults used."""
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)

        exporter = create_exporter('csv', {'output_dir': temp_dir})

        self.assertEqual(str(exporter.output_dir), temp_dir)

    def test_unaccepted_option_raises_naming_the_accepted_ones(self):
        with self.assertRaises(ValueError) as context:
            create_exporter('console', {'output_dir': 'results'})

        message = str(context.exception)
        self.assertIn('does not accept: output_dir', message)
        self.assertIn('It accepts: config', message)


class TestOneEditRegistration(unittest.TestCase):
    """A new exporter is one dict entry, and its option arrives intact."""

    def setUp(self):
        patcher = mock.patch.dict(
            EXPORTER_CLASSES, {'probe': _ProbeExporter}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_new_exporter_is_offered_to_the_cli(self):
        self.assertIn('probe', get_available_exporters())

    def test_the_new_exporter_receives_its_own_option(self):
        exporter = create_exporter('probe', {'probe_token': 'niffler-probe'})

        self.assertEqual(exporter.probe_token, 'niffler-probe')

    def test_an_option_the_new_exporter_lacks_still_raises(self):
        with self.assertRaises(ValueError) as context:
            create_exporter('probe', {'output_dir': 'results'})

        self.assertIn('probe_token', str(context.exception))


if __name__ == '__main__':
    unittest.main()
