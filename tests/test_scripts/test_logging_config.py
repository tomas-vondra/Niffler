import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.config.logging import VALID_LOG_LEVELS, resolve_log_level, setup_logging


class TestResolveLogLevel(unittest.TestCase):
    """Tests for log level validation."""

    def test_known_levels(self):
        """Every advertised level resolves to its logging constant."""
        for name in VALID_LOG_LEVELS:
            self.assertEqual(resolve_log_level(name), getattr(logging, name))

    def test_level_is_case_insensitive_and_stripped(self):
        """Levels are normalised before lookup."""
        self.assertEqual(resolve_log_level(' debug '), logging.DEBUG)

    def test_typo_raises_value_error(self):
        """A typo raises a clear ValueError instead of AttributeError."""
        with self.assertRaises(ValueError) as context:
            resolve_log_level('INFOO')
        self.assertIn('Invalid log level', str(context.exception))
        self.assertIn('DEBUG', str(context.exception))

    def test_non_level_attribute_raises_value_error(self):
        """An unrelated logging attribute is not accepted as a level."""
        # getattr(logging, "handlers".upper()) used to be the failure mode here.
        with self.assertRaises(ValueError):
            resolve_log_level('handlers')

    def test_non_string_raises_type_error(self):
        """A non-string level raises TypeError with guidance."""
        with self.assertRaises(TypeError):
            resolve_log_level(10)


class TestSetupLogging(unittest.TestCase):
    """Tests for setup_logging."""

    def setUp(self):
        """Remember the root logger configuration."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = logging.getLogger()
        self.original_handlers = list(self.root.handlers)
        self.original_level = self.root.level

    def tearDown(self):
        """Restore the root logger configuration."""
        for handler in list(self.root.handlers):
            if handler not in self.original_handlers:
                handler.close()
                self.root.removeHandler(handler)
        for handler in self.original_handlers:
            if handler not in self.root.handlers:
                self.root.addHandler(handler)
        self.root.setLevel(self.original_level)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_level_raises_before_configuring(self):
        """An invalid level is rejected without touching logging config."""
        with patch('logging.basicConfig') as mock_basic_config:
            with self.assertRaises(ValueError):
                setup_logging(level='verbose')
            mock_basic_config.assert_not_called()

    def test_valid_level_is_passed_through(self):
        """A valid level reaches basicConfig as a numeric level."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(level='debug')

        _, kwargs = mock_basic_config.call_args
        self.assertEqual(kwargs['level'], logging.DEBUG)

    def test_log_to_file_creates_parent_directory(self):
        """A log file in a missing directory is created on demand."""
        log_file = os.path.join(self.temp_dir, 'nested', 'niffler.log')

        setup_logging(level='INFO', log_to_file=True, log_file=log_file)

        self.assertTrue(os.path.isdir(os.path.dirname(log_file)))


if __name__ == '__main__':
    unittest.main()
