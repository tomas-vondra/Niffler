"""Importing a CLI script must not reconfigure logging.

``scripts/download_data.py`` and ``scripts/preprocessor.py`` used to call
``setup_logging(level="INFO")`` at module scope. That made a plain
``import scripts.preprocessor`` - which every test module doing so performs -
attach handlers to the *root* logger and pin its level, silently overriding
whatever the importing process had already configured. It also ignored
``--log-level`` entirely, because the call had already happened by the time the
arguments were parsed.

The invariant this file protects: ``setup_logging`` is called from ``main()``,
never at import time, in every CLI script.
"""

import importlib
import logging
import sys
import unittest
from unittest.mock import patch

# Add project root to path for imports
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

#: Every module under scripts/ that owns a CLI entry point.
CLI_SCRIPT_MODULES = [
    'scripts.analyze',
    'scripts.backtest',
    'scripts.compare',
    'scripts.download_data',
    'scripts.optimize',
    'scripts.preprocessor',
    'scripts.screen',
]


class TestScriptImportSideEffects(unittest.TestCase):
    """Importing a script module must be free of logging side effects."""

    def test_import_does_not_call_setup_logging(self):
        """Re-importing each CLI script must not invoke setup_logging.

        The module body is re-executed with ``setup_logging`` patched out. A
        module that configures logging at import time calls the mock; one that
        defers it to ``main()`` does not.
        """
        for module_name in CLI_SCRIPT_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)

                with patch('niffler.config.logging.setup_logging') as mock_setup:
                    importlib.reload(module)

                mock_setup.assert_not_called()

    def test_import_does_not_touch_the_root_logger(self):
        """Re-importing each CLI script must not attach root logger handlers.

        A direct check on the observable damage, independent of how the script
        reaches the logging configuration.

        The root handlers are cleared first, deliberately. ``setup_logging``
        goes through ``logging.basicConfig``, which is a no-op once the root
        logger already has handlers - and by the time this runs, an earlier
        import has invariably configured them. Without the reset the assertion
        could never fail, which would make this test decorative.
        """
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level

        try:
            for module_name in CLI_SCRIPT_MODULES:
                with self.subTest(module=module_name):
                    module = importlib.import_module(module_name)

                    root.handlers = []
                    importlib.reload(module)

                    self.assertEqual(
                        root.handlers, [],
                        f"importing {module_name} attached a root log handler",
                    )
        finally:
            root.handlers = saved_handlers
            root.setLevel(saved_level)

    def test_every_cli_script_configures_logging_inside_a_function(self):
        """Each CLI script must still call setup_logging - just not at module scope.

        Guards the opposite failure: deleting the import-time call without
        adding a call inside ``main()`` would satisfy the tests above while
        leaving the scripts with no logging configuration at all. An indented
        call is inside a function; an unindented one runs on import.
        """
        for module_name in CLI_SCRIPT_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                with open(module.__file__, encoding='utf-8') as handle:
                    lines = handle.read().splitlines()

                calls = [ln for ln in lines if 'setup_logging(level=' in ln]
                self.assertTrue(
                    calls,
                    f"{module_name} no longer configures logging at all",
                )
                for line in calls:
                    self.assertNotEqual(
                        line, line.lstrip(),
                        f"{module_name} configures logging at module scope: {line!r}",
                    )


if __name__ == '__main__':
    unittest.main()
