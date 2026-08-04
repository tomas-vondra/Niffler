"""
Unit tests for the shared JSON helpers.
"""

import io
import json
import math
import unittest

import numpy as np

from niffler.exporters.json_utils import (
    safe_json_dump,
    safe_json_dumps,
    sanitize_numeric_values,
)


class TestSanitizeNumericValues(unittest.TestCase):
    """Test cases for sanitize_numeric_values."""

    def test_replaces_non_finite_scalars(self):
        """inf, -inf and NaN become None."""
        self.assertIsNone(sanitize_numeric_values(float('inf')))
        self.assertIsNone(sanitize_numeric_values(float('-inf')))
        self.assertIsNone(sanitize_numeric_values(float('nan')))

    def test_keeps_finite_values(self):
        """Finite values, strings, booleans and None pass through unchanged."""
        self.assertEqual(sanitize_numeric_values(1.5), 1.5)
        self.assertEqual(sanitize_numeric_values(0), 0)
        self.assertEqual(sanitize_numeric_values("text"), "text")
        self.assertIs(sanitize_numeric_values(True), True)
        self.assertIsNone(sanitize_numeric_values(None))

    def test_nested_structures(self):
        """Nested dicts and lists are sanitized recursively."""
        data = {
            'a': float('inf'),
            'b': {'c': float('nan'), 'd': 1},
            'e': [1.0, float('-inf'), {'f': float('nan')}],
            'g': (float('inf'), 2)
        }

        result = sanitize_numeric_values(data)

        self.assertEqual(result, {
            'a': None,
            'b': {'c': None, 'd': 1},
            'e': [1.0, None, {'f': None}],
            'g': [None, 2]
        })

    def test_numpy_scalars_are_converted(self):
        """Numpy scalars become plain Python types so json can serialise them."""
        result = sanitize_numeric_values({
            'f': np.float64(1.5),
            'i': np.int64(7),
            'b': np.bool_(True),
            'nan': np.float64('nan'),
            'inf': np.float32('inf')
        })

        self.assertIsInstance(result['f'], float)
        self.assertEqual(result['f'], 1.5)
        self.assertIsInstance(result['i'], int)
        self.assertEqual(result['i'], 7)
        self.assertIs(result['b'], True)
        self.assertIsNone(result['nan'])
        self.assertIsNone(result['inf'])

    def test_does_not_mutate_input(self):
        """The original structure is left untouched."""
        data = {'a': float('inf')}

        sanitize_numeric_values(data)

        self.assertTrue(math.isinf(data['a']))


class TestSafeJsonDumps(unittest.TestCase):
    """Test cases for safe_json_dumps / safe_json_dump."""

    def test_no_invalid_literals(self):
        """Output never contains the non-standard Infinity/NaN literals."""
        text = safe_json_dumps({'a': float('inf'), 'b': float('nan')})

        self.assertNotIn('Infinity', text)
        self.assertNotIn('NaN', text)
        self.assertEqual(json.loads(text), {'a': None, 'b': None})

    def test_forwards_kwargs(self):
        """Formatting kwargs are forwarded to json.dumps."""
        text = safe_json_dumps({'a': 1}, indent=2)
        self.assertIn('\n', text)

    def test_allow_nan_is_forced_off(self):
        """A caller cannot re-enable the invalid literals."""
        text = safe_json_dumps({'a': float('nan')}, allow_nan=True)
        self.assertEqual(json.loads(text), {'a': None})

    def test_default_hook_still_applies(self):
        """Non-serialisable objects can still be handled via default=."""
        class Custom:
            def __str__(self):
                return "custom"

        text = safe_json_dumps({'a': Custom()}, default=str)
        self.assertEqual(json.loads(text), {'a': 'custom'})

    def test_safe_json_dump_writes_valid_json(self):
        """safe_json_dump writes sanitized JSON to a file object."""
        buffer = io.StringIO()

        safe_json_dump({'profit_factor': float('inf'), 'trades': 3}, buffer, indent=2)

        buffer.seek(0)
        loaded = json.load(buffer)
        self.assertEqual(loaded, {'profit_factor': None, 'trades': 3})


class TestJsonUtilsLocation(unittest.TestCase):
    """The helpers live in the neutral utils package, with a compat re-export."""

    def test_canonical_module_exposes_the_helpers(self):
        from niffler.utils.json_utils import (
            safe_json_dump as canonical_dump,
            safe_json_dumps as canonical_dumps,
            sanitize_numeric_values as canonical_sanitize,
        )

        self.assertTrue(callable(canonical_dump))
        self.assertTrue(callable(canonical_dumps))
        self.assertTrue(callable(canonical_sanitize))

    def test_old_import_path_still_resolves_to_the_same_functions(self):
        from niffler.exporters import json_utils as legacy
        from niffler.utils import json_utils as canonical

        self.assertIs(legacy.safe_json_dump, canonical.safe_json_dump)
        self.assertIs(legacy.safe_json_dumps, canonical.safe_json_dumps)
        self.assertIs(legacy.sanitize_numeric_values, canonical.sanitize_numeric_values)


if __name__ == '__main__':
    unittest.main()
