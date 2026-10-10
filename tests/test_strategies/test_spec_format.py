"""The strategy spec format: what loads, and what is refused with a reason.

Each test starts from a valid document and breaks one thing, so a failure names
exactly the rule that stopped holding.
"""

import copy
import datetime
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.strategies.spec import (
    POSITION_SIZE_SEARCH,
    SPEC_DIR,
    STATUS_IMPLEMENTED,
    STATUS_SPECIFIED,
    STATUS_UNSUPPORTED,
    SpecError,
    load_spec,
    parse_spec,
)


def valid_document() -> dict:
    """A valid spec document, freshly parsed, not yet implemented."""
    with (SPEC_DIR / 'rsi.toml').open('rb') as handle:
        raw = tomllib.load(handle)
    raw['key'] = 'example'
    raw['class_name'] = 'ExampleStrategy'
    del raw['dates']['implemented']
    return raw


class SpecFormatTestCase(unittest.TestCase):

    def setUp(self):
        self.raw = valid_document()

    def assert_refused(self, raw: dict, *fragments: str) -> SpecError:
        with self.assertRaises(SpecError) as raised:
            parse_spec(raw)
        message = str(raised.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return raised.exception


class TestAValidSpec(SpecFormatTestCase):

    def test_loads(self):
        spec = parse_spec(self.raw)
        self.assertEqual('example', spec.key)
        self.assertEqual('example_strategy', spec.module)
        self.assertEqual(['rsi_period', 'oversold', 'overbought'], [p.name for p in spec.parameters])
        self.assertEqual(('close',), spec.rules.inputs)
        self.assertEqual(datetime.date(2026, 8, 7), spec.found)

    def test_without_an_implementation_date_it_is_specified(self):
        self.assertEqual(STATUS_SPECIFIED, parse_spec(self.raw).status)

    def test_with_one_it_is_implemented(self):
        self.raw['dates']['implemented'] = datetime.date(2026, 9, 1)
        self.assertEqual(STATUS_IMPLEMENTED, parse_spec(self.raw).status)

    def test_position_size_is_added_to_the_search_and_the_defaults(self):
        spec = parse_spec(self.raw)
        self.assertEqual(POSITION_SIZE_SEARCH, spec.parameter_spec()['position_size'])
        self.assertEqual(1.0, spec.defaults()['position_size'])

    def test_a_parameter_without_search_is_left_out_of_parameter_spec(self):
        del self.raw['parameters']['oversold']['search']
        spec = parse_spec(self.raw)
        self.assertNotIn('oversold', spec.parameter_spec())
        self.assertIn('oversold', spec.defaults())

    def test_a_float_parameter_accepts_a_toml_integer(self):
        self.raw['parameters']['oversold'] = {
            'type': 'float', 'default': 30, 'description': 'level',
            'search': {'min': 20, 'max': 35, 'step': 2.5},
        }
        parameter = parse_spec(self.raw).parameters[1]
        self.assertIsInstance(parameter.default, float)
        self.assertEqual({'type': 'float', 'min': 20.0, 'max': 35.0, 'step': 2.5},
                         parameter.search_entry())

    def test_a_choice_parameter(self):
        self.raw['parameters']['smoothing'] = {
            'type': 'choice', 'default': 'wilder', 'description': 'average',
            'search': {'choices': ['wilder', 'simple']},
        }
        spec = parse_spec(self.raw)
        self.assertEqual({'type': 'choice', 'choices': ['wilder', 'simple']},
                         spec.parameter_spec()['smoothing'])

    def test_a_web_source_with_url_and_retrieval_date(self):
        self.raw['source'] = {
            'kind': 'web', 'reference': 'A blog post',
            'url': 'https://example.com/strategy', 'retrieved': datetime.date(2026, 8, 1),
        }
        self.assertEqual('https://example.com/strategy', parse_spec(self.raw).source.url)


class TestRefusals(SpecFormatTestCase):

    def test_every_problem_is_reported_at_once(self):
        self.raw['family'] = 'astrology'
        self.raw['source']['kind'] = 'rumour'
        error = self.assert_refused(self.raw, 'astrology', 'rumour')
        self.assertEqual(2, len(error.problems))

    def test_an_unknown_key_is_an_error_not_ignored(self):
        self.raw['parameters']['rsi_period']['publshed'] = 14
        self.assert_refused(self.raw, "unknown key 'parameters.rsi_period.publshed'")

    def test_an_unknown_top_level_key(self):
        self.raw['author'] = 'someone'
        self.assert_refused(self.raw, "unknown key 'author'")

    def test_a_missing_required_key(self):
        del self.raw['rules']['exit']
        self.assert_refused(self.raw, "missing required key 'rules.exit'")

    def test_the_format_version_is_checked(self):
        self.raw['spec_version'] = 2
        self.assert_refused(self.raw, 'spec_version')

    def test_key_must_be_snake_case(self):
        self.raw['key'] = 'My-Idea'
        self.assert_refused(self.raw, 'snake_case')

    def test_class_name_must_end_in_strategy(self):
        self.raw['class_name'] = 'Example'
        self.assert_refused(self.raw, "end in 'Strategy'")

    def test_position_size_is_reserved(self):
        self.raw['parameters']['position_size'] = {
            'type': 'float', 'default': 1.0, 'description': 'size'}
        self.assert_refused(self.raw, "'parameters.position_size' is reserved")

    def test_risk_manager_is_reserved(self):
        self.raw['parameters']['risk_manager'] = {
            'type': 'choice', 'default': 'none', 'description': 'x'}
        self.assert_refused(self.raw, 'reserved')

    def test_no_parameters(self):
        self.raw['parameters'] = {}
        self.assert_refused(self.raw, 'declares no parameter')

    def test_a_default_of_the_wrong_type(self):
        self.raw['parameters']['rsi_period']['default'] = 14.5
        self.assert_refused(self.raw, "'parameters.rsi_period.default' 14.5 is not of type int")

    def test_a_boolean_is_not_a_number(self):
        self.raw['parameters']['rsi_period']['default'] = True
        self.assert_refused(self.raw, 'is not of type int')

    def test_an_unknown_parameter_type(self):
        self.raw['parameters']['rsi_period']['type'] = 'decimal'
        self.assert_refused(self.raw, "'parameters.rsi_period.type' must be one of")

    def test_search_min_must_be_below_max(self):
        self.raw['parameters']['rsi_period']['search'] = {'min': 21, 'max': 7, 'step': 1}
        self.assert_refused(self.raw, 'min must be below max')

    def test_search_step_must_be_positive(self):
        self.raw['parameters']['rsi_period']['search'] = {'min': 7, 'max': 21, 'step': 0}
        self.assert_refused(self.raw, 'step must be positive')

    def test_search_step_is_required(self):
        self.raw['parameters']['rsi_period']['search'] = {'min': 7, 'max': 21}
        self.assert_refused(self.raw, "missing required key 'parameters.rsi_period.search.step'")

    def test_the_default_must_lie_in_the_searched_range(self):
        self.raw['parameters']['rsi_period']['default'] = 30
        self.assert_refused(self.raw, 'outside the searched range')

    def test_a_choice_default_must_be_a_choice(self):
        self.raw['parameters']['smoothing'] = {
            'type': 'choice', 'default': 'ema', 'description': 'average',
            'search': {'choices': ['wilder', 'simple']},
        }
        self.assert_refused(self.raw, "must include the default 'ema'")

    def test_inputs_must_be_ohlcv_columns(self):
        self.raw['rules']['inputs'] = ['close', 'vwap']
        self.assert_refused(self.raw, "['vwap']")

    def test_inputs_must_not_be_empty(self):
        self.raw['rules']['inputs'] = []
        self.assert_refused(self.raw, 'rules.inputs must be a non-empty list')

    def test_a_date_must_be_a_toml_date_not_a_string(self):
        self.raw['dates']['found'] = '2026-08-07'
        self.assert_refused(self.raw, "'dates.found' must be a date")

    def test_a_datetime_is_not_a_date(self):
        self.raw['dates']['found'] = datetime.datetime(2026, 8, 7, 12, 0)
        self.assert_refused(self.raw, "'dates.found' must be a date")

    def test_implemented_before_found(self):
        self.raw['dates']['implemented'] = datetime.date(2026, 1, 1)
        self.assert_refused(self.raw, 'dates.implemented is before dates.found')

    def test_a_web_source_needs_a_url(self):
        self.raw['source'] = {'kind': 'web', 'reference': 'A blog post'}
        self.assert_refused(self.raw, "source.url is required when source.kind is 'web'")

    def test_a_url_needs_a_retrieval_date(self):
        self.raw['source']['url'] = 'https://example.com'
        self.assert_refused(self.raw, 'source.retrieved is required with source.url')

    def test_a_url_must_be_http(self):
        self.raw['source']['url'] = 'file:///etc/passwd'
        self.raw['source']['retrieved'] = datetime.date(2026, 8, 1)
        self.assert_refused(self.raw, 'must start with https://')

    def test_an_unknown_capability(self):
        self.raw['requires']['capabilities'].append('time_travel')
        self.assert_refused(self.raw, "['time_travel']", 'requires.unlisted')

    def test_a_capability_listed_twice(self):
        self.raw['requires']['capabilities'].append('long_entry')
        self.assert_refused(self.raw, 'twice')

    def test_an_empty_unlisted_entry(self):
        self.raw['requires']['unlisted'] = ['  ']
        self.assert_refused(self.raw, 'requires.unlisted')


class TestUnsupported(SpecFormatTestCase):
    """An idea the engine cannot express is recorded as such, with the reason."""

    def test_an_unsupported_capability_makes_the_spec_unsupported(self):
        self.raw['requires']['capabilities'].append('short_entry')
        spec = parse_spec(self.raw)
        self.assertEqual(STATUS_UNSUPPORTED, spec.status)
        self.assertEqual(1, len(spec.unsupported_reasons()))
        self.assertTrue(spec.unsupported_reasons()[0].startswith('short_entry: '))
        self.assertIn('long-only', spec.unsupported_reasons()[0])

    def test_an_unlisted_need_makes_the_spec_unsupported(self):
        self.raw['requires']['unlisted'] = ['options implied volatility']
        spec = parse_spec(self.raw)
        self.assertEqual(STATUS_UNSUPPORTED, spec.status)
        self.assertEqual(['unlisted: options implied volatility'], spec.unsupported_reasons())

    def test_an_unsupported_spec_cannot_be_implemented(self):
        self.raw['requires']['capabilities'].append('take_profit')
        self.raw['dates']['implemented'] = datetime.date(2026, 9, 1)
        self.assert_refused(self.raw, 'dates.implemented is set', 'take_profit')


class TestLoadSpec(unittest.TestCase):

    def write(self, directory: str, name: str, text: str) -> Path:
        path = Path(directory) / name
        path.write_text(text, encoding='utf-8')
        return path

    def test_the_key_must_match_the_file_name(self):
        source = (SPEC_DIR / 'rsi.toml').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, 'other.toml', source)
            with self.assertRaises(SpecError) as raised:
                load_spec(path)
        self.assertIn("must match the file name 'other.toml'", str(raised.exception))

    def test_invalid_toml_names_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, 'broken.toml', 'key = \n')
            with self.assertRaises(SpecError) as raised:
                load_spec(path)
        self.assertIn('broken.toml', str(raised.exception))
        self.assertIn('not valid TOML', str(raised.exception))

    def test_a_missing_file(self):
        with self.assertRaises(SpecError) as raised:
            load_spec(SPEC_DIR / 'no_such_spec.toml')
        self.assertIn('cannot read', str(raised.exception))

    def test_the_document_is_not_mutated(self):
        raw = valid_document()
        before = copy.deepcopy(raw)
        parse_spec(raw)
        self.assertEqual(before, raw)


if __name__ == '__main__':
    unittest.main()
