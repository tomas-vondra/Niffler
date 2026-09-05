"""Exporter option handling in backtest.py.

Since the exporter registry made construction generic, ``backtest.py`` no longer
knows which option belongs to which exporter. These tests pin the rule that
replaced that knowledge: an option none of the chosen exporters accepts is an
error, never silently dropped - the same discipline ``--params`` follows for
strategies. The dropped option used to be invisible: the exporter was built on
its defaults and the run still exited 0.
"""

import argparse
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.backtest import build_exporter_options, main


def args_for(**overrides) -> argparse.Namespace:
    """Build a Namespace matching the parser's exporter-option defaults."""
    values = {
        'exporter_params': None,
        'csv_output_dir': None,
        'es_host': None,
        'es_port': None,
        'es_index_prefix': None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestBuildExporterOptions(unittest.TestCase):

    def test_no_flags_yields_no_options(self):
        """Unset flags must not be forwarded, or every exporter would see them."""
        self.assertEqual({}, build_exporter_options(args_for()))

    def test_named_flags_are_collected_under_their_constructor_names(self):
        options = build_exporter_options(
            args_for(csv_output_dir='results', es_host='es.example', es_port=9201)
        )

        self.assertEqual(
            {'output_dir': 'results', 'host': 'es.example', 'port': 9201}, options
        )

    def test_exporter_params_json_is_parsed(self):
        options = build_exporter_options(
            args_for(exporter_params='{"index_prefix": "custom", "timeout": 5}')
        )

        self.assertEqual({'index_prefix': 'custom', 'timeout': 5}, options)

    def test_an_explicit_flag_overrides_exporter_params_json(self):
        options = build_exporter_options(
            args_for(exporter_params='{"output_dir": "from-json"}',
                     csv_output_dir='from-flag')
        )

        self.assertEqual({'output_dir': 'from-flag'}, options)

    def test_invalid_json_is_reported_against_the_flag(self):
        with self.assertRaises(ValueError) as context:
            build_exporter_options(args_for(exporter_params='{not json'))

        self.assertIn('--exporter-params', str(context.exception))

    def test_non_object_json_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            build_exporter_options(args_for(exporter_params='[1, 2]'))

        self.assertIn('must be a JSON object', str(context.exception))


class TestExporterOptionsThroughMain(unittest.TestCase):
    """End to end: the option reaches the exporter, or the run fails loudly."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def _run(self, argv):
        data = pd.DataFrame({
            'open': [100.0 + i for i in range(30)],
            'high': [105.0 + i for i in range(30)],
            'low': [95.0 + i for i in range(30)],
            'close': [102.0 + i for i in range(30)],
            'volume': [1000.0] * 30,
        }, index=pd.date_range('2024-01-01', periods=30, freq='D'))
        stderr = io.StringIO()

        with patch('sys.argv', argv), \
                patch('scripts.backtest.load_data', return_value=data), \
                patch('scripts.backtest.setup_logging'), \
                patch('sys.stdout', new=io.StringIO()), \
                patch('sys.stderr', new=stderr):
            exit_code = main()

        return exit_code, stderr.getvalue()

    def test_an_option_no_chosen_exporter_accepts_exits_non_zero(self):
        exit_code, stderr = self._run(
            ['backtest.py', '--data', 'ok.csv', '--exporters', 'console',
             '--csv-output-dir', self.temp_dir]
        )

        self.assertEqual(1, exit_code)
        self.assertIn('No requested exporter accepts: output_dir', stderr)

    def test_the_csv_exporter_actually_receives_the_output_dir(self):
        """Positive control: the file lands where --csv-output-dir said."""
        target = Path(self.temp_dir) / 'results'

        exit_code, stderr = self._run(
            ['backtest.py', '--data', 'ok.csv', '--exporters', 'csv',
             '--csv-output-dir', str(target)]
        )

        self.assertEqual(0, exit_code, f"unexpected failure: {stderr}")
        self.assertTrue(any(target.iterdir()))

    def test_options_reach_an_exporter_through_exporter_params(self):
        """The generic path: no new flag is needed to configure an exporter."""
        target = Path(self.temp_dir) / 'generic'

        exit_code, stderr = self._run(
            ['backtest.py', '--data', 'ok.csv', '--exporters', 'csv',
             '--exporter-params', '{"output_dir": "%s"}' % target.as_posix()]
        )

        self.assertEqual(0, exit_code, f"unexpected failure: {stderr}")
        self.assertTrue(any(target.iterdir()))


if __name__ == '__main__':
    unittest.main()
