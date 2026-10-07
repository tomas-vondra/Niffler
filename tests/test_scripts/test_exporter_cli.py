"""The exporter flags every script shares.

The behaviour worth pinning is what happens at the edges: ``--output`` on a
script whose default exporter is the console must still write a file, an
exporter that cannot export the run must stop it before any work is done, and
the file an optimization writes must be readable as a params file by the next
step, parent link included.
"""

import argparse
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.exporters import ExporterManager
from scripts import analyze, backtest, compare, optimize, screen
from scripts.common import (
    add_exporter_arguments,
    configure_exporters,
    read_params_file,
)


def parse(argv, default='console', with_output=True):
    parser = argparse.ArgumentParser(add_help=False)
    add_exporter_arguments(parser, default=default)
    if with_output:
        parser.add_argument('--output', default=None)
    return parser.parse_args(argv)


def exporter_classes(manager):
    return [type(exporter).__name__ for exporter in manager.exporters]


class TestConfigureExporters(unittest.TestCase):

    def test_the_default_list_is_used(self):
        manager = ExporterManager()
        configure_exporters(manager, parse([]), 'walk_forward')
        self.assertEqual(exporter_classes(manager), ['ConsoleExporter'])

    def test_output_implies_the_json_exporter(self):
        """analyze.py --output x.json predates shared exporters and must still write."""
        manager = ExporterManager()
        configure_exporters(manager, parse(['--output', 'x.json']), 'walk_forward')
        self.assertEqual(exporter_classes(manager), ['ConsoleExporter', 'JsonExporter'])
        self.assertEqual(manager.exporters[1].output_path, 'x.json')

    def test_output_does_not_add_json_twice(self):
        manager = ExporterManager()
        configure_exporters(
            manager, parse(['--exporters', 'json', '--output', 'x.json']), 'optimize')
        self.assertEqual(exporter_classes(manager), ['JsonExporter'])

    def test_the_default_output_applies_only_when_json_is_requested(self):
        manager = ExporterManager()
        configure_exporters(manager, parse([], default='console,json'), 'optimize',
                            default_output='auto.json')
        self.assertEqual(manager.exporters[1].output_path, 'auto.json')

        console_only = ExporterManager()
        # No json exporter, so the path must not be broadcast as an orphan option.
        configure_exporters(console_only, parse(['--exporters', 'console']), 'optimize',
                            default_output='auto.json')
        self.assertEqual(exporter_classes(console_only), ['ConsoleExporter'])

    def test_output_beats_the_default_output(self):
        manager = ExporterManager()
        configure_exporters(manager, parse(['--output', 'mine.json'], default='json'),
                            'optimize', default_output='auto.json')
        self.assertEqual(manager.exporters[0].output_path, 'mine.json')

    def test_an_exporter_that_cannot_export_the_kind_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            configure_exporters(ExporterManager(), parse(['--exporters', 'csv']),
                                'monte_carlo')
        self.assertIn('monte_carlo', str(ctx.exception))

    def test_an_option_no_exporter_accepts_is_still_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            configure_exporters(
                ExporterManager(), parse(['--csv-output-dir', 'results']), 'optimize')
        self.assertIn('output_dir', str(ctx.exception))

    def test_every_script_declares_the_same_exporter_flags(self):
        expected = {'exporters', 'exporter_params', 'csv_output_dir',
                    'es_host', 'es_port', 'es_index_prefix'}
        self.assertLessEqual(expected, set(vars(parse([]))))
        for module in (analyze, backtest, compare, optimize, screen):
            with self.subTest(script=module.__name__):
                source = Path(module.__file__).read_text(encoding='utf-8')
                self.assertIn('add_exporter_arguments(parser', source)


class TestOptimizeMain(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def run_main(self, *argv):
        optimizer = Mock()
        optimizer.optimize.return_value = [Mock()]
        optimizer.analyze_best_metrics.return_value = {}
        optimizer.results_truncated = False
        optimizer.results_document.return_value = {
            'metadata': {'sort_by': 'total_return'},
            'results': [{'parameters': {'short_window': 7, 'long_window': 33},
                         'metrics': {}}],
        }
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('sys.argv', ['optimize.py', '--data', 'test.csv', '--strategy',
                                'simple_ma', '--top-n', '0', '--no-plateau']
                   + list(argv)), \
                patch('scripts.optimize.setup_logging'), \
                patch('scripts.optimize.load_and_validate_data', return_value=Mock()), \
                patch('scripts.optimize.collect_provenance', return_value={'code': None}), \
                patch('scripts.optimize.create_optimizer', return_value=optimizer), \
                patch('sys.stderr', new=stderr), redirect_stdout(stdout):
            code = optimize.main()
        return code, optimizer, stdout.getvalue()

    def test_an_unusable_exporter_stops_the_run_before_the_search(self):
        code, optimizer, _ = self.run_main('--exporters', 'csv')
        self.assertEqual(code, 1)
        optimizer.optimize.assert_not_called()

    def test_the_saved_file_is_a_params_file_that_names_its_run(self):
        """optimize -> --params-file is the chain the whole design exists for."""
        output = os.path.join(self.temp_dir, 'opt.json')
        code, _, stdout = self.run_main('--output', output, '--experiment', 'exp-a')
        self.assertEqual(code, 0)

        parameters, parent = read_params_file(output)
        self.assertEqual(parameters, {'short_window': 7, 'long_window': 33})
        self.assertEqual(parent.experiment, 'exp-a')
        self.assertIn(parent.run_id, stdout)

    def test_console_only_writes_no_file(self):
        original = os.getcwd()
        os.chdir(self.temp_dir)
        try:
            code, _, _ = self.run_main('--exporters', 'console')
        finally:
            os.chdir(original)
        self.assertEqual(code, 0)
        self.assertEqual(os.listdir(self.temp_dir), [])

    def test_a_failed_export_fails_the_run(self):
        missing = os.path.join(self.temp_dir, 'missing', 'opt.json')
        code, _, _ = self.run_main('--output', missing)
        self.assertEqual(code, 1)


if __name__ == '__main__':
    unittest.main()
