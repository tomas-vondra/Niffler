"""
Tests for niffler.utils.provenance.

Two rules drive these tests:

* **Provenance must never raise and never block.** Every failure mode the collector can
  meet in the wild - no git binary, not a repository, a git call that hangs, a missing
  or unreadable data file, an uninstalled package - is exercised here and must degrade
  to ``None`` rather than propagate.
* **They must not depend on the git state of the machine running them.** Every git
  lookup is mocked at the ``subprocess.run`` boundary, so the suite behaves identically
  on a clean CI checkout and on a developer's dirty working tree.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from importlib import metadata as importlib_metadata
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from niffler.utils import provenance as prov
from niffler.utils.json_utils import safe_json_dumps
from niffler.utils.provenance import (
    GIT_TIMEOUT_SECONDS,
    TRACKED_PACKAGES,
    collect_provenance,
    format_provenance_summary,
)


def _completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    """Build a CompletedProcess stand-in for a mocked ``git`` invocation."""
    return subprocess.CompletedProcess(
        args=['git'], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _git_responses(sha: str = 'a' * 40, branch: str = 'master', status: str = ""):
    """
    Build a side_effect sequence matching the three calls _collect_code_provenance makes.

    Order matters: rev-parse HEAD, rev-parse --abbrev-ref HEAD, status --porcelain.
    """
    return [_completed(sha), _completed(branch), _completed(status)]


class ProvenanceTestCase(unittest.TestCase):
    """Base case that isolates the module's per-process memoisation between tests."""

    def setUp(self):
        self._clear_caches()
        self.addCleanup(self._clear_caches)

    @staticmethod
    def _clear_caches():
        prov._collect_code_provenance.cache_clear()
        prov._collect_environment_provenance.cache_clear()
        prov._hash_file.cache_clear()


class TestCodeProvenance(ProvenanceTestCase):
    """Git identity collection, including the failure modes that must not raise."""

    def test_collects_sha_branch_and_clean_flag(self):
        """A clean repository reports its SHA, branch and dirty=False."""
        with patch('niffler.utils.provenance.subprocess.run',
                   side_effect=_git_responses(sha='b' * 40, branch='feat/provenance')):
            code = collect_provenance()['code']

        self.assertEqual(code['git_sha'], 'b' * 40)
        self.assertEqual(code['git_sha_short'], 'b' * 12)
        self.assertEqual(code['branch'], 'feat/provenance')
        self.assertFalse(code['dirty'])

    def test_dirty_working_tree_is_flagged(self):
        """Uncommitted changes must be recorded: the SHA alone would be a lie."""
        with patch('niffler.utils.provenance.subprocess.run',
                   side_effect=_git_responses(status=' M niffler/utils/provenance.py')):
            code = collect_provenance()['code']

        self.assertTrue(code['dirty'])

    def test_not_a_git_repository_degrades_to_unknown(self):
        """Outside a repository every git field is None - and dirty is None, not False."""
        with patch('niffler.utils.provenance.subprocess.run',
                   return_value=_completed(returncode=128, stderr='not a git repository')) as run:
            code = collect_provenance()['code']

        self.assertIsNone(code['git_sha'])
        self.assertIsNone(code['git_sha_short'])
        self.assertIsNone(code['branch'])
        # None means "not determined". False would assert a cleanliness never checked.
        self.assertIsNone(code['dirty'])
        # One failed call is enough to know the rest will fail; do not shell out again.
        self.assertEqual(run.call_count, 1)

    def test_missing_git_binary_degrades_to_unknown(self):
        """A machine without git still runs backtests; provenance just says 'unknown'."""
        with patch('niffler.utils.provenance.subprocess.run',
                   side_effect=FileNotFoundError('git not found')):
            code = collect_provenance()['code']

        self.assertIsNone(code['git_sha'])
        self.assertIsNone(code['dirty'])

    def test_git_timeout_degrades_to_unknown(self):
        """A hung git call is bounded and degrades - it must never stall a run."""
        with patch('niffler.utils.provenance.subprocess.run',
                   side_effect=subprocess.TimeoutExpired(cmd='git', timeout=GIT_TIMEOUT_SECONDS)):
            code = collect_provenance()['code']

        self.assertIsNone(code['git_sha'])

    def test_git_is_called_with_a_timeout_and_without_check(self):
        """The subprocess contract: bounded, and non-zero exits handled, not raised."""
        with patch('niffler.utils.provenance.subprocess.run',
                   side_effect=_git_responses()) as run:
            collect_provenance()

        for call in run.call_args_list:
            self.assertEqual(call.kwargs['timeout'], GIT_TIMEOUT_SECONDS)
            self.assertFalse(call.kwargs['check'])

    def test_status_failure_leaves_dirty_unknown(self):
        """If only `git status` fails, the SHA is still recorded but dirty stays None."""
        responses = [_completed('c' * 40), _completed('master'), _completed(returncode=1)]
        with patch('niffler.utils.provenance.subprocess.run', side_effect=responses):
            code = collect_provenance()['code']

        self.assertEqual(code['git_sha'], 'c' * 40)
        self.assertIsNone(code['dirty'])

    def test_git_lookups_are_memoised_per_process(self):
        """optimize.py may export hundreds of results; git must be shelled out once."""
        with patch('niffler.utils.provenance.subprocess.run',
                   side_effect=_git_responses()) as run:
            first = collect_provenance()
            second = collect_provenance()

        self.assertEqual(run.call_count, 3)
        self.assertEqual(first['code'], second['code'])

    def test_memoised_record_is_copied_per_call(self):
        """A caller mutating its record must not poison every later run in the process."""
        with patch('niffler.utils.provenance.subprocess.run',
                   side_effect=_git_responses(branch='master')):
            first = collect_provenance()
            first['code']['branch'] = 'tampered'
            second = collect_provenance()

        self.assertEqual(second['code']['branch'], 'master')


class TestDataProvenance(ProvenanceTestCase):
    """Input-file fingerprinting."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_file = Path(self.temp_dir.name) / 'BTCUSDT_binance_1d.csv'
        self.content = b'timestamp,open,high,low,close,volume\n2024-01-01,1,2,0,1.5,100\n'
        self.data_file.write_bytes(self.content)

    def _collect(self, path):
        with patch('niffler.utils.provenance.subprocess.run', side_effect=_git_responses()):
            return collect_provenance(path)['data']

    def test_hashes_the_data_file(self):
        """The recorded SHA-256 is the real digest of the file's bytes."""
        data = self._collect(self.data_file)

        self.assertEqual(data['sha256'], hashlib.sha256(self.content).hexdigest())
        self.assertEqual(data['size_bytes'], len(self.content))
        self.assertEqual(data['path'], str(self.data_file.resolve()))
        self.assertIsNotNone(data['modified_utc'])

    def test_no_data_path_yields_no_data_block(self):
        """Runs without a single input file report data=None rather than a stub."""
        with patch('niffler.utils.provenance.subprocess.run', side_effect=_git_responses()):
            self.assertIsNone(collect_provenance()['data'])

    def test_missing_file_degrades_but_keeps_the_path(self):
        """A path that does not exist must not raise - and the path is still recorded."""
        missing = Path(self.temp_dir.name) / 'does_not_exist.csv'

        data = self._collect(missing)

        self.assertIsNone(data['sha256'])
        self.assertIsNone(data['size_bytes'])
        self.assertIsNone(data['modified_utc'])
        # Knowing which file was meant is more useful than an empty block.
        self.assertEqual(data['path'], str(missing.resolve()))

    def test_unreadable_file_degrades_to_no_hash(self):
        """A permission error while hashing loses the digest, not the size or mtime."""
        with patch('builtins.open', side_effect=PermissionError('denied')):
            data = self._collect(self.data_file)

        self.assertIsNone(data['sha256'])
        self.assertEqual(data['size_bytes'], len(self.content))

    def test_large_file_is_hashed_in_chunks(self):
        """A multi-hundred-MB CSV must be streamed, never slurped into memory."""
        big = Path(self.temp_dir.name) / 'big.csv'
        payload = b'x' * (prov._HASH_CHUNK_BYTES * 2 + 17)
        big.write_bytes(payload)

        read_sizes = []
        real_open = open

        class RecordingFile:
            """Real file handle that records the size argument of every read."""

            def __init__(self, handle):
                self._handle = handle

            def read(self, *args):
                read_sizes.append(args)
                return self._handle.read(*args)

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                self._handle.close()
                return False

        def recording_open(path, *args, **kwargs):
            handle = real_open(path, *args, **kwargs)
            # Only the data file is instrumented: patching open globally would also
            # catch the pyproject.toml fallback read and pollute the recording.
            if str(path) == str(big.resolve()):
                return RecordingFile(handle)
            return handle

        with patch('builtins.open', side_effect=recording_open):
            data = self._collect(big)

        self.assertEqual(data['sha256'], hashlib.sha256(payload).hexdigest())
        # Two full chunks, the 17-byte remainder and the empty read that ends the loop -
        # never a single unbounded read() of the whole file.
        self.assertEqual(len(read_sizes), 4)
        for args in read_sizes:
            self.assertEqual(args, (prov._HASH_CHUNK_BYTES,))

    def test_hash_is_cached_per_file_identity(self):
        """The same unchanged file is hashed once, however many runs ask for it."""
        real_open = open
        opened = []

        def counting_open(path, *args, **kwargs):
            opened.append(str(path))
            return real_open(path, *args, **kwargs)

        with patch('builtins.open', side_effect=counting_open):
            first = self._collect(self.data_file)
            second = self._collect(self.data_file)

        self.assertEqual(first['sha256'], second['sha256'])
        self.assertEqual(opened.count(str(self.data_file.resolve())), 1)

    def test_rewritten_file_is_rehashed(self):
        """mtime and size are part of the cache key, so an edited file is not stale."""
        first = self._collect(self.data_file)

        new_content = self.content + b'2024-01-02,2,3,1,2.5,200\n'
        self.data_file.write_bytes(new_content)
        second = self._collect(self.data_file)

        self.assertNotEqual(first['sha256'], second['sha256'])
        self.assertEqual(second['sha256'], hashlib.sha256(new_content).hexdigest())


class TestEnvironmentProvenance(ProvenanceTestCase):
    """Interpreter, platform and library versions."""

    def test_reports_python_platform_and_package_versions(self):
        with patch('niffler.utils.provenance.subprocess.run', side_effect=_git_responses()):
            environment = collect_provenance()['environment']

        self.assertTrue(environment['python_version'])
        self.assertTrue(environment['platform'])
        self.assertEqual(set(environment['packages']), set(TRACKED_PACKAGES))
        self.assertTrue(environment['packages']['pandas'])
        self.assertTrue(environment['packages']['numpy'])

    def test_uninstalled_package_maps_to_none(self):
        """'We looked and it was absent' is a fact worth recording, so None, not omitted."""
        def fake_version(name):
            if name == 'ccxt':
                raise importlib_metadata.PackageNotFoundError(name)
            return '1.2.3'

        with patch('niffler.utils.provenance.importlib_metadata.version', side_effect=fake_version), \
                patch('niffler.utils.provenance.subprocess.run', side_effect=_git_responses()):
            packages = collect_provenance()['environment']['packages']

        self.assertIn('ccxt', packages)
        self.assertIsNone(packages['ccxt'])
        self.assertEqual(packages['pandas'], '1.2.3')

    def test_niffler_version_falls_back_to_pyproject(self):
        """Niffler runs as a checkout, not an installed distribution - read pyproject."""
        with patch('niffler.utils.provenance.importlib_metadata.version',
                   side_effect=importlib_metadata.PackageNotFoundError('niffler')), \
                patch('niffler.utils.provenance.subprocess.run', side_effect=_git_responses()):
            code = collect_provenance()['code']

        self.assertIsNotNone(code['niffler_version'])

    def test_unreadable_pyproject_leaves_version_none(self):
        with patch('niffler.utils.provenance.importlib_metadata.version',
                   side_effect=importlib_metadata.PackageNotFoundError('niffler')), \
                patch('builtins.open', side_effect=OSError('gone')), \
                patch('niffler.utils.provenance.subprocess.run', side_effect=_git_responses()):
            code = collect_provenance()['code']

        self.assertIsNone(code['niffler_version'])

    def test_environment_lookup_is_memoised(self):
        with patch('niffler.utils.provenance.importlib_metadata.version',
                   return_value='1.0.0') as version, \
                patch('niffler.utils.provenance.subprocess.run', side_effect=_git_responses()):
            collect_provenance()
            collect_provenance()

        # One niffler lookup for the code block plus one per tracked package.
        self.assertEqual(version.call_count, 1 + len(TRACKED_PACKAGES))


class TestCollectProvenanceContract(ProvenanceTestCase):
    """Shape, serialisability and the last-resort guard."""

    def test_record_has_the_expected_top_level_shape(self):
        with patch('niffler.utils.provenance.subprocess.run', side_effect=_git_responses()):
            record = collect_provenance()

        self.assertEqual(
            set(record), {'run_timestamp_utc', 'code', 'data', 'environment'}
        )
        self.assertTrue(record['run_timestamp_utc'].endswith('+00:00'))

    def test_record_round_trips_through_safe_json_dumps(self):
        """Everything exporters write goes through safe_json_dumps(allow_nan=False)."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as handle:
            handle.write(b'timestamp,close\n2024-01-01,1\n')
            path = handle.name
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        with patch('niffler.utils.provenance.subprocess.run', side_effect=_git_responses()):
            record = collect_provenance(path)

        restored = json.loads(safe_json_dumps(record))
        self.assertEqual(restored['code']['git_sha'], 'a' * 40)
        self.assertEqual(restored['data']['size_bytes'], record['data']['size_bytes'])

    def test_unexpected_failure_still_returns_a_record(self):
        """The last-resort guard: provenance must never take a backtest down with it."""
        with patch('niffler.utils.provenance._collect_code_provenance',
                   side_effect=RuntimeError('boom')):
            record = collect_provenance()

        self.assertEqual(
            set(record), {'run_timestamp_utc', 'code', 'data', 'environment'}
        )
        self.assertIsNone(record['code'])


class TestLayering(unittest.TestCase):
    """niffler/utils/ must not import from the layers above it."""

    def test_module_imports_only_the_standard_library(self):
        """A helper must never drag an optional third-party dependency along with it."""
        import ast

        source = Path(prov.__file__).read_text(encoding='utf-8')
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.level:
                self.fail(f"relative import in a layer-neutral module: {node.module}")

        third_party = imported - set(sys.stdlib_module_names)
        self.assertEqual(third_party, set())


class TestFormatProvenanceSummary(unittest.TestCase):
    """The one-line console rendering."""

    def test_none_and_empty_render_as_none(self):
        self.assertIsNone(format_provenance_summary(None))
        self.assertIsNone(format_provenance_summary({}))

    def test_clean_run_shows_sha_and_branch(self):
        summary = format_provenance_summary({
            'code': {'git_sha_short': 'abc123def456', 'branch': 'master', 'dirty': False},
            'data': {'sha256': 'f' * 64},
        })

        self.assertIn('abc123def456', summary)
        self.assertIn('master', summary)
        self.assertNotIn('DIRTY', summary)
        self.assertIn('f' * 12, summary)

    def test_dirty_run_is_marked_loudly(self):
        summary = format_provenance_summary({
            'code': {'git_sha_short': 'abc123def456', 'branch': 'master', 'dirty': True},
        })

        self.assertIn('DIRTY', summary)

    def test_unknown_dirty_state_is_distinguished_from_clean(self):
        summary = format_provenance_summary({
            'code': {'git_sha_short': 'abc123def456', 'branch': None, 'dirty': None},
        })

        self.assertIn('dirty-unknown', summary)

    def test_missing_code_block_renders_unknown(self):
        summary = format_provenance_summary({'code': None, 'data': None})

        self.assertEqual(summary, 'code unknown')


if __name__ == '__main__':
    unittest.main()
