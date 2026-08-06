"""
Run Provenance

Answers the question "what produced this number?" for every artefact Niffler writes.

A backtest result is only meaningful next to the code and the data that produced it. A
Sharpe ratio of 1.8 sitting in an Elasticsearch index with nothing but a UUID attached
cannot be reproduced, cannot be compared against a later run, and cannot be trusted -
it may have come from a commit that was later reverted, from a data file that has since
been re-downloaded, or from a working tree with uncommitted experiments in it.

:func:`collect_provenance` gathers three things:

* **code** - git SHA, branch and, crucially, whether the working tree was dirty. A SHA
  recorded against a dirty tree is a lie about reproducibility, so the flag is recorded
  and surfaced rather than quietly dropped.
* **data** - the resolved path, a streaming SHA-256 of the input file, its size and its
  modification time. Two runs over "the same" CSV are only the same run if the hashes
  match.
* **environment** - Python version, platform and the versions of the libraries whose
  behaviour moves the numbers (pandas, numpy, ccxt, yfinance).

Design rules that must not be undone:

* **Nothing here ever raises.** Provenance is metadata about a run, not part of it.
  Every failure path degrades to ``None`` and a debug/warning log. A backtest must
  never fail because ``git`` is missing.
* **Nothing here blocks.** Subprocess calls are bounded by
  :data:`GIT_TIMEOUT_SECONDS` and never raise on a non-zero exit.
* **Unknown is ``None``, never a plausible-looking default.** In particular ``dirty``
  is ``None`` when it could not be determined - reporting ``False`` would assert
  cleanliness that was never checked.
* **Lookups are memoised per process.** ``optimize.py`` may serialise hundreds of
  results from one run; the git and environment answers are constant for the process
  and the file hash is cached on (path, mtime, size).
* **Layer-neutral.** Like everything under ``niffler/utils/``, this module imports only
  from the standard library.

Usage::

    from niffler.utils.provenance import collect_provenance

    provenance = collect_provenance(args.data)   # collect ONCE per run, at the CLI
"""

import copy
import hashlib
import logging
import platform
import subprocess
import sys
import tomllib
from datetime import datetime, UTC
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    'collect_provenance',
    'format_provenance_summary',
    'GIT_TIMEOUT_SECONDS',
    'TRACKED_PACKAGES',
]

# Hard ceiling on any `git` invocation. Provenance collection must not be able to hang
# a research run - a repository on a stalled network mount degrades to "unknown".
GIT_TIMEOUT_SECONDS = 5.0

# Read the data file in 1 MiB chunks: a multi-hundred-megabyte CSV must never be
# slurped into memory just to fingerprint it.
_HASH_CHUNK_BYTES = 1024 * 1024

# Number of hex characters kept for the abbreviated SHA. Long enough to stay unique in
# any realistic repository, short enough to read in a console line or a Grafana legend.
_SHORT_SHA_LENGTH = 12

# Libraries whose version can move the numbers a backtest produces. `niffler` itself is
# included so a result can be tied back to a released version of the project.
TRACKED_PACKAGES = (
    'niffler',
    'pandas',
    'numpy',
    'ccxt',
    'yfinance',
    'elasticsearch',
    'python-dateutil',
)

# Repository root: niffler/utils/provenance.py -> niffler/utils -> niffler -> root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_git(*args: str) -> Optional[str]:
    """
    Run a ``git`` command in the repository root and return its trimmed stdout.

    Never raises and never blocks indefinitely: a missing ``git`` binary, a directory
    that is not a repository, a non-zero exit and a timeout all resolve to ``None``.

    Args:
        *args: Arguments passed to ``git`` (the executable itself is prepended)

    Returns:
        The command's stdout with surrounding whitespace stripped, or None if the
        command could not be run or failed
    """
    command = ['git', '-C', str(_REPO_ROOT), *args]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        # OSError covers a missing git binary (FileNotFoundError); SubprocessError
        # covers TimeoutExpired. Neither is worth failing a backtest over.
        logger.debug(f"Could not run {' '.join(command)}: {e}")
        return None

    if completed.returncode != 0:
        logger.debug(
            f"git exited {completed.returncode} for {' '.join(args)}: "
            f"{(completed.stderr or '').strip()}"
        )
        return None

    return completed.stdout.strip()


def _read_version_from_pyproject() -> Optional[str]:
    """
    Read ``project.version`` from the repository's ``pyproject.toml``.

    Niffler is developed as a checkout rather than an installed distribution, so
    ``importlib.metadata.version('niffler')`` raises ``PackageNotFoundError`` in the
    normal development environment. Falling back to ``pyproject.toml`` keeps the field
    useful instead of permanently ``None``.

    Returns:
        The declared project version, or None if it could not be read
    """
    pyproject = _REPO_ROOT / 'pyproject.toml'
    try:
        with open(pyproject, 'rb') as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.debug(f"Could not read version from {pyproject}: {e}")
        return None

    version = data.get('project', {}).get('version')
    return str(version) if version is not None else None


def _package_version(name: str) -> Optional[str]:
    """
    Resolve an installed distribution's version.

    Args:
        name: Distribution name as it appears on PyPI (e.g. "python-dateutil")

    Returns:
        The installed version, or None when the package is not installed
    """
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        if name == 'niffler':
            # Expected in a source checkout - fall back to the declared version.
            return _read_version_from_pyproject()
        logger.debug(f"Package not installed, version unknown: {name}")
        return None
    except Exception as e:  # pragma: no cover - defensive, metadata can be corrupt
        logger.debug(f"Could not resolve version for {name}: {e}")
        return None


@lru_cache(maxsize=1)
def _collect_code_provenance() -> Dict[str, Any]:
    """
    Collect git identity for the running code. Memoised: constant for the process.

    ``dirty`` is deliberately tri-state. ``True``/``False`` are answers; ``None`` means
    the question could not be asked, which is not the same as "clean".

    Returns:
        Dictionary with git_sha, git_sha_short, branch, dirty and niffler_version
    """
    niffler_version = _package_version('niffler')

    git_sha = _run_git('rev-parse', 'HEAD')
    if not git_sha:
        # Not a repository, no git binary, or no commits yet. One failed call is enough
        # to know the rest will fail too - do not shell out three more times.
        logger.warning(
            "Could not determine the git revision of the running code; "
            "this run will not be reproducible from its provenance record"
        )
        return {
            'git_sha': None,
            'git_sha_short': None,
            'branch': None,
            'dirty': None,
            'niffler_version': niffler_version,
        }

    branch = _run_git('rev-parse', '--abbrev-ref', 'HEAD')
    status = _run_git('status', '--porcelain')

    dirty: Optional[bool] = None
    if status is not None:
        dirty = bool(status)
        if dirty:
            logger.warning(
                f"Working tree is dirty at {git_sha[:_SHORT_SHA_LENGTH]}; results from "
                f"this run cannot be reproduced from the recorded commit alone"
            )

    return {
        'git_sha': git_sha,
        'git_sha_short': git_sha[:_SHORT_SHA_LENGTH],
        'branch': branch,
        'dirty': dirty,
        'niffler_version': niffler_version,
    }


@lru_cache(maxsize=1)
def _collect_environment_provenance() -> Dict[str, Any]:
    """
    Collect interpreter, platform and library versions. Memoised for the process.

    Returns:
        Dictionary with python_version, platform and a packages mapping
    """
    try:
        python_version = platform.python_version()
        platform_name = platform.platform()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"Could not determine platform information: {e}")
        python_version = '.'.join(str(part) for part in sys.version_info[:3])
        platform_name = sys.platform

    return {
        'python_version': python_version,
        'platform': platform_name,
        # A package that is not installed maps to None rather than being omitted: the
        # record then says "we looked and it was absent", not "we forgot to look".
        'packages': {name: _package_version(name) for name in TRACKED_PACKAGES},
    }


@lru_cache(maxsize=32)
def _hash_file(path: str, mtime_ns: int, size: int) -> Optional[str]:
    """
    Compute a streaming SHA-256 of a file, cached on its identity.

    ``mtime_ns`` and ``size`` are part of the cache key rather than being ignored: a
    file rewritten in place under the same path must produce a fresh hash.

    Args:
        path: Resolved path to the file
        mtime_ns: Modification time in nanoseconds, part of the cache key
        size: File size in bytes, part of the cache key

    Returns:
        Lowercase hex digest, or None when the file could not be read
    """
    digest = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b''):
                digest.update(chunk)
    except OSError as e:
        logger.warning(f"Could not hash data file {path}: {e}")
        return None

    return digest.hexdigest()


def _collect_data_provenance(data_path: Union[Path, str]) -> Dict[str, Any]:
    """
    Fingerprint the input data file.

    A missing or unreadable file degrades to a record that still carries the path that
    was asked for - knowing which file was *meant* is more useful than an empty block.

    Args:
        data_path: Path to the input data file

    Returns:
        Dictionary with path, sha256, size_bytes and modified_utc
    """
    try:
        resolved = Path(data_path).resolve()
    except (OSError, ValueError) as e:  # pragma: no cover - defensive
        logger.warning(f"Could not resolve data path {data_path!r}: {e}")
        return {
            'path': str(data_path),
            'sha256': None,
            'size_bytes': None,
            'modified_utc': None,
        }

    try:
        stat = resolved.stat()
    except OSError as e:
        logger.warning(f"Could not stat data file {resolved}: {e}")
        return {
            'path': str(resolved),
            'sha256': None,
            'size_bytes': None,
            'modified_utc': None,
        }

    try:
        modified_utc = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
    except (OSError, OverflowError, ValueError) as e:  # pragma: no cover - defensive
        logger.debug(f"Could not read modification time of {resolved}: {e}")
        modified_utc = None

    return {
        'path': str(resolved),
        'sha256': _hash_file(str(resolved), stat.st_mtime_ns, stat.st_size),
        'size_bytes': stat.st_size,
        'modified_utc': modified_utc,
    }


def collect_provenance(data_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Collect a full provenance record for the current run.

    Call this **once per run, at the CLI boundary** and thread the result through to the
    exporters. Collecting it inside each exporter would hash the input file once per
    exporter and shell out to git once per exported result.

    This function never raises: every sub-collector degrades to ``None`` for the fields
    it could not determine.

    Args:
        data_path: Path to the input data file. When omitted the ``data`` block is
            ``None`` - appropriate for runs that have no single input file

    Returns:
        Dictionary with run_timestamp_utc, code, data and environment blocks
    """
    try:
        return {
            'run_timestamp_utc': datetime.now(UTC).isoformat(),
            # Memoised results are shared; hand out copies so a caller that annotates
            # its own record cannot poison every later run in the process.
            'code': copy.deepcopy(_collect_code_provenance()),
            'data': _collect_data_provenance(data_path) if data_path is not None else None,
            'environment': copy.deepcopy(_collect_environment_provenance()),
        }
    except Exception as e:  # pragma: no cover - last-resort guard
        # Provenance is metadata. Whatever went wrong here, it must not take a backtest
        # down with it.
        logger.warning(f"Provenance collection failed: {e}")
        return {
            'run_timestamp_utc': None,
            'code': None,
            'data': None,
            'environment': None,
        }


def format_provenance_summary(provenance: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Render a one-line human summary of a provenance record.

    Used by the console exporter. A dirty working tree is called out explicitly,
    because that is the single fact that invalidates reproducibility.

    Args:
        provenance: Provenance record as returned by :func:`collect_provenance`

    Returns:
        A single line such as ``"code a1b2c3d4e5f6 (feat/provenance, DIRTY) | data
        9f86d081b2f6"``, or None when there is nothing to report
    """
    if not provenance:
        return None

    parts = []

    code = provenance.get('code') or {}
    short_sha = code.get('git_sha_short')
    if short_sha:
        descriptor = short_sha
        branch = code.get('branch')
        dirty = code.get('dirty')
        annotations = []
        if branch:
            annotations.append(branch)
        if dirty is True:
            annotations.append('DIRTY')
        elif dirty is None:
            annotations.append('dirty-unknown')
        if annotations:
            descriptor = f"{descriptor} ({', '.join(annotations)})"
        parts.append(f"code {descriptor}")
    else:
        parts.append("code unknown")

    data = provenance.get('data') or {}
    data_hash = data.get('sha256')
    if data_hash:
        parts.append(f"data {data_hash[:_SHORT_SHA_LENGTH]}")

    return ' | '.join(parts)
