"""Persisted defaults for the Niffler command line scripts.

``scripts/common.py`` already shares the flag *definitions*, so backtest,
optimize and analyze can express the same market assumption. It cannot share
the *values*: those are retyped on every run, and a ``--half-spread-bps`` typed
into two of the three commands and forgotten in the third silently invalidates
the comparison between them - the exact failure the shared cost-model CLI was
written to prevent. A file the three read is what closes that loop.

The format is TOML, parsed by the standard library's ``tomllib``: Python 3.13
is already required, ``niffler/utils/provenance.py`` already reads
``pyproject.toml`` with it, TOML carries comments - a research setting nobody
can annotate is a research setting nobody will trust - and it adds no
dependency.

Precedence, lowest to highest::

    argparse defaults  <  [common]/[costs]/[engine]/[risk]  <  [<script>]
                       <  [profile.<name>]  <  command line flags

The command line always wins, so every invocation that worked before a file
existed still behaves exactly as it did.

Two properties are what make the mechanism safe rather than merely convenient:

* **It is generic.** Keys are matched against the dests the parser actually
  declares, never a hardcoded list, and each value is checked against that
  action's ``type``, ``choices`` and ``nargs``. A flag added to a script is
  configurable the same day, and ``set_defaults`` - which bypasses all three of
  those checks - can no longer smuggle an unusable value past argparse.
* **A supplied value is not a missing one.** A dest the file provides has its
  ``required`` flag cleared, which is how ``--data`` can live in the file; a
  dest nothing provides still fails with argparse's own message.

The per-script section is strict: an unknown key is an error naming the valid
ones, because a user who believes a setting applied must never be wrong. The
shared sections and the profiles are lenient about keys the current parser does
not declare - ``[common]`` is read by seven scripts that share no single flag
between them, so rejecting there would make the section unusable. The cost,
worth stating plainly, is that a typo in a shared section is skipped instead of
reported.
"""

import argparse
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Read when neither ``--config`` nor the environment names a file. Missing is
#: not an error: no file is the state every existing checkout is in.
DEFAULT_CONFIG_FILENAME = 'niffler.toml'

#: Overrides the default path. Set it to an empty string to switch file loading
#: off entirely - which is what the test suite does, so that a developer's own
#: ``niffler.toml`` cannot change what the tests assert.
CONFIG_PATH_ENV_VAR = 'NIFFLER_CONFIG'

#: Namespace attribute carrying ``dest -> "niffler.toml [section]"`` for every
#: value the file supplied. ``build_cost_model`` reads it so that rejecting a
#: flag can say where the flag came from, instead of sending the user hunting
#: for something they never typed.
CONFIG_ORIGINS_ATTR = '_config_origins'

#: Sections every script reads, applied in this order before the script's own.
SHARED_SECTIONS: Tuple[str, ...] = ('common', 'costs', 'engine', 'risk')

#: Table holding the named overlays, applied last: ``[profile.quick]``.
PROFILE_SECTION = 'profile'

#: Dests a file must never set: they select the file itself.
_RESERVED_DESTS: Tuple[str, ...] = ('help', 'config', 'profile')


class ConfigError(ValueError):
    """A configuration file could not be read, or asks for something invalid."""


@dataclass
class LoadedConfig:
    """What a configuration file contributed to one script's defaults."""

    path: Path
    profile: Optional[str] = None
    sections: List[str] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)
    origins: Dict[str, str] = field(default_factory=dict)
    tables: Dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def describe(self) -> str:
        """One line naming the file and the sections that were applied."""
        sections = ', '.join(self.sections) if self.sections else 'no matching sections'
        return f"Config: {self.path} [{sections}]"


def script_sections() -> Tuple[str, ...]:
    """Return the per-script section names, derived from the scripts directory.

    Returns:
        Section names, e.g. ``('analyze', 'backtest', ...)``.
    """
    here = Path(__file__).resolve().parent
    return tuple(sorted(
        path.stem for path in here.glob('*.py')
        if path.stem not in ('common', 'config_file', '__init__')
    ))


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` and ``--profile`` so they appear in ``--help``.

    Both are also read ahead of time by :func:`load_config`; declaring them
    here is what stops argparse rejecting them on the real pass.

    Args:
        parser: Parser to extend.
    """
    group = parser.add_argument_group('configuration file')
    group.add_argument(
        '--config', default=None,
        help=(f"TOML file supplying defaults for this script's flags "
              f"(default: ./{DEFAULT_CONFIG_FILENAME} when it exists, or "
              f"${CONFIG_PATH_ENV_VAR}). Command line flags always win")
    )
    group.add_argument(
        '--profile', default=None,
        help="Named overlay from the file's [profile.<name>] table, applied "
             "over both the shared and the per-script sections"
    )


def load_config(parser: argparse.ArgumentParser,
                section: str,
                argv: Optional[Sequence[str]] = None,
                tables: Sequence[str] = ()) -> Optional[LoadedConfig]:
    """Read the configuration file that applies to one script.

    Args:
        parser: The script's fully built parser. Its actions are the only
            definition of which keys are valid and what they may hold.
        section: The script's own section name, e.g. ``'backtest'``.
        argv: Argument list to pre-scan for ``--config``/``--profile``
            (default: ``sys.argv[1:]``).
        tables: Sub-tables of the script's section that hold structured data
            rather than flag values, e.g. ``('parameter_space',)``. Any other
            sub-table is an error.

    Returns:
        What the file contributed, or None when there is no file to read.

    Raises:
        ConfigError: If a named file is missing, the file is malformed, or a
            section asks for something the parser cannot accept.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre.add_argument('--profile', default=None)
    known, _ = pre.parse_known_args(argv)

    path = _resolve_path(known.config)
    if path is None:
        return None

    document = _read(path)
    _validate_top_level(document, path)

    actions = _dest_actions(parser)
    loaded = LoadedConfig(path=path, profile=known.profile)

    for name in SHARED_SECTIONS:
        _apply_section(loaded, document.get(name), name, actions, strict=False)

    _apply_section(loaded, document.get(section), section, actions,
                   strict=True, tables=tables)

    if known.profile is not None:
        profiles = document.get(PROFILE_SECTION) or {}
        if known.profile not in profiles:
            available = ', '.join(sorted(profiles)) if profiles else 'none defined'
            raise ConfigError(
                f"{path}: no [profile.{known.profile}] section. "
                f"Available profiles: {available}"
            )
        _apply_section(loaded, profiles[known.profile],
                       f'{PROFILE_SECTION}.{known.profile}', actions, strict=False)

    return loaded


def apply_config(parser: argparse.ArgumentParser,
                 section: str,
                 argv: Optional[Sequence[str]] = None,
                 tables: Sequence[str] = ()) -> Optional[LoadedConfig]:
    """Fold a configuration file into a parser's defaults, before ``parse_args``.

    Call this after every ``add_*_arguments`` helper and before parsing, so the
    file can only be overridden by flags actually typed on the command line.

    Args:
        parser: The script's fully built parser.
        section: The script's own section name.
        argv: Argument list to pre-scan (default: ``sys.argv[1:]``).
        tables: Structured sub-tables of the script's section.

    Returns:
        What the file contributed, or None when there is no file to read. A
        configuration error exits through ``parser.error``, as a bad flag does.
    """
    try:
        loaded = load_config(parser, section, argv=argv, tables=tables)
    except ConfigError as e:
        parser.error(str(e))
        raise  # pragma: no cover - parser.error never returns

    if loaded is None:
        return None

    parser.set_defaults(**loaded.values)
    parser.set_defaults(**{CONFIG_ORIGINS_ATTR: dict(loaded.origins)})

    # A value the file supplies satisfies a required flag. A dest nothing
    # supplies keeps argparse's own "the following arguments are required".
    for action in parser._actions:
        if action.dest in loaded.values:
            action.required = False

    return loaded


def report_config(loaded: Optional[LoadedConfig]) -> None:
    """Print which file and sections are in force, if any.

    Silence would mean a run whose settings came from a file nobody reading the
    output can see.

    Args:
        loaded: The value returned by :func:`apply_config`.
    """
    if loaded is not None:
        print(loaded.describe())


def _resolve_path(explicit: Optional[str]) -> Optional[Path]:
    """Decide which file to read, or None when there is nothing to read.

    An explicitly named file - by flag or by environment - must exist. The
    implicit default must not.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        return path

    from_env = os.environ.get(CONFIG_PATH_ENV_VAR)
    if from_env is not None:
        if not from_env.strip():
            return None
        path = Path(from_env)
        if not path.exists():
            raise ConfigError(
                f"Config file not found: {path} (from ${CONFIG_PATH_ENV_VAR})"
            )
        return path

    path = Path(DEFAULT_CONFIG_FILENAME)
    return path if path.exists() else None


def _read(path: Path) -> Mapping[str, Any]:
    """Parse a TOML file, translating failures into ConfigError."""
    try:
        with open(path, 'rb') as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Could not parse config file {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Could not read config file {path}: {e}") from e


def _validate_top_level(document: Mapping[str, Any], path: Path) -> None:
    """Reject unknown top-level tables and stray top-level values."""
    allowed = set(SHARED_SECTIONS) | {PROFILE_SECTION} | set(script_sections())

    loose = sorted(key for key, value in document.items() if not isinstance(value, dict))
    if loose:
        raise ConfigError(
            f"{path}: {', '.join(loose)} must live in a section, not at the top "
            f"level. Sections: {', '.join(sorted(allowed))}"
        )

    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ConfigError(
            f"{path}: unknown section(s) {', '.join(unknown)}. "
            f"Valid sections: {', '.join(sorted(allowed))}"
        )


def _dest_actions(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
    """Map every configurable dest of a parser to its action."""
    return {
        action.dest: action
        for action in parser._actions
        if action.dest not in _RESERVED_DESTS and action.dest != argparse.SUPPRESS
    }


def _apply_section(loaded: LoadedConfig,
                   body: Optional[Mapping[str, Any]],
                   label: str,
                   actions: Mapping[str, argparse.Action],
                   strict: bool,
                   tables: Sequence[str] = ()) -> None:
    """Fold one section over what the earlier sections contributed."""
    if body is None:
        return
    if not isinstance(body, dict):
        raise ConfigError(f"{loaded.path}: [{label}] must be a table")

    loaded.sections.append(label)

    for key, value in body.items():
        if isinstance(value, dict):
            if key not in tables:
                expected = ', '.join(tables) if tables else 'none'
                raise ConfigError(
                    f"{loaded.path}: [{label}.{key}] is not a table this script "
                    f"reads. Sub-tables of [{label}]: {expected}"
                )
            loaded.tables[key] = value
            continue

        action = actions.get(key)
        if action is None:
            if not strict:
                # A shared section is written once for seven scripts; a key
                # this one does not declare belongs to one of the others.
                continue
            valid = ', '.join(sorted(actions))
            raise ConfigError(
                f"{loaded.path}: [{label}] has no setting '{key}'. "
                f"Valid settings: {valid}"
            )

        loaded.values[key] = _coerce(action, value, f"{loaded.path} [{label}] {key}")
        loaded.origins[key] = f"{loaded.path} [{label}]"


def _coerce(action: argparse.Action, value: Any, where: str) -> Any:
    """Check one value against what the argparse action actually accepts.

    ``set_defaults`` skips ``type``, ``choices`` and ``nargs`` entirely, so a
    file value would otherwise reach the script unvalidated - a misspelt
    ``sort_by`` failing deep inside the optimizer rather than at startup.
    """
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value

    if isinstance(value, bool):
        raise ConfigError(f"{where}: expected a value, not true/false")

    if action.nargs in ('+', '*'):
        # One value is what a single occurrence of the flag would produce on
        # the command line, so accept it rather than demanding a one-item list.
        items = value if isinstance(value, list) else [value]
        return [_scalar(action, item, where) for item in items]

    if isinstance(action.nargs, int):
        if not isinstance(value, list) or len(value) != action.nargs:
            raise ConfigError(
                f"{where}: expected a list of {action.nargs}, got {value!r}")
        return [_scalar(action, item, where) for item in value]

    if isinstance(value, list):
        raise ConfigError(f"{where}: expected a single value, got a list")

    return _scalar(action, value, where)


def _scalar(action: argparse.Action, value: Any, where: str) -> Any:
    """Convert and range-check one scalar the way the command line would."""
    if action.type is not None:
        try:
            value = action.type(value)
        except (TypeError, ValueError) as e:
            name = getattr(action.type, '__name__', str(action.type))
            raise ConfigError(f"{where}: {value!r} is not a valid {name} ({e})") from e
    elif not isinstance(value, str):
        raise ConfigError(f"{where}: expected a string, got {value!r}")

    if action.choices is not None and value not in action.choices:
        choices = ', '.join(str(choice) for choice in action.choices)
        raise ConfigError(f"{where}: {value!r} is not one of: {choices}")

    return value
