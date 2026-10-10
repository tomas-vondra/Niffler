"""Strategy specs: one TOML file per strategy that says what it is.

A strategy used to be only a class and a registry line. A spec adds what the code
cannot say: where the idea came from, its rules in prose, the parameters the
source published, what it needs from the engine, and when it was found and
implemented. The format is documented in ``docs/strategy-specs.md``; the specs
live in :data:`SPEC_DIR`, named ``<key>.toml``.

The spec is the input the rest of the pipeline reads. The scaffold
(:mod:`niffler.strategies.scaffold`) turns it into a class skeleton, a registry
line and a rule-test template; the rule test and the algorithm are then written
from the spec alone, by different hands. ``tests/test_strategies/test_specs.py``
keeps every registered strategy and its spec in agreement, so a spec cannot drift
from the class it documents.

Loading is strict. An unknown key is an error, not ignored, because a misspelt
``publshed`` would otherwise vanish without a trace. Every problem in a file is
reported at once, so whoever is writing a spec fixes it in one pass.

Like the rest of :mod:`niffler.strategies`, this module imports nothing from
:mod:`niffler.optimization`: :meth:`StrategySpec.parameter_spec` returns the same
plain dict a strategy declares as ``PARAMETER_SPEC``.
"""

import datetime
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .capabilities import CAPABILITIES

#: Where the specs live, one ``<key>.toml`` per strategy.
SPEC_DIR = Path(__file__).resolve().parent / 'specs'

#: The only format version this loader reads. A change to the format that an
#: old file cannot satisfy bumps it.
SPEC_VERSION = 1

FAMILIES = (
    'trend_following',
    'mean_reversion',
    'breakout',
    'momentum',
    'volatility',
    'seasonal',
    'other',
)

SOURCE_KINDS = (
    'textbook',  # a rule so widely described that no single page owns it
    'book',
    'paper',
    'web',       # a page on the internet: url and retrieved are required
    'original',  # written here, not taken from anywhere
)

OHLCV_COLUMNS = ('open', 'high', 'low', 'close', 'volume')

PARAMETER_TYPES = ('int', 'float', 'choice')

#: Parameters every strategy takes, which a spec therefore never declares.
#: ``position_size`` is a framework knob, not part of any published rule, and
#: ``risk_manager`` is plumbing.
RESERVED_PARAMETERS = ('position_size', 'risk_manager')

#: The ``position_size`` every scaffolded strategy gets, and the search range the
#: three shipped strategies already use.
POSITION_SIZE_DEFAULT = 1.0
POSITION_SIZE_SEARCH = {'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.1}

STATUS_IMPLEMENTED = 'implemented'
STATUS_SPECIFIED = 'specified'
STATUS_UNSUPPORTED = 'unsupported'

_KEY_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
_CLASS_PATTERN = re.compile(r'^[A-Z][A-Za-z0-9]*Strategy$')
_PARAMETER_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')

_TOP_LEVEL_KEYS = {
    'spec_version': True, 'key': True, 'class_name': True, 'name': True,
    'family': True, 'summary': True,
    'source': True, 'dates': True, 'rules': True, 'parameters': True, 'requires': True,
}
_SOURCE_KEYS = {'kind': True, 'reference': True, 'url': False, 'retrieved': False}
_DATES_KEYS = {'found': True, 'implemented': False}
_RULES_KEYS = {'entry': True, 'exit': True, 'inputs': True, 'notes': False, 'deviations': False}
_PARAMETER_KEYS = {'type': True, 'default': True, 'description': True,
                   'published': False, 'search': False}
_REQUIRES_KEYS = {'capabilities': True, 'unlisted': False}

ParameterValue = Union[int, float, str]


class SpecError(ValueError):
    """A spec file that cannot be read, or does not satisfy the format.

    Attributes:
        path: The file, when there was one.
        problems: Every problem found, so all of them can be fixed in one pass.
    """

    def __init__(self, problems: List[str], path: Optional[Path] = None):
        self.path = path
        self.problems = list(problems)
        where = f"{path}: " if path else ''
        listed = '\n'.join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"{where}invalid strategy spec\n{listed}")


@dataclass(frozen=True)
class SpecSource:
    """Where the idea came from."""

    kind: str
    reference: str
    url: Optional[str] = None
    retrieved: Optional[datetime.date] = None


@dataclass(frozen=True)
class SpecRules:
    """The rules in prose, as the source states them."""

    entry: str
    exit: str
    inputs: Tuple[str, ...]
    notes: Optional[str] = None
    deviations: Optional[str] = None


@dataclass(frozen=True)
class SpecParameter:
    """One strategy parameter.

    Attributes:
        name: Constructor keyword argument.
        type: ``'int'``, ``'float'`` or ``'choice'``.
        default: The constructor default.
        description: What the parameter means, for the docstring.
        published: The value the source states, when it states one.
        search: The optimisation range: ``{'min', 'max', 'step'}`` for a number,
            ``{'choices'}`` for a choice. ``None`` keeps the parameter out of
            the search.
    """

    name: str
    type: str
    default: ParameterValue
    description: str
    published: Optional[ParameterValue] = None
    search: Optional[Dict[str, Any]] = None

    @property
    def annotation(self) -> str:
        """The Python type annotation for the constructor."""
        return {'int': 'int', 'float': 'float', 'choice': 'str'}[self.type]

    def search_entry(self) -> Optional[Dict[str, Any]]:
        """The ``PARAMETER_SPEC`` entry for this parameter, or None if not searched."""
        if self.search is None:
            return None
        return {'type': self.type, **self.search}


@dataclass(frozen=True)
class StrategySpec:
    """A validated strategy spec.

    Attributes:
        key: Registry name, also the file name and the module stem.
        class_name: Name of the strategy class.
        name: Display name, which the class passes to ``BaseStrategy``.
        family: One of :data:`FAMILIES`.
        summary: One sentence.
        source: Where it came from.
        found: When the idea was found.
        implemented: When the class was merged; None while it is not.
        rules: The rules in prose.
        parameters: Strategy parameters, in declaration order.
        capabilities: Names from :data:`niffler.strategies.capabilities.CAPABILITIES`.
        unlisted: Needs the capability list does not name. Any entry makes the
            spec unsupported, since the engine cannot be claimed to cover a need
            nobody has listed.
        path: The file it was read from, if any.
    """

    key: str
    class_name: str
    name: str
    family: str
    summary: str
    source: SpecSource
    found: datetime.date
    implemented: Optional[datetime.date]
    rules: SpecRules
    parameters: Tuple[SpecParameter, ...]
    capabilities: Tuple[str, ...]
    unlisted: Tuple[str, ...] = ()
    path: Optional[Path] = None

    @property
    def module(self) -> str:
        """Module stem inside ``niffler/strategies/``."""
        return f"{self.key}_strategy"

    def unsupported_reasons(self) -> List[str]:
        """Why the engine cannot express this spec; empty when it can."""
        reasons = [
            f"{name}: {CAPABILITIES[name].note}"
            for name in self.capabilities
            if not CAPABILITIES[name].supported
        ]
        reasons.extend(f"unlisted: {need}" for need in self.unlisted)
        return reasons

    @property
    def is_supported(self) -> bool:
        return not self.unsupported_reasons()

    @property
    def status(self) -> str:
        """``implemented``, ``specified`` (buildable, not built) or ``unsupported``."""
        if not self.is_supported:
            return STATUS_UNSUPPORTED
        if self.implemented is not None:
            return STATUS_IMPLEMENTED
        return STATUS_SPECIFIED

    def defaults(self) -> Dict[str, ParameterValue]:
        """Constructor defaults, ``position_size`` included."""
        values: Dict[str, ParameterValue] = {p.name: p.default for p in self.parameters}
        values['position_size'] = POSITION_SIZE_DEFAULT
        return values

    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        """The ``PARAMETER_SPEC`` the class declares, ``position_size`` included."""
        spec = {}
        for parameter in self.parameters:
            entry = parameter.search_entry()
            if entry is not None:
                spec[parameter.name] = entry
        spec['position_size'] = dict(POSITION_SIZE_SEARCH)
        return spec


# --- loading ---------------------------------------------------------------


def load_spec(path: Union[str, Path]) -> StrategySpec:
    """Read and validate one spec file.

    Args:
        path: A ``<key>.toml`` file.

    Returns:
        The validated spec.

    Raises:
        SpecError: If the file cannot be read or parsed, or breaks the format.
            Every problem found is listed.
    """
    path = Path(path)
    try:
        with path.open('rb') as handle:
            raw = tomllib.load(handle)
    except OSError as e:
        raise SpecError([f"cannot read the file: {e}"], path) from e
    except tomllib.TOMLDecodeError as e:
        raise SpecError([f"not valid TOML: {e}"], path) from e

    spec = parse_spec(raw, path=path)
    if spec.key != path.stem:
        raise SpecError([f"key '{spec.key}' must match the file name '{path.stem}.toml'"], path)
    return spec


def load_all_specs(directory: Union[str, Path] = SPEC_DIR) -> Dict[str, StrategySpec]:
    """Read every ``*.toml`` spec in a directory.

    Args:
        directory: Where to look. Defaults to :data:`SPEC_DIR`.

    Returns:
        Specs by key, sorted by key.

    Raises:
        SpecError: On the first file that does not validate.
    """
    specs = {}
    for path in sorted(Path(directory).glob('*.toml')):
        spec = load_spec(path)
        specs[spec.key] = spec
    return specs


def parse_spec(raw: Dict[str, Any], path: Optional[Path] = None) -> StrategySpec:
    """Validate an already-parsed TOML document.

    Args:
        raw: The document, as ``tomllib`` returns it.
        path: The file it came from, for error messages.

    Returns:
        The validated spec.

    Raises:
        SpecError: Listing every problem found.
    """
    problems: List[str] = []
    _check_keys(raw, _TOP_LEVEL_KEYS, '', problems)

    version = raw.get('spec_version')
    if 'spec_version' in raw and version != SPEC_VERSION:
        problems.append(f"spec_version is {version!r}; this loader reads {SPEC_VERSION}")

    key = _string(raw, 'key', '', problems)
    if key and not _KEY_PATTERN.match(key):
        problems.append(f"key '{key}' must be snake_case: lowercase letters, digits, underscores")

    class_name = _string(raw, 'class_name', '', problems)
    if class_name and not _CLASS_PATTERN.match(class_name):
        problems.append(f"class_name '{class_name}' must be CamelCase and end in 'Strategy'")

    name = _string(raw, 'name', '', problems)
    summary = _string(raw, 'summary', '', problems)
    family = _string(raw, 'family', '', problems)
    if family and family not in FAMILIES:
        problems.append(f"family '{family}' is not one of: {', '.join(FAMILIES)}")

    source = _parse_source(_table(raw, 'source', problems), problems)
    found, implemented = _parse_dates(_table(raw, 'dates', problems), problems)
    rules = _parse_rules(_table(raw, 'rules', problems), problems)
    parameters = _parse_parameters(_table(raw, 'parameters', problems), problems)
    capabilities, unlisted = _parse_requires(_table(raw, 'requires', problems), problems)

    if source and source.retrieved and found and source.retrieved > found:
        problems.append("source.retrieved is after dates.found; a source is read before it is found")

    if problems:
        raise SpecError(problems, path)

    spec = StrategySpec(
        key=key, class_name=class_name, name=name, family=family, summary=summary,
        source=source, found=found, implemented=implemented, rules=rules,
        parameters=parameters, capabilities=capabilities, unlisted=unlisted, path=path,
    )

    if spec.implemented is not None and not spec.is_supported:
        raise SpecError(
            ["dates.implemented is set, but the spec needs what the engine cannot do: "
             + '; '.join(spec.unsupported_reasons())],
            path,
        )
    return spec


def _check_keys(table: Dict[str, Any], allowed: Dict[str, bool], where: str,
                problems: List[str]) -> None:
    """Report unknown keys and missing required ones in one table."""
    prefix = f"{where}." if where else ''
    for unknown in sorted(set(table) - set(allowed)):
        problems.append(f"unknown key '{prefix}{unknown}'; allowed: {', '.join(allowed)}")
    for required, is_required in allowed.items():
        if is_required and required not in table:
            problems.append(f"missing required key '{prefix}{required}'")


def _table(raw: Dict[str, Any], name: str, problems: List[str]) -> Optional[Dict[str, Any]]:
    value = raw.get(name)
    if value is None:
        return None  # already reported as missing
    if not isinstance(value, dict):
        problems.append(f"'{name}' must be a table")
        return None
    return value


def _string(table: Dict[str, Any], name: str, where: str, problems: List[str],
            required: bool = True) -> Optional[str]:
    """Read a non-empty string, or report why it is not one."""
    if name not in table:
        return None  # a missing required key is reported by _check_keys
    value = table[name]
    label = f"{where}.{name}" if where else name
    if not isinstance(value, str) or not value.strip():
        problems.append(f"'{label}' must be a non-empty string")
        return None
    return value.strip()


def _date(table: Dict[str, Any], name: str, where: str,
          problems: List[str]) -> Optional[datetime.date]:
    """Read a TOML local date (``2026-10-10``, unquoted)."""
    if name not in table:
        return None
    value = table[name]
    if isinstance(value, datetime.datetime) or not isinstance(value, datetime.date):
        problems.append(f"'{where}.{name}' must be a date written as 2026-10-10, unquoted")
        return None
    return value


def _parse_source(table: Optional[Dict[str, Any]], problems: List[str]) -> Optional[SpecSource]:
    if table is None:
        return None
    _check_keys(table, _SOURCE_KEYS, 'source', problems)
    kind = _string(table, 'kind', 'source', problems)
    if kind and kind not in SOURCE_KINDS:
        problems.append(f"source.kind '{kind}' is not one of: {', '.join(SOURCE_KINDS)}")
    reference = _string(table, 'reference', 'source', problems)
    url = _string(table, 'url', 'source', problems)
    if url and not url.startswith(('https://', 'http://')):
        problems.append(f"source.url '{url}' must start with https:// or http://")
    retrieved = _date(table, 'retrieved', 'source', problems)

    if kind == 'web' and 'url' not in table:
        problems.append("source.url is required when source.kind is 'web'")
    if 'url' in table and 'retrieved' not in table:
        problems.append("source.retrieved is required with source.url: a page changes")
    return SpecSource(kind=kind, reference=reference, url=url, retrieved=retrieved)


def _parse_dates(table: Optional[Dict[str, Any]],
                 problems: List[str]) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    if table is None:
        return None, None
    _check_keys(table, _DATES_KEYS, 'dates', problems)
    found = _date(table, 'found', 'dates', problems)
    implemented = _date(table, 'implemented', 'dates', problems)
    if found and implemented and implemented < found:
        problems.append("dates.implemented is before dates.found")
    return found, implemented


def _parse_rules(table: Optional[Dict[str, Any]], problems: List[str]) -> Optional[SpecRules]:
    if table is None:
        return None
    _check_keys(table, _RULES_KEYS, 'rules', problems)
    entry = _string(table, 'entry', 'rules', problems)
    exit_rule = _string(table, 'exit', 'rules', problems)
    notes = _string(table, 'notes', 'rules', problems)
    deviations = _string(table, 'deviations', 'rules', problems)

    inputs: Tuple[str, ...] = ()
    if 'inputs' in table:
        raw_inputs = table['inputs']
        if (not isinstance(raw_inputs, list) or not raw_inputs
                or not all(isinstance(column, str) for column in raw_inputs)):
            problems.append(f"rules.inputs must be a non-empty list drawn from: {', '.join(OHLCV_COLUMNS)}")
        else:
            unknown = [column for column in raw_inputs if column not in OHLCV_COLUMNS]
            if unknown:
                problems.append(f"rules.inputs names {unknown}; the columns are: {', '.join(OHLCV_COLUMNS)}")
            if len(set(raw_inputs)) != len(raw_inputs):
                problems.append("rules.inputs lists a column twice")
            inputs = tuple(raw_inputs)
    return SpecRules(entry=entry, exit=exit_rule, inputs=inputs, notes=notes, deviations=deviations)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _typed_value(value: Any, param_type: str) -> bool:
    """Whether a value fits a parameter type. A float parameter accepts a TOML integer."""
    if param_type == 'int':
        return isinstance(value, int) and not isinstance(value, bool)
    if param_type == 'float':
        return _is_number(value)
    return isinstance(value, str)


def _parse_parameters(table: Optional[Dict[str, Any]],
                      problems: List[str]) -> Tuple[SpecParameter, ...]:
    if table is None:
        return ()
    if not table:
        problems.append("'parameters' declares no parameter; a rule with nothing to set needs no spec")
        return ()

    parameters = []
    for name, body in table.items():
        where = f"parameters.{name}"
        if name in RESERVED_PARAMETERS:
            problems.append(f"'{where}' is reserved: every strategy takes {name}, so a spec never declares it")
            continue
        if not _PARAMETER_PATTERN.match(name):
            problems.append(f"'{where}': a parameter name must be snake_case")
            continue
        if not isinstance(body, dict):
            problems.append(f"'{where}' must be a table")
            continue
        parameter = _parse_parameter(name, body, where, problems)
        if parameter is not None:
            parameters.append(parameter)
    return tuple(parameters)


def _parse_parameter(name: str, body: Dict[str, Any], where: str,
                     problems: List[str]) -> Optional[SpecParameter]:
    before = len(problems)
    _check_keys(body, _PARAMETER_KEYS, where, problems)
    param_type = body.get('type')
    if 'type' not in body:
        return None  # reported as missing; nothing else can be checked without it
    if param_type not in PARAMETER_TYPES:
        problems.append(f"'{where}.type' must be one of: {', '.join(PARAMETER_TYPES)}")
        return None
    description = _string(body, 'description', where, problems)

    default = body.get('default')
    if 'default' in body and not _typed_value(default, param_type):
        problems.append(f"'{where}.default' {default!r} is not of type {param_type}")
    published = body.get('published')
    if 'published' in body and not _typed_value(published, param_type):
        problems.append(f"'{where}.published' {published!r} is not of type {param_type}")

    search = None
    if 'search' in body:
        search = _parse_search(body['search'], param_type, default, f"{where}.search", problems)

    if len(problems) > before:
        return None
    if param_type == 'float':
        default = float(default)
        published = float(published) if published is not None else None
    return SpecParameter(name=name, type=param_type, default=default, description=description,
                         published=published, search=search)


def _parse_search(search: Any, param_type: str, default: Any, where: str,
                  problems: List[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(search, dict):
        problems.append(f"'{where}' must be a table")
        return None

    if param_type == 'choice':
        _check_keys(search, {'choices': True}, where, problems)
        choices = search.get('choices')
        if 'choices' in search and (not isinstance(choices, list) or not choices
                                    or not all(isinstance(c, str) for c in choices)):
            problems.append(f"'{where}.choices' must be a non-empty list of strings")
            return None
        if choices and default not in choices:
            problems.append(f"'{where}.choices' must include the default {default!r}")
        return {'choices': list(choices)} if choices else None

    _check_keys(search, {'min': True, 'max': True, 'step': True}, where, problems)
    values = {bound: search.get(bound) for bound in ('min', 'max', 'step')}
    for bound, value in values.items():
        if bound in search and not _typed_value(value, param_type):
            problems.append(f"'{where}.{bound}' {value!r} is not of type {param_type}")
            return None
    if any(bound not in search for bound in values):
        return None
    if values['min'] >= values['max']:
        problems.append(f"'{where}': min must be below max")
    if values['step'] <= 0:
        problems.append(f"'{where}': step must be positive")
    if _typed_value(default, param_type) and not values['min'] <= default <= values['max']:
        problems.append(f"'{where}': the default {default!r} lies outside the searched range")
    if param_type == 'float':
        values = {bound: float(value) for bound, value in values.items()}
    return values


def _parse_requires(table: Optional[Dict[str, Any]],
                    problems: List[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if table is None:
        return (), ()
    _check_keys(table, _REQUIRES_KEYS, 'requires', problems)

    capabilities: Tuple[str, ...] = ()
    raw = table.get('capabilities')
    if 'capabilities' in table:
        if not isinstance(raw, list) or not raw or not all(isinstance(c, str) for c in raw):
            problems.append("requires.capabilities must be a non-empty list of capability names")
        else:
            unknown = [c for c in raw if c not in CAPABILITIES]
            if unknown:
                problems.append(
                    f"requires.capabilities names {unknown}, which are not declared in "
                    f"niffler/strategies/capabilities.py. A need the list does not cover "
                    f"goes in requires.unlisted"
                )
            if len(set(raw)) != len(raw):
                problems.append("requires.capabilities lists a capability twice")
            capabilities = tuple(raw)

    unlisted: Tuple[str, ...] = ()
    raw_unlisted = table.get('unlisted', [])
    if not isinstance(raw_unlisted, list) or not all(
            isinstance(need, str) and need.strip() for need in raw_unlisted):
        problems.append("requires.unlisted must be a list of non-empty strings")
    else:
        unlisted = tuple(need.strip() for need in raw_unlisted)
    return capabilities, unlisted
