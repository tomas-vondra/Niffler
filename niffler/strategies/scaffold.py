"""Turn a strategy spec into a class skeleton, a registry line and a rule-test template.

The scaffold writes everything a spec fully determines - the constructor, the
defaults, ``PARAMETER_SPEC``, the display name, the registry entry - and nothing
it does not. The algorithm and the rule test are left for someone to write from
the spec, and both are left *failing*: ``generate_signals`` raises
``NotImplementedError`` and each rule test calls ``self.fail``. A scaffold on its
own is therefore red in CI, which is the point - it cannot merge as a strategy
that silently never trades.

The roadmap has the rule test and the algorithm written by different hands, each
reading only the spec. That is why the scaffold puts the spec's rules into both
files verbatim: neither writer needs to open the other's file.

Rendering is pure (strings in, strings out); :func:`plan_scaffold` and
:func:`write_scaffold` do the file work so that tests can aim them at a temporary
tree. A spec the engine cannot express is refused with its reasons.
"""

import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from .spec import (
    POSITION_SIZE_DEFAULT,
    STATUS_UNSUPPORTED,
    SpecParameter,
    StrategySpec,
)

STRATEGIES_DIR = Path('niffler') / 'strategies'
TESTS_DIR = Path('tests') / 'test_strategies'
REGISTRY_PATH = STRATEGIES_DIR / 'registry.py'

_REGISTRY_OPEN = re.compile(r'^STRATEGY_CLASSES\b.*=[ \t]*\{[ \t]*$', re.MULTILINE)
_RELATIVE_IMPORT = re.compile(r'^from \.\w+ import \w+[ \t]*$', re.MULTILINE)


class ScaffoldError(ValueError):
    """The scaffold cannot be generated, or would overwrite something."""


@dataclass(frozen=True)
class PlannedFile:
    """One file the scaffold would write.

    Attributes:
        path: Path relative to the repository root.
        content: The full new content.
        created: True for a new file, False for an edit of an existing one.
    """

    path: Path
    content: str
    created: bool


def strategy_module_path(spec: StrategySpec) -> Path:
    return STRATEGIES_DIR / f"{spec.module}.py"


def test_module_path(spec: StrategySpec) -> Path:
    return TESTS_DIR / f"test_{spec.module}.py"


# --- rendering -------------------------------------------------------------


def _literal(value) -> str:
    """A Python literal for a spec value."""
    return repr(value)


def _doc_safe(text: str) -> str:
    """Make spec prose safe to place inside a triple-quoted docstring."""
    return text.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')


def _indent_prose(text: str, indent: str, width: int = 88) -> str:
    """Re-wrap a prose block from the spec at a fixed indent, paragraph breaks kept."""
    paragraphs = [' '.join(p.split()) for p in _doc_safe(text).strip().split('\n\n') if p.strip()]
    return '\n\n'.join(
        textwrap.fill(p, width=width, initial_indent=indent, subsequent_indent=indent)
        for p in paragraphs
    )


def _combinations(spec: StrategySpec) -> int:
    total = 1
    for entry in spec.parameter_spec().values():
        if 'choices' in entry:
            total *= len(entry['choices'])
        else:
            total *= int(math.floor((entry['max'] - entry['min']) / entry['step'] + 1e-9)) + 1
    return total


def _parameter_spec_lines(spec: StrategySpec) -> str:
    lines = []
    for name, entry in spec.parameter_spec().items():
        body = ', '.join(f"'{k}': {_literal(v)}" for k, v in entry.items())
        lines.append(f"        '{name}': {{{body}}},")
    return '\n'.join(lines)


def _signature(parameters: List[SpecParameter]) -> str:
    parts = [f"{p.name}: {p.annotation} = {_literal(p.default)}" for p in parameters]
    parts.append(f"position_size: float = {_literal(POSITION_SIZE_DEFAULT)}")
    parts.append('risk_manager=None')
    return ',\n                 '.join(parts)


def _args_doc(parameters: List[SpecParameter]) -> str:
    lines = []
    for p in parameters:
        text = textwrap.fill(
            f"{p.name}: {' '.join(_doc_safe(p.description).split())}",
            width=88, initial_indent=' ' * 12, subsequent_indent=' ' * 16,
        )
        lines.append(text)
    lines.append(' ' * 12 + 'position_size: Fraction of portfolio to use for each trade (0.0 to 1.0).')
    lines.append(' ' * 12 + 'risk_manager: Risk manager instance for position sizing and stop-loss.')
    return '\n'.join(lines)


def render_strategy_module(spec: StrategySpec) -> str:
    """Render ``niffler/strategies/<key>_strategy.py`` for a spec.

    Args:
        spec: A spec the engine can express.

    Returns:
        The module source. ``generate_signals`` raises ``NotImplementedError``.
    """
    parameters = list(spec.parameters)
    all_names = [p.name for p in parameters] + ['position_size']
    parameters_dict = '\n'.join(f"            '{n}': {n}," for n in all_names)
    attributes = '\n'.join(f"        self.{n} = {n}" for n in all_names)
    inputs = ', '.join(spec.rules.inputs)

    notes = ''
    if spec.rules.notes:
        notes = f"\n\n    Notes:\n\n{_indent_prose(spec.rules.notes, ' ' * 8)}"
    deviations = ''
    if spec.rules.deviations:
        deviations = f"\n\n    Departures from the source:\n\n{_indent_prose(spec.rules.deviations, ' ' * 8)}"

    return f'''import pandas as pd

from .base_strategy import BaseStrategy


class {spec.class_name}(BaseStrategy):
    """{spec.name}.

{_indent_prose(spec.summary, '    ')}

    Scaffolded from ``niffler/strategies/specs/{spec.key}.toml``, which is the
    source of truth for the rules and the parameters. The rules below are copied
    from it.

    Entry:

{_indent_prose(spec.rules.entry, ' ' * 8)}

    Exit:

{_indent_prose(spec.rules.exit, ' ' * 8)}{notes}{deviations}

    Reads: {inputs}. Signals are computed from the close of the bar they belong
    to, which is bias-free because ``BacktestEngine`` fills them at the next
    bar's open.
    """

    #: Optimisation search space, generated from the spec. Keys must be
    #: ``__init__`` keyword arguments - see :mod:`niffler.strategies.registry`.
    #: {_combinations(spec)} combinations.
    PARAMETER_SPEC = {{
{_parameter_spec_lines(spec)}
    }}

    def __init__(self, {_signature(parameters)}):
        """Initialize the strategy.

        Args:
{_args_doc(parameters)}
        """
        parameters = {{
{parameters_dict}
        }}
        super().__init__({spec.name!r}, parameters, risk_manager)

{attributes}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from the rules in the spec.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            A copy of ``data`` with a ``signal`` column (1 buy, -1 sell, 0 hold)
            and a ``position_size`` column, on the same index.

        Raises:
            ValueError: If the data does not have the required OHLCV format.
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data format")

        raise NotImplementedError(
            "{spec.class_name}.generate_signals is not written yet: implement the "
            "rules in niffler/strategies/specs/{spec.key}.toml"
        )

    def get_description(self) -> str:
        """Return strategy description."""
        risk_desc = ""
        if self.risk_manager is not None:
            risk_metrics = self.risk_manager.get_risk_metrics()
            risk_desc = f" Risk management: {{risk_metrics.get('risk_management_type', 'Unknown')}}"

        settings = ', '.join(f"{{name}}={{value}}" for name, value in self.parameters.items()
                             if name != 'position_size')
        return (f"{{self.name}} ({{settings}}). "
                f"Position size: {{self.position_size * 100}}%.{{risk_desc}}")
'''


def render_test_module(spec: StrategySpec) -> str:
    """Render ``tests/test_strategies/test_<key>_strategy.py`` for a spec.

    The rule tests are stubs that fail until written. When the spec reads a
    bar's high or low, a third stub asks for the intra-bar check that the shared
    look-ahead test cannot make.

    Args:
        spec: A spec the engine can express.

    Returns:
        The test module source.
    """
    reads_range = any(column in spec.rules.inputs for column in ('high', 'low'))
    intrabar = ''
    if reads_range:
        intrabar = f'''

    def test_a_bar_does_not_judge_its_own_close_by_its_own_range(self):
        """The spec reads high or low. A bar's own high or low must not decide its signal.

        The shared look-ahead test in test_registry.py truncates the series after
        each bar, which cannot see a bar using its own high to judge its own
        close. Build a series where including the current bar would change the
        answer, and assert it does not.
        """
        self.fail("Not written yet: the intra-bar test for specs/{spec.key}.toml")'''

    return f'''"""Rule tests for {spec.class_name}, written from its spec alone.

Spec: ``niffler/strategies/specs/{spec.key}.toml``. Each test builds a short
synthetic series on which a rule must fire on known bars and nowhere else, and
asserts exactly that. Write them from the rules below without reading the
implementation.

The shared contract - constructor defaults, the corners of ``PARAMETER_SPEC``,
the look-ahead check - already runs against this strategy from
``test_registry.py`` and needs no copy here.

Entry:

{_indent_prose(spec.rules.entry, '    ')}

Exit:

{_indent_prose(spec.rules.exit, '    ')}
"""

import sys
import unittest
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.strategies.{spec.module} import {spec.class_name}


def frame(closes: Sequence[float],
          highs: Optional[Sequence[float]] = None,
          lows: Optional[Sequence[float]] = None,
          opens: Optional[Sequence[float]] = None,
          volumes: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Build a daily OHLCV frame. Columns not given default to the close (volume to 1000)."""
    closes = list(closes)
    return pd.DataFrame(
        {{
            'open': list(opens) if opens is not None else closes,
            'high': list(highs) if highs is not None else closes,
            'low': list(lows) if lows is not None else closes,
            'close': closes,
            'volume': list(volumes) if volumes is not None else [1000.0] * len(closes),
        }},
        index=pd.date_range('2024-01-01', periods=len(closes), freq='D'),
    )


class Test{spec.class_name}Rules(unittest.TestCase):
    """Each rule fires on the bars the spec says, and on no other bar."""

    def assert_signals_exactly(self, result: pd.DataFrame,
                               buys: Iterable[int], sells: Iterable[int]) -> None:
        """Assert the buy and sell bars by position, and that every other bar holds."""
        signal = result['signal'].tolist()
        self.assertEqual(sorted(buys), [i for i, s in enumerate(signal) if s == 1], "buy bars")
        self.assertEqual(sorted(sells), [i for i, s in enumerate(signal) if s == -1], "sell bars")

    def test_entry_fires_on_known_bars_only(self):
        """Build a series on which the entry rule fires on known bars, and assert it fires there only."""
        self.fail("Not written yet: the entry rule test for specs/{spec.key}.toml")

    def test_exit_fires_on_known_bars_only(self):
        """Build a series on which the exit rule fires on known bars, and assert it fires there only."""
        self.fail("Not written yet: the exit rule test for specs/{spec.key}.toml"){intrabar}


if __name__ == '__main__':
    unittest.main()
'''


def add_registry_entry(source: str, spec: StrategySpec) -> str:
    """Add a spec's import and its ``STRATEGY_CLASSES`` line to the registry source.

    Args:
        source: The current text of ``niffler/strategies/registry.py``.
        spec: The spec being scaffolded.

    Returns:
        The edited text.

    Raises:
        ScaffoldError: If the key is already registered, or the registry no
            longer has the shape this edit expects.
    """
    if re.search(rf"^\s*'{re.escape(spec.key)}'\s*:", source, re.MULTILINE):
        raise ScaffoldError(f"'{spec.key}' is already in STRATEGY_CLASSES")

    opening = _REGISTRY_OPEN.search(source)
    if opening is None:
        raise ScaffoldError("cannot find 'STRATEGY_CLASSES = {' in the registry")
    closing = source.find('\n}', opening.end())
    if closing < 0:
        raise ScaffoldError("cannot find the closing '}' of STRATEGY_CLASSES")
    entry = f"\n    '{spec.key}': {spec.class_name},"
    source = source[:closing] + entry + source[closing:]

    imports = list(_RELATIVE_IMPORT.finditer(source, 0, opening.start()))
    if not imports:
        raise ScaffoldError("cannot find the strategy imports in the registry")
    new_import = f"from .{spec.module} import {spec.class_name}"
    # Keep the block sorted by module, as it is today.
    for match in imports:
        if match.group(0).strip() > new_import:
            return source[:match.start()] + new_import + '\n' + source[match.start():]
    end = imports[-1].end()
    return source[:end] + '\n' + new_import + source[end:]


# --- files -----------------------------------------------------------------


def plan_scaffold(spec: StrategySpec, root: Union[str, Path]) -> List[PlannedFile]:
    """Work out every file the scaffold would write, without writing any.

    Args:
        spec: The spec to scaffold.
        root: The repository root.

    Returns:
        The strategy module, the test module and the edited registry.

    Raises:
        ScaffoldError: If the spec is unsupported or already implemented, or a
            file it would create already exists.
    """
    if spec.status == STATUS_UNSUPPORTED:
        reasons = '\n'.join(f"  - {reason}" for reason in spec.unsupported_reasons())
        raise ScaffoldError(
            f"'{spec.key}' needs what the engine cannot do, so it is recorded as "
            f"unsupported rather than approximated:\n{reasons}"
        )
    if spec.implemented is not None:
        raise ScaffoldError(
            f"'{spec.key}' already has dates.implemented = {spec.implemented}; "
            f"there is nothing to scaffold"
        )

    root = Path(root)
    planned = []
    for relative, content in (
        (strategy_module_path(spec), render_strategy_module(spec)),
        (test_module_path(spec), render_test_module(spec)),
    ):
        if (root / relative).exists():
            raise ScaffoldError(f"{relative} already exists; the scaffold never overwrites")
        planned.append(PlannedFile(relative, content, created=True))

    registry = root / REGISTRY_PATH
    try:
        registry_source = registry.read_text(encoding='utf-8')
    except OSError as e:
        raise ScaffoldError(f"cannot read {REGISTRY_PATH}: {e}") from e
    planned.append(PlannedFile(REGISTRY_PATH, add_registry_entry(registry_source, spec),
                               created=False))
    return planned


def write_scaffold(planned: List[PlannedFile], root: Union[str, Path]) -> Dict[Path, bool]:
    """Write planned files under a root.

    Args:
        planned: From :func:`plan_scaffold`.
        root: The repository root.

    Returns:
        Each written path mapped to whether it was created (else edited).
    """
    root = Path(root)
    written = {}
    for item in planned:
        target = root / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding='utf-8')
        written[item.path] = item.created
    return written
