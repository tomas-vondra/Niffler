#!/usr/bin/env python3
"""Turn a strategy spec into a class skeleton, a registry line and a rule-test template.

    python scripts/scaffold_strategy.py my_idea            # niffler/strategies/specs/my_idea.toml
    python scripts/scaffold_strategy.py path/to/spec.toml --dry-run
    python scripts/scaffold_strategy.py --list             # every spec and its status

The scaffold writes what the spec determines and leaves the algorithm and the rule
tests failing until someone writes them from the spec. It never overwrites a file,
and it refuses a spec the engine cannot express, printing why. The format is in
docs/strategy-specs.md.

Exit codes: 0 done, 1 the spec is invalid or cannot be scaffolded.
"""

import argparse
import logging
import sys
from pathlib import Path

# Running "python scripts/scaffold_strategy.py" puts scripts/ on sys.path but not
# the repository root, so the root has to be added for "import niffler" to work.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niffler.config.logging import setup_logging
from niffler.strategies.scaffold import (
    REGISTRY_PATH,
    ScaffoldError,
    plan_scaffold,
    write_scaffold,
)
from niffler.strategies.spec import (
    STATUS_UNSUPPORTED,
    SpecError,
    load_all_specs,
    load_spec,
)
from scripts.config_file import add_config_arguments, apply_config, report_config

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SPECS_RELATIVE = Path('niffler') / 'strategies' / 'specs'


def resolve_spec_path(spec: str, root: Path) -> Path:
    """A path to a ``.toml`` file is used as given; anything else is a key under the root's specs."""
    candidate = Path(spec)
    if candidate.suffix == '.toml' or candidate.exists():
        return candidate
    return root / SPECS_RELATIVE / f"{spec}.toml"


def list_specs(root: Path) -> int:
    """Print every spec under the root with its status, and why when unsupported."""
    try:
        specs = load_all_specs(root / SPECS_RELATIVE)
    except SpecError as e:
        print(e, file=sys.stderr)
        return 1

    if not specs:
        print(f"No specs in {root / SPECS_RELATIVE}")
        return 0
    width = max(len(key) for key in specs)
    for key, spec in specs.items():
        print(f"{key:<{width}}  {spec.status:<11}  {spec.name}")
        if spec.status == STATUS_UNSUPPORTED:
            for reason in spec.unsupported_reasons():
                print(f"{'':<{width}}    {reason}")
    return 0


def main() -> int:
    """Scaffold one strategy from its spec, or list the specs.

    Returns:
        Process exit code: 0 on success, 1 if the spec is invalid or cannot be
        scaffolded.
    """
    parser = argparse.ArgumentParser(
        description='Scaffold a strategy class, its registry line and a rule-test '
                    'template from a spec (docs/strategy-specs.md).'
    )
    parser.add_argument('spec', nargs='?', default=None,
                        help='A spec key (looked up in niffler/strategies/specs/) or a path to a .toml file')
    parser.add_argument('--list', action='store_true',
                        help='List every spec with its status instead of scaffolding')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be written without writing anything')
    parser.add_argument('--root', default=str(REPOSITORY_ROOT),
                        help='Repository root to write into (default: this checkout)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set logging level (default: INFO)')

    add_config_arguments(parser)
    config = apply_config(parser, 'scaffold_strategy')

    args = parser.parse_args()
    setup_logging(level=args.log_level)
    report_config(config)

    root = Path(args.root)
    if args.list:
        return list_specs(root)
    if args.spec is None:
        parser.error('name a spec to scaffold, or pass --list')

    try:
        spec = load_spec(resolve_spec_path(args.spec, root))
        planned = plan_scaffold(spec, root)
    except (SpecError, ScaffoldError) as e:
        print(f"Cannot scaffold: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        for item in planned:
            action = 'create' if item.created else 'edit  '
            print(f"would {action} {item.path}")
        return 0

    try:
        written = write_scaffold(planned, root)
    except OSError as e:
        print(f"Cannot scaffold: {e}", file=sys.stderr)
        return 1

    for path, created in written.items():
        print(f"{'created' if created else 'edited '} {path}")
    print(
        f"\nNext, from the spec alone:\n"
        f"  1. write the rule tests in {planned[1].path}\n"
        f"  2. write generate_signals in {planned[0].path}, without reading those tests\n"
        f"  3. set dates.implemented in the spec in the same change\n"
        f"Until 1 and 2 are done the suite fails, by design. {REGISTRY_PATH} already lists it."
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
