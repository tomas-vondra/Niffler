"""Contract tests that apply to *every* registered strategy.

These are what turn :mod:`niffler.strategies.registry` from a convention into a
contract. A newly registered strategy is picked up here automatically, so the
cost of adding one stays "write the class, add one registry line" without that
also meaning "and hope the CLIs still work".

The look-ahead test in :class:`TestStrategyContract` is the important one: it
runs against every strategy in the registry, present and future.
"""

import io
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.optimization.optimizer_factory import get_parameter_space
from niffler.optimization.parameter_space import ParameterSpace
from niffler.strategies.registry import (
    STRATEGY_CLASSES,
    create_strategy,
    get_available_strategies,
    get_parameter_spec,
    get_strategy_class,
    get_strategy_parameter_names,
)


def make_ohlcv(periods: int = 250, seed: int = 7) -> pd.DataFrame:
    """Build a deterministic OHLCV frame long enough for every strategy's warm-up."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, periods)
    close = 100.0 * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(0.0, 0.01, periods)) * close

    return pd.DataFrame(
        {
            'open': close - spread * 0.5,
            'high': close + spread,
            'low': close - spread,
            'close': close,
            'volume': rng.uniform(1000, 5000, periods),
        },
        index=pd.date_range('2020-01-01', periods=periods, freq='D'),
    )


def spec_corner(spec: dict, corner: str) -> dict:
    """Take the lowest or highest lattice point of a parameter spec."""
    return {name: config[corner] for name, config in spec.items()}


class TestStrategyContract(unittest.TestCase):
    """Invariants every registered strategy must satisfy."""

    def setUp(self):
        self.data = make_ohlcv()

    def test_registry_is_not_empty(self):
        self.assertTrue(STRATEGY_CLASSES, "The strategy registry must not be empty")

    def test_every_strategy_declares_a_parameter_spec(self):
        for name in get_available_strategies():
            with self.subTest(strategy=name):
                spec = get_parameter_spec(name)
                self.assertTrue(spec, f"{name} declares an empty PARAMETER_SPEC")

    def test_spec_keys_are_constructor_parameters(self):
        """A spec key the constructor does not accept breaks every optimizer.

        The library builds strategies as ``strategy_class(**parameters)``, so a
        spec that names something ``__init__`` does not take fails only once an
        optimisation is already running.
        """
        for name in get_available_strategies():
            with self.subTest(strategy=name):
                accepted = get_strategy_parameter_names(name)
                unknown = set(get_parameter_spec(name)) - accepted
                self.assertEqual(
                    set(), unknown,
                    f"{name} declares parameters its constructor rejects: {sorted(unknown)}"
                )

    def test_every_strategy_accepts_position_size_and_risk_manager(self):
        for name in get_available_strategies():
            with self.subTest(strategy=name):
                self.assertIn('position_size', get_strategy_parameter_names(name))
                # risk_manager is excluded from the reported names, so check the
                # signature accepts it by constructing with one.
                strategy = create_strategy(name, {}, risk_manager=None)
                self.assertIsNone(strategy.risk_manager)

    def test_every_strategy_constructs_with_no_arguments(self):
        """Every parameter must have a default, so a bare CLI run works."""
        for name in get_available_strategies():
            with self.subTest(strategy=name):
                strategy = create_strategy(name)
                self.assertTrue(strategy.get_description())
                self.assertTrue(strategy.name)

    def test_every_spec_corner_constructs_and_generates_signals(self):
        """Both extremes of the search space must be valid parameter sets.

        An optimizer will visit the corners of the lattice; if a corner raises,
        the run reports error cells for combinations that were meant to be legal.
        """
        for name in get_available_strategies():
            spec = get_parameter_spec(name)
            for corner in ('min', 'max'):
                with self.subTest(strategy=name, corner=corner):
                    strategy = create_strategy(name, spec_corner(spec, corner))
                    result = strategy.generate_signals(self.data)

                    self.assertIn('signal', result.columns)
                    self.assertIn('position_size', result.columns)
                    self.assertTrue(result.index.equals(self.data.index))
                    self.assertTrue(
                        set(result['signal'].unique()).issubset({-1, 0, 1}),
                        f"{name} emitted a signal outside -1/0/1"
                    )

    def test_no_strategy_looks_ahead(self):
        """The signal on the final bar must not need bars that come after it.

        For every cutoff, the last signal of ``data[:c]`` must equal the
        full-series signal at ``c - 1``. On the truncated frame that bar has no
        future at all, so any dependence on a later bar - a centred rolling
        window, a negative shift, a backward fill - changes the answer.

        Sweeping *every* cutoff rather than sampling matters: a one-bar leak only
        perturbs the bar adjacent to the boundary, so a strided sweep can walk
        straight past it. The assertion at the end guards against the test going
        vacuous if a strategy or the fixture stops producing signals - a run
        where every compared bar was a flat 0 would pass without testing anything.

        This runs against every registered strategy, so a future one is covered
        without touching this file.

        Note what this does *not* cover: using the current bar's own high or low
        to judge its close is an intra-bar leak, invisible to truncation. Those
        are caught per-strategy, e.g.
        ``test_breakout_strategy.TestDonchianChannel``.
        """
        first_cutoff = 40

        for name in get_available_strategies():
            with self.subTest(strategy=name):
                strategy = create_strategy(name)
                full = strategy.generate_signals(self.data)['signal']

                compared_signals = 0
                for cutoff in range(first_cutoff, len(self.data) + 1):
                    prefix = strategy.generate_signals(self.data.iloc[:cutoff])
                    expected = full.iloc[cutoff - 1]

                    self.assertEqual(
                        expected, prefix['signal'].iloc[-1],
                        f"{name}: the signal at bar {cutoff - 1} changed once the "
                        f"bars after it were removed, so it depends on the future"
                    )
                    compared_signals += int(expected != 0)

                self.assertGreaterEqual(
                    compared_signals, 3,
                    f"{name} produced almost no signals on the fixture, so this "
                    f"look-ahead check proved nothing"
                )

    def test_parameter_space_builds_for_every_strategy(self):
        for name in get_available_strategies():
            with self.subTest(strategy=name):
                space = get_parameter_space(name)
                self.assertIsInstance(space, ParameterSpace)
                self.assertEqual(set(space.parameters), set(get_parameter_spec(name)))

    def test_grid_stays_under_the_cli_memory_cap(self):
        """A grid larger than the cap makes the optimizer withhold statistics.

        ``optimize.py`` retains ``CLI_MAX_RESULTS_IN_MEMORY`` results; beyond it
        the result set is marked truncated and whole-grid distribution reporting
        is suppressed. A shipped strategy should not silently land in that state.
        """
        from scripts.optimize import CLI_MAX_RESULTS_IN_MEMORY

        for name in get_available_strategies():
            with self.subTest(strategy=name):
                combinations = 1
                for config in get_parameter_spec(name).values():
                    step = config.get('step', 1)
                    span = config['max'] - config['min']
                    combinations *= int(round(span / step)) + 1

                self.assertLessEqual(
                    combinations, CLI_MAX_RESULTS_IN_MEMORY,
                    f"{name}'s grid ({combinations}) exceeds the CLI retention cap"
                )


class TestRegistryLookups(unittest.TestCase):
    """Behaviour of the registry's own lookup functions."""

    def test_get_strategy_class_returns_the_registered_class(self):
        for name, expected in STRATEGY_CLASSES.items():
            with self.subTest(strategy=name):
                self.assertIs(get_strategy_class(name), expected)

    def test_unknown_strategy_names_the_available_ones(self):
        with self.assertRaises(ValueError) as ctx:
            get_strategy_class('does_not_exist')

        message = str(ctx.exception)
        self.assertIn('does_not_exist', message)
        for name in get_available_strategies():
            self.assertIn(name, message)

    def test_get_parameter_spec_returns_a_copy(self):
        """Mutating a returned spec must not corrupt the class attribute."""
        first = get_parameter_spec('simple_ma')
        first['short_window']['min'] = 999
        first['injected'] = {'type': 'int', 'min': 1, 'max': 2}

        second = get_parameter_spec('simple_ma')
        self.assertNotIn('injected', second)
        self.assertNotEqual(999, second['short_window']['min'])

    def test_create_strategy_rejects_an_unknown_parameter(self):
        with self.assertRaises(ValueError) as ctx:
            create_strategy('simple_ma', {'not_a_parameter': 1})

        self.assertIn('simple_ma', str(ctx.exception))

    def test_create_strategy_applies_the_parameters(self):
        strategy = create_strategy('simple_ma', {'short_window': 6, 'long_window': 40})
        self.assertEqual(6, strategy.short_window)
        self.assertEqual(40, strategy.long_window)

    def test_parameter_spec_missing_is_reported(self):
        class SpeclessStrategy:
            pass

        with patch.dict(STRATEGY_CLASSES, {'specless': SpeclessStrategy}):
            with self.assertRaises(ValueError) as ctx:
                get_parameter_spec('specless')

        self.assertIn('PARAMETER_SPEC', str(ctx.exception))


class TestLayering(unittest.TestCase):
    """niffler.strategies must not depend on niffler.optimization."""

    def test_importing_the_registry_pulls_in_no_optimization_module(self):
        """The one-way dependency is what keeps the registry importable at all.

        ``niffler/optimization/__init__.py`` imports ``optimizer_factory``, which
        imports this registry. If a strategy module imported ``ParameterSpace``
        (or anything else from the optimization layer) that would close the loop
        into a circular import. Run in a subprocess because the rest of the suite
        has already imported the optimization layer in this process.
        """
        probe = (
            "import sys;"
            f"sys.path.insert(0, r'{project_root}');"
            "import niffler.strategies.registry;"
            "print([m for m in sys.modules if m.startswith('niffler.optimization')])"
        )
        result = subprocess.run(
            [sys.executable, '-c', probe], capture_output=True, text=True
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            '[]', result.stdout.strip(),
            "niffler.strategies must not import from niffler.optimization; "
            f"it pulled in {result.stdout.strip()}"
        )


class TestCLIsExposeTheWholeRegistry(unittest.TestCase):
    """Every CLI must offer every registered strategy.

    This is the regression test for the bug this registry replaced: ``analyze.py``
    used to define its own ``get_strategy_class`` shadowing the shared one, so a
    strategy added to the registry was accepted by ``optimize.py`` and rejected by
    ``analyze.py`` with "Unknown strategy".
    """

    def test_analyze_parser_offers_every_registered_strategy(self):
        from scripts.analyze import create_parser

        action = next(
            a for a in create_parser()._actions if a.dest == 'strategy'
        )
        self.assertEqual(sorted(get_available_strategies()), sorted(action.choices))

    def test_analyze_resolves_every_registered_strategy(self):
        """analyze.py must resolve names through the shared registry, not a local map."""
        import scripts.analyze as analyze

        for name in get_available_strategies():
            with self.subTest(strategy=name):
                self.assertIs(analyze.get_strategy_class(name), STRATEGY_CLASSES[name])

    def _argparse_accepts(self, main, argv):
        """True if argv survived argument parsing, whatever happened afterwards."""
        with patch('sys.argv', argv), \
                patch('sys.stderr', new=io.StringIO()), \
                patch('sys.stdout', new=io.StringIO()):
            try:
                main()
            except SystemExit as e:
                # argparse exits 2 on an invalid choice; anything else means the
                # arguments parsed and the run failed later, which is fine here.
                return e.code != 2
            except Exception:
                return True
        return True

    def test_backtest_accepts_every_registered_strategy(self):
        from scripts.backtest import main

        for name in get_available_strategies():
            with self.subTest(strategy=name):
                self.assertTrue(
                    self._argparse_accepts(
                        main,
                        ['backtest.py', '--data', 'no_such_file.csv', '--strategy', name],
                    ),
                    f"backtest.py rejected the registered strategy '{name}'"
                )

    def test_optimize_accepts_every_registered_strategy(self):
        from scripts.optimize import main

        for name in get_available_strategies():
            with self.subTest(strategy=name):
                self.assertTrue(
                    self._argparse_accepts(
                        main,
                        ['optimize.py', '--data', 'no_such_file.csv', '--strategy', name],
                    ),
                    f"optimize.py rejected the registered strategy '{name}'"
                )


if __name__ == '__main__':
    unittest.main()
