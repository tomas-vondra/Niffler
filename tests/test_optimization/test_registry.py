"""Contract tests for the optimizer registry.

The optimizer was the last name-to-class map not shaped like the strategy, risk
and exporter registries. These pin what makes it a seam: registering an
optimizer is one edit, every CLI's method choices come from the registry, and
the old ``optimizer_factory`` import path still reaches the same objects.
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.optimization import optimizer_factory
from niffler.optimization.base_optimizer import BaseOptimizer
from niffler.optimization.grid_search_optimizer import GridSearchOptimizer
from niffler.optimization.parameter_space import ParameterSpace
from niffler.optimization.random_search_optimizer import RandomSearchOptimizer
from niffler.optimization.registry import (
    OPTIMIZER_CLASSES,
    create_optimizer,
    get_available_optimizers,
    get_optimizer_class,
)
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
from scripts import analyze, compare, optimize, screen


class DummyOptimizer(BaseOptimizer):
    """A minimal optimizer, used to prove registration costs exactly one edit."""

    def optimize(self):
        return []


def make_data():
    index = pd.date_range('2024-01-01', periods=30, freq='D')
    closes = [100.0 + i for i in range(30)]
    return pd.DataFrame({
        'open': closes, 'high': closes, 'low': closes, 'close': closes,
        'volume': [1000.0] * 30,
    }, index=index)


def make_space():
    return ParameterSpace({'short_window': {'type': 'int', 'min': 3, 'max': 5, 'step': 1}})


def help_text(main, script):
    buf = io.StringIO()
    with patch.object(sys, 'argv', [script, '--help']), redirect_stdout(buf):
        try:
            main()
        except SystemExit:
            pass
    return buf.getvalue()


class TestOptimizerRegistry(unittest.TestCase):

    def test_registered_names(self):
        self.assertEqual(get_available_optimizers(), ['grid', 'random'])

    def test_lookup_returns_the_class(self):
        self.assertIs(get_optimizer_class('grid'), GridSearchOptimizer)
        self.assertIs(get_optimizer_class('random'), RandomSearchOptimizer)

    def test_unknown_name_lists_what_is_registered(self):
        with self.assertRaises(ValueError) as raised:
            get_optimizer_class('annealing')

        self.assertIn("'annealing'", str(raised.exception))
        self.assertIn('grid, random', str(raised.exception))

    def test_create_builds_the_registered_class(self):
        optimizer = create_optimizer('random', SimpleMAStrategy, make_space(), make_data(),
                                     sort_by='sharpe_ratio', n_jobs=1)

        self.assertIsInstance(optimizer, RandomSearchOptimizer)
        self.assertEqual(optimizer.sort_by, 'sharpe_ratio')

    def test_create_rejects_an_unknown_method(self):
        with self.assertRaises(ValueError):
            create_optimizer('annealing', SimpleMAStrategy, make_space(), make_data())


class TestRegistrationIsOneEdit(unittest.TestCase):

    def setUp(self):
        registered = patch.dict(OPTIMIZER_CLASSES, {'dummy': DummyOptimizer})
        registered.start()
        self.addCleanup(registered.stop)

    def test_it_is_listed_and_constructible(self):
        self.assertIn('dummy', get_available_optimizers())
        optimizer = create_optimizer('dummy', SimpleMAStrategy, make_space(), make_data(),
                                     n_jobs=1)
        self.assertIsInstance(optimizer, DummyOptimizer)

    def test_every_cli_offers_it(self):
        for module, script in ((optimize, 'optimize.py'), (analyze, 'analyze.py'),
                               (compare, 'compare.py'), (screen, 'screen.py')):
            with self.subTest(script=script):
                self.assertIn('dummy', help_text(module.main, script))


class TestFactoryShim(unittest.TestCase):
    """``optimizer_factory`` re-exports the registry; it holds no second map."""

    def test_the_old_import_path_reaches_the_same_objects(self):
        self.assertIs(optimizer_factory.OPTIMIZER_CLASSES, OPTIMIZER_CLASSES)
        self.assertIs(optimizer_factory.create_optimizer, create_optimizer)
        self.assertIs(optimizer_factory.get_available_optimizers, get_available_optimizers)

    def test_the_parameter_space_helper_is_still_there(self):
        space = optimizer_factory.get_parameter_space('simple_ma')
        self.assertIsInstance(space, ParameterSpace)


if __name__ == '__main__':
    unittest.main()
