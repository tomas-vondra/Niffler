"""Run identity: the id of an execution and the experiment it belongs to.

The rules under test are the ones that decide which box a run lands in. Each
wrong answer is silent in production - a run filed under the wrong experiment
looks exactly like one filed correctly - so every branch is pinned here.
"""

import unittest

from niffler.utils.run_identity import (
    RUN_KINDS,
    ExperimentMismatchError,
    RunIdentity,
    format_run_identity,
    mint_run_id,
    new_run_identity,
    resolve_experiment,
)


class TestRunIdentity(unittest.TestCase):

    def test_minted_ids_are_unique(self):
        self.assertNotEqual(mint_run_id(), mint_run_id())

    def test_every_kind_is_accepted(self):
        for kind in RUN_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(new_run_identity(kind).kind, kind)

    def test_an_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            RunIdentity(run_id='r1', kind='paper_trade')
        self.assertIn('paper_trade', str(ctx.exception))

    def test_an_empty_run_id_is_rejected(self):
        with self.assertRaises(ValueError):
            RunIdentity(run_id='', kind='backtest')

    def test_a_blank_experiment_is_rejected_rather_than_stored(self):
        with self.assertRaises(ValueError):
            RunIdentity(run_id='r1', kind='backtest', experiment='  ')

    def test_the_experiment_is_never_minted(self):
        """An unnamed run stays unnamed: None, not an invented id."""
        identity = new_run_identity('optimize')
        self.assertIsNone(identity.experiment)
        self.assertIsNone(identity.to_metadata()['experiment'])

    def test_metadata_carries_every_field(self):
        identity = RunIdentity(run_id='r1', kind='backtest', experiment='exp',
                               parent_run_id='p1', profile='prof')
        self.assertEqual(identity.to_metadata(), {
            'run_id': 'r1', 'kind': 'backtest', 'experiment': 'exp',
            'parent_run_id': 'p1', 'profile': 'prof',
        })

    def test_the_summary_line_shows_an_unnamed_experiment_as_none(self):
        line = format_run_identity(RunIdentity(run_id='r1', kind='backtest'))
        self.assertIn('experiment: (none)', line)
        self.assertNotIn('parent', line)

    def test_the_summary_line_names_parent_and_profile(self):
        line = format_run_identity(RunIdentity(
            run_id='r1', kind='backtest', experiment='exp',
            parent_run_id='p1', profile='prof'))
        self.assertIn('parent: p1', line)
        self.assertIn('profile: prof', line)


class TestResolveExperiment(unittest.TestCase):

    def test_no_parent_keeps_the_configured_experiment(self):
        self.assertEqual(resolve_experiment('a', None), ('a', None))
        self.assertEqual(resolve_experiment(None, None), (None, None))

    def test_an_unnamed_run_inherits_its_parents_experiment(self):
        experiment, note = resolve_experiment(None, 'breakout-btc')
        self.assertEqual(experiment, 'breakout-btc')
        self.assertIn('inherited', note)

    def test_the_same_experiment_needs_no_note(self):
        self.assertEqual(resolve_experiment('a', 'a'), ('a', None))

    def test_a_different_experiment_is_an_error_naming_both(self):
        with self.assertRaises(ExperimentMismatchError) as ctx:
            resolve_experiment('b', 'a', own_origin='niffler.toml [profile.b]',
                               parent_origin='opt.json')
        message = str(ctx.exception)
        self.assertIn('"a"', message)
        self.assertIn('"b"', message)
        self.assertIn('niffler.toml [profile.b]', message)
        self.assertIn('opt.json', message)
        self.assertIn('--experiment', message)

    def test_a_typed_experiment_overrides_the_mismatch(self):
        experiment, note = resolve_experiment('b', 'a', own_is_explicit=True)
        self.assertEqual(experiment, 'b')
        self.assertIn('"a"', note)

    def test_a_typed_experiment_does_not_block_inheritance_when_equal(self):
        self.assertEqual(resolve_experiment('a', 'a', own_is_explicit=True), ('a', None))


if __name__ == '__main__':
    unittest.main()
