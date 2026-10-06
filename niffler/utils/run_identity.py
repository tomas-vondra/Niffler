"""Run identity: which execution a result is, and which results belong together.

Provenance (:mod:`niffler.utils.provenance`) answers "what code and data produced
this?". It cannot answer "what belongs together?": an optimization, the
walk-forward that validates it and the final backtest all run on the same commit
and the same file, so their provenance is identical. Three fields close that gap:

* ``run_id`` - one execution of one script. Always minted, here and nowhere else.
* ``parent_run_id`` - the run whose output this run was fed (read from a params
  file). A fact about the input, recorded whether or not an experiment is named.
* ``experiment`` - the research question, as a name the user chose.

**The experiment is never minted.** An invented id for an unlabelled run is a
plausible-looking default: a forgotten flag would create a one-run experiment
indistinguishable from a deliberate one. Unknown is ``None``.

Standard library only, like the rest of :mod:`niffler.utils`.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

#: Every kind of run a script can be. One tuple, so an exporter or a dashboard
#: filter never carries its own spelling of it.
RUN_KIND_BACKTEST = 'backtest'
RUN_KIND_OPTIMIZE = 'optimize'
RUN_KIND_WALK_FORWARD = 'walk_forward'
RUN_KIND_MONTE_CARLO = 'monte_carlo'
RUN_KIND_COMPARE = 'compare'
RUN_KIND_SCREEN = 'screen'

RUN_KINDS: Tuple[str, ...] = (
    RUN_KIND_BACKTEST,
    RUN_KIND_OPTIMIZE,
    RUN_KIND_WALK_FORWARD,
    RUN_KIND_MONTE_CARLO,
    RUN_KIND_COMPARE,
    RUN_KIND_SCREEN,
)

#: How an unnamed experiment is shown to a person. Never stored.
NO_EXPERIMENT_LABEL = '(none)'


class ExperimentMismatchError(ValueError):
    """A run was fed by one experiment while configured as another."""


def mint_run_id() -> str:
    """Return a new unique run id. The only mint site in the codebase."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class RunIdentity:
    """Identity of one script execution.

    Attributes:
        run_id: Unique id of this execution.
        kind: One of :data:`RUN_KINDS`.
        experiment: Name of the research question, or None when unnamed.
        parent_run_id: The run whose output fed this one, or None.
        profile: The ``niffler.toml`` profile in force, or None.
    """

    run_id: str
    kind: str
    experiment: Optional[str] = None
    parent_run_id: Optional[str] = None
    profile: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.kind not in RUN_KINDS:
            raise ValueError(
                f"Unknown run kind '{self.kind}'. Kinds: {', '.join(RUN_KINDS)}"
            )
        if self.experiment is not None and not self.experiment.strip():
            raise ValueError("experiment must be a non-empty name or None")

    def to_metadata(self) -> Dict[str, Any]:
        """Render the identity for JSON output and exported documents."""
        return {
            'run_id': self.run_id,
            'kind': self.kind,
            'experiment': self.experiment,
            'parent_run_id': self.parent_run_id,
            'profile': self.profile,
        }


def new_run_identity(kind: str,
                     experiment: Optional[str] = None,
                     parent_run_id: Optional[str] = None,
                     profile: Optional[str] = None) -> RunIdentity:
    """Mint the identity of a run that is starting now.

    Args:
        kind: One of :data:`RUN_KINDS`.
        experiment: Name of the research question, or None.
        parent_run_id: The run whose output feeds this one, or None.
        profile: The configuration profile in force, or None.

    Returns:
        A new identity with a freshly minted ``run_id``.
    """
    return RunIdentity(
        run_id=mint_run_id(),
        kind=kind,
        experiment=experiment,
        parent_run_id=parent_run_id,
        profile=profile,
    )


def resolve_experiment(own: Optional[str],
                       parent: Optional[str],
                       own_is_explicit: bool = False,
                       own_origin: Optional[str] = None,
                       parent_origin: Optional[str] = None
                       ) -> Tuple[Optional[str], Optional[str]]:
    """Decide which experiment a run belongs to, given the run that fed it.

    The results of an optimization belong to its experiment, so the steps fed
    by them do too: an unnamed run inherits. A *different* name is nearly
    always a forgotten or wrong profile, so it stops the run - unless the name
    was typed on the command line, which is a statement of intent (reusing
    tuned parameters as a baseline under a new question).

    Args:
        own: The experiment this run was configured with, or None.
        parent: The experiment of the run that fed this one, or None.
        own_is_explicit: True when ``own`` was typed on the command line.
        own_origin: Where ``own`` came from, for the error message.
        parent_origin: Where ``parent`` was read from, for the error message.

    Returns:
        ``(experiment, note)``. ``note`` is a line worth printing, or None when
        nothing happened that the user did not ask for.

    Raises:
        ExperimentMismatchError: If the two names differ and ``own`` was not
            typed on the command line.
    """
    if parent is None or own == parent:
        return own, None

    source = f" ({parent_origin})" if parent_origin else ''

    if own is None:
        return parent, f'Experiment "{parent}" inherited from the parent run{source}'

    if own_is_explicit:
        return own, (
            f'Parent run{source} belonged to experiment "{parent}"; this run is '
            f'"{own}" because --experiment was given'
        )

    configured = f" (set in {own_origin})" if own_origin else ''
    raise ExperimentMismatchError(
        f'Experiment mismatch: this run is configured as "{own}"{configured}, but '
        f'the parent run{source} belongs to "{parent}". Use the matching profile, '
        f'or pass --experiment on the command line to place this run in a '
        f'different experiment on purpose.'
    )


def format_run_identity(identity: RunIdentity) -> str:
    """Render the one-line summary a script prints for its run."""
    parts = [
        f"Run: {identity.run_id}",
        f"kind: {identity.kind}",
        f"experiment: {identity.experiment or NO_EXPERIMENT_LABEL}",
    ]
    if identity.parent_run_id:
        parts.append(f"parent: {identity.parent_run_id}")
    if identity.profile:
        parts.append(f"profile: {identity.profile}")
    return ' | '.join(parts)
