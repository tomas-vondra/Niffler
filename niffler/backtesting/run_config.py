"""The engine settings of one run, built once and carried everywhere.

``BacktestEngine`` takes ten configuration knobs. Before this module existed the
engine was constructed at six places and five of them passed three: the two
analyzers, the optimizer and both of their spawn-safe worker paths built an
engine from ``initial_capital``, ``commission`` and ``cost_model`` alone. Every
other knob silently reverted to its default somewhere inside a walk-forward
fold, so there was no way to run walk-forward with ``periods_per_year=252``, or
without a benchmark, or with a different execution timing - the flags simply did
not reach the engine that did the work.

``RunConfig`` is that set of knobs as one frozen value. A caller either has one
or does not; there is no way to populate half of it, and adding a knob is one
field here rather than six constructor edits, five of which get forgotten.

Why it lives in ``niffler/backtesting/``
----------------------------------------
It is the argument list of ``BacktestEngine`` and it carries a ``CostModel``, so
it belongs to the backtesting layer. It cannot live in ``niffler/utils/``, which
may not import from backtesting, and putting it in ``niffler/optimization/`` or
``niffler/analysis/`` would make the two layers that consume it depend on each
other. It imports only leaf modules of its own layer (``cost_model``,
``benchmark``, ``significance``) and never imports ``backtest_engine``, so the
engine can import it: ``BacktestEngine.from_config`` is the single place a
config becomes an engine.

Validation lives here
---------------------
The ranges used to be checked in the engine and again, partially, in
``BaseOptimizer``, ``WalkForwardAnalyzer`` and ``MonteCarloAnalyzer``. There is
now one copy: constructing a ``RunConfig`` validates it, and ``BacktestEngine``
builds one from its own arguments rather than repeating the checks. An invalid
setting is therefore rejected at the same place with the same message whether it
came from a CLI flag, a library caller or a worker process.

Defaults are the engine's defaults, unchanged. ``bootstrap_samples`` in
particular defaults to 0: the bootstrap Sharpe interval is the only expensive
part of the assessment and nothing inside an optimisation or Monte Carlo loop
reads it, so only ``backtest.py`` turns it on.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .benchmark import BENCHMARK_BUY_AND_HOLD, BENCHMARK_CHOICES, BENCHMARK_NONE
from .cost_model import CostModel
from .significance import DEFAULT_BOOTSTRAP_SEED, DEFAULT_MIN_TRADES

#: Supported execution-timing policies. Defined here so the engine, the config
#: and any validation share one tuple.
EXECUTION_TIMINGS = ('next_bar_open', 'same_bar_close')

#: The only bias-free timing: a signal from bar i fills at bar i+1's open.
DEFAULT_EXECUTION_TIMING = 'next_bar_open'

DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_COMMISSION = 0.001
DEFAULT_MIN_ORDER_VALUE = 1.0

#: Resample count that skips the bootstrap Sharpe interval. It is the engine's
#: default because the interval is the only expensive part of the assessment and
#: nothing inside an optimisation or a Monte Carlo loop reads it;
#: ``significance.DEFAULT_BOOTSTRAP_SAMPLES`` is the count the backtest CLI
#: turns it on with.
BOOTSTRAP_DISABLED = 0


@dataclass(frozen=True)
class RunConfig:
    """Every setting ``BacktestEngine`` accepts, as one immutable value.

    Frozen so a config handed to an analyzer cannot be mutated behind the
    backtests that already used it, and picklable so it crosses the spawn
    boundary into worker processes intact - which is exactly where the dropped
    knobs used to be dropped.

    Attributes:
        initial_capital: Starting capital.
        commission: Commission rate per trade (0.001 = 0.1%).
        min_order_value: Smallest order notional that executes.
        execution_timing: ``next_bar_open`` (default) or the look-ahead-biased
            ``same_bar_close``. Deliberately not exposed on any CLI.
        periods_per_year: Annualisation factor. ``None`` means "infer from the
            index", which is not the same as 252 and must stay distinguishable
            from it.
        cost_model: Transaction cost model. ``None`` means the engine's
            frictionless ``ZeroCostModel``; it is kept as ``None`` rather than
            normalised so "no model was configured" stays visible in metadata.
        benchmark: ``buy_and_hold`` (default) or ``none``. ``None`` is accepted
            and normalised to ``none``, as the engine has always done.
        min_trades_for_significance: Round trips below which no significance
            verdict is rendered.
        bootstrap_samples: Resamples for the bootstrap Sharpe interval; 0 skips
            it.
        bootstrap_seed: Seed for that bootstrap.
    """

    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    commission: float = DEFAULT_COMMISSION
    min_order_value: float = DEFAULT_MIN_ORDER_VALUE
    execution_timing: str = DEFAULT_EXECUTION_TIMING
    periods_per_year: Optional[float] = None
    cost_model: Optional[CostModel] = None
    benchmark: Optional[str] = BENCHMARK_BUY_AND_HOLD
    min_trades_for_significance: int = DEFAULT_MIN_TRADES
    bootstrap_samples: int = BOOTSTRAP_DISABLED
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED

    def __post_init__(self) -> None:
        """Validate every setting and normalise a ``None`` benchmark.

        Raises:
            ValueError: If any setting is outside its valid range.
            TypeError: If ``cost_model`` is not a ``CostModel``.
        """
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
        if self.min_order_value < 0:
            raise ValueError("Minimum order value cannot be negative")
        if self.execution_timing not in EXECUTION_TIMINGS:
            raise ValueError(
                f"Execution timing must be one of {EXECUTION_TIMINGS}, "
                f"got '{self.execution_timing}'"
            )
        if self.periods_per_year is not None and self.periods_per_year <= 0:
            raise ValueError("Periods per year must be positive")
        if self.cost_model is not None and not isinstance(self.cost_model, CostModel):
            raise TypeError(
                f"cost_model must be a CostModel, got {type(self.cost_model).__name__}"
            )
        # An explicit None has always meant "no benchmark"; it is normalised
        # rather than rejected so the engine's old contract still holds.
        if self.benchmark is None:
            object.__setattr__(self, 'benchmark', BENCHMARK_NONE)
        if self.benchmark not in BENCHMARK_CHOICES:
            raise ValueError(
                f"Benchmark must be one of {BENCHMARK_CHOICES}, got '{self.benchmark}'"
            )
        if self.min_trades_for_significance < 0:
            raise ValueError("Minimum trades for significance cannot be negative")
        if self.bootstrap_samples < 0:
            raise ValueError("Bootstrap samples cannot be negative")

    def to_metadata(self) -> Dict[str, Any]:
        """Render the config for JSON output and console reporting.

        The cost model is reduced to its description, which is the only part of
        it that is serialisable and the only part a reader needs.

        Returns:
            A JSON-safe dict of every setting.
        """
        return {
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'min_order_value': self.min_order_value,
            'execution_timing': self.execution_timing,
            'periods_per_year': self.periods_per_year,
            'cost_model': (self.cost_model.description
                           if self.cost_model is not None else None),
            'benchmark': self.benchmark,
            'min_trades_for_significance': self.min_trades_for_significance,
            'bootstrap_samples': self.bootstrap_samples,
            'bootstrap_seed': self.bootstrap_seed,
        }


def resolve_run_config(run_config: Optional[RunConfig]) -> RunConfig:
    """Return the given config, or the default one when a caller passed none.

    Args:
        run_config: A config, or None.

    Returns:
        ``run_config`` when it is not None, otherwise ``RunConfig()`` - whose
        defaults are the engine's own defaults, so an unconfigured caller gets
        exactly the behaviour it had before this type existed.
    """
    return run_config if run_config is not None else RunConfig()
