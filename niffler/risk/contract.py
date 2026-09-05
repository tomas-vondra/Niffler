"""The contract between the backtest engine and a risk manager.

Two types live here and nothing else, because these are the only things the
engine and the risk layer have to agree on:

* :class:`PortfolioSnapshot` - the portfolio state a risk manager is allowed to
  see, handed to it at the call site.
* :class:`RiskManager` - the methods the engine actually calls.

Why the state is passed in rather than held
-------------------------------------------
A risk manager used to keep its own ``Dict[str, PositionInfo]``, mutated by the
engine through ``update_position_state`` / ``clear_position``. That made a
manager instance a *history*: reusing one across two runs let the first run's
open position veto the second run's first entry through ``max_positions``.
Walk-forward runs its folds in parallel and every fold is an independent
hypothetical history, so a shared manager could not be threaded into validation
at all - which is why it never was.

:class:`niffler.backtesting.portfolio.Portfolio` already owns cash, position,
entry price, stop and side for the run in progress. It is therefore the single
owner of position state, and the risk manager is handed a frozen snapshot of the
part it needs. A manager provably cannot inherit another run's history, because
it has nowhere to put one.

Why a snapshot and not ``Portfolio`` itself
-------------------------------------------
The dependency has to run one way: ``niffler/backtesting`` imports
``niffler/risk``, never the reverse. Passing ``Portfolio`` would invert that the
moment a manager type-annotated it, and the engine imports :class:`RiskManager`
from here, so it would be a cycle. A snapshot is also the *minimum* the manager
needs, which keeps the contract small enough to read.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class PortfolioSnapshot:
    """
    Immutable view of portfolio state at the moment a trade is evaluated.

    Frozen on purpose: a risk manager that cannot mutate the snapshot cannot use
    it as a place to accumulate state between calls.

    Attributes:
        open_positions: Number of positions currently open across the portfolio
        total_exposure: Sum of absolute position fractions across the portfolio,
            valued at the current price
        current_position: Signed position fraction in the symbol being evaluated
            (positive long, negative short, 0.0 when flat)
    """

    open_positions: int
    total_exposure: float
    current_position: float

    @classmethod
    def flat(cls) -> "PortfolioSnapshot":
        """A snapshot of a portfolio holding nothing.

        Only for callers that genuinely are flat - it is never a default, because
        a flat snapshot silently satisfies every ``max_positions`` and exposure
        check.

        Returns:
            A snapshot with no positions and no exposure.
        """
        return cls(open_positions=0, total_exposure=0.0, current_position=0.0)


@runtime_checkable
class RiskManager(Protocol):
    """
    What :class:`~niffler.backtesting.backtest_engine.BacktestEngine` requires of
    a risk manager.

    The engine previously reached its manager purely through duck typing, via
    ``strategy.risk_manager``, and never imported :mod:`niffler.risk` at all - so
    renaming a method broke a backtest at the first signal rather than at import.
    The engine now checks against this Protocol before the first bar.

    ``runtime_checkable`` verifies that the attributes *exist*, not that their
    signatures match; a wrong signature still surfaces as a ``TypeError`` at the
    call. That is a smaller guarantee than a full structural check, and still
    turns a rename into an immediate, named failure.
    """

    def evaluate_trade(self, signal: int, current_price: float,
                       portfolio_value: float, historical_data: pd.DataFrame,
                       portfolio: PortfolioSnapshot):
        """Size, stop and veto a pending order. Returns a ``RiskDecision``."""
        ...

    def should_close_position(self, current_price: float, entry_price: float,
                              stop_loss_price: Optional[float], signal: int,
                              unrealized_pnl: float) -> Tuple[bool, str]:
        """Decide whether an open position must be closed now."""
        ...
