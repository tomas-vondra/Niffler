from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from .base_risk_manager import BaseRiskManager


class KellyRiskManager(BaseRiskManager):
    """
    Kelly Criterion position sizing. **Deliberately unimplemented** - see below.

    The constructor, validation and risk metrics are real; all three sizing
    methods raise :class:`NotImplementedError`. ``FixedRiskManager`` is the only
    usable risk manager, and this class is not selectable from the
    ``backtest.py`` CLI. That is a decision, not an abandoned branch, and this
    docstring exists so a reader does not have to guess which.

    What Kelly computes
    -------------------
    The growth-optimal fraction of capital to commit to a bet - the fraction
    maximising the expected log of terminal wealth. For a discrete bet,
    ``f* = (bp - q) / b``, where ``b`` is the payoff odds (average win over
    average loss), ``p`` the win probability and ``q = 1 - p``. For continuously
    compounded returns it reduces to ``f* ~= mu / sigma^2``.

    Why it is not implemented here
    ------------------------------
    Kelly is a **position-sizing rule, not an edge-detector**. ``f*`` is a
    function of the estimated edge, so it can only amplify an edge that has
    already been established - it never finds one. Niffler is currently
    single-strategy, single-symbol and long-only, which collapses the sizing
    question to a single scalar that ``FixedRiskManager`` already answers well
    enough. Kelly starts earning its keep when capital must be *allocated*
    between several competing opportunities with different edges, which is the
    multi-asset work that does not exist yet. Shipping it now would add
    estimation risk without answering a question the framework is currently
    asking.

    What a correct implementation would require
    -------------------------------------------
    Recorded so the reader can see the shape of the problem rather than assume
    it was merely skipped:

    * **An expanding-window estimate over closed round trips only.** ``p`` and
      ``b`` must come from trades that closed *strictly before* the current bar.
      Estimating them from the whole backtest's trades and using that to size an
      early trade is look-ahead bias - precisely the class of bug the
      correctness audit removed elsewhere in this codebase.
    * **A minimum-sample gate before Kelly activates at all**, mirroring
      :data:`niffler.backtesting.significance.DEFAULT_MIN_TRADES`. ``f*`` from a
      handful of trades is noise scaled up by leverage. Note that the
      ``min_trades_for_kelly`` default below is 10, well under that gate, and
      would need revisiting.
    * **A fractional cap.** The growth curve is concave and returns to zero at
      twice the optimal fraction, so overestimating the edge by 2x buys all of
      the variance and none of the growth. Hence ``fractional_kelly`` and
      ``max_kelly_fraction``; half-Kelly or quarter-Kelly is the practical
      choice, and full Kelly on an *estimated* edge is not.
    * **A clamp of negative ``f*`` to zero.** A negative Kelly fraction is an
      instruction to take the other side, and the engine is long-only, so the
      only faithful response is to stand aside.
    """
    
    def __init__(self, lookback_periods: int = 50, max_kelly_fraction: float = 0.25,
                 stop_loss_pct: float = 0.05, min_trades_for_kelly: int = 10,
                 max_risk_per_trade: float = 0.05, fractional_kelly: float = 1.0,
                 max_positions: int = 5):
        """
        Initialize Kelly risk manager.
        
        Args:
            lookback_periods: Number of historical periods to analyze for Kelly calculation
            max_kelly_fraction: Maximum position size regardless of Kelly calculation
            stop_loss_pct: Fixed stop loss percentage (used if Kelly stop not viable)
            min_trades_for_kelly: Minimum number of trades needed to use Kelly criterion
            max_risk_per_trade: Maximum risk per trade as fraction of portfolio
            fractional_kelly: Fraction of Kelly to use (0.25=quarter-Kelly, 0.5=half-Kelly, 1.0=full-Kelly)
            max_positions: Maximum number of concurrent positions
        """
        # Set instance attributes first
        self.lookback_periods = lookback_periods
        self.max_kelly_fraction = max_kelly_fraction
        self.stop_loss_pct = stop_loss_pct
        self.min_trades_for_kelly = min_trades_for_kelly
        self.max_risk_per_trade = max_risk_per_trade
        self.fractional_kelly = fractional_kelly
        self.max_positions = max_positions
        
        config = {
            'lookback_periods': lookback_periods,
            'max_kelly_fraction': max_kelly_fraction,
            'stop_loss_pct': stop_loss_pct,
            'min_trades_for_kelly': min_trades_for_kelly,
            'max_risk_per_trade': max_risk_per_trade,
            'fractional_kelly': fractional_kelly,
            'max_positions': max_positions,
            'max_position_size': max_kelly_fraction,  # Map to base class expected key
            'max_total_exposure': max_positions * max_kelly_fraction  # Total possible exposure
        }
        super().__init__(config)
        
    def calculate_position_size(self, signal: int, current_price: float,
                              portfolio_value: float, historical_data: pd.DataFrame,
                              current_position: float = 0.0) -> float:
        """
        Calculate position size using Kelly Criterion.

        Unimplemented by design; raises. See the class docstring for why Kelly
        is not the sizing rule this framework currently needs, and for the
        look-ahead, sample-size, fractional-cap and long-only constraints any
        implementation would have to satisfy.
        """
        raise NotImplementedError("Kelly position sizing not yet implemented")

    def calculate_stop_loss(self, entry_price: float, signal: int,
                          historical_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate stop-loss based on historical volatility or fixed percentage.

        Unimplemented by design; raises. See the class docstring. A volatility
        stop would be estimated from bars strictly before the current one, for
        the same look-ahead reason that governs the sizing estimate.
        """
        raise NotImplementedError("Kelly stop loss calculation not yet implemented")

    def should_close_position(self, current_price: float, entry_price: float,
                            stop_loss_price: Optional[float], signal: int,
                            unrealized_pnl: float) -> Tuple[bool, str]:
        """
        Determine if position should be closed based on stop-loss or Kelly criteria.

        Unimplemented by design; raises. See the class docstring. Note that the
        engine sizes *entries* from the risk manager only - an exit uses the
        strategy's own position size - so this method would gate a close, not
        resize it.
        """
        raise NotImplementedError("Kelly position closing logic not yet implemented")
        
    def _validate_config(self):
        """Validate Kelly risk manager configuration."""
        if self.lookback_periods <= 0:
            raise ValueError("lookback_periods must be positive")
            
        if not 0 < self.max_kelly_fraction <= 1.0:
            raise ValueError("max_kelly_fraction must be between 0 and 1")
            
        if not 0 <= self.stop_loss_pct <= 1.0:
            raise ValueError("stop_loss_pct must be between 0 and 1")
            
        if self.min_trades_for_kelly <= 0:
            raise ValueError("min_trades_for_kelly must be positive")
            
        if not 0 < self.max_risk_per_trade <= 1.0:
            raise ValueError("max_risk_per_trade must be between 0 and 1")
            
        if not 0 < self.fractional_kelly <= 1.0:
            raise ValueError("fractional_kelly must be between 0 and 1")
            
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
            
    def _validate_subclass_relationships(self):
        """Validate Kelly risk manager specific parameter relationships."""
        # Check if lookback period is reasonable for Kelly calculation
        if self.lookback_periods > 252:  # More than 1 year of daily data
            warnings.warn(
                f"Large lookback period ({self.lookback_periods}) might use outdated data for Kelly calculation. "
                f"Consider using 20-60 periods for more responsive risk management."
            )
            
        # Check if min_trades_for_kelly is reasonable relative to lookback_periods
        if self.min_trades_for_kelly > self.lookback_periods:
            raise ValueError(
                f"min_trades_for_kelly ({self.min_trades_for_kelly}) cannot exceed "
                f"lookback_periods ({self.lookback_periods})"
            )
            
        # Warning if max_kelly_fraction is very high
        if self.max_kelly_fraction > 0.5:
            warnings.warn(
                f"High max Kelly fraction ({self.max_kelly_fraction*100:.1f}%) detected. "
                f"Kelly criterion can suggest aggressive position sizes. Consider capping at 25%."
            )
            
        # Recommend fractional Kelly for practical use
        if self.fractional_kelly == 1.0 and self.max_kelly_fraction > 0.1:
            warnings.warn(
                f"Using full Kelly ({self.fractional_kelly}) with high max fraction. "
                f"Consider using fractional Kelly (0.25-0.5) for reduced volatility and drawdown."
            )
            
        # Warning about fractional Kelly benefits
        if self.fractional_kelly < 0.1:
            warnings.warn(
                f"Very conservative fractional Kelly ({self.fractional_kelly}) detected. "
                f"This may significantly underperform optimal growth."
            )
            
        # Check if stop loss makes sense with Kelly approach
        if self.stop_loss_pct > 0.15:  # More than 15% stop loss
            warnings.warn(
                f"Large stop loss ({self.stop_loss_pct*100:.1f}%) might conflict with Kelly optimization. "
                f"Kelly assumes optimal position sizing, but large stops suggest high volatility."
            )
            
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive Kelly risk manager metrics."""
        metrics = super().get_risk_metrics()
        
        # Calculate current positions and utilization
        current_positions = len([pos for pos in self._positions.values() if pos.position_size != 0])
        position_utilization = current_positions / self.max_positions if self.max_positions > 0 else 0
        
        # Calculate effective Kelly fraction being used
        effective_max_kelly = self.max_kelly_fraction * self.fractional_kelly
        
        metrics.update({
            'lookback_periods': self.lookback_periods,
            'max_kelly_fraction': self.max_kelly_fraction,
            'fractional_kelly': self.fractional_kelly,
            'effective_max_kelly': effective_max_kelly,
            'stop_loss_pct': self.stop_loss_pct,
            'min_trades_for_kelly': self.min_trades_for_kelly,
            'max_positions': self.max_positions,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_total_exposure': self.max_positions * self.max_kelly_fraction,
            'current_positions': current_positions,
            'position_utilization': position_utilization,
            'risk_management_type': 'Kelly Criterion (Not Implemented)'
        })
        return metrics
    
    def __str__(self):
        """String representation of KellyRiskManager."""
        return (f"KellyRiskManager(max_kelly_fraction={self.max_kelly_fraction}, "
                f"max_positions={self.max_positions}) - NOT IMPLEMENTED")
    
    def __repr__(self):
        """Repr representation of KellyRiskManager."""
        return (f"KellyRiskManager(lookback_periods={self.lookback_periods}, "
                f"max_kelly_fraction={self.max_kelly_fraction}, "
                f"stop_loss_pct={self.stop_loss_pct}, "
                f"max_positions={self.max_positions})")