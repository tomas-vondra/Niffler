from typing import Dict, Any, Optional, Tuple
import pandas as pd
import warnings
from .base_risk_manager import BaseRiskManager


class FixedRiskManager(BaseRiskManager):
    """
    Fixed risk manager using constant position sizing and stop-loss percentages.
    
    This risk manager uses:
    - Fixed position size as percentage of portfolio
    - Fixed stop-loss percentage from entry price
    - Simple position limits
    """
    
    def __init__(self, position_size_pct: float = 0.1, stop_loss_pct: float = 0.05,
                 max_positions: int = 5, max_risk_per_trade: float = 0.02):
        """
        Initialize fixed risk manager.
        
        Args:
            position_size_pct: Fixed position size as fraction of portfolio (e.g., 0.1 = 10%)
            stop_loss_pct: Fixed stop loss as fraction of entry price (e.g., 0.05 = 5%)
            max_positions: Maximum number of concurrent positions
            max_risk_per_trade: Maximum risk per trade as fraction of portfolio
        """
        # Set instance attributes first
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        
        config = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'max_positions': max_positions,
            'max_risk_per_trade': max_risk_per_trade,
            'max_position_size': position_size_pct,  # Map to base class expected key
            'max_total_exposure': max_positions * position_size_pct  # Total possible exposure
        }
        super().__init__(config)
        
    def calculate_position_size(self, signal: int, current_price: float,
                              portfolio_value: float, historical_data: pd.DataFrame,
                              current_position: float = 0.0) -> float:
        """
        Calculate fixed position size with portfolio risk validation.
        
        Args:
            signal: Trading signal (1 for buy, -1 for sell, 0 for hold)
            current_price: Current asset price
            portfolio_value: Current portfolio value
            historical_data: Historical price data (not used in fixed strategy)
            current_position: Current position size
            
        Returns:
            Fixed position size as fraction of portfolio
        """
        if signal == 0:
            return 0.0
            
        # For sells, close the entire current position
        if signal == -1:
            return abs(current_position) if current_position != 0 else 0.0
            
        # For buys, return fixed position size (portfolio checks handled by base class)
        if signal == 1:
            return self.position_size_pct
        
    def calculate_stop_loss(self, entry_price: float, signal: int,
                          historical_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate fixed percentage stop-loss price.
        
        Args:
            entry_price: Price at which position was entered
            signal: Trading signal (1 for long, -1 for short)
            historical_data: Historical price data (not used in fixed strategy)
            
        Returns:
            Stop loss price based on fixed percentage
        """
        if signal == 1:  # Long position
            return entry_price * (1 - self.stop_loss_pct)
        elif signal == -1:  # Short position
            return entry_price * (1 + self.stop_loss_pct)
        else:
            return None
            
    def should_close_position(self, current_price: float, entry_price: float,
                            stop_loss_price: Optional[float], signal: int,
                            unrealized_pnl: float) -> Tuple[bool, str]:
        """
        Determine if position should be closed based on stop-loss.
        
        Args:
            current_price: Current asset price
            entry_price: Entry price of the position
            stop_loss_price: Stop loss price
            signal: Original signal (1 for long, -1 for short)
            unrealized_pnl: Current unrealized profit/loss
            
        Returns:
            Tuple of (should_close, reason)
        """
        if stop_loss_price is None:
            return False, "No stop loss set"
            
        if signal == 1:  # Long position
            if current_price <= stop_loss_price:
                return True, f"Stop loss hit: {current_price:.2f} <= {stop_loss_price:.2f}"
        elif signal == -1:  # Short position
            if current_price >= stop_loss_price:
                return True, f"Stop loss hit: {current_price:.2f} >= {stop_loss_price:.2f}"
                
        return False, "Stop loss not triggered"
        
    def _validate_config(self):
        """Validate fixed risk manager configuration."""
        if not 0 < self.position_size_pct <= 1.0:
            raise ValueError("position_size_pct must be between 0 and 1")
            
        if not 0 <= self.stop_loss_pct <= 1.0:
            raise ValueError("stop_loss_pct must be between 0 and 1")
            
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
            
        if not 0 < self.max_risk_per_trade <= 1.0:
            raise ValueError("max_risk_per_trade must be between 0 and 1")
            
        # Ensure position size doesn't exceed maximum risk per trade
        if self.position_size_pct * self.stop_loss_pct > self.max_risk_per_trade:
            raise ValueError(
                f"Position size ({self.position_size_pct}) * stop loss ({self.stop_loss_pct}) "
                f"= {self.position_size_pct * self.stop_loss_pct:.3f} "
                f"exceeds max risk per trade ({self.max_risk_per_trade})"
            )
            
    def _validate_subclass_relationships(self):
        """Validate Fixed risk manager specific parameter relationships."""
        # Check if position size and stop loss combination makes sense
        estimated_risk = self.position_size_pct * self.stop_loss_pct
        
        # Warning if estimated risk is very small (might indicate conservative settings)
        if estimated_risk < 0.001:  # Less than 0.1% risk per trade
            warnings.warn(
                f"Very conservative risk settings detected: "
                f"Estimated risk per trade is only {estimated_risk:.4f} ({estimated_risk*100:.2f}%). "
                f"This might result in very small position sizes."
            )
            
        # Warning if stop loss is very large (might indicate aggressive settings)  
        if self.stop_loss_pct > 0.2:  # More than 20% stop loss
            warnings.warn(
                f"Large stop loss detected: {self.stop_loss_pct*100:.1f}%. "
                f"This might result in significant losses per trade."
            )
            
        # Validate portfolio-level risk exposure
        max_portfolio_risk = self.max_positions * estimated_risk
        if max_portfolio_risk > 0.5:  # More than 50% portfolio risk
            warnings.warn(
                f"High portfolio risk detected: Maximum possible risk exposure is "
                f"{max_portfolio_risk*100:.1f}% if all {self.max_positions} positions hit stop loss."
            )
            
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive fixed risk manager metrics."""
        metrics = super().get_risk_metrics()
        
        # Calculate current portfolio utilization
        current_positions = len([pos for pos in self._positions.values() if pos.position_size != 0])
        position_utilization = current_positions / self.max_positions if self.max_positions > 0 else 0
        
        # Calculate theoretical maximum risk
        estimated_risk_per_trade = self.position_size_pct * self.stop_loss_pct
        max_portfolio_risk = self.max_positions * estimated_risk_per_trade
        
        # Calculate current actual risk exposure
        current_risk_exposure = current_positions * estimated_risk_per_trade
        
        metrics.update({
            'position_size_pct': self.position_size_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'max_positions': self.max_positions,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_total_exposure': self.max_positions * self.position_size_pct,
            'current_positions': current_positions,
            'position_utilization': position_utilization,
            'estimated_risk_per_trade': estimated_risk_per_trade,
            'max_portfolio_risk': max_portfolio_risk,
            'current_risk_exposure': current_risk_exposure,
            'risk_efficiency': estimated_risk_per_trade / self.max_risk_per_trade if self.max_risk_per_trade > 0 else 0,
            'risk_management_type': 'Fixed Risk Manager'
        })
        return metrics
    
    def __str__(self):
        """String representation of FixedRiskManager."""
        return (f"FixedRiskManager(position_size_pct={self.position_size_pct}, "
                f"stop_loss_pct={self.stop_loss_pct}, max_positions={self.max_positions})")
    
    def __repr__(self):
        """Repr representation of FixedRiskManager."""
        return (f"FixedRiskManager(position_size_pct={self.position_size_pct}, "
                f"stop_loss_pct={self.stop_loss_pct}, max_positions={self.max_positions}, "
                f"max_risk_per_trade={self.max_risk_per_trade})")