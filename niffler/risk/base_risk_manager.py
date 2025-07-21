from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass


@dataclass
class PositionInfo:
    """
    Information about a current position for risk management tracking.
    """
    symbol: str
    position_size: float  # Fraction of portfolio
    entry_price: float
    stop_loss_price: Optional[float]
    entry_timestamp: pd.Timestamp
    unrealized_pnl: float = 0.0


@dataclass
class RiskDecision:
    """
    Risk management decision containing position sizing and stop-loss information.
    """
    position_size: float  # Fraction of portfolio to allocate (0.0 to 1.0)
    stop_loss_price: Optional[float]  # Stop loss price level, None if no stop
    max_risk_per_trade: float  # Maximum risk as fraction of portfolio
    allow_trade: bool = True  # Whether trade should be executed
    reason: str = ""  # Reason for the decision


class BaseRiskManager(ABC):
    """
    Abstract base class for risk management systems.
    
    Risk managers are responsible for:
    - Position sizing based on risk tolerance and strategy performance
    - Stop-loss placement and management
    - Portfolio-level risk controls
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager with configuration.
        
        Args:
            config: Configuration dictionary with risk management parameters
        """
        self.config = config
        self._positions: Dict[str, PositionInfo] = {}  # Track current positions by symbol
        self._validate_config()
        self.validate_config_relationships()
        
    @abstractmethod
    def calculate_position_size(self, signal: int, current_price: float, 
                              portfolio_value: float, historical_data: pd.DataFrame,
                              current_position: float = 0.0) -> float:
        """
        Calculate position size for a given trading signal.
        
        Args:
            signal: Trading signal (1 for buy, -1 for sell, 0 for hold)
            current_price: Current asset price
            portfolio_value: Current portfolio value
            historical_data: Historical price and performance data
            current_position: Current position size (positive for long, negative for short)
            
        Returns:
            Position size as fraction of portfolio (0.0 to 1.0)
        """
        pass
        
    @abstractmethod
    def calculate_stop_loss(self, entry_price: float, signal: int,
                          historical_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate stop-loss price for a position.
        
        Args:
            entry_price: Price at which position was entered
            signal: Trading signal (1 for long, -1 for short)
            historical_data: Historical price data for volatility calculation
            
        Returns:
            Stop loss price, or None if no stop loss
        """
        pass
        
    @abstractmethod
    def should_close_position(self, current_price: float, entry_price: float,
                            stop_loss_price: Optional[float], signal: int,
                            unrealized_pnl: float) -> Tuple[bool, str]:
        """
        Determine if a position should be closed due to risk management rules.
        
        Args:
            current_price: Current asset price
            entry_price: Entry price of the position
            stop_loss_price: Stop loss price (can be None)
            signal: Original signal (1 for long, -1 for short)
            unrealized_pnl: Current unrealized profit/loss
            
        Returns:
            Tuple of (should_close, reason)
        """
        pass
        
    def update_position_state(self, symbol: str, position_size: float, entry_price: float,
                            stop_loss_price: Optional[float], entry_timestamp: pd.Timestamp):
        """
        Update or add position state for risk tracking.
        
        Args:
            symbol: Trading symbol
            position_size: Position size as fraction of portfolio
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price (can be None)
            entry_timestamp: When the position was entered
        """
        if abs(position_size) < 1e-10:  # Effectively zero with floating point tolerance
            # Remove position if size is effectively zero
            self.clear_position(symbol)
        else:
            self._positions[symbol] = PositionInfo(
                symbol=symbol,
                position_size=position_size,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                entry_timestamp=entry_timestamp
            )
            
    def clear_position(self, symbol: str):
        """
        Remove position from tracking when closed.
        
        Args:
            symbol: Trading symbol to remove
        """
        if symbol in self._positions:
            del self._positions[symbol]
            
    def get_position_info(self, symbol: str) -> Optional[PositionInfo]:
        """
        Get current position information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            PositionInfo if position exists, None otherwise
        """
        return self._positions.get(symbol)
        
    def get_total_exposure(self) -> float:
        """
        Calculate total portfolio exposure across all positions.
        
        Returns:
            Total exposure as fraction of portfolio
        """
        return sum(abs(pos.position_size) for pos in self._positions.values())
        
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get summary of current portfolio state.
        
        Returns:
            Dictionary with portfolio metrics
        """
        total_positions = len(self._positions)
        total_exposure = self.get_total_exposure()
        
        long_positions = sum(1 for pos in self._positions.values() if pos.position_size > 0)
        short_positions = total_positions - long_positions
        
        avg_position_size = total_exposure / total_positions if total_positions > 0 else 0.0
        
        return {
            'total_positions': total_positions,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'total_exposure': total_exposure,
            'avg_position_size': avg_position_size,
            'symbols': list(self._positions.keys())
        }
        
    def evaluate_trade(self, signal: int, current_price: float, 
                      portfolio_value: float, historical_data: pd.DataFrame,
                      current_position: float = 0.0) -> RiskDecision:
        """
        Comprehensive risk evaluation for a potential trade.
        
        Args:
            signal: Trading signal (1 for buy, -1 for sell, 0 for hold)
            current_price: Current asset price
            portfolio_value: Current portfolio value
            historical_data: Historical price and performance data
            current_position: Current position size
            
        Returns:
            RiskDecision object with position sizing and risk parameters
        """
        if signal == 0:
            return RiskDecision(
                position_size=0.0,
                stop_loss_price=None,
                max_risk_per_trade=0.0,
                allow_trade=False,
                reason="No signal"
            )
            
        # Calculate position size
        position_size = self.calculate_position_size(
            signal, current_price, portfolio_value, historical_data, current_position
        )
        
        # Calculate stop loss
        stop_loss_price = self.calculate_stop_loss(
            current_price, signal, historical_data
        )
        
        # Calculate maximum risk per trade
        if stop_loss_price is not None:
            if signal == 1:  # Long position
                risk_per_share = abs(current_price - stop_loss_price)
            else:  # Short position  
                risk_per_share = abs(stop_loss_price - current_price)
            
            max_risk_per_trade = (risk_per_share / current_price) * position_size
        else:
            max_risk_per_trade = position_size  # Full position at risk if no stop
            
        # Portfolio-level risk checks
        allow_trade = self._portfolio_risk_check(position_size, max_risk_per_trade, 
                                               portfolio_value)
        
        # Adjust position size and reason if trade is blocked
        if not allow_trade:
            position_size = 0.0
            stop_loss_price = None
            reason = "Portfolio risk check failed"
        else:
            reason = "Risk evaluation completed"
        
        return RiskDecision(
            position_size=position_size,
            stop_loss_price=stop_loss_price,
            max_risk_per_trade=max_risk_per_trade,
            allow_trade=allow_trade,
            reason=reason
        )
        
    def _portfolio_risk_check(self, position_size: float, max_risk: float,
                            portfolio_value: float) -> bool:
        """
        Check portfolio-level risk constraints using position tracking.
        
        Args:
            position_size: Proposed position size
            max_risk: Maximum risk per trade
            portfolio_value: Current portfolio value
            
        Returns:
            True if trade passes portfolio risk checks
        """
        # Maximum position size check
        max_position = self.config.get('max_position_size', 1.0)
        if position_size > max_position:
            return False
            
        # Maximum risk per trade check
        max_risk_per_trade = self.config.get('max_risk_per_trade', 0.1)
        if max_risk > max_risk_per_trade:
            return False
            
        # Portfolio exposure check
        current_exposure = self.get_total_exposure()
        max_total_exposure = self.config.get('max_total_exposure', 2.0)  # Allow some leverage
        if current_exposure + position_size > max_total_exposure:
            return False
            
        # Maximum positions check
        max_positions = self.config.get('max_positions', 10)
        if len(self._positions) >= max_positions:
            return False
            
        return True
        
    @abstractmethod
    def _validate_config(self):
        """Validate configuration parameters for the specific risk manager."""
        pass
        
    def validate_config_relationships(self):
        """
        Validate that configuration parameters work well together.
        This method checks for potentially problematic parameter combinations.
        """
        # Get common config values with defaults
        max_position = self.config.get('max_position_size', 1.0)
        max_risk_per_trade = self.config.get('max_risk_per_trade', 0.1)
        
        # Basic sanity checks
        if max_position <= 0 or max_position > 1.0:
            raise ValueError("max_position_size must be between 0 and 1")
            
        if max_risk_per_trade <= 0 or max_risk_per_trade > 1.0:
            raise ValueError("max_risk_per_trade must be between 0 and 1")
            
        # Check if max_risk_per_trade is reasonable relative to max_position_size
        if max_risk_per_trade > max_position:
            raise ValueError(
                f"max_risk_per_trade ({max_risk_per_trade}) cannot exceed "
                f"max_position_size ({max_position})"
            )
            
        # Subclass-specific validation (can be overridden)
        self._validate_subclass_relationships()
        
    def _validate_subclass_relationships(self):
        """
        Validate subclass-specific parameter relationships.
        Override in subclasses for custom validation logic.
        """
        pass
        
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk management metrics and settings.
        
        Returns:
            Dictionary containing risk metrics and configuration
        """
        portfolio_summary = self.get_portfolio_summary()
        
        return {
            'config': self.config.copy(),
            'type': self.__class__.__name__,
            'portfolio_summary': portfolio_summary,
            'current_exposure': self.get_total_exposure(),
            'positions_tracked': len(self._positions)
        }