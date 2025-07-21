import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.risk.base_risk_manager import BaseRiskManager, PositionInfo, RiskDecision


class ConcreteRiskManager(BaseRiskManager):
    """Concrete implementation of BaseRiskManager for testing."""
    
    def calculate_position_size(self, signal: int, current_price: float, 
                              portfolio_value: float, historical_data: pd.DataFrame,
                              current_position: float = 0.0) -> float:
        # Simple implementation for testing
        if signal == 0:
            return 0.0
        elif signal == -1:
            return abs(current_position)  # Close position
        else:
            return 0.1  # 10% position size
    
    def calculate_stop_loss(self, entry_price: float, signal: int,
                          historical_data: pd.DataFrame):
        # Simple 5% stop loss for testing
        if signal == 1:
            return entry_price * 0.95
        elif signal == -1:
            return entry_price * 1.05
        return None
        
    def should_close_position(self, current_price: float, entry_price: float,
                            stop_loss_price, signal: int, unrealized_pnl: float):
        if stop_loss_price is None:
            return False, "No stop loss"
        
        if signal == 1 and current_price <= stop_loss_price:
            return True, "Stop loss hit"
        elif signal == -1 and current_price >= stop_loss_price:
            return True, "Stop loss hit"
        return False, "Stop loss not triggered"
        
    def _validate_config(self):
        # Simple validation for testing
        max_position = self.config.get('max_position_size', 1.0)
        if max_position <= 0 or max_position > 1.0:
            raise ValueError("max_position_size must be between 0 and 1")


class TestPositionInfo(unittest.TestCase):
    
    def test_position_info_creation(self):
        """Test PositionInfo dataclass creation."""
        timestamp = pd.Timestamp('2024-01-01')
        position = PositionInfo(
            symbol="BTC",
            position_size=0.1,
            entry_price=50000.0,
            stop_loss_price=47500.0,
            entry_timestamp=timestamp,
            unrealized_pnl=100.0
        )
        
        self.assertEqual(position.symbol, "BTC")
        self.assertEqual(position.position_size, 0.1)
        self.assertEqual(position.entry_price, 50000.0)
        self.assertEqual(position.stop_loss_price, 47500.0)
        self.assertEqual(position.entry_timestamp, timestamp)
        self.assertEqual(position.unrealized_pnl, 100.0)
        
    def test_position_info_defaults(self):
        """Test PositionInfo with default unrealized_pnl."""
        timestamp = pd.Timestamp('2024-01-01')
        position = PositionInfo(
            symbol="ETH",
            position_size=0.2,
            entry_price=3000.0,
            stop_loss_price=None,
            entry_timestamp=timestamp
        )
        
        self.assertEqual(position.unrealized_pnl, 0.0)
        self.assertIsNone(position.stop_loss_price)


class TestRiskDecision(unittest.TestCase):
    
    def test_risk_decision_creation(self):
        """Test RiskDecision dataclass creation."""
        decision = RiskDecision(
            position_size=0.15,
            stop_loss_price=95.0,
            max_risk_per_trade=0.025,
            allow_trade=True,
            reason="Risk evaluation passed"
        )
        
        self.assertEqual(decision.position_size, 0.15)
        self.assertEqual(decision.stop_loss_price, 95.0)
        self.assertEqual(decision.max_risk_per_trade, 0.025)
        self.assertTrue(decision.allow_trade)
        self.assertEqual(decision.reason, "Risk evaluation passed")
        
    def test_risk_decision_defaults(self):
        """Test RiskDecision with default values."""
        decision = RiskDecision(
            position_size=0.1,
            stop_loss_price=None,
            max_risk_per_trade=0.02
        )
        
        self.assertTrue(decision.allow_trade)  # Default True
        self.assertEqual(decision.reason, "")  # Default empty string


class TestBaseRiskManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'max_position_size': 0.2,
            'max_risk_per_trade': 0.05,
            'max_total_exposure': 1.5,
            'max_positions': 5
        }
        self.risk_manager = ConcreteRiskManager(self.config)
        
        # Sample historical data
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        self.sample_data = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0, 103.0, 101.0, 104.0, 105.0, 103.0, 106.0, 107.0, 104.0, 108.0],
            'volume': [1000] * 10
        }, index=dates)
    
    def test_initialization(self):
        """Test BaseRiskManager initialization."""
        self.assertEqual(self.risk_manager.config, self.config)
        self.assertEqual(len(self.risk_manager._positions), 0)
        
    def test_initialization_invalid_config(self):
        """Test BaseRiskManager initialization with invalid config."""
        invalid_config = {'max_position_size': 1.5}  # > 1.0
        
        with self.assertRaises(ValueError):
            ConcreteRiskManager(invalid_config)
            
    def test_update_position_state(self):
        """Test updating position state."""
        timestamp = pd.Timestamp('2024-01-01')
        
        self.risk_manager.update_position_state(
            symbol="BTC",
            position_size=0.1,
            entry_price=50000.0,
            stop_loss_price=47500.0,
            entry_timestamp=timestamp
        )
        
        position = self.risk_manager.get_position_info("BTC")
        self.assertIsNotNone(position)
        self.assertEqual(position.symbol, "BTC")
        self.assertEqual(position.position_size, 0.1)
        self.assertEqual(position.entry_price, 50000.0)
        self.assertEqual(position.stop_loss_price, 47500.0)
        self.assertEqual(position.entry_timestamp, timestamp)
        
    def test_update_position_state_zero_size(self):
        """Test updating position with zero size removes it."""
        timestamp = pd.Timestamp('2024-01-01')
        
        # Add position first
        self.risk_manager.update_position_state("BTC", 0.1, 50000.0, 47500.0, timestamp)
        self.assertIsNotNone(self.risk_manager.get_position_info("BTC"))
        
        # Update with zero size should remove it
        self.risk_manager.update_position_state("BTC", 0.0, 50000.0, None, timestamp)
        self.assertIsNone(self.risk_manager.get_position_info("BTC"))
        
    def test_clear_position(self):
        """Test clearing a position."""
        timestamp = pd.Timestamp('2024-01-01')
        
        # Add position
        self.risk_manager.update_position_state("BTC", 0.1, 50000.0, 47500.0, timestamp)
        self.assertIsNotNone(self.risk_manager.get_position_info("BTC"))
        
        # Clear position
        self.risk_manager.clear_position("BTC")
        self.assertIsNone(self.risk_manager.get_position_info("BTC"))
        
    def test_clear_nonexistent_position(self):
        """Test clearing a position that doesn't exist."""
        # Should not raise an error
        self.risk_manager.clear_position("NONEXISTENT")
        
    def test_get_position_info_nonexistent(self):
        """Test getting info for nonexistent position."""
        position = self.risk_manager.get_position_info("NONEXISTENT")
        self.assertIsNone(position)
        
    def test_get_total_exposure_empty(self):
        """Test total exposure with no positions."""
        exposure = self.risk_manager.get_total_exposure()
        self.assertEqual(exposure, 0.0)
        
    def test_get_total_exposure_with_positions(self):
        """Test total exposure with multiple positions."""
        timestamp = pd.Timestamp('2024-01-01')
        
        self.risk_manager.update_position_state("BTC", 0.1, 50000.0, None, timestamp)
        self.risk_manager.update_position_state("ETH", 0.15, 3000.0, None, timestamp)
        
        exposure = self.risk_manager.get_total_exposure()
        self.assertEqual(exposure, 0.25)  # 0.1 + 0.15
        
    def test_get_total_exposure_with_short_positions(self):
        """Test total exposure includes absolute values of short positions."""
        timestamp = pd.Timestamp('2024-01-01')
        
        self.risk_manager.update_position_state("BTC", 0.1, 50000.0, None, timestamp)
        self.risk_manager.update_position_state("ETH", -0.15, 3000.0, None, timestamp)
        
        exposure = self.risk_manager.get_total_exposure()
        self.assertEqual(exposure, 0.25)  # |0.1| + |-0.15|
        
    def test_get_portfolio_summary_empty(self):
        """Test portfolio summary with no positions."""
        summary = self.risk_manager.get_portfolio_summary()
        
        expected = {
            'total_positions': 0,
            'long_positions': 0,
            'short_positions': 0,
            'total_exposure': 0.0,
            'avg_position_size': 0.0,
            'symbols': []
        }
        self.assertEqual(summary, expected)
        
    def test_get_portfolio_summary_with_positions(self):
        """Test portfolio summary with multiple positions."""
        timestamp = pd.Timestamp('2024-01-01')
        
        self.risk_manager.update_position_state("BTC", 0.1, 50000.0, None, timestamp)
        self.risk_manager.update_position_state("ETH", -0.2, 3000.0, None, timestamp)
        self.risk_manager.update_position_state("ADA", 0.05, 1.0, None, timestamp)
        
        summary = self.risk_manager.get_portfolio_summary()
        
        self.assertEqual(summary['total_positions'], 3)
        self.assertEqual(summary['long_positions'], 2)  # BTC, ADA
        self.assertEqual(summary['short_positions'], 1)  # ETH
        self.assertAlmostEqual(summary['total_exposure'], 0.35, places=6)  # |0.1| + |-0.2| + |0.05|
        self.assertAlmostEqual(summary['avg_position_size'], 0.35/3, places=6)
        self.assertEqual(set(summary['symbols']), {'BTC', 'ETH', 'ADA'})
        
    def test_evaluate_trade_hold_signal(self):
        """Test evaluate_trade with hold signal."""
        decision = self.risk_manager.evaluate_trade(
            signal=0,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data
        )
        
        self.assertFalse(decision.allow_trade)
        self.assertEqual(decision.position_size, 0.0)
        self.assertIsNone(decision.stop_loss_price)
        self.assertEqual(decision.reason, "No signal")
        
    def test_evaluate_trade_buy_signal(self):
        """Test evaluate_trade with buy signal."""
        decision = self.risk_manager.evaluate_trade(
            signal=1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data
        )
        
        self.assertTrue(decision.allow_trade)
        self.assertEqual(decision.position_size, 0.1)  # From ConcreteRiskManager
        self.assertEqual(decision.stop_loss_price, 95.0)  # 100 * 0.95
        self.assertAlmostEqual(decision.max_risk_per_trade, 0.005, places=6)  # (100-95)/100 * 0.1
        
    def test_evaluate_trade_sell_signal(self):
        """Test evaluate_trade with sell signal."""
        decision = self.risk_manager.evaluate_trade(
            signal=-1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            current_position=0.05
        )
        
        self.assertTrue(decision.allow_trade)
        self.assertEqual(decision.position_size, 0.05)  # Close current position
        self.assertEqual(decision.stop_loss_price, 105.0)  # 100 * 1.05
        
    def test_portfolio_risk_check_valid(self):
        """Test portfolio risk check with valid parameters."""
        result = self.risk_manager._portfolio_risk_check(
            position_size=0.15,
            max_risk=0.03,
            portfolio_value=10000.0
        )
        
        self.assertTrue(result)
        
    def test_portfolio_risk_check_position_too_large(self):
        """Test portfolio risk check with position size too large."""
        result = self.risk_manager._portfolio_risk_check(
            position_size=0.25,  # > max_position_size (0.2)
            max_risk=0.03,
            portfolio_value=10000.0
        )
        
        self.assertFalse(result)
        
    def test_portfolio_risk_check_risk_too_high(self):
        """Test portfolio risk check with risk too high."""
        result = self.risk_manager._portfolio_risk_check(
            position_size=0.15,
            max_risk=0.06,  # > max_risk_per_trade (0.05)
            portfolio_value=10000.0
        )
        
        self.assertFalse(result)
        
    def test_portfolio_risk_check_exposure_too_high(self):
        """Test portfolio risk check with total exposure too high."""
        # Add existing positions to push exposure high
        timestamp = pd.Timestamp('2024-01-01')
        self.risk_manager.update_position_state("BTC", 1.0, 50000.0, None, timestamp)
        self.risk_manager.update_position_state("ETH", 0.4, 3000.0, None, timestamp)
        
        result = self.risk_manager._portfolio_risk_check(
            position_size=0.2,  # Would push total over max_total_exposure (1.5)
            max_risk=0.03,
            portfolio_value=10000.0
        )
        
        self.assertFalse(result)
        
    def test_portfolio_risk_check_too_many_positions(self):
        """Test portfolio risk check with too many positions."""
        # Add maximum positions
        timestamp = pd.Timestamp('2024-01-01')
        for i in range(5):  # max_positions = 5
            self.risk_manager.update_position_state(f"COIN{i}", 0.1, 100.0, None, timestamp)
            
        result = self.risk_manager._portfolio_risk_check(
            position_size=0.1,
            max_risk=0.01,
            portfolio_value=10000.0
        )
        
        self.assertFalse(result)
        
    def test_validate_config_relationships(self):
        """Test config relationship validation."""
        # Should pass with valid config
        try:
            self.risk_manager.validate_config_relationships()
        except ValueError:
            self.fail("validate_config_relationships raised ValueError with valid config")
            
    def test_validate_config_relationships_invalid_max_position(self):
        """Test config validation with invalid max position size."""
        config = {'max_position_size': 1.5}  # > 1.0
        
        with self.assertRaises(ValueError):
            ConcreteRiskManager(config)
            
    def test_validate_config_relationships_invalid_max_risk(self):
        """Test config validation with invalid max risk."""
        config = {
            'max_position_size': 0.2,
            'max_risk_per_trade': 1.5  # > 1.0
        }
        
        with self.assertRaises(ValueError):
            ConcreteRiskManager(config)
            
    def test_validate_config_relationships_risk_exceeds_position(self):
        """Test config validation when risk exceeds position size."""
        config = {
            'max_position_size': 0.1,
            'max_risk_per_trade': 0.2  # > max_position_size
        }
        
        with self.assertRaises(ValueError):
            ConcreteRiskManager(config)
            
    def test_get_risk_metrics(self):
        """Test getting risk metrics."""
        # Add some positions
        timestamp = pd.Timestamp('2024-01-01')
        self.risk_manager.update_position_state("BTC", 0.1, 50000.0, None, timestamp)
        
        metrics = self.risk_manager.get_risk_metrics()
        
        self.assertIn('config', metrics)
        self.assertIn('type', metrics)
        self.assertIn('portfolio_summary', metrics)
        self.assertIn('current_exposure', metrics)
        self.assertIn('positions_tracked', metrics)
        
        self.assertEqual(metrics['type'], 'ConcreteRiskManager')
        self.assertEqual(metrics['current_exposure'], 0.1)
        self.assertEqual(metrics['positions_tracked'], 1)
        self.assertEqual(metrics['config'], self.config)


if __name__ == '__main__':
    unittest.main()