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

from niffler.risk.fixed_risk_manager import FixedRiskManager
from niffler.risk.base_risk_manager import PositionInfo, RiskDecision


class TestFixedRiskManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'position_size_pct': 0.1,
            'stop_loss_pct': 0.05,
            'max_positions': 3,
            'max_risk_per_trade': 0.03,
            'max_position_size': 0.1,
            'max_total_exposure': 0.3  # 3 positions * 0.1 each
        }
        self.risk_manager = FixedRiskManager(
            position_size_pct=0.1,
            stop_loss_pct=0.05,
            max_positions=3,
            max_risk_per_trade=0.03
        )
        
        # Sample historical data
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        self.sample_data = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0, 103.0, 101.0, 104.0, 105.0, 103.0, 106.0, 107.0, 104.0, 108.0],
            'volume': [1000] * 10
        }, index=dates)
        
    def test_initialization_default_values(self):
        """Test FixedRiskManager initialization with default values."""
        manager = FixedRiskManager()
        
        self.assertEqual(manager.config['position_size_pct'], 0.1)
        self.assertEqual(manager.config['stop_loss_pct'], 0.05)
        self.assertEqual(manager.config['max_positions'], 5)
        self.assertEqual(manager.config['max_risk_per_trade'], 0.02)
        self.assertEqual(manager.config['max_total_exposure'], 0.5)  # 5 * 0.1
        
    def test_initialization_custom_values(self):
        """Test FixedRiskManager initialization with custom values."""
        manager = FixedRiskManager(
            position_size_pct=0.15,
            stop_loss_pct=0.03,
            max_positions=2,
            max_risk_per_trade=0.025
        )
        
        self.assertEqual(manager.config['position_size_pct'], 0.15)
        self.assertEqual(manager.config['stop_loss_pct'], 0.03)
        self.assertEqual(manager.config['max_positions'], 2)
        self.assertEqual(manager.config['max_risk_per_trade'], 0.025)
        self.assertEqual(manager.config['max_total_exposure'], 0.3)  # 2 * 0.15
        
    def test_initialization_invalid_position_size(self):
        """Test initialization with invalid position size."""
        with self.assertRaises(ValueError):
            FixedRiskManager(position_size_pct=0.0)
            
        with self.assertRaises(ValueError):
            FixedRiskManager(position_size_pct=1.5)
            
    def test_initialization_invalid_stop_loss(self):
        """Test initialization with invalid stop loss."""
        with self.assertRaises(ValueError):
            FixedRiskManager(stop_loss_pct=-0.01)
            
        with self.assertRaises(ValueError):
            FixedRiskManager(stop_loss_pct=1.5)
            
    def test_initialization_invalid_max_positions(self):
        """Test initialization with invalid max positions."""
        with self.assertRaises(ValueError):
            FixedRiskManager(max_positions=0)
            
        with self.assertRaises(ValueError):
            FixedRiskManager(max_positions=-1)
            
    def test_calculate_position_size_hold_signal(self):
        """Test position size calculation for hold signal."""
        position_size = self.risk_manager.calculate_position_size(
            signal=0,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data
        )
        
        self.assertEqual(position_size, 0.0)
        
    def test_calculate_position_size_buy_signal(self):
        """Test position size calculation for buy signal."""
        position_size = self.risk_manager.calculate_position_size(
            signal=1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data
        )
        
        self.assertEqual(position_size, 0.1)  # position_size_pct
        
    def test_calculate_position_size_sell_signal_with_position(self):
        """Test position size calculation for sell signal with current position."""
        position_size = self.risk_manager.calculate_position_size(
            signal=-1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            current_position=0.15
        )
        
        self.assertEqual(position_size, 0.15)  # Close entire position
        
    def test_calculate_position_size_sell_signal_no_position(self):
        """Test position size calculation for sell signal with no position."""
        position_size = self.risk_manager.calculate_position_size(
            signal=-1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            current_position=0.0
        )
        
        self.assertEqual(position_size, 0.0)  # No position to close
        
    def test_calculate_position_size_sell_signal_short_position(self):
        """Test position size calculation for sell signal with short position."""
        position_size = self.risk_manager.calculate_position_size(
            signal=-1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            current_position=-0.12
        )
        
        self.assertEqual(position_size, 0.12)  # Absolute value of short position
        
    def test_calculate_stop_loss_buy_signal(self):
        """Test stop loss calculation for buy signal."""
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=100.0,
            signal=1,
            historical_data=self.sample_data
        )
        
        self.assertEqual(stop_loss, 95.0)  # 100 * (1 - 0.05)
        
    def test_calculate_stop_loss_sell_signal(self):
        """Test stop loss calculation for sell signal."""
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=100.0,
            signal=-1,
            historical_data=self.sample_data
        )
        
        self.assertEqual(stop_loss, 105.0)  # 100 * (1 + 0.05)
        
    def test_calculate_stop_loss_hold_signal(self):
        """Test stop loss calculation for hold signal."""
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=100.0,
            signal=0,
            historical_data=self.sample_data
        )
        
        self.assertIsNone(stop_loss)
        
    def test_should_close_position_buy_stop_loss_hit(self):
        """Test position closure when buy stop loss is hit."""
        should_close, reason = self.risk_manager.should_close_position(
            current_price=94.0,  # Below stop loss of 95.0
            entry_price=100.0,
            stop_loss_price=95.0,
            signal=1,
            unrealized_pnl=-100.0
        )
        
        self.assertTrue(should_close)
        self.assertIn("Stop loss", reason)
        
    def test_should_close_position_buy_stop_loss_not_hit(self):
        """Test position closure when buy stop loss is not hit."""
        should_close, reason = self.risk_manager.should_close_position(
            current_price=96.0,  # Above stop loss of 95.0
            entry_price=100.0,
            stop_loss_price=95.0,
            signal=1,
            unrealized_pnl=-50.0
        )
        
        self.assertFalse(should_close)
        self.assertEqual(reason, "Stop loss not triggered")
        
    def test_should_close_position_sell_stop_loss_hit(self):
        """Test position closure when sell stop loss is hit."""
        should_close, reason = self.risk_manager.should_close_position(
            current_price=106.0,  # Above stop loss of 105.0
            entry_price=100.0,
            stop_loss_price=105.0,
            signal=-1,
            unrealized_pnl=-100.0
        )
        
        self.assertTrue(should_close)
        self.assertIn("Stop loss", reason)
        
    def test_should_close_position_sell_stop_loss_not_hit(self):
        """Test position closure when sell stop loss is not hit."""
        should_close, reason = self.risk_manager.should_close_position(
            current_price=104.0,  # Below stop loss of 105.0
            entry_price=100.0,
            stop_loss_price=105.0,
            signal=-1,
            unrealized_pnl=50.0
        )
        
        self.assertFalse(should_close)
        self.assertEqual(reason, "Stop loss not triggered")
        
    def test_should_close_position_no_stop_loss(self):
        """Test position closure with no stop loss set."""
        should_close, reason = self.risk_manager.should_close_position(
            current_price=80.0,  # Much lower price
            entry_price=100.0,
            stop_loss_price=None,
            signal=1,
            unrealized_pnl=-200.0
        )
        
        self.assertFalse(should_close)
        self.assertEqual(reason, "No stop loss set")
        
    def test_get_risk_metrics(self):
        """Test getting risk metrics."""
        # Add a position to test with state
        timestamp = pd.Timestamp('2024-01-01')
        self.risk_manager.update_position_state(
            symbol="BTC",
            position_size=0.1,
            entry_price=50000.0,
            stop_loss_price=47500.0,
            entry_timestamp=timestamp
        )
        
        metrics = self.risk_manager.get_risk_metrics()
        
        self.assertIn('risk_management_type', metrics)
        self.assertIn('position_size_pct', metrics)
        self.assertIn('stop_loss_pct', metrics)
        self.assertIn('max_positions', metrics)
        self.assertIn('max_risk_per_trade', metrics)
        self.assertIn('max_total_exposure', metrics)
        self.assertIn('current_exposure', metrics)
        self.assertIn('positions_tracked', metrics)
        
        self.assertEqual(metrics['risk_management_type'], 'Fixed Risk Manager')
        self.assertEqual(metrics['position_size_pct'], 0.1)
        self.assertEqual(metrics['stop_loss_pct'], 0.05)
        self.assertEqual(metrics['max_positions'], 3)
        self.assertEqual(metrics['max_risk_per_trade'], 0.03)
        self.assertAlmostEqual(metrics['max_total_exposure'], 0.3, places=6)  # 3 * 0.1
        self.assertEqual(metrics['current_exposure'], 0.1)
        self.assertEqual(metrics['positions_tracked'], 1)
        
    def test_evaluate_trade_integration_buy_allowed(self):
        """Test full trade evaluation integration for allowed buy trade."""
        decision = self.risk_manager.evaluate_trade(
            signal=1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            current_position=0.0
        )
        
        self.assertTrue(decision.allow_trade)
        self.assertEqual(decision.position_size, 0.1)
        self.assertEqual(decision.stop_loss_price, 95.0)
        self.assertAlmostEqual(decision.max_risk_per_trade, 0.005, places=6)  # (100-95)/100 * 0.1
        self.assertEqual(decision.reason, "Risk evaluation completed")
        
    def test_evaluate_trade_integration_sell_allowed(self):
        """Test full trade evaluation integration for allowed sell trade."""
        decision = self.risk_manager.evaluate_trade(
            signal=-1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            current_position=0.05
        )
        
        self.assertTrue(decision.allow_trade)
        self.assertEqual(decision.position_size, 0.05)
        self.assertEqual(decision.stop_loss_price, 105.0)
        self.assertAlmostEqual(decision.max_risk_per_trade, 0.0025, places=6)  # (105-100)/100 * 0.05
        self.assertEqual(decision.reason, "Risk evaluation completed")
        
    def test_evaluate_trade_blocked_by_position_limit(self):
        """Test trade blocked by maximum position limit."""
        # Add maximum positions
        timestamp = pd.Timestamp('2024-01-01')
        for i in range(3):  # max_positions = 3
            self.risk_manager.update_position_state(
                symbol=f"COIN{i}",
                position_size=0.1,
                entry_price=100.0,
                stop_loss_price=95.0,
                entry_timestamp=timestamp
            )
        
        decision = self.risk_manager.evaluate_trade(
            signal=1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            current_position=0.0
        )
        
        self.assertFalse(decision.allow_trade)
        self.assertEqual(decision.position_size, 0.0)
        self.assertIsNone(decision.stop_loss_price)
        self.assertIn("Portfolio risk check failed", decision.reason)
        
    def test_evaluate_trade_blocked_by_exposure_limit(self):
        """Test trade blocked by total exposure limit."""
        # Add positions near exposure limit
        timestamp = pd.Timestamp('2024-01-01')
        self.risk_manager.update_position_state(
            symbol="BTC",
            position_size=0.25,  # Near max_total_exposure of 0.3
            entry_price=50000.0,
            stop_loss_price=47500.0,
            entry_timestamp=timestamp
        )
        
        decision = self.risk_manager.evaluate_trade(
            signal=1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            current_position=0.0
        )
        
        self.assertFalse(decision.allow_trade)
        self.assertEqual(decision.position_size, 0.0)
        self.assertIsNone(decision.stop_loss_price)
        self.assertIn("Portfolio risk check failed", decision.reason)
        
    def test_config_validation_relationships(self):
        """Test configuration relationship validation."""
        # Valid config should not raise
        try:
            self.risk_manager.validate_config_relationships()
        except ValueError:
            self.fail("validate_config_relationships raised ValueError with valid config")
            
    def test_config_validation_invalid_max_risk_vs_position(self):
        """Test config validation when max risk exceeds position size."""
        with self.assertRaises(ValueError):
            FixedRiskManager(
                position_size_pct=0.1,
                stop_loss_pct=0.05,
                max_risk_per_trade=0.2  # > position_size_pct
            )
            
    def test_config_validation_invalid_stop_loss_vs_max_risk(self):
        """Test config validation when stop loss risk exceeds max risk."""
        with self.assertRaises(ValueError):
            FixedRiskManager(
                position_size_pct=0.2,
                stop_loss_pct=0.3,  # Would create 0.06 risk (0.3 * 0.2)
                max_risk_per_trade=0.05  # Less than 0.06
            )
            
    def test_str_representation(self):
        """Test string representation of FixedRiskManager."""
        str_repr = str(self.risk_manager)
        
        self.assertIn("FixedRiskManager", str_repr)
        self.assertIn("position_size_pct=0.1", str_repr)
        self.assertIn("stop_loss_pct=0.05", str_repr)
        
    def test_repr_representation(self):
        """Test repr representation of FixedRiskManager."""
        repr_str = repr(self.risk_manager)
        
        self.assertIn("FixedRiskManager", repr_str)
        self.assertIn("position_size_pct=0.1", repr_str)
        self.assertIn("stop_loss_pct=0.05", repr_str)
        

if __name__ == '__main__':
    unittest.main()