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

from niffler.risk.kelly_risk_manager import KellyRiskManager
from niffler.risk.base_risk_manager import PositionInfo, RiskDecision


class TestKellyRiskManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'max_kelly_fraction': 0.25,
            'max_risk_per_trade': 0.05,
            'max_positions': 5,
            'max_total_exposure': 1.0
        }
        self.risk_manager = KellyRiskManager(
            max_kelly_fraction=0.25,
            max_risk_per_trade=0.05
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
        """Test KellyRiskManager initialization with default values."""
        manager = KellyRiskManager()
        
        self.assertEqual(manager.config['max_kelly_fraction'], 0.25)
        self.assertEqual(manager.config['max_risk_per_trade'], 0.05)
        self.assertEqual(manager.config['max_positions'], 5)
        self.assertEqual(manager.config['max_total_exposure'], 1.25)  # 5 * 0.25
        
    def test_initialization_custom_values(self):
        """Test KellyRiskManager initialization with custom values."""
        manager = KellyRiskManager(
            max_kelly_fraction=0.5,
            max_risk_per_trade=0.03,
            max_positions=3
        )
        
        self.assertEqual(manager.config['max_kelly_fraction'], 0.5)
        self.assertEqual(manager.config['max_risk_per_trade'], 0.03)
        self.assertEqual(manager.config['max_positions'], 3)
        self.assertEqual(manager.config['max_total_exposure'], 1.5)  # 3 * 0.5
        
    def test_initialization_invalid_kelly_fraction(self):
        """Test initialization with invalid Kelly fraction."""
        with self.assertRaises(ValueError):
            KellyRiskManager(max_kelly_fraction=0.0)
            
        with self.assertRaises(ValueError):
            KellyRiskManager(max_kelly_fraction=1.5)
            
    def test_initialization_invalid_max_position_size(self):
        """Test initialization with invalid max position size."""
        with self.assertRaises(ValueError):
            KellyRiskManager(max_kelly_fraction=0.0)
            
        with self.assertRaises(ValueError):
            KellyRiskManager(max_kelly_fraction=1.5)
            
    def test_calculate_position_size_not_implemented(self):
        """Test that calculate_position_size raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as context:
            self.risk_manager.calculate_position_size(
                signal=1,
                current_price=100.0,
                portfolio_value=10000.0,
                historical_data=self.sample_data
            )
        
        self.assertIn("Kelly position sizing not yet implemented", str(context.exception))
        
    def test_calculate_stop_loss_not_implemented(self):
        """Test that calculate_stop_loss raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as context:
            self.risk_manager.calculate_stop_loss(
                entry_price=100.0,
                signal=1,
                historical_data=self.sample_data
            )
        
        self.assertIn("Kelly stop loss calculation not yet implemented", str(context.exception))
        
    def test_should_close_position_not_implemented(self):
        """Test that should_close_position raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as context:
            self.risk_manager.should_close_position(
                current_price=95.0,
                entry_price=100.0,
                stop_loss_price=None,
                signal=1,
                unrealized_pnl=-50.0
            )
        
        self.assertIn("Kelly position closing logic not yet implemented", str(context.exception))
        
    def test_get_risk_metrics(self):
        """Test getting risk metrics."""
        metrics = self.risk_manager.get_risk_metrics()
        
        self.assertIn('risk_management_type', metrics)
        self.assertIn('max_kelly_fraction', metrics)
        self.assertIn('max_risk_per_trade', metrics)
        self.assertIn('max_positions', metrics)
        self.assertIn('current_exposure', metrics)
        self.assertIn('positions_tracked', metrics)
        
        self.assertEqual(metrics['risk_management_type'], 'Kelly Criterion (Not Implemented)')
        self.assertEqual(metrics['max_kelly_fraction'], 0.25)
        self.assertEqual(metrics['max_risk_per_trade'], 0.05)
        self.assertEqual(metrics['max_positions'], 5)
        self.assertEqual(metrics['max_total_exposure'], 1.25)  # 5 * 0.25
        self.assertEqual(metrics['current_exposure'], 0.0)  # No positions initially
        self.assertEqual(metrics['positions_tracked'], 0)  # No positions initially
        
    def test_get_risk_metrics_with_positions(self):
        """Test getting risk metrics with tracked positions."""
        # Add a position to test metrics with state
        timestamp = pd.Timestamp('2024-01-01')
        self.risk_manager.update_position_state(
            symbol="BTC",
            position_size=0.15,
            entry_price=50000.0,
            stop_loss_price=None,
            entry_timestamp=timestamp
        )
        
        metrics = self.risk_manager.get_risk_metrics()
        
        self.assertEqual(metrics['current_exposure'], 0.15)
        self.assertEqual(metrics['positions_tracked'], 1)
        
    def test_evaluate_trade_not_implemented_for_signals(self):
        """Test that evaluate_trade raises NotImplementedError for non-zero signals."""
        # Hold signal should work (inherited from base class)
        decision = self.risk_manager.evaluate_trade(
            signal=0,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data
        )
        self.assertFalse(decision.allow_trade)
        self.assertEqual(decision.reason, "No signal")
        
        # Buy signal should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.risk_manager.evaluate_trade(
                signal=1,
                current_price=100.0,
                portfolio_value=10000.0,
                historical_data=self.sample_data
            )
        
        # Sell signal should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.risk_manager.evaluate_trade(
                signal=-1,
                current_price=100.0,
                portfolio_value=10000.0,
                historical_data=self.sample_data,
                current_position=0.1
            )
            
    def test_config_validation_relationships(self):
        """Test configuration relationship validation."""
        # Valid config should not raise
        try:
            self.risk_manager.validate_config_relationships()
        except ValueError:
            self.fail("validate_config_relationships raised ValueError with valid config")
            
    def test_config_validation_invalid_kelly_fraction_vs_max_position(self):
        """Test config validation when Kelly fraction is too high."""
        with self.assertRaises(ValueError):
            KellyRiskManager(
                max_kelly_fraction=1.5  # > 1.0 limit
            )
            
    def test_str_representation(self):
        """Test string representation of KellyRiskManager."""
        str_repr = str(self.risk_manager)
        
        self.assertIn("KellyRiskManager", str_repr)
        self.assertIn("max_kelly_fraction=0.25", str_repr)
        self.assertIn("NOT IMPLEMENTED", str_repr)
        
    def test_repr_representation(self):
        """Test repr representation of KellyRiskManager."""
        repr_str = repr(self.risk_manager)
        
        self.assertIn("KellyRiskManager", repr_str)
        self.assertIn("max_kelly_fraction=0.25", repr_str)
        
    def test_inheritance_from_base_risk_manager(self):
        """Test that KellyRiskManager inherits base functionality."""
        # Should have all base class methods and attributes
        self.assertTrue(hasattr(self.risk_manager, 'config'))
        self.assertTrue(hasattr(self.risk_manager, '_positions'))
        self.assertTrue(hasattr(self.risk_manager, 'update_position_state'))
        self.assertTrue(hasattr(self.risk_manager, 'get_position_info'))
        self.assertTrue(hasattr(self.risk_manager, 'clear_position'))
        self.assertTrue(hasattr(self.risk_manager, 'get_total_exposure'))
        self.assertTrue(hasattr(self.risk_manager, 'get_portfolio_summary'))
        self.assertTrue(hasattr(self.risk_manager, '_portfolio_risk_check'))
        
    def test_position_state_management_inheritance(self):
        """Test that position state management works through inheritance."""
        timestamp = pd.Timestamp('2024-01-01')
        
        # Should be able to update position state
        self.risk_manager.update_position_state(
            symbol="ETH",
            position_size=0.08,
            entry_price=3000.0,
            stop_loss_price=2850.0,
            entry_timestamp=timestamp
        )
        
        # Should be able to get position info
        position = self.risk_manager.get_position_info("ETH")
        self.assertIsNotNone(position)
        self.assertEqual(position.symbol, "ETH")
        self.assertEqual(position.position_size, 0.08)
        self.assertEqual(position.entry_price, 3000.0)
        self.assertEqual(position.stop_loss_price, 2850.0)
        
        # Should be able to clear position
        self.risk_manager.clear_position("ETH")
        position = self.risk_manager.get_position_info("ETH")
        self.assertIsNone(position)
        
    def test_portfolio_summary_inheritance(self):
        """Test that portfolio summary works through inheritance."""
        timestamp = pd.Timestamp('2024-01-01')
        
        # Add some positions
        self.risk_manager.update_position_state("BTC", 0.1, 50000.0, None, timestamp)
        self.risk_manager.update_position_state("ETH", -0.05, 3000.0, None, timestamp)
        
        summary = self.risk_manager.get_portfolio_summary()
        
        self.assertEqual(summary['total_positions'], 2)
        self.assertEqual(summary['long_positions'], 1)
        self.assertEqual(summary['short_positions'], 1)
        self.assertAlmostEqual(summary['total_exposure'], 0.15, places=6)  # |0.1| + |-0.05|
        self.assertEqual(set(summary['symbols']), {'BTC', 'ETH'})


if __name__ == '__main__':
    unittest.main()