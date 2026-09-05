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
from niffler.risk.base_risk_manager import RiskDecision
from niffler.risk.contract import PortfolioSnapshot


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

        self.assertEqual(metrics['risk_management_type'], 'Kelly Criterion (Not Implemented)')
        self.assertEqual(metrics['max_kelly_fraction'], 0.25)
        self.assertEqual(metrics['max_risk_per_trade'], 0.05)
        self.assertEqual(metrics['max_positions'], 5)
        self.assertEqual(metrics['max_total_exposure'], 1.25)  # 5 * 0.25
        self.assertNotIn('current_exposure', metrics)
        self.assertNotIn('positions_tracked', metrics)

    def test_evaluate_trade_not_implemented_for_signals(self):
        """Test that evaluate_trade raises NotImplementedError for non-zero signals."""
        # Hold signal should work (inherited from base class)
        decision = self.risk_manager.evaluate_trade(
            signal=0,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            portfolio=PortfolioSnapshot.flat()
        )
        self.assertFalse(decision.allow_trade)
        self.assertEqual(decision.reason, "No signal")

        # Buy signal should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.risk_manager.evaluate_trade(
                signal=1,
                current_price=100.0,
                portfolio_value=10000.0,
                historical_data=self.sample_data,
                portfolio=PortfolioSnapshot.flat()
            )

        # Sell signal should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.risk_manager.evaluate_trade(
                signal=-1,
                current_price=100.0,
                portfolio_value=10000.0,
                historical_data=self.sample_data,
                portfolio=PortfolioSnapshot(open_positions=1, total_exposure=0.1,
                                            current_position=0.1)
            )

    def test_kelly_is_not_registered(self):
        """A stub must not be selectable: --risk-manager kelly would crash mid-run."""
        from niffler.risk.registry import RISK_MANAGER_CLASSES, get_available_risk_managers

        self.assertNotIn('kelly', RISK_MANAGER_CLASSES)
        self.assertNotIn('kelly', get_available_risk_managers())

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
        self.assertTrue(hasattr(self.risk_manager, 'config'))
        self.assertTrue(hasattr(self.risk_manager, '_portfolio_risk_check'))
        self.assertTrue(hasattr(self.risk_manager, 'evaluate_trade'))

    def test_holds_no_position_state(self):
        """The stub inherits statelessness too, so it is fold-safe by construction."""
        for attribute in ('_positions', 'update_position_state', 'clear_position',
                          'get_position_info', 'get_total_exposure',
                          'get_portfolio_summary'):
            self.assertFalse(hasattr(self.risk_manager, attribute))


if __name__ == '__main__':
    unittest.main()