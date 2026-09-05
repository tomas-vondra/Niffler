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

from niffler.risk.base_risk_manager import BaseRiskManager, RiskDecision
from niffler.risk.contract import PortfolioSnapshot


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


class TestPortfolioSnapshot(unittest.TestCase):

    def test_snapshot_is_frozen(self):
        """A manager must not be able to use the snapshot as scratch storage."""
        snapshot = PortfolioSnapshot(open_positions=1, total_exposure=0.3,
                                     current_position=0.3)

        with self.assertRaises(Exception):
            snapshot.open_positions = 2

    def test_flat_snapshot_holds_nothing(self):
        """Test the explicit empty-portfolio constructor."""
        snapshot = PortfolioSnapshot.flat()

        self.assertEqual(snapshot.open_positions, 0)
        self.assertEqual(snapshot.total_exposure, 0.0)
        self.assertEqual(snapshot.current_position, 0.0)


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

    def test_initialization_invalid_config(self):
        """Test BaseRiskManager initialization with invalid config."""
        invalid_config = {'max_position_size': 1.5}  # > 1.0

        with self.assertRaises(ValueError):
            ConcreteRiskManager(invalid_config)

    def test_manager_holds_no_position_state(self):
        """A manager must expose nowhere to accumulate one run's positions."""
        for attribute in ('_positions', 'update_position_state', 'clear_position',
                          'get_position_info', 'get_total_exposure',
                          'get_portfolio_summary'):
            self.assertFalse(
                hasattr(self.risk_manager, attribute),
                f"{attribute} is back - position state belongs to Portfolio"
            )

    def test_evaluate_trade_hold_signal(self):
        """Test evaluate_trade with hold signal."""
        decision = self.risk_manager.evaluate_trade(
            signal=0,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            portfolio=PortfolioSnapshot.flat()
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
            historical_data=self.sample_data,
            portfolio=PortfolioSnapshot.flat()
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
            portfolio=PortfolioSnapshot(open_positions=1, total_exposure=0.05,
                                        current_position=0.05)
        )

        self.assertTrue(decision.allow_trade)
        self.assertEqual(decision.position_size, 0.05)  # Close current position
        self.assertEqual(decision.stop_loss_price, 105.0)  # 100 * 1.05

    def test_evaluate_trade_requires_a_snapshot(self):
        """The snapshot has no default - a flat one would disable max_positions."""
        with self.assertRaises(TypeError):
            self.risk_manager.evaluate_trade(
                signal=1,
                current_price=100.0,
                portfolio_value=10000.0,
                historical_data=self.sample_data
            )

    def test_evaluate_trade_blocked_at_the_position_limit(self):
        """A snapshot at max_positions blocks the trade end to end."""
        at_limit = PortfolioSnapshot(open_positions=5, total_exposure=0.5,
                                     current_position=0.0)

        decision = self.risk_manager.evaluate_trade(
            signal=1,
            current_price=100.0,
            portfolio_value=10000.0,
            historical_data=self.sample_data,
            portfolio=at_limit
        )

        self.assertFalse(decision.allow_trade)
        self.assertEqual(decision.position_size, 0.0)
        self.assertEqual(decision.reason, "Portfolio risk check failed")

    def test_evaluate_trade_does_not_remember_the_previous_call(self):
        """Two calls in sequence are independent - the first cannot veto the second."""
        at_limit = PortfolioSnapshot(open_positions=5, total_exposure=0.5,
                                     current_position=0.0)

        blocked = self.risk_manager.evaluate_trade(
            signal=1, current_price=100.0, portfolio_value=10000.0,
            historical_data=self.sample_data, portfolio=at_limit
        )
        allowed = self.risk_manager.evaluate_trade(
            signal=1, current_price=100.0, portfolio_value=10000.0,
            historical_data=self.sample_data, portfolio=PortfolioSnapshot.flat()
        )

        self.assertFalse(blocked.allow_trade)
        self.assertTrue(allowed.allow_trade)

    def test_portfolio_risk_check_valid(self):
        """Test portfolio risk check with valid parameters."""
        result = self.risk_manager._portfolio_risk_check(
            position_size=0.15,
            max_risk=0.03,
            portfolio_value=10000.0,
            portfolio=PortfolioSnapshot.flat()
        )

        self.assertTrue(result)

    def test_portfolio_risk_check_position_too_large(self):
        """Test portfolio risk check with position size too large."""
        result = self.risk_manager._portfolio_risk_check(
            position_size=0.25,  # > max_position_size (0.2)
            max_risk=0.03,
            portfolio_value=10000.0,
            portfolio=PortfolioSnapshot.flat()
        )

        self.assertFalse(result)

    def test_portfolio_risk_check_risk_too_high(self):
        """Test portfolio risk check with risk too high."""
        result = self.risk_manager._portfolio_risk_check(
            position_size=0.15,
            max_risk=0.06,  # > max_risk_per_trade (0.05)
            portfolio_value=10000.0,
            portfolio=PortfolioSnapshot.flat()
        )

        self.assertFalse(result)

    def test_portfolio_risk_check_exposure_too_high(self):
        """Test portfolio risk check with total exposure too high."""
        crowded = PortfolioSnapshot(open_positions=2, total_exposure=1.4,
                                    current_position=0.0)

        result = self.risk_manager._portfolio_risk_check(
            position_size=0.2,  # Would push total over max_total_exposure (1.5)
            max_risk=0.03,
            portfolio_value=10000.0,
            portfolio=crowded
        )

        self.assertFalse(result)

    def test_portfolio_risk_check_too_many_positions(self):
        """Test portfolio risk check with too many positions."""
        at_limit = PortfolioSnapshot(open_positions=5,  # max_positions = 5
                                     total_exposure=0.5, current_position=0.0)

        result = self.risk_manager._portfolio_risk_check(
            position_size=0.1,
            max_risk=0.01,
            portfolio_value=10000.0,
            portfolio=at_limit
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
        """Metrics report configuration only - there is no live state to report."""
        metrics = self.risk_manager.get_risk_metrics()

        self.assertIn('config', metrics)
        self.assertIn('type', metrics)
        self.assertEqual(metrics['type'], 'ConcreteRiskManager')
        self.assertEqual(metrics['config'], self.config)
        self.assertNotIn('positions_tracked', metrics)
        self.assertNotIn('current_exposure', metrics)


if __name__ == '__main__':
    unittest.main()
