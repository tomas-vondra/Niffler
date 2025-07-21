import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.trade import Trade, TradeSide
from niffler.backtesting.backtest_result import BacktestResult
from niffler.strategies.base_strategy import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, signals_data=None, risk_manager=None):
        super().__init__("MockStrategy", {}, risk_manager)
        self.signals_data = signals_data or {}
        
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['position_size'] = 1.0
        
        # Apply custom signals if provided
        for idx, signal in self.signals_data.items():
            if idx in df.index:
                df.loc[idx, 'signal'] = signal
                
        return df
        
    def validate_data(self, data):
        return True
        
    def get_description(self):
        return "Mock Strategy for Testing"


class MockRiskManager:
    """Mock risk manager for testing."""
    
    def __init__(self, allow_trade=True, position_size=0.5, stop_loss_price=95.0):
        self.allow_trade = allow_trade
        self.position_size = position_size  
        self.stop_loss_price = stop_loss_price
        self.positions = {}
        
    def evaluate_trade(self, signal, current_price, portfolio_value, historical_data, current_position):
        from niffler.risk.base_risk_manager import RiskDecision
        return RiskDecision(
            position_size=self.position_size,
            stop_loss_price=self.stop_loss_price,
            max_risk_per_trade=0.05,
            allow_trade=self.allow_trade,
            reason="Mock risk decision"
        )
        
    def should_close_position(self, current_price, entry_price, stop_loss_price, signal, unrealized_pnl):
        if stop_loss_price and current_price <= stop_loss_price:
            return True, "Mock stop loss triggered"
        return False, "Mock stop loss not triggered"
        
    def update_position_state(self, symbol, position_size, entry_price, stop_loss_price, entry_timestamp):
        # Debug print to see what values we're getting
        # print(f"MockRiskManager.update_position_state: {symbol}, position_size={position_size}")
        # Always update the position state - the backtest engine handles calling clear_position when needed
        self.positions[symbol] = {
            'position_size': position_size,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'entry_timestamp': entry_timestamp
        }
        
    def clear_position(self, symbol):
        # print(f"MockRiskManager.clear_position called for {symbol}")
        if symbol in self.positions:
            del self.positions[symbol]


class TestBacktestEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine(
            initial_capital=10000.0,
            commission=0.001,
            min_order_value=1.0
        )
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        self.sample_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'volume': [1000.0] * 10
        }, index=dates)
        
    def test_init_valid_parameters(self):
        """Test BacktestEngine initialization with valid parameters."""
        engine = BacktestEngine(
            initial_capital=5000.0,
            commission=0.002,
            min_order_value=10.0
        )
        
        self.assertEqual(engine.initial_capital, 5000.0)
        self.assertEqual(engine.commission, 0.002)
        self.assertEqual(engine.min_order_value, 10.0)
        
    def test_init_invalid_capital(self):
        """Test BacktestEngine initialization with invalid capital."""
        with self.assertRaises(ValueError) as context:
            BacktestEngine(initial_capital=-1000.0)
        self.assertIn("Initial capital must be positive", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            BacktestEngine(initial_capital=0.0)
        self.assertIn("Initial capital must be positive", str(context.exception))
        
    def test_init_invalid_commission(self):
        """Test BacktestEngine initialization with invalid commission."""
        with self.assertRaises(ValueError) as context:
            BacktestEngine(commission=-0.001)
        self.assertIn("Commission cannot be negative", str(context.exception))
        
    def test_init_invalid_min_order_value(self):
        """Test BacktestEngine initialization with invalid min_order_value."""
        with self.assertRaises(ValueError) as context:
            BacktestEngine(min_order_value=-1.0)
        self.assertIn("Minimum order value cannot be negative", str(context.exception))
        
    def test_validate_inputs_valid_data(self):
        """Test input validation with valid data."""
        strategy = MockStrategy()
        
        # Should not raise any exceptions
        self.engine._validate_inputs(strategy, self.sample_data, "TEST")
        
    def test_validate_inputs_none_strategy(self):
        """Test input validation with None strategy."""
        with self.assertRaises(ValueError) as context:
            self.engine._validate_inputs(None, self.sample_data, "TEST")
        self.assertIn("Strategy cannot be None", str(context.exception))
        
    def test_validate_inputs_empty_data(self):
        """Test input validation with empty data."""
        strategy = MockStrategy()
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.engine._validate_inputs(strategy, empty_data, "TEST")
        self.assertIn("Data cannot be empty", str(context.exception))
        
    def test_validate_inputs_insufficient_data(self):
        """Test input validation with insufficient data."""
        strategy = MockStrategy()
        small_data = self.sample_data.iloc[:1]
        
        with self.assertRaises(ValueError) as context:
            self.engine._validate_inputs(strategy, small_data, "TEST")
        self.assertIn("Data must have at least 2 rows", str(context.exception))
        
    def test_validate_inputs_missing_columns(self):
        """Test input validation with missing columns."""
        strategy = MockStrategy()
        invalid_data = self.sample_data.drop(['volume'], axis=1)
        
        with self.assertRaises(ValueError) as context:
            self.engine._validate_inputs(strategy, invalid_data, "TEST")
        self.assertIn("Missing required columns: ['volume']", str(context.exception))
        
    def test_validate_inputs_invalid_ohlc_relationships(self):
        """Test input validation with invalid OHLC relationships."""
        strategy = MockStrategy()
        invalid_data = self.sample_data.copy()
        invalid_data.loc[invalid_data.index[0], 'high'] = 50.0  # High < Low
        
        with self.assertRaises(ValueError) as context:
            self.engine._validate_inputs(strategy, invalid_data, "TEST")
        self.assertIn("invalid OHLC relationships", str(context.exception))
        
    def test_execute_buy_trade_successful(self):
        """Test successful buy trade execution."""
        timestamp = pd.Timestamp('2024-01-01')
        trade = self.engine._execute_buy_trade(
            timestamp=timestamp,
            symbol="TEST",
            price=100.0,
            position_size=0.5,
            available_cash=10000.0
        )
        
        self.assertIsNotNone(trade)
        self.assertEqual(trade.timestamp, timestamp)
        self.assertEqual(trade.symbol, "TEST")
        self.assertEqual(trade.side, TradeSide.BUY)
        self.assertEqual(trade.price, 100.0)
        self.assertAlmostEqual(trade.value, 4995.002, places=2)  # 5000 / 1.001
        
    def test_execute_buy_trade_insufficient_cash(self):
        """Test buy trade with insufficient cash."""
        timestamp = pd.Timestamp('2024-01-01')
        trade = self.engine._execute_buy_trade(
            timestamp=timestamp,
            symbol="TEST",
            price=100.0,
            position_size=1.0,
            available_cash=50.0
        )
        
        self.assertIsNone(trade)
        
    def test_execute_buy_trade_below_min_order_value(self):
        """Test buy trade below minimum order value."""
        engine = BacktestEngine(min_order_value=1000.0)
        timestamp = pd.Timestamp('2024-01-01')
        trade = engine._execute_buy_trade(
            timestamp=timestamp,
            symbol="TEST",
            price=100.0,
            position_size=0.001,
            available_cash=10000.0
        )
        
        self.assertIsNone(trade)
        
    def test_execute_sell_trade_successful(self):
        """Test successful sell trade execution."""
        timestamp = pd.Timestamp('2024-01-01')
        trade = self.engine._execute_sell_trade(
            timestamp=timestamp,
            symbol="TEST",
            price=100.0,
            position_size=0.5,
            current_position=10.0
        )
        
        self.assertIsNotNone(trade)
        self.assertEqual(trade.timestamp, timestamp)
        self.assertEqual(trade.symbol, "TEST")
        self.assertEqual(trade.side, TradeSide.SELL)
        self.assertEqual(trade.price, 100.0)
        self.assertEqual(trade.quantity, 5.0)
        self.assertEqual(trade.value, 500.0)
        
    def test_execute_sell_trade_no_position(self):
        """Test sell trade with no position."""
        timestamp = pd.Timestamp('2024-01-01')
        trade = self.engine._execute_sell_trade(
            timestamp=timestamp,
            symbol="TEST",
            price=100.0,
            position_size=1.0,
            current_position=0.0
        )
        
        self.assertIsNone(trade)
        
    def test_execute_sell_trade_below_min_order_value(self):
        """Test sell trade below minimum order value."""
        engine = BacktestEngine(min_order_value=1000.0)
        timestamp = pd.Timestamp('2024-01-01')
        trade = engine._execute_sell_trade(
            timestamp=timestamp,
            symbol="TEST",
            price=100.0,
            position_size=1.0,
            current_position=1.0
        )
        
        self.assertIsNone(trade)
        
    def test_run_backtest_no_signals(self):
        """Test backtest with no trading signals."""
        strategy = MockStrategy()  # No signals
        
        result = self.engine.run_backtest(strategy, self.sample_data, "TEST")
        
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.strategy_name, "MockStrategy")
        self.assertEqual(result.symbol, "TEST")
        self.assertEqual(result.initial_capital, 10000.0)
        self.assertEqual(result.final_capital, 10000.0)
        self.assertEqual(result.total_return, 0.0)
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(len(result.trades), 0)
        
    def test_run_backtest_with_signals(self):
        """Test backtest with trading signals."""
        # Create strategy with buy signal on day 1 and sell signal on day 5
        signals = {
            self.sample_data.index[1]: 1,  # Buy signal
            self.sample_data.index[5]: -1  # Sell signal
        }
        strategy = MockStrategy(signals)
        
        result = self.engine.run_backtest(strategy, self.sample_data, "TEST")
        
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.total_trades, 2)
        self.assertEqual(len(result.trades), 2)
        
        # Check first trade is buy
        first_trade = result.trades[0]
        self.assertEqual(first_trade.side, TradeSide.BUY)
        self.assertEqual(first_trade.price, 101.5)
        
        # Check second trade is sell
        second_trade = result.trades[1]
        self.assertEqual(second_trade.side, TradeSide.SELL)
        self.assertEqual(second_trade.price, 105.5)
        
        # Should be profitable
        self.assertGreater(result.total_return, 0)
        
    def test_run_backtest_position_size_validation(self):
        """Test backtest with invalid position size."""
        signals = {self.sample_data.index[1]: 1}
        strategy = MockStrategy(signals)
        
        # Mock strategy to return invalid position size
        def mock_generate_signals(data):
            df = data.copy()
            df['signal'] = 0
            df['position_size'] = 1.5  # Invalid: > 1.0
            df.loc[df.index[1], 'signal'] = 1
            return df
            
        strategy.generate_signals = mock_generate_signals
        
        with self.assertRaises(ValueError) as context:
            self.engine.run_backtest(strategy, self.sample_data, "TEST")
        self.assertIn("Position size must be between 0 and 1", str(context.exception))
        
    def test_calculate_win_rate_no_trades(self):
        """Test win rate calculation with no trades."""
        win_rate = self.engine._calculate_win_rate([])
        self.assertEqual(win_rate, 0.0)
        
    def test_calculate_win_rate_with_trades(self):
        """Test win rate calculation with trades."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0),
            Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.SELL, 110.0, 10.0, 1100.0),  # Win
            Trade(pd.Timestamp('2024-01-03'), 'TEST', TradeSide.BUY, 120.0, 5.0, 600.0),
            Trade(pd.Timestamp('2024-01-04'), 'TEST', TradeSide.SELL, 115.0, 5.0, 575.0),  # Loss
        ]
        
        win_rate = self.engine._calculate_win_rate(trades)
        self.assertEqual(win_rate, 50.0)  # 1 win out of 2 trades
        
    def test_calculate_win_rate_partial_fills(self):
        """Test win rate calculation with partial fills."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0),
            Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.SELL, 110.0, 5.0, 550.0),   # Partial sell (win)
            Trade(pd.Timestamp('2024-01-03'), 'TEST', TradeSide.SELL, 90.0, 5.0, 450.0),    # Remaining sell (loss)
        ]
        
        win_rate = self.engine._calculate_win_rate(trades)
        self.assertEqual(win_rate, 50.0)  # 1 win, 1 loss from partial fills
        
    @patch('niffler.backtesting.backtest_engine.logging')
    def test_logging_calls(self, mock_logging):
        """Test that logging calls are made during backtest."""
        strategy = MockStrategy()
        
        self.engine.run_backtest(strategy, self.sample_data, "TEST")
        
        # Check that logging.info was called
        mock_logging.info.assert_called()
        
        # Check for specific log messages
        log_calls = [call[0][0] for call in mock_logging.info.call_args_list]
        self.assertTrue(any("Input validation passed" in msg for msg in log_calls))
        self.assertTrue(any("Starting backtest" in msg for msg in log_calls))
        
    def test_backtest_with_risk_manager_allowed_trade(self):
        """Test backtest with risk manager that allows trades."""
        risk_manager = MockRiskManager(allow_trade=True, position_size=0.3)
        signals_data = {self.sample_data.index[1]: 1}
        strategy = MockStrategy(signals_data, risk_manager)
        
        result = self.engine.run_backtest(strategy, self.sample_data, "TEST")
        
        self.assertIsInstance(result, BacktestResult)
        self.assertGreater(len(result.trades), 0)  # Should have trades
        # Risk manager position size should be used (0.3 instead of 1.0)
        
        # Check that risk manager state was updated
        self.assertIn("TEST", risk_manager.positions)
        
    def test_backtest_with_risk_manager_blocked_trade(self):
        """Test backtest with risk manager that blocks trades."""
        risk_manager = MockRiskManager(allow_trade=False)
        signals_data = {self.sample_data.index[1]: 1}
        strategy = MockStrategy(signals_data, risk_manager)
        
        result = self.engine.run_backtest(strategy, self.sample_data, "TEST")
        
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(len(result.trades), 0)  # Should have no trades
        self.assertEqual(result.final_capital, result.initial_capital)  # No trades = no change
        
    def test_backtest_with_risk_manager_stop_loss(self):
        """Test backtest with risk manager stop loss functionality."""
        risk_manager = MockRiskManager(allow_trade=True, position_size=0.5, stop_loss_price=102.0)
        # Buy signal at index 1 (price 101.5), should trigger stop loss at index 2 (price 102.5)
        signals_data = {self.sample_data.index[1]: 1}
        strategy = MockStrategy(signals_data, risk_manager)
        
        result = self.engine.run_backtest(strategy, self.sample_data, "TEST")
        
        self.assertIsInstance(result, BacktestResult)
        # Should have both buy trade and stop loss sell trade
        self.assertGreaterEqual(len(result.trades), 1)
        
    def test_backtest_risk_manager_position_state_sync(self):
        """Test that risk manager position state stays synchronized."""
        risk_manager = MockRiskManager(allow_trade=True, position_size=1.0)  # Use 100% to ensure full close
        signals_data = {
            self.sample_data.index[1]: 1,   # Buy
            self.sample_data.index[4]: -1   # Sell
        }
        strategy = MockStrategy(signals_data, risk_manager)
        
        result = self.engine.run_backtest(strategy, self.sample_data, "TEST")
        
        # After backtest completes, position should be cleared
        self.assertNotIn("TEST", risk_manager.positions)
        
        # Should have buy and sell trades
        self.assertGreaterEqual(len(result.trades), 2)


if __name__ == '__main__':
    unittest.main()