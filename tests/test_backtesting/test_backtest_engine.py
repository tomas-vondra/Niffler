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
from niffler.backtesting.portfolio import Portfolio
from niffler.backtesting.round_trip import RoundTrip
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
    """Mock risk manager for testing.

    Holds no position state - it records the snapshots the engine hands it so a
    test can assert on what the engine reported, not on what the manager kept.
    """

    def __init__(self, allow_trade=True, position_size=0.5, stop_loss_price=95.0):
        self.allow_trade = allow_trade
        self.position_size = position_size
        self.stop_loss_price = stop_loss_price
        self.snapshots = []

    def evaluate_trade(self, signal, current_price, portfolio_value, historical_data, portfolio):
        from niffler.risk.base_risk_manager import RiskDecision
        self.snapshots.append(portfolio)
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


class RecordingEngine(BacktestEngine):
    """Engine that records portfolio state after each executed buy.

    The risk manager used to be the only place a test could observe the engine's
    view of an open position. It no longer keeps one, so a test that needs the
    cost basis or the armed stop reads them off the Portfolio the engine owns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_prices = []
        self.stops = []

    def _process_buy(self, portfolio, *args, **kwargs):
        super()._process_buy(portfolio, *args, **kwargs)
        self.entry_prices.append(portfolio.entry_price)
        self.stops.append(portfolio.stop_loss)


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
        
    def test_execute_buy_trade_small_balance_buys_a_fraction(self):
        """
        A balance smaller than one share is not insufficient cash.

        The engine trades fractional units, so $50 at $100 a share buys half a
        share. This used to return None - not by design, but because
        ``budget / (1 + commission)`` recomposed to a value one ULP above the
        budget and the affordability check rejected it. The old test asserted the
        rounding bug rather than a rule.
        """
        timestamp = pd.Timestamp('2024-01-01')
        trade = self.engine._execute_buy_trade(
            timestamp=timestamp,
            symbol="TEST",
            price=100.0,
            position_size=1.0,
            available_cash=50.0
        )

        self.assertIsNotNone(trade)
        self.assertAlmostEqual(trade.value, 50.0 / 1.001, places=6)
        self.assertLessEqual(trade.value + trade.commission, 50.0)

    def test_execute_buy_trade_no_cash(self):
        """With nothing to spend there is nothing to buy."""
        timestamp = pd.Timestamp('2024-01-01')

        for available_cash in (0.0, -10.0):
            with self.subTest(available_cash=available_cash):
                trade = self.engine._execute_buy_trade(
                    timestamp=timestamp,
                    symbol="TEST",
                    price=100.0,
                    position_size=1.0,
                    available_cash=available_cash
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
        """Test backtest with trading signals (filled on the next bar's open)."""
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

        # Check first trade is buy, executed at the OPEN of the bar after the signal
        first_trade = result.trades[0]
        self.assertEqual(first_trade.side, TradeSide.BUY)
        self.assertEqual(first_trade.timestamp, self.sample_data.index[2])
        self.assertEqual(first_trade.price, self.sample_data['open'].iloc[2])

        # Check second trade is sell, likewise deferred by one bar
        second_trade = result.trades[1]
        self.assertEqual(second_trade.side, TradeSide.SELL)
        self.assertEqual(second_trade.timestamp, self.sample_data.index[6])
        self.assertEqual(second_trade.price, self.sample_data['open'].iloc[6])

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
        # The risk manager's 0.3 is used in place of the strategy's 1.0.
        self.assertAlmostEqual(result.trades[0].value / 10000.0, 0.3, places=2)
        # It was handed a snapshot of a flat portfolio, and kept nothing.
        self.assertEqual(len(risk_manager.snapshots), 1)
        self.assertEqual(risk_manager.snapshots[0].open_positions, 0)
        
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
        
    def test_backtest_risk_manager_sees_the_portfolio_it_created(self):
        """The snapshot at the exit reflects the entry the engine just made."""
        risk_manager = MockRiskManager(allow_trade=True, position_size=1.0)  # Use 100% to ensure full close
        signals_data = {
            self.sample_data.index[1]: 1,   # Buy
            self.sample_data.index[4]: -1   # Sell
        }
        strategy = MockStrategy(signals_data, risk_manager)

        result = self.engine.run_backtest(strategy, self.sample_data, "TEST")

        self.assertEqual(len(risk_manager.snapshots), 2)
        self.assertEqual(risk_manager.snapshots[0].open_positions, 0)  # flat before the buy
        self.assertEqual(risk_manager.snapshots[1].open_positions, 1)  # positioned before the sell

        # Everything bought is sold back.
        bought = sum(t.quantity for t in result.trades if t.side == TradeSide.BUY)
        sold = sum(t.quantity for t in result.trades if t.side == TradeSide.SELL)
        self.assertAlmostEqual(bought, sold, places=9)
        self.assertGreaterEqual(len(result.trades), 2)


class TestExecutionTiming(unittest.TestCase):
    """Regression tests for the look-ahead bias fix (signals fill on the next bar)."""

    def setUp(self):
        self.engine = BacktestEngine(initial_capital=10000.0, commission=0.001,
                                     min_order_value=1.0)
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        self.data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'volume': [1000.0] * 10
        }, index=dates)

    def test_default_execution_timing_is_next_bar_open(self):
        """The bias-free policy must be the default."""
        self.assertEqual(self.engine.execution_timing, 'next_bar_open')
        self.assertEqual(self.engine.execution_lag, 1)
        self.assertEqual(self.engine.execution_price_column, 'open')

    def test_invalid_execution_timing_rejected(self):
        """Unknown execution timings must fail loudly."""
        with self.assertRaises(ValueError) as context:
            BacktestEngine(execution_timing='midnight')
        self.assertIn("Execution timing must be one of", str(context.exception))

    def test_signal_is_never_filled_on_its_own_bar(self):
        """A signal on bar i must not be filled at bar i (no same-bar fill)."""
        signal_bar = 3
        strategy = MockStrategy({self.data.index[signal_bar]: 1})

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertNotEqual(trade.timestamp, self.data.index[signal_bar])
        self.assertEqual(trade.timestamp, self.data.index[signal_bar + 1])
        self.assertEqual(trade.price, self.data['open'].iloc[signal_bar + 1])
        # The signal bar's close is exactly the price we must never trade at.
        self.assertNotEqual(trade.price, self.data['close'].iloc[signal_bar])

    def test_every_fill_uses_its_own_bar_open_and_lags_the_signal(self):
        """Sweep several signals: every fill is one bar late and at that bar's open."""
        signal_bars = [1, 3, 5, 7]
        signals = {self.data.index[b]: (1 if k % 2 == 0 else -1)
                   for k, b in enumerate(signal_bars)}
        strategy = MockStrategy(signals)

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        self.assertGreater(len(result.trades), 0)
        signal_timestamps = {self.data.index[b] for b in signal_bars}
        for trade in result.trades:
            self.assertNotIn(trade.timestamp, signal_timestamps)
            bar = self.data.index.get_loc(trade.timestamp)
            self.assertEqual(trade.price, self.data['open'].iloc[bar])

    def test_signal_on_final_bar_is_never_executed(self):
        """There is no bar after the last one, so the signal simply expires."""
        strategy = MockStrategy({self.data.index[-1]: 1})

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.final_capital, result.initial_capital)

    def test_same_bar_close_timing_is_opt_in(self):
        """The legacy (biased) policy is still reachable explicitly."""
        engine = BacktestEngine(initial_capital=10000.0, commission=0.001,
                                execution_timing='same_bar_close')
        strategy = MockStrategy({self.data.index[1]: 1})

        result = engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].timestamp, self.data.index[1])
        self.assertEqual(result.trades[0].price, self.data['close'].iloc[1])


class TestFirstBarRiskManagement(unittest.TestCase):
    """Regression tests for the units-vs-value bug and the unbound portfolio_value."""

    def setUp(self):
        self.engine = BacktestEngine(initial_capital=10000.0, commission=0.001,
                                     min_order_value=1.0)
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        self.data = pd.DataFrame({
            'open': [100.0, 102.0, 104.0, 106.0, 108.0],
            'high': [103.0, 105.0, 107.0, 109.0, 111.0],
            'low': [99.0, 101.0, 103.0, 105.0, 107.0],
            'close': [101.0, 103.0, 105.0, 107.0, 109.0],
            'volume': [1000.0] * 5
        }, index=dates)

    def test_buy_on_very_first_bar_with_risk_manager(self):
        """Buying on bar 0 with a risk manager used to raise UnboundLocalError."""
        engine = BacktestEngine(initial_capital=10000.0, commission=0.001,
                                execution_timing='same_bar_close')
        risk_manager = MockRiskManager(allow_trade=True, position_size=1.0,
                                       stop_loss_price=None)
        strategy = MockStrategy({self.data.index[0]: 1}, risk_manager)

        result = engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].timestamp, self.data.index[0])

    def test_buy_on_first_executable_bar_with_risk_manager(self):
        """Same case under the default next-bar-open policy."""
        risk_manager = MockRiskManager(allow_trade=True, position_size=1.0,
                                       stop_loss_price=None)
        strategy = MockStrategy({self.data.index[0]: 1}, risk_manager)

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].timestamp, self.data.index[1])

    def test_risk_manager_receives_value_based_exposure(self):
        """Exposure reported to the risk manager is value/value, not units/value."""
        risk_manager = MockRiskManager(allow_trade=True, position_size=1.0,
                                       stop_loss_price=None)
        strategy = MockStrategy({self.data.index[0]: 1, self.data.index[2]: -1},
                                risk_manager)

        self.engine.run_backtest(strategy, self.data, "TEST")

        # All-in buy => essentially the whole portfolio is in the position.
        # The old code divided units by currency and reported ~0.01 here.
        self.assertAlmostEqual(risk_manager.snapshots[1].total_exposure, 1.0, places=6)

    def test_risk_manager_current_position_is_value_based(self):
        """The current_position on the snapshot is a value fraction."""
        risk_manager = MockRiskManager(allow_trade=True, position_size=1.0,
                                       stop_loss_price=None)
        strategy = MockStrategy({self.data.index[0]: 1, self.data.index[2]: -1},
                                risk_manager)

        self.engine.run_backtest(strategy, self.data, "TEST")

        observed = [snapshot.current_position for snapshot in risk_manager.snapshots]
        self.assertEqual(len(observed), 2)
        self.assertAlmostEqual(observed[0], 0.0, places=6)   # flat before the buy
        self.assertAlmostEqual(observed[1], 1.0, places=6)   # fully invested before the sell


class TestTradePairing(unittest.TestCase):
    """Tests for the single commission-aware FIFO pairing routine."""

    def setUp(self):
        self.engine = BacktestEngine(initial_capital=10000.0, commission=0.001)

    def test_trade_commission_defaults_to_zero(self):
        """Trade.commission is optional so existing callers keep working."""
        trade = Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0)
        self.assertEqual(trade.commission, 0.0)

    def test_pairing_is_fifo_across_partial_fills(self):
        """One sell spanning two buys produces two correctly sized round trips."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0),
            Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.BUY, 110.0, 10.0, 1100.0),
            Trade(pd.Timestamp('2024-01-03'), 'TEST', TradeSide.SELL, 120.0, 15.0, 1800.0),
        ]

        round_trips = self.engine.pair_trades(trades)

        self.assertEqual(len(round_trips), 2)
        self.assertTrue(all(isinstance(rt, RoundTrip) for rt in round_trips))
        self.assertAlmostEqual(round_trips[0].quantity, 10.0)
        self.assertAlmostEqual(round_trips[0].entry_price, 100.0)
        self.assertAlmostEqual(round_trips[0].pnl, 200.0)
        self.assertAlmostEqual(round_trips[1].quantity, 5.0)
        self.assertAlmostEqual(round_trips[1].entry_price, 110.0)
        self.assertAlmostEqual(round_trips[1].pnl, 50.0)

    def test_unclosed_buy_does_not_overwrite_an_open_position(self):
        """A second buy must not silently replace the first (old overwrite bug)."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0),
            Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.BUY, 200.0, 10.0, 2000.0),
            Trade(pd.Timestamp('2024-01-03'), 'TEST', TradeSide.SELL, 110.0, 10.0, 1100.0),
        ]

        round_trips = self.engine.pair_trades(trades)

        self.assertEqual(len(round_trips), 1)
        # FIFO: the sell closes the 100.0 lot for +100, it does not close the 200.0 lot.
        self.assertAlmostEqual(round_trips[0].entry_price, 100.0)
        self.assertAlmostEqual(round_trips[0].pnl, 100.0)
        stats = self.engine._calculate_trade_statistics(trades)
        self.assertEqual(stats['num_winning_trades'], 1)
        self.assertEqual(stats['num_losing_trades'], 0)

    def test_mismatched_quantities_are_not_compared_by_value(self):
        """Selling fewer units than bought must not count the whole buy as a loss."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0),
            Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.SELL, 110.0, 2.0, 220.0),
        ]

        round_trips = self.engine.pair_trades(trades)

        self.assertEqual(len(round_trips), 1)
        self.assertAlmostEqual(round_trips[0].quantity, 2.0)
        self.assertAlmostEqual(round_trips[0].pnl, 20.0)
        self.assertEqual(self.engine._calculate_win_rate(trades), 100.0)

    def test_commission_can_turn_a_gross_win_into_a_net_loss(self):
        """P&L is net of both entry and exit commission."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0, 1.0),
            Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.SELL, 100.05, 10.0, 1000.5, 1.0),
        ]

        round_trips = self.engine.pair_trades(trades)

        self.assertEqual(len(round_trips), 1)
        self.assertAlmostEqual(round_trips[0].gross_pnl, 0.5)
        self.assertAlmostEqual(round_trips[0].pnl, -1.5)
        self.assertTrue(round_trips[0].is_loss)
        self.assertEqual(self.engine._calculate_win_rate(trades), 0.0)

    def test_commission_is_apportioned_pro_rata_on_partial_exits(self):
        """Half a lot carries half its entry commission."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0, 2.0),
            Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.SELL, 110.0, 5.0, 550.0, 1.0),
        ]

        round_trips = self.engine.pair_trades(trades)

        self.assertEqual(len(round_trips), 1)
        self.assertAlmostEqual(round_trips[0].entry_commission, 1.0)
        self.assertAlmostEqual(round_trips[0].exit_commission, 1.0)
        self.assertAlmostEqual(round_trips[0].pnl, 50.0 - 2.0)

    def test_all_statistics_come_from_the_same_pairing(self):
        """Win rate, profit factor and trade stats must never disagree."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0, 1.0),
            Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.SELL, 120.0, 10.0, 1200.0, 1.2),
            Trade(pd.Timestamp('2024-01-03'), 'TEST', TradeSide.BUY, 120.0, 10.0, 1200.0, 1.2),
            Trade(pd.Timestamp('2024-01-04'), 'TEST', TradeSide.SELL, 100.0, 10.0, 1000.0, 1.0),
        ]

        stats = self.engine._calculate_trade_statistics(trades)
        round_trips = self.engine.pair_trades(trades)

        self.assertEqual(len(round_trips), 2)
        self.assertEqual(stats['num_winning_trades'] + stats['num_losing_trades'],
                         len(round_trips))
        self.assertEqual(stats['win_rate'], self.engine._calculate_win_rate(trades))
        self.assertEqual(stats['profit_factor'], self.engine._calculate_profit_factor(trades))

        expected_win = 200.0 - 1.0 - 1.2
        expected_loss = 200.0 + 1.2 + 1.0
        self.assertAlmostEqual(stats['average_win'], expected_win)
        self.assertAlmostEqual(stats['largest_win'], expected_win)
        self.assertAlmostEqual(stats['average_loss'], expected_loss)
        self.assertAlmostEqual(stats['largest_loss'], expected_loss)
        self.assertAlmostEqual(stats['profit_factor'], expected_win / expected_loss)

    def test_no_round_trips_yields_zeroed_statistics(self):
        """An unmatched buy produces no statistics rather than garbage."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0, 1.0),
        ]

        stats = self.engine._calculate_trade_statistics(trades)

        self.assertEqual(stats['win_rate'], 0.0)
        self.assertEqual(stats['profit_factor'], 0.0)
        self.assertEqual(stats['num_winning_trades'], 0)
        self.assertEqual(stats['num_losing_trades'], 0)

    def test_profit_factor_is_infinite_without_losses(self):
        """All-winning sequences keep the documented inf semantics."""
        trades = [
            Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0),
            Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.SELL, 110.0, 10.0, 1100.0),
        ]

        self.assertEqual(self.engine._calculate_profit_factor(trades), float('inf'))

    def test_engine_records_commission_on_executed_trades(self):
        """Trades emitted by the engine carry the commission the cash account paid."""
        dates = pd.date_range('2024-01-01', periods=6, freq='D')
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            'high': [106.0] * 6,
            'low': [99.0] * 6,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            'volume': [1000.0] * 6
        }, index=dates)
        strategy = MockStrategy({dates[0]: 1, dates[3]: -1})

        result = self.engine.run_backtest(strategy, data, "TEST")

        self.assertEqual(len(result.trades), 2)
        for trade in result.trades:
            self.assertAlmostEqual(trade.commission, trade.value * self.engine.commission)


class TestAnnualisation(unittest.TestCase):
    """Tests for inferring the Sharpe annualisation factor from the data."""

    def setUp(self):
        self.engine = BacktestEngine()

    def test_daily_crypto_index_infers_365(self):
        index = pd.date_range('2024-01-01', periods=60, freq='D')
        self.assertAlmostEqual(self.engine.resolve_periods_per_year(index), 365.0)

    def test_daily_equity_index_infers_252(self):
        index = pd.date_range('2024-01-01', periods=60, freq='B')
        self.assertAlmostEqual(self.engine.resolve_periods_per_year(index), 252.0)

    def test_hourly_crypto_index_infers_8760(self):
        index = pd.date_range('2024-01-01', periods=24 * 14, freq='h')
        self.assertAlmostEqual(self.engine.resolve_periods_per_year(index), 8760.0)

    def test_weekly_crypto_index_infers_52(self):
        index = pd.date_range('2024-01-06', periods=30, freq='7D')
        self.assertAlmostEqual(self.engine.resolve_periods_per_year(index), 365.0 / 7)

    def test_explicit_override_wins(self):
        engine = BacktestEngine(periods_per_year=100.0)
        index = pd.date_range('2024-01-01', periods=60, freq='D')
        self.assertEqual(engine.resolve_periods_per_year(index), 100.0)

    def test_non_datetime_index_falls_back_to_252(self):
        self.assertAlmostEqual(self.engine.resolve_periods_per_year(pd.RangeIndex(10)), 252.0)

    def test_invalid_override_rejected(self):
        with self.assertRaises(ValueError) as context:
            BacktestEngine(periods_per_year=0)
        self.assertIn("Periods per year must be positive", str(context.exception))

    def test_sharpe_uses_the_resolved_annualisation_factor(self):
        index = pd.date_range('2024-01-01', periods=30, freq='D')  # crypto -> 365
        values = pd.Series(np.linspace(10000.0, 11000.0, 30)
                           + np.tile([0.0, 50.0], 15), index=index)

        sharpe = self.engine._calculate_metrics(values, [])['sharpe_ratio']

        returns = values.pct_change().dropna()
        expected = np.sqrt(365.0) * returns.mean() / returns.std()
        self.assertAlmostEqual(sharpe, expected)

        overridden = BacktestEngine(periods_per_year=252.0)._calculate_metrics(
            values, [])['sharpe_ratio']
        self.assertAlmostEqual(overridden, np.sqrt(252.0) * returns.mean() / returns.std())


class TestPortfolio(unittest.TestCase):
    """Tests for the extracted portfolio state object."""

    def setUp(self):
        self.portfolio = Portfolio(10000.0)

    def test_initial_state_is_flat(self):
        self.assertEqual(self.portfolio.cash, 10000.0)
        self.assertEqual(self.portfolio.position, 0.0)
        self.assertTrue(self.portfolio.is_flat)
        self.assertIsNone(self.portfolio.entry_price)
        self.assertIsNone(self.portfolio.stop_loss)
        self.assertEqual(self.portfolio.side, 0)

    def test_invalid_capital_rejected(self):
        with self.assertRaises(ValueError):
            Portfolio(0.0)

    def test_buy_then_sell_round_trip_accounting(self):
        buy = Trade(pd.Timestamp('2024-01-01'), 'TEST', TradeSide.BUY, 100.0, 10.0, 1000.0, 1.0)
        self.portfolio.apply_buy(buy)
        self.assertAlmostEqual(self.portfolio.cash, 8999.0)
        self.assertAlmostEqual(self.portfolio.position, 10.0)

        sell = Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.SELL, 110.0, 10.0, 1100.0, 1.1)
        self.portfolio.apply_sell(sell)
        self.assertAlmostEqual(self.portfolio.cash, 10097.9)
        self.assertAlmostEqual(self.portfolio.position, 0.0)
        self.assertTrue(self.portfolio.is_flat)

    def test_apply_buy_rejects_a_sell(self):
        sell = Trade(pd.Timestamp('2024-01-02'), 'TEST', TradeSide.SELL, 110.0, 10.0, 1100.0)
        with self.assertRaises(ValueError):
            self.portfolio.apply_buy(sell)

    def test_position_fraction_is_value_based(self):
        """position_fraction converts units to currency before dividing."""
        self.portfolio.cash = 5000.0
        self.portfolio.position = 50.0
        # 50 units at 100 = 5000 of a 10000 portfolio
        self.assertAlmostEqual(self.portfolio.position_fraction(100.0), 0.5)
        self.assertAlmostEqual(self.portfolio.market_value(100.0), 10000.0)

    def test_unrealized_pnl_tracks_side(self):
        self.portfolio.position = 10.0
        self.portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)
        self.assertAlmostEqual(self.portfolio.unrealized_pnl(110.0), 100.0)

        self.portfolio.close_position()
        self.assertEqual(self.portfolio.unrealized_pnl(110.0), 0.0)
        self.assertIsNone(self.portfolio.stop_loss)


class TestStopLossExecution(unittest.TestCase):
    """The stop-loss path must also respect execution timing and commission."""

    def setUp(self):
        self.engine = BacktestEngine(initial_capital=10000.0, commission=0.001)
        dates = pd.date_range('2024-01-01', periods=6, freq='D')
        self.data = pd.DataFrame({
            'open': [100.0, 100.0, 96.0, 90.0, 88.0, 86.0],
            'high': [101.0, 101.0, 97.0, 91.0, 89.0, 87.0],
            'low': [99.0, 95.0, 89.0, 87.0, 85.0, 83.0],
            'close': [100.0, 96.0, 90.0, 88.0, 86.0, 84.0],
            'volume': [1000.0] * 6
        }, index=dates)

    def test_stop_loss_closes_position_and_leaves_the_portfolio_flat(self):
        risk_manager = MockRiskManager(allow_trade=True, position_size=1.0,
                                       stop_loss_price=95.0)
        strategy = MockStrategy({self.data.index[0]: 1}, risk_manager)

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(len(result.trades), 2)
        buy, stop_exit = result.trades
        self.assertEqual(buy.side, TradeSide.BUY)
        self.assertEqual(buy.timestamp, self.data.index[1])
        self.assertEqual(stop_exit.side, TradeSide.SELL)
        # Bar 2 opens at 96.0, above the 95.0 stop, but trades down to 89.0: a
        # resting stop fills the moment price trades through it, so the exit is on
        # bar 2 at the stop price, not on bar 3's open.
        self.assertEqual(stop_exit.timestamp, self.data.index[2])
        self.assertEqual(stop_exit.price, 95.0)
        self.assertAlmostEqual(stop_exit.quantity, buy.quantity, places=9)
        self.assertLess(result.total_return, 0)
        self.assertEqual(result.num_losing_trades, 1)


class EchoingRiskManager(MockRiskManager):
    """
    Risk manager that mirrors FixedRiskManager's sizing contract.

    Entries are sized as a fixed fraction of portfolio *value*; exits echo the
    current position fraction back, exactly like
    ``FixedRiskManager.calculate_position_size`` does for ``signal == -1``.
    """

    def __init__(self, entry_fraction=0.1, stop_loss_price=None):
        super().__init__(allow_trade=True, position_size=entry_fraction,
                         stop_loss_price=stop_loss_price)
        self.entry_fraction = entry_fraction

    def evaluate_trade(self, signal, current_price, portfolio_value, historical_data,
                       portfolio):
        from niffler.risk.base_risk_manager import RiskDecision
        self.snapshots.append(portfolio)
        size = self.entry_fraction if signal == 1 else abs(portfolio.current_position)
        return RiskDecision(
            position_size=size,
            stop_loss_price=self.stop_loss_price,
            max_risk_per_trade=0.05,
            allow_trade=True,
            reason="Echoing risk decision"
        )


class TestRiskManagedExit(unittest.TestCase):
    """A sell signal must flatten the position even with a risk manager attached."""

    def setUp(self):
        self.engine = BacktestEngine(initial_capital=10000.0, commission=0.001,
                                     min_order_value=1.0)
        dates = pd.date_range('2024-01-01', periods=8, freq='D')
        self.data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            'high': [101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
            'volume': [1000.0] * 8
        }, index=dates)

    def test_sell_signal_flattens_position_with_risk_manager(self):
        """The exit sells every held unit, not `position_fraction` of them."""
        risk_manager = EchoingRiskManager(entry_fraction=0.1)
        strategy = MockStrategy({self.data.index[0]: 1, self.data.index[3]: -1},
                                risk_manager)

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(len(result.trades), 2)
        buy, sell = result.trades
        self.assertEqual(buy.side, TradeSide.BUY)
        self.assertEqual(sell.side, TradeSide.SELL)
        # The whole position is liquidated: the old code sold only ~10% of it.
        self.assertAlmostEqual(sell.quantity, buy.quantity, places=9)

    def test_portfolio_is_flat_after_a_risk_managed_exit(self):
        """Everything bought is sold back, so the strategy really exits."""
        risk_manager = EchoingRiskManager(entry_fraction=0.1)
        strategy = MockStrategy({self.data.index[0]: 1, self.data.index[3]: -1},
                                risk_manager)

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        bought = sum(t.quantity for t in result.trades if t.side == TradeSide.BUY)
        sold = sum(t.quantity for t in result.trades if t.side == TradeSide.SELL)
        self.assertAlmostEqual(bought, sold, places=9)
        self.assertEqual(result.num_winning_trades + result.num_losing_trades, 1)

    def test_exit_does_not_leak_exposure_across_repeated_signals(self):
        """Alternating buy/sell signals must not grow the position monotonically."""
        risk_manager = EchoingRiskManager(entry_fraction=0.1)
        signals = {self.data.index[0]: 1, self.data.index[2]: -1,
                   self.data.index[4]: 1, self.data.index[6]: -1}
        strategy = MockStrategy(signals, risk_manager)

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        sells = [t for t in result.trades if t.side == TradeSide.SELL]
        buys = [t for t in result.trades if t.side == TradeSide.BUY]
        self.assertEqual(len(buys), 2)
        self.assertEqual(len(sells), 2)
        # Each entry is fully closed before the next one, so both entries are the
        # same 10% slice of a comparable portfolio rather than a growing stack.
        self.assertAlmostEqual(sum(b.quantity for b in buys),
                               sum(s.quantity for s in sells), places=9)

    def test_strategy_can_still_request_a_partial_exit(self):
        """The exit fraction comes from the strategy, and it is still honoured."""
        class HalfExitStrategy(MockStrategy):
            def generate_signals(self, data):
                df = super().generate_signals(data)
                df['position_size'] = 0.5
                return df

        risk_manager = EchoingRiskManager(entry_fraction=1.0)
        strategy = HalfExitStrategy({self.data.index[0]: 1, self.data.index[3]: -1},
                                    risk_manager)

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        buy, sell = result.trades
        self.assertAlmostEqual(sell.quantity, buy.quantity * 0.5, places=9)

    def test_risk_manager_can_still_veto_an_exit(self):
        """allow_trade=False keeps vetoing sells."""
        risk_manager = MockRiskManager(allow_trade=False)
        strategy = MockStrategy({self.data.index[0]: 1, self.data.index[3]: -1},
                                risk_manager)

        result = self.engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(len(result.trades), 0)


class TestScalingIntoAPosition(unittest.TestCase):
    """A second buy must not discard the cost basis or the stop of the first."""

    def setUp(self):
        self.engine = BacktestEngine(initial_capital=10000.0, commission=0.0,
                                     min_order_value=1.0)
        dates = pd.date_range('2024-01-01', periods=6, freq='D')
        self.data = pd.DataFrame({
            'open': [100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
            'high': [101.0, 101.0, 101.0, 51.0, 51.0, 51.0],
            'low': [99.0, 99.0, 99.0, 49.0, 49.0, 49.0],
            'close': [100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
            'volume': [1000.0] * 6
        }, index=dates)

    def test_add_to_position_keeps_weighted_average_entry_price(self):
        portfolio = Portfolio(10000.0)
        portfolio.position = 10.0
        portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)

        portfolio.add_to_position(entry_price=50.0, quantity=10.0, stop_loss=47.5)

        self.assertAlmostEqual(portfolio.entry_price, 75.0)

    def test_add_to_position_never_disarms_an_existing_stop(self):
        portfolio = Portfolio(10000.0)
        portfolio.position = 10.0
        portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)

        portfolio.add_to_position(entry_price=50.0, quantity=10.0, stop_loss=None)

        self.assertEqual(portfolio.stop_loss, 95.0)

    def test_add_to_position_keeps_the_tighter_stop(self):
        portfolio = Portfolio(10000.0)
        portfolio.position = 10.0
        portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)

        # A looser stop from the new lot must not weaken the protection.
        portfolio.add_to_position(entry_price=50.0, quantity=10.0, stop_loss=47.5)
        self.assertEqual(portfolio.stop_loss, 95.0)

        portfolio.add_to_position(entry_price=110.0, quantity=20.0, stop_loss=104.5)
        self.assertEqual(portfolio.stop_loss, 104.5)

    def test_add_to_position_rejects_non_positive_quantity(self):
        portfolio = Portfolio(10000.0)
        portfolio.position = 10.0
        portfolio.open_position(entry_price=100.0, stop_loss=95.0, side=1)

        with self.assertRaises(ValueError):
            portfolio.add_to_position(entry_price=50.0, quantity=0.0)

    def test_add_to_position_from_flat_opens_the_position(self):
        portfolio = Portfolio(10000.0)

        portfolio.add_to_position(entry_price=100.0, quantity=5.0, stop_loss=95.0)

        self.assertEqual(portfolio.entry_price, 100.0)
        self.assertEqual(portfolio.stop_loss, 95.0)
        self.assertEqual(portfolio.side, 1)

    def test_second_buy_averages_the_entry_price_in_a_backtest(self):
        """Two engine-executed buys leave a volume-weighted entry price."""
        engine = RecordingEngine(initial_capital=10000.0, commission=0.0,
                                 min_order_value=1.0)
        risk_manager = MockRiskManager(allow_trade=True, position_size=0.5,
                                       stop_loss_price=None)
        strategy = MockStrategy({self.data.index[0]: 1, self.data.index[2]: 1},
                                risk_manager)

        result = engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(len(result.trades), 2)
        first, second = result.trades
        expected = ((first.price * first.quantity + second.price * second.quantity)
                    / (first.quantity + second.quantity))
        self.assertAlmostEqual(engine.entry_prices[-1], expected, places=9)
        # The last fill's price is NOT the whole book's entry price.
        self.assertNotAlmostEqual(engine.entry_prices[-1], second.price, places=6)

    def test_second_buy_without_a_stop_leaves_the_original_stop_armed(self):
        """A risk decision carrying no stop must not disarm the position."""
        class StopThenNoneRiskManager(MockRiskManager):
            def __init__(self):
                super().__init__(allow_trade=True, position_size=0.5,
                                 stop_loss_price=95.0)
                self.calls = 0

            def evaluate_trade(self, signal, current_price, portfolio_value,
                               historical_data, portfolio):
                from niffler.risk.base_risk_manager import RiskDecision
                self.calls += 1
                stop = 95.0 if self.calls == 1 else None
                return RiskDecision(position_size=0.5, stop_loss_price=stop,
                                    max_risk_per_trade=0.05, allow_trade=True,
                                    reason="mock")

            def should_close_position(self, current_price, entry_price,
                                      stop_loss_price, signal, unrealized_pnl):
                return False, "not triggered"

        engine = RecordingEngine(initial_capital=10000.0, commission=0.0,
                                 min_order_value=1.0)
        risk_manager = StopThenNoneRiskManager()
        strategy = MockStrategy({self.data.index[0]: 1, self.data.index[2]: 1},
                                risk_manager)

        engine.run_backtest(strategy, self.data, "TEST")

        self.assertEqual(engine.stops[-1], 95.0)


class TestIntrabarStopLoss(unittest.TestCase):
    """The stop is probed against the bar's traded range, not only its open."""

    def setUp(self):
        self.engine = BacktestEngine(initial_capital=10000.0, commission=0.0,
                                     min_order_value=1.0)

    def _run(self, data, stop_loss_price=95.0):
        risk_manager = MockRiskManager(allow_trade=True, position_size=1.0,
                                       stop_loss_price=stop_loss_price)
        strategy = MockStrategy({data.index[0]: 1}, risk_manager)
        return self.engine.run_backtest(strategy, data, "TEST")

    def test_stop_fires_when_the_low_pierces_it_and_price_recovers(self):
        dates = pd.date_range('2024-01-01', periods=4, freq='D')
        data = pd.DataFrame({
            # Bar 2 opens and closes above the 95.0 stop but trades down to 90.0.
            'open': [100.0, 100.0, 99.0, 99.0],
            'high': [101.0, 101.0, 101.0, 101.0],
            'low': [99.0, 99.0, 90.0, 98.0],
            'close': [100.0, 100.0, 100.0, 100.0],
            'volume': [1000.0] * 4
        }, index=dates)

        result = self._run(data)

        self.assertEqual(len(result.trades), 2)
        stop_exit = result.trades[1]
        self.assertEqual(stop_exit.side, TradeSide.SELL)
        self.assertEqual(stop_exit.timestamp, dates[2])
        # Filled at the resting stop, which the bar traded through.
        self.assertAlmostEqual(stop_exit.price, 95.0)

    def test_gap_through_the_stop_fills_at_the_open_not_the_stop(self):
        dates = pd.date_range('2024-01-01', periods=4, freq='D')
        data = pd.DataFrame({
            'open': [100.0, 100.0, 80.0, 80.0],
            'high': [101.0, 101.0, 82.0, 82.0],
            'low': [99.0, 99.0, 78.0, 78.0],
            'close': [100.0, 100.0, 81.0, 81.0],
            'volume': [1000.0] * 4
        }, index=dates)

        result = self._run(data)

        stop_exit = result.trades[1]
        self.assertEqual(stop_exit.timestamp, dates[2])
        # The stop was unreachable: the bar opened below it.
        self.assertAlmostEqual(stop_exit.price, 80.0)

    def test_stop_is_not_triggered_when_the_range_stays_above_it(self):
        dates = pd.date_range('2024-01-01', periods=4, freq='D')
        data = pd.DataFrame({
            'open': [100.0, 100.0, 99.0, 99.0],
            'high': [101.0, 101.0, 101.0, 101.0],
            'low': [99.0, 96.0, 96.0, 96.0],
            'close': [100.0, 100.0, 100.0, 100.0],
            'volume': [1000.0] * 4
        }, index=dates)

        result = self._run(data)

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].side, TradeSide.BUY)

    def test_unexecutable_stop_warns_and_leaves_the_position_open(self):
        """A stop below min_order_value must be visible in the run log."""
        engine = BacktestEngine(initial_capital=10000.0, commission=0.0,
                                min_order_value=1_000_000.0)
        dates = pd.date_range('2024-01-01', periods=4, freq='D')
        data = pd.DataFrame({
            'open': [100.0, 100.0, 99.0, 99.0],
            'high': [101.0, 101.0, 101.0, 101.0],
            'low': [99.0, 99.0, 90.0, 98.0],
            'close': [100.0, 100.0, 100.0, 100.0],
            'volume': [1000.0] * 4
        }, index=dates)
        risk_manager = MockRiskManager(allow_trade=True, position_size=1.0,
                                       stop_loss_price=95.0)
        strategy = MockStrategy({dates[0]: 1}, risk_manager)

        # Force an open position that is too small to liquidate.
        with patch.object(BacktestEngine, '_execute_buy_trade',
                          return_value=Trade(dates[1], "TEST", TradeSide.BUY,
                                             100.0, 0.001, 0.1, 0.0)):
            with patch('niffler.backtesting.backtest_engine.logging') as mock_logging:
                result = engine.run_backtest(strategy, data, "TEST")

        warnings = [str(call) for call in mock_logging.warning.call_args_list]
        self.assertTrue(any('STOP LOSS NOT EXECUTED' in w for w in warnings), warnings)
        # Only the buy was executed; the position is still open.
        self.assertEqual(len(result.trades), 1)


class TestAnnualisationAcrossFrequencies(unittest.TestCase):
    """Bar spacings coarser than a day are calendar time, not trading time."""

    def setUp(self):
        self.engine = BacktestEngine()

    def test_frequency_table(self):
        cases = [
            ('business daily', pd.date_range('2024-01-01', periods=250, freq='B'), 252.0, 0.1),
            ('calendar daily', pd.date_range('2024-01-01', periods=250, freq='D'), 365.0, 0.1),
            ('weekly equity', pd.date_range('2024-01-05', periods=60, freq='W-FRI'), 52.0, 0.5),
            ('monthly equity', pd.date_range('2020-01-31', periods=48, freq='BME'), 12.0, 0.3),
            ('hourly crypto', pd.date_range('2024-01-01', periods=24 * 30, freq='h'), 8760.0, 0.1),
        ]

        for label, index, expected, tolerance in cases:
            with self.subTest(frequency=label):
                inferred = self.engine.resolve_periods_per_year(index)
                self.assertAlmostEqual(inferred, expected, delta=tolerance)

    def test_weekly_equity_is_not_scaled_off_the_trading_calendar(self):
        """The old code divided 252 trading days by a 7 calendar-day spacing."""
        index = pd.date_range('2024-01-05', periods=60, freq='W-FRI')
        self.assertGreater(self.engine.resolve_periods_per_year(index), 50.0)


if __name__ == '__main__':
    unittest.main()