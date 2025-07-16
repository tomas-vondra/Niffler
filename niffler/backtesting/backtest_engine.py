import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from niffler.strategies.base_strategy import BaseStrategy
from .trade import Trade, TradeSide
from .backtest_result import BacktestResult


class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001, 
                 min_order_value: float = 1.0):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: Starting capital amount
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
            min_order_value: Minimum order value to execute trades
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if commission < 0:
            raise ValueError("Commission cannot be negative")
        if min_order_value < 0:
            raise ValueError("Minimum order value cannot be negative")
            
        self.initial_capital = initial_capital
        self.commission = commission
        self.min_order_value = min_order_value
        
    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame, 
                    symbol: str = "UNKNOWN") -> BacktestResult:
        """
        Run a backtest for the given strategy and data.
        
        Args:
            strategy: Trading strategy to test
            data: Price data with OHLCV columns
            symbol: Symbol identifier for the data
            
        Returns:
            BacktestResult object containing all backtest metrics
        """
        # Comprehensive input validation
        self._validate_inputs(strategy, data, symbol)
        
        logging.info(f"Starting backtest for {symbol} with {len(data)} data points")
        
        # Generate trading signals
        signals_df = strategy.generate_signals(data.copy())
        
        # Initialize tracking variables
        cash = self.initial_capital
        position = 0.0  # Number of shares/units held
        portfolio_values = np.zeros(len(data))  # Pre-allocate for performance
        trades = []
        
        # Process each signal
        for i, (idx, row) in enumerate(signals_df.iterrows()):
            current_price = row['close']
            signal = row.get('signal', 0)
            position_size = row.get('position_size', 1.0)
            
            # Validate position size
            if position_size <= 0 or position_size > 1.0:
                raise ValueError(f"Position size must be between 0 and 1, got {position_size}")
            
            # Execute trades based on signals
            if signal == 1 and cash > 0:  # Buy signal
                trade = self._execute_buy_trade(idx, symbol, current_price, position_size, cash)
                if trade:
                    trades.append(trade)
                    # Commission is already included in trade execution logic
                    commission_cost = trade.value * self.commission
                    cash -= trade.value + commission_cost
                    position += trade.quantity
                    logging.info(f"BUY: {trade.quantity:.4f} shares at ${trade.price:.2f} "
                                 f"(Value: ${trade.value:.2f}, Commission: ${commission_cost:.2f}, Cash: ${cash:.2f})")
                    
            elif signal == -1 and position > 0:  # Sell signal
                # Validate sell doesn't exceed position
                max_sell_quantity = position
                requested_sell_quantity = position * position_size
                
                if requested_sell_quantity > max_sell_quantity:
                    logging.warning(f"Sell quantity {requested_sell_quantity:.4f} exceeds position {max_sell_quantity:.4f}")
                    # Adjust to maximum available
                    adjusted_position_size = max_sell_quantity / position if position > 0 else 0
                    trade = self._execute_sell_trade(idx, symbol, current_price, adjusted_position_size, position)
                else:
                    trade = self._execute_sell_trade(idx, symbol, current_price, position_size, position)
                    
                if trade:
                    trades.append(trade)
                    commission_cost = trade.value * self.commission
                    cash += trade.value - commission_cost
                    position -= trade.quantity
                    logging.info(f"SELL: {trade.quantity:.4f} shares at ${trade.price:.2f} "
                                 f"(Value: ${trade.value:.2f}, Commission: ${commission_cost:.2f}, Cash: ${cash:.2f})")
            
            # Calculate portfolio value AFTER trades
            portfolio_value = cash + position * current_price
            portfolio_values[i] = portfolio_value
        
        # Final portfolio value
        final_price = data['close'].iloc[-1]
        final_portfolio_value = cash + position * final_price
        
        # Convert numpy array to pandas Series
        portfolio_series = pd.Series(portfolio_values, index=data.index)
        metrics = self._calculate_metrics(portfolio_series, trades)
        
        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=final_portfolio_value,
            total_return=final_portfolio_value - self.initial_capital,
            total_return_pct=(final_portfolio_value - self.initial_capital) / self.initial_capital * 100,
            trades=trades,
            portfolio_values=portfolio_series,
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics['sharpe_ratio'],
            win_rate=metrics['win_rate'],
            total_trades=len(trades)
        )
    
    def _calculate_metrics(self, portfolio_values: pd.Series, trades: List[Trade]) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        
        # Max drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min() * 100
        
        # Sharpe ratio (assuming 252 trading days per year)
        if len(returns) > 1 and returns.std() > 0:
            metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Win rate - calculate based on properly paired trades
        if trades:
            metrics['win_rate'] = self._calculate_win_rate(trades)
        else:
            metrics['win_rate'] = 0.0
        
        return metrics
    
    def _execute_buy_trade(self, timestamp: pd.Timestamp, symbol: str, price: float, 
                          position_size: float, available_cash: float) -> Optional[Trade]:
        """Execute a buy trade if conditions are met."""
        # Calculate max investment accounting for commission
        max_investment_with_commission = available_cash * position_size
        # Solve for trade_value where trade_value + (trade_value * commission) = max_investment
        trade_value = max_investment_with_commission / (1 + self.commission)
        shares_to_buy = trade_value / price
        commission_cost = trade_value * self.commission
        total_cost = trade_value + commission_cost
        
        # Check minimum order value and sufficient cash
        if trade_value >= self.min_order_value and available_cash >= total_cost:
            return Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=TradeSide.BUY,
                price=price,
                quantity=shares_to_buy,
                value=trade_value
            )
        return None
    
    def _execute_sell_trade(self, timestamp: pd.Timestamp, symbol: str, price: float,
                           position_size: float, current_position: float) -> Optional[Trade]:
        """Execute a sell trade if conditions are met."""
        shares_to_sell = current_position * position_size
        trade_value = shares_to_sell * price
        
        # Check minimum order value
        if trade_value >= self.min_order_value and shares_to_sell > 0:
            return Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=TradeSide.SELL,
                price=price,
                quantity=shares_to_sell,
                value=trade_value
            )
        return None
    
    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """Calculate win rate based on properly paired buy/sell trades."""
        if not trades:
            return 0.0
        
        # Group trades by timestamp to properly pair them
        position_tracker = 0.0
        trade_pairs = []
        open_buys = []
        
        for trade in trades:
            if trade.side == TradeSide.BUY:
                # Create a copy to avoid mutating original trade data
                buy_copy = Trade(
                    timestamp=trade.timestamp,
                    symbol=trade.symbol,
                    side=trade.side,
                    price=trade.price,
                    quantity=trade.quantity,
                    value=trade.value
                )
                open_buys.append(buy_copy)
                position_tracker += trade.quantity
            elif trade.side == TradeSide.SELL and open_buys:
                # Match sells with buys on FIFO basis
                remaining_to_sell = trade.quantity
                sell_price = trade.price
                
                while remaining_to_sell > 0 and open_buys:
                    buy_trade = open_buys[0]
                    
                    if buy_trade.quantity <= remaining_to_sell:
                        # Full buy trade is closed
                        trade_pairs.append((buy_trade.price, sell_price))
                        remaining_to_sell -= buy_trade.quantity
                        open_buys.pop(0)
                    else:
                        # Partial buy trade is closed
                        trade_pairs.append((buy_trade.price, sell_price))
                        buy_trade.quantity -= remaining_to_sell  # Now safe to modify copy
                        remaining_to_sell = 0
                
                position_tracker -= trade.quantity
        
        # Calculate win rate from paired trades
        if not trade_pairs:
            return 0.0
        
        winning_trades = sum(1 for buy_price, sell_price in trade_pairs if sell_price > buy_price)
        return (winning_trades / len(trade_pairs)) * 100
    
    def _validate_inputs(self, strategy: BaseStrategy, data: pd.DataFrame, symbol: str) -> None:
        """Comprehensive input validation for backtest data."""
        # Validate strategy
        if not strategy:
            raise ValueError("Strategy cannot be None")
        
        if not strategy.validate_data(data):
            raise ValueError("Invalid data format for backtesting")
        
        # Validate DataFrame
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        if len(data) < 2:
            raise ValueError("Data must have at least 2 rows for backtesting")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types and values
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be numeric")
            
            if data[col].isnull().any():
                raise ValueError(f"Column '{col}' contains null values")
            
            if col in ['open', 'high', 'low', 'close'] and (data[col] <= 0).any():
                raise ValueError(f"Column '{col}' contains non-positive values")
            
            if col == 'volume' and (data[col] < 0).any():
                raise ValueError(f"Column '{col}' contains negative values")
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            raise ValueError(f"Found {invalid_count} rows with invalid OHLC relationships")
        
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Validate index (should be datetime)
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")
        
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index must be sorted in ascending order")
        
        logging.info(f"Input validation passed for {symbol}")
        logging.info(f"Data range: {data.index[0]} to {data.index[-1]}")
        logging.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")