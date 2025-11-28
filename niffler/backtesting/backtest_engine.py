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
        
        # Risk management will be applied during execution loop for real-time portfolio state
        
        # Initialize tracking variables
        cash = self.initial_capital
        position = 0.0  # Number of shares/units held
        portfolio_values = np.zeros(len(data))  # Pre-allocate for performance
        trades = []
        
        # Risk management tracking
        current_stop_loss = None
        entry_price = None
        position_signal = 0  # Track whether current position is long (1) or short (-1)
        
        # Process each signal
        for i, (idx, row) in enumerate(signals_df.iterrows()):
            current_price = row['close']
            signal = row.get('signal', 0)
            position_size = row.get('position_size', 1.0)
            stop_loss_price = None
            
            # Apply risk management for this specific signal
            if signal != 0 and hasattr(strategy, 'risk_manager') and strategy.risk_manager is not None:
                # Get historical data up to current point  
                historical_data = data.iloc[:i+1] if i > 0 else data.iloc[:1]
                
                # Get current position size as fraction of portfolio
                portfolio_value_current = cash + position * current_price
                current_position_fraction = (position * current_price) / portfolio_value_current if portfolio_value_current > 0 else 0.0
                
                # Evaluate risk for this trade
                risk_decision = strategy.risk_manager.evaluate_trade(
                    signal=signal,
                    current_price=current_price,
                    portfolio_value=portfolio_value_current,
                    historical_data=historical_data,
                    current_position=current_position_fraction
                )
                
                # Apply risk management decisions
                if risk_decision.allow_trade:
                    position_size = risk_decision.position_size
                    stop_loss_price = risk_decision.stop_loss_price
                else:
                    signal = 0  # Block the trade
                    position_size = 0.0
            
            # Validate position size
            if position_size < 0 or position_size > 1.0:
                raise ValueError(f"Position size must be between 0 and 1, got {position_size}")
            
            # Check stop-loss conditions for existing position
            stop_loss_triggered = False
            if position != 0 and current_stop_loss is not None:
                if hasattr(strategy, 'risk_manager') and strategy.risk_manager is not None:
                    should_close, reason = strategy.risk_manager.should_close_position(
                        current_price=current_price,
                        entry_price=entry_price,
                        stop_loss_price=current_stop_loss,
                        signal=position_signal,
                        unrealized_pnl=(current_price - entry_price) * position * position_signal
                    )
                    if should_close:
                        # Execute stop-loss trade
                        stop_trade = self._execute_sell_trade(idx, symbol, current_price, 1.0, position)
                        if stop_trade:
                            trades.append(stop_trade)
                            commission_cost = stop_trade.value * self.commission
                            cash += stop_trade.value - commission_cost
                            position = 0.0
                            current_stop_loss = None
                            entry_price = None
                            position_signal = 0
                            stop_loss_triggered = True
                            
                            # Clear risk manager position state for stop loss
                            if hasattr(strategy, 'risk_manager') and strategy.risk_manager is not None:
                                strategy.risk_manager.clear_position(symbol)
                            
                            logging.info(f"STOP LOSS: {stop_trade.quantity:.4f} shares at ${stop_trade.price:.2f} - {reason}")
            
            # Execute trades based on signals (if no stop-loss was triggered)
            if not stop_loss_triggered:
                if signal == 1 and cash > 0:  # Buy signal
                    trade = self._execute_buy_trade(idx, symbol, current_price, position_size, cash)
                    if trade:
                        trades.append(trade)
                        # Commission is already included in trade execution logic
                        commission_cost = trade.value * self.commission
                        cash -= trade.value + commission_cost
                        position += trade.quantity
                        
                        # Set risk management tracking for new position
                        entry_price = current_price
                        current_stop_loss = stop_loss_price
                        position_signal = 1
                        
                        # Update risk manager position state
                        if hasattr(strategy, 'risk_manager') and strategy.risk_manager is not None:
                            position_size_fraction = position / portfolio_value if portfolio_value > 0 else 0.0
                            strategy.risk_manager.update_position_state(
                                symbol=symbol,
                                position_size=position_size_fraction,
                                entry_price=entry_price,
                                stop_loss_price=current_stop_loss,
                                entry_timestamp=idx
                            )
                        
                        logging.info(f"BUY: {trade.quantity:.4f} shares at ${trade.price:.2f} "
                                     f"(Value: ${trade.value:.2f}, Commission: ${commission_cost:.2f}, Cash: ${cash:.2f})")
                        if current_stop_loss:
                            logging.info(f"Stop loss set at ${current_stop_loss:.2f}")
                        
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
                        
                        # Clear risk management tracking if position fully closed
                        if abs(position) < 0.005:  # Use tolerance for positions smaller than 0.5%
                            current_stop_loss = None
                            entry_price = None
                            position_signal = 0
                            
                            # Clear risk manager position state
                            if hasattr(strategy, 'risk_manager') and strategy.risk_manager is not None:
                                strategy.risk_manager.clear_position(symbol)
                        else:
                            # Update risk manager with new position size
                            if hasattr(strategy, 'risk_manager') and strategy.risk_manager is not None:
                                position_size_fraction = position / portfolio_value if portfolio_value > 0 else 0.0
                                strategy.risk_manager.update_position_state(
                                    symbol=symbol,
                                    position_size=position_size_fraction,
                                    entry_price=entry_price,
                                    stop_loss_price=current_stop_loss,
                                    entry_timestamp=idx
                                )
                        
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
            total_trades=len(trades),
            profit_factor=metrics['profit_factor'],
            average_win=metrics['average_win'],
            average_loss=metrics['average_loss'],
            largest_win=metrics['largest_win'],
            largest_loss=metrics['largest_loss'],
            num_winning_trades=metrics['num_winning_trades'],
            num_losing_trades=metrics['num_losing_trades']
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

        # Win rate and profit factor - calculate based on properly paired trades
        if trades:
            metrics['win_rate'] = self._calculate_win_rate(trades)
            metrics['profit_factor'] = self._calculate_profit_factor(trades)
            trade_stats = self._calculate_trade_statistics(trades)
            metrics['average_win'] = trade_stats['average_win']
            metrics['average_loss'] = trade_stats['average_loss']
            metrics['largest_win'] = trade_stats['largest_win']
            metrics['largest_loss'] = trade_stats['largest_loss']
            metrics['num_winning_trades'] = trade_stats['num_winning_trades']
            metrics['num_losing_trades'] = trade_stats['num_losing_trades']
        else:
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
            metrics['average_win'] = 0.0
            metrics['average_loss'] = 0.0
            metrics['largest_win'] = 0.0
            metrics['largest_loss'] = 0.0
            metrics['num_winning_trades'] = 0
            metrics['num_losing_trades'] = 0

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

    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """
        Calculate profit factor based on properly paired buy/sell trades.

        Profit Factor = Gross Profit / Gross Loss

        Returns:
            Profit factor (0 if no losses, None if no winning trades)
        """
        if not trades:
            return 0.0

        # Pair buy/sell trades and calculate P&L for each position
        open_position = None
        gross_profit = 0.0
        gross_loss = 0.0

        for trade in trades:
            if trade.side == TradeSide.BUY:
                open_position = trade
            elif trade.side == TradeSide.SELL and open_position is not None:
                # Calculate P&L for this position
                pnl = trade.value - open_position.value

                if pnl > 0:
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)

                open_position = None

        # Calculate profit factor
        if gross_loss > 0:
            return gross_profit / gross_loss
        elif gross_profit > 0:
            return float('inf')  # All wins, no losses
        else:
            return 0.0  # No trades or all break-even

    def _calculate_trade_statistics(self, trades: List[Trade]) -> Dict[str, float]:
        """
        Calculate detailed trade statistics from paired buy/sell trades.

        Returns:
            Dictionary containing average_win, average_loss, largest_win, largest_loss,
            num_winning_trades, and num_losing_trades
        """
        if not trades:
            return {
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'num_winning_trades': 0,
                'num_losing_trades': 0
            }

        # Pair buy/sell trades and calculate P&L for each position
        open_position = None
        winning_trades = []
        losing_trades = []

        for trade in trades:
            if trade.side == TradeSide.BUY:
                open_position = trade
            elif trade.side == TradeSide.SELL and open_position is not None:
                # Calculate P&L for this position
                pnl = trade.value - open_position.value

                if pnl > 0:
                    winning_trades.append(pnl)
                elif pnl < 0:
                    losing_trades.append(abs(pnl))
                # Ignore break-even trades (pnl == 0)

                open_position = None

        # Calculate statistics
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)

        average_win = sum(winning_trades) / num_winning if num_winning > 0 else 0.0
        average_loss = sum(losing_trades) / num_losing if num_losing > 0 else 0.0
        largest_win = max(winning_trades) if num_winning > 0 else 0.0
        largest_loss = max(losing_trades) if num_losing > 0 else 0.0

        return {
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'num_winning_trades': num_winning,
            'num_losing_trades': num_losing
        }

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