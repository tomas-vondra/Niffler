import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from niffler.strategies.base_strategy import BaseStrategy
from .trade import Trade, TradeSide
from .backtest_result import BacktestResult
from .portfolio import Portfolio
from .cost_model import CostModel, FillRequest, ZeroCostModel
from .round_trip import RoundTrip, QUANTITY_EPSILON, pair_trades


class BacktestEngine:
    """
    Engine for backtesting trading strategies.

    Execution timing
    ----------------
    A signal observed on bar ``i`` cannot be traded at bar ``i``'s close: the
    close is only known once the bar is over. The engine therefore defers every
    order by one bar and fills it at the *next* bar's open (``next_bar_open``,
    the default). ``same_bar_close`` reproduces the old, look-ahead-biased
    behaviour and exists only for comparison; it must not be used to evaluate a
    strategy. Because execution timing is enforced here, strategies stay free to
    compute their signals from the closing price of the bar they see.

    Transaction costs
    -----------------
    Every fill is priced by a :class:`~niffler.backtesting.cost_model.CostModel`.
    It defaults to :class:`~niffler.backtesting.cost_model.ZeroCostModel` -
    frictionless, so numbers produced before transaction costs existed stay
    reproducible - and a configured model can only make fills worse: a buy pays
    up, a sell gives up, stop exits included. A model may also cap how much of a
    bar's volume one order takes; the order is then truncated to a partial fill
    and logged, never silently dropped.
    """

    #: Supported execution-timing policies.
    EXECUTION_TIMINGS = ('next_bar_open', 'same_bar_close')

    #: Quantities below this are treated as fully matched during trade pairing.
    _QUANTITY_EPSILON = QUANTITY_EPSILON

    #: Fallback annualisation factor when the index frequency cannot be inferred.
    _DEFAULT_PERIODS_PER_YEAR = 252.0

    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001,
                 min_order_value: float = 1.0, execution_timing: str = 'next_bar_open',
                 periods_per_year: Optional[float] = None,
                 cost_model: Optional[CostModel] = None):
        """
        Initialize the backtest engine.

        Args:
            initial_capital: Starting capital amount
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
            min_order_value: Minimum order value to execute trades
            execution_timing: When a signal is filled. 'next_bar_open' (default)
                fills a signal from bar i at bar i+1's open, which is the only
                bias-free choice. 'same_bar_close' fills at the signal bar's own
                close and is look-ahead biased.
            periods_per_year: Annualisation factor for the Sharpe ratio. When
                None (default) it is inferred from the data's index frequency:
                daily bars including weekends -> 365, daily bars without
                weekends -> 252, intraday bars -> bars per day times the same
                calendar figure (e.g. hourly crypto -> 8760).
            cost_model: Transaction cost model applied to every fill. When None
                (default) a ZeroCostModel is used: fills happen at the exact
                reference price in unlimited size, which is frictionless and
                therefore not realistic.

        Raises:
            ValueError: If any parameter is outside its valid range
            TypeError: If cost_model is not a CostModel
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if commission < 0:
            raise ValueError("Commission cannot be negative")
        if min_order_value < 0:
            raise ValueError("Minimum order value cannot be negative")
        if execution_timing not in self.EXECUTION_TIMINGS:
            raise ValueError(
                f"Execution timing must be one of {self.EXECUTION_TIMINGS}, got '{execution_timing}'"
            )
        if periods_per_year is not None and periods_per_year <= 0:
            raise ValueError("Periods per year must be positive")
        if cost_model is not None and not isinstance(cost_model, CostModel):
            raise TypeError(
                f"cost_model must be a CostModel, got {type(cost_model).__name__}"
            )

        self.initial_capital = initial_capital
        self.commission = commission
        self.min_order_value = min_order_value
        self.execution_timing = execution_timing
        self.periods_per_year = periods_per_year
        self.cost_model: CostModel = cost_model if cost_model is not None else ZeroCostModel()

    @property
    def execution_lag(self) -> int:
        """Number of bars between signal generation and execution."""
        return 1 if self.execution_timing == 'next_bar_open' else 0

    @property
    def execution_price_column(self) -> str:
        """OHLC column used as the fill price."""
        return 'open' if self.execution_timing == 'next_bar_open' else 'close'

    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame,
                     symbol: str = "UNKNOWN") -> BacktestResult:
        """
        Run a backtest for the given strategy and data.

        A signal produced on bar i is executed on bar i+1 at that bar's open
        price (see the class docstring); a signal on the final bar is therefore
        never filled.

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
        signals, position_sizes = self._extract_signal_columns(signals_df, len(data))

        portfolio = Portfolio(self.initial_capital)
        trades: List[Trade] = []
        portfolio_values = np.zeros(len(data))  # Pre-allocate for performance

        lag = self.execution_lag
        execution_prices = data[self.execution_price_column].to_numpy(dtype=float)
        close_prices = data['close'].to_numpy(dtype=float)
        low_prices = data['low'].to_numpy(dtype=float)
        high_prices = data['high'].to_numpy(dtype=float)
        # Liquidity of the bar the order fills on, not of the signal bar.
        volumes = data['volume'].to_numpy(dtype=float)
        timestamps = data.index

        for i in range(len(data)):
            execution_price = float(execution_prices[i])
            timestamp = timestamps[i]

            # Orders are placed on bar i - lag and filled on bar i.
            signal_index = i - lag
            if signal_index >= 0:
                signal = int(signals[signal_index])
                position_size = float(position_sizes[signal_index])
            else:
                signal, position_size = 0, 0.0

            signal, position_size, stop_loss_price = self._apply_risk_management(
                strategy=strategy,
                data=data,
                signal=signal,
                position_size=position_size,
                signal_index=signal_index,
                execution_price=execution_price,
                portfolio=portfolio,
            )

            bar_volume = float(volumes[i])

            # An existing position is checked against its stop before new orders.
            stop_loss_triggered = self._process_stop_loss(
                strategy, portfolio, trades, timestamp, symbol, execution_price,
                bar_low=float(low_prices[i]), bar_high=float(high_prices[i]),
                bar_volume=bar_volume
            )

            if not stop_loss_triggered:
                if signal == 1 and portfolio.cash > 0:
                    self._process_buy(strategy, portfolio, trades, timestamp, symbol,
                                      execution_price, position_size, stop_loss_price,
                                      bar_volume=bar_volume)
                elif signal == -1 and portfolio.position > 0:
                    self._process_sell(strategy, portfolio, trades, timestamp, symbol,
                                       execution_price, position_size,
                                       bar_volume=bar_volume)

            # Mark to market at the bar's close, AFTER trades
            portfolio_values[i] = portfolio.market_value(float(close_prices[i]))

        final_portfolio_value = portfolio.market_value(float(data['close'].iloc[-1]))

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
            num_losing_trades=metrics['num_losing_trades'],
            total_commission=float(sum(trade.commission for trade in trades)),
            total_slippage=float(sum(trade.slippage_cost for trade in trades))
        )

    def _extract_signal_columns(self, signals_df: pd.DataFrame,
                                expected_length: int) -> tuple:
        """
        Pull the signal and position_size columns out of a strategy's output.

        Args:
            signals_df: DataFrame returned by the strategy
            expected_length: Number of bars in the price data

        Returns:
            Tuple of (signals, position_sizes) numpy arrays

        Raises:
            ValueError: If the strategy returned a different number of rows, or
                any position size falls outside [0, 1]
        """
        if len(signals_df) != expected_length:
            raise ValueError(
                f"Strategy returned {len(signals_df)} signal rows for {expected_length} data rows"
            )

        if 'signal' in signals_df.columns:
            signals = signals_df['signal'].fillna(0).to_numpy(dtype=float)
        else:
            signals = np.zeros(expected_length)

        if 'position_size' in signals_df.columns:
            position_sizes = signals_df['position_size'].fillna(0.0).to_numpy(dtype=float)
        else:
            position_sizes = np.ones(expected_length)

        invalid = (position_sizes < 0) | (position_sizes > 1.0)
        if invalid.any():
            offending = position_sizes[invalid][0]
            raise ValueError(f"Position size must be between 0 and 1, got {offending}")

        return signals, position_sizes

    def _apply_risk_management(self, strategy: BaseStrategy, data: pd.DataFrame,
                               signal: int, position_size: float, signal_index: int,
                               execution_price: float,
                               portfolio: Portfolio) -> tuple:
        """
        Let the strategy's risk manager veto or resize a pending order.

        The risk manager only sees data up to the bar that produced the signal,
        so it cannot look ahead either.

        ``RiskDecision.position_size`` is an *entry* sizing target: a fraction of
        portfolio **value** to deploy. Exits are sized as a fraction of the held
        **units**, which is a different quantity entirely, so the risk manager's
        number is only applied to buys. A sell keeps the exit fraction the
        strategy asked for (1.0 by default, i.e. flatten the position); the risk
        manager can still veto the exit through ``allow_trade``.

        Args:
            strategy: Strategy being tested
            data: Full price data
            signal: Pending signal (1 buy, -1 sell, 0 none)
            position_size: Position size requested by the strategy
            signal_index: Index of the bar that produced the signal
            execution_price: Price the order would be filled at
            portfolio: Current portfolio state

        Returns:
            Tuple of (signal, position_size, stop_loss_price)

        Raises:
            ValueError: If the risk manager returns a position size outside [0, 1]
        """
        if signal == 0 or strategy.risk_manager is None:
            return signal, position_size, None

        historical_data = data.iloc[:max(signal_index, 0) + 1]

        risk_decision = strategy.risk_manager.evaluate_trade(
            signal=signal,
            current_price=execution_price,
            portfolio_value=portfolio.market_value(execution_price),
            historical_data=historical_data,
            current_position=portfolio.position_fraction(execution_price)
        )

        if not risk_decision.allow_trade:
            return 0, 0.0, None

        if signal == -1:
            # An exit is a fraction of the open position, not a fraction of
            # portfolio value: reinterpreting the risk manager's entry-sizing
            # target here would liquidate only a sliver of the position.
            return signal, position_size, None

        if risk_decision.position_size < 0 or risk_decision.position_size > 1.0:
            raise ValueError(
                f"Position size must be between 0 and 1, got {risk_decision.position_size}"
            )

        return signal, risk_decision.position_size, risk_decision.stop_loss_price

    def _process_stop_loss(self, strategy: BaseStrategy, portfolio: Portfolio,
                           trades: List[Trade], timestamp: pd.Timestamp, symbol: str,
                           price: float, bar_low: Optional[float] = None,
                           bar_high: Optional[float] = None,
                           bar_volume: Optional[float] = None) -> bool:
        """
        Close the open position if the risk manager says the stop was hit.

        A resting stop order fills the moment price trades through it, so the stop
        is probed against the worst price the bar actually traded at - the low for
        a long, the high for a short - not only against the bar's execution price.
        Sampling opens alone credited every dip-through-and-recover to the strategy
        as a loss it would not have avoided in reality.

        The fill respects gaps: a long exits at ``min(open, stop)``, so a bar that
        opens below the stop fills at the open rather than at the unreachable stop
        price. What remains unmodelled is intra-bar ordering - within the bar that
        triggers, the engine cannot tell whether the stop or some other level was
        reached first - and the entry bar itself, which is checked before the entry
        fills and so never against the entry's own stop.

        Transaction costs apply to stops too, and only ever make them worse:
        ``min(open, stop)`` is the *reference* price handed to the cost model, so a
        long's stop fills at or below it and never above. A liquidity cap can leave
        part of the position unsold; the remainder stays open with its stop armed
        rather than being conjured away.

        Args:
            strategy: Strategy being tested
            portfolio: Current portfolio state
            trades: Trade log to append to
            timestamp: Current bar timestamp
            symbol: Traded symbol
            price: Execution price for this bar
            bar_low: Lowest price traded on this bar; falls back to `price`
            bar_high: Highest price traded on this bar; falls back to `price`
            bar_volume: Volume traded on this bar, for liquidity-aware cost models

        Returns:
            True if a stop-loss exit was executed on this bar
        """
        if portfolio.position == 0 or portfolio.stop_loss is None:
            return False
        if strategy.risk_manager is None:
            return False

        is_long = portfolio.side >= 0
        worst_price = (bar_low if is_long else bar_high)
        if worst_price is None:
            worst_price = price

        should_close, reason = strategy.risk_manager.should_close_position(
            current_price=worst_price,
            entry_price=portfolio.entry_price,
            stop_loss_price=portfolio.stop_loss,
            signal=portfolio.side,
            unrealized_pnl=portfolio.unrealized_pnl(worst_price)
        )
        if not should_close:
            return False

        # Gapping through the stop fills at the open, never at the better stop price.
        fill_price = min(price, portfolio.stop_loss) if is_long else max(price, portfolio.stop_loss)

        stop_trade = self._execute_sell_trade(timestamp, symbol, fill_price, 1.0,
                                              portfolio.position, bar_volume=bar_volume)
        if stop_trade is None:
            # The stop was hit but the exit did not execute - the residual position
            # is below the minimum order value, or the cost model reports the bar
            # cannot absorb it. Staying silent here is indistinguishable from
            # "stop not hit".
            logging.warning(
                f"STOP LOSS NOT EXECUTED: {reason}; position {portfolio.position:.6f} units "
                f"at ${fill_price:.2f} is worth ${portfolio.position * fill_price:.2f}, which "
                f"is either below the minimum order value of ${self.min_order_value:.2f} or "
                f"more than the bar can absorb. Position stays open."
            )
            return False

        trades.append(stop_trade)
        portfolio.apply_sell(stop_trade)

        if not portfolio.is_flat:
            # A liquidity cap kept the stop from filling in full. Reporting the
            # partial exit beats pretending the whole position was liquidated.
            logging.warning(
                f"STOP LOSS PARTIALLY FILLED: sold {stop_trade.quantity:.6f} of "
                f"{stop_trade.quantity + portfolio.position:.6f} units at "
                f"${stop_trade.price:.2f} - {reason}. {portfolio.position:.6f} units "
                f"remain open with the stop still armed."
            )
            strategy.risk_manager.update_position_state(
                symbol=symbol,
                position_size=portfolio.position_fraction(price),
                entry_price=portfolio.entry_price,
                stop_loss_price=portfolio.stop_loss,
                entry_timestamp=timestamp
            )
            return True

        portfolio.position = 0.0
        portfolio.close_position()
        strategy.risk_manager.clear_position(symbol)

        logging.info(f"STOP LOSS: {stop_trade.quantity:.4f} shares at ${stop_trade.price:.2f} - {reason}")
        return True

    def _process_buy(self, strategy: BaseStrategy, portfolio: Portfolio,
                     trades: List[Trade], timestamp: pd.Timestamp, symbol: str,
                     price: float, position_size: float,
                     stop_loss_price: Optional[float],
                     bar_volume: Optional[float] = None) -> None:
        """
        Execute a buy order and update portfolio and risk-manager state.

        A buy into an already open position scales in: the entry price becomes
        the quantity-weighted average of both lots and the existing stop is kept
        unless the new order carries a tighter one. Only a buy from flat opens a
        fresh position and adopts the new order's stop verbatim.

        Args:
            strategy: Strategy being tested
            portfolio: Current portfolio state
            trades: Trade log to append to
            timestamp: Current bar timestamp
            symbol: Traded symbol
            price: Reference price for this bar; the cost model prices the
                actual fill against it
            position_size: Fraction of available cash to deploy
            stop_loss_price: Stop-loss level for the new position, if any
            bar_volume: Volume traded on this bar, for liquidity-aware cost models
        """
        trade = self._execute_buy_trade(timestamp, symbol, price, position_size,
                                        portfolio.cash, bar_volume=bar_volume)
        if trade is None:
            return

        trades.append(trade)
        # Risk state first: scaling in has to average against the units held so
        # far, so it must be recorded before apply_buy adds the new ones. The cost
        # basis is the price actually paid, not the reference price the order was
        # benchmarked against, so stop distances measure from the real fill.
        if portfolio.is_flat:
            portfolio.open_position(entry_price=trade.price, stop_loss=stop_loss_price,
                                    side=1)
        else:
            portfolio.add_to_position(entry_price=trade.price, quantity=trade.quantity,
                                      stop_loss=stop_loss_price)
        portfolio.apply_buy(trade)

        if strategy.risk_manager is not None:
            strategy.risk_manager.update_position_state(
                symbol=symbol,
                position_size=portfolio.position_fraction(price),
                entry_price=portfolio.entry_price,
                stop_loss_price=portfolio.stop_loss,
                entry_timestamp=timestamp
            )

        logging.info(f"BUY: {trade.quantity:.4f} shares at ${trade.price:.2f} "
                     f"(Value: ${trade.value:.2f}, Commission: ${trade.commission:.2f}, "
                     f"Cash: ${portfolio.cash:.2f})")
        if portfolio.stop_loss:
            logging.info(f"Stop loss set at ${portfolio.stop_loss:.2f}")

    def _process_sell(self, strategy: BaseStrategy, portfolio: Portfolio,
                      trades: List[Trade], timestamp: pd.Timestamp, symbol: str,
                      price: float, position_size: float,
                      bar_volume: Optional[float] = None) -> None:
        """
        Execute a sell order and update portfolio and risk-manager state.

        Args:
            strategy: Strategy being tested
            portfolio: Current portfolio state
            trades: Trade log to append to
            timestamp: Current bar timestamp
            symbol: Traded symbol
            price: Reference price for this bar; the cost model prices the
                actual fill against it
            position_size: Fraction of the open position to liquidate
            bar_volume: Volume traded on this bar, for liquidity-aware cost models
        """
        trade = self._execute_sell_trade(timestamp, symbol, price,
                                         min(position_size, 1.0), portfolio.position,
                                         bar_volume=bar_volume)
        if trade is None:
            return

        trades.append(trade)
        portfolio.apply_sell(trade)

        if portfolio.is_flat:
            portfolio.close_position()
            if strategy.risk_manager is not None:
                strategy.risk_manager.clear_position(symbol)
        elif strategy.risk_manager is not None:
            strategy.risk_manager.update_position_state(
                symbol=symbol,
                position_size=portfolio.position_fraction(price),
                entry_price=portfolio.entry_price,
                stop_loss_price=portfolio.stop_loss,
                entry_timestamp=timestamp
            )

        logging.info(f"SELL: {trade.quantity:.4f} shares at ${trade.price:.2f} "
                     f"(Value: ${trade.value:.2f}, Commission: ${trade.commission:.2f}, "
                     f"Cash: ${portfolio.cash:.2f})")

    def _calculate_metrics(self, portfolio_values: pd.Series, trades: List[Trade]) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            portfolio_values: Mark-to-market portfolio value per bar
            trades: Executed trades

        Returns:
            Dictionary of performance metrics
        """
        metrics: Dict[str, float] = {}

        # Calculate returns
        returns = portfolio_values.pct_change().dropna()

        # Max drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min() * 100

        # Sharpe ratio, annualised with the data's own bar frequency
        periods_per_year = self.resolve_periods_per_year(portfolio_values.index)
        if len(returns) > 1 and returns.std() > 0:
            metrics['sharpe_ratio'] = np.sqrt(periods_per_year) * returns.mean() / returns.std()
        else:
            metrics['sharpe_ratio'] = 0.0

        # Every trade statistic comes from the same FIFO pairing routine
        metrics.update(self._calculate_trade_statistics(trades))

        return metrics

    def resolve_periods_per_year(self, index: pd.Index) -> float:
        """
        Resolve the annualisation factor for the Sharpe ratio.

        Args:
            index: Index of the portfolio value series

        Returns:
            The explicit `periods_per_year` override if one was given, otherwise
            a value inferred from the index frequency.
        """
        if self.periods_per_year is not None:
            return float(self.periods_per_year)
        return self._infer_periods_per_year(index)

    @classmethod
    def _infer_periods_per_year(cls, index: pd.Index) -> float:
        """
        Infer how many bars of this data make up one year.

        One bar per session on a market that skips weekends gives ~252 bars a
        year; on a market that trades every day it gives 365. Anything coarser
        than one bar per day is counted on the calendar instead, because a weekly
        or monthly bar spans whole weeks regardless of whether the market is open
        at weekends: weekly -> ~52, monthly -> ~12. Intraday data is scaled by the
        observed number of bars per day, so hourly crypto yields 24 * 365 = 8760
        and hourly equity data roughly 7 * 252.

        Args:
            index: Index of the portfolio value series

        Returns:
            Number of bars per year; falls back to 252 when the index carries no
            usable frequency information.
        """
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
            return cls._DEFAULT_PERIODS_PER_YEAR

        deltas = index.to_series().diff().dropna()
        deltas = deltas[deltas > pd.Timedelta(0)]
        if deltas.empty:
            return cls._DEFAULT_PERIODS_PER_YEAR

        median_seconds = deltas.median().total_seconds()
        if median_seconds <= 0:
            return cls._DEFAULT_PERIODS_PER_YEAR

        # Weekend bars mean a 24/7 market, so a year has 365 sessions, not 252.
        trades_at_weekends = bool(index.dayofweek.isin([5, 6]).any())
        days_per_year = 365.0 if trades_at_weekends else cls._DEFAULT_PERIODS_PER_YEAR

        seconds_per_day = 86400.0
        if median_seconds >= seconds_per_day:
            days_per_bar = median_seconds / seconds_per_day
            if days_per_bar <= 1.0:
                # One bar per session: the session count is the trading calendar.
                return days_per_year
            # Coarser than daily: the spacing is calendar time, so scaling the
            # 252-day trading calendar by it would undercount (a weekly equity
            # bar is 7 calendar days, not 7 trading days).
            return 365.0 / days_per_bar

        # Intraday: count how many bars a typical day actually contains.
        bars_per_day = float(index.normalize().value_counts().median())
        return bars_per_day * days_per_year

    def _execute_buy_trade(self, timestamp: pd.Timestamp, symbol: str, price: float,
                           position_size: float, available_cash: float,
                           bar_volume: Optional[float] = None) -> Optional[Trade]:
        """
        Execute a buy trade if conditions are met.

        The cash budget is solved against the price the order *fills* at, not
        against the reference price: sizing on the reference price and paying
        slippage on top would spend cash the portfolio does not have, and could
        drive it negative on a full-capital order.

        The order's market footprint is measured at the reference price, i.e.
        before slippage shrinks the quantity. That is deliberate: it is the larger
        of the two candidate sizes, so a size-dependent impact term is never
        under-charged, and the sizing stays one pass instead of a fixed point.

        Args:
            timestamp: Bar timestamp of the fill
            symbol: Traded symbol
            price: Reference price the fill is benchmarked against
            position_size: Fraction of available cash to deploy
            available_cash: Cash available for the trade
            bar_volume: Volume traded on the execution bar, for liquidity-aware
                cost models

        Returns:
            The Trade, or None if it fails the minimum order value / cash checks,
            or if the bar cannot absorb any of the order
        """
        # Calculate max investment accounting for commission
        max_investment_with_commission = available_cash * position_size
        if max_investment_with_commission <= 0:
            return None

        # Solve for trade_value where trade_value + (trade_value * commission) = max_investment
        trade_value = max_investment_with_commission / (1 + self.commission)

        request = FillRequest(side=1, reference_price=price,
                              quantity=trade_value / price, bar_volume=bar_volume,
                              timestamp=timestamp)

        # Liquidity before price: a bar that cannot absorb the order has no fill
        # price to quote, so asking for one would be asking a nonsense question.
        fillable = self._fillable_quantity(request, symbol, 'BUY')
        if fillable <= 0:
            return None

        fill_price = self.cost_model.fill_price(request)

        shares_to_buy = trade_value / fill_price
        if shares_to_buy > fillable:
            # A truncated order buys fewer units and therefore spends less cash.
            self._log_partial_fill(request, shares_to_buy, fillable, symbol, 'BUY')
            shares_to_buy = fillable
            trade_value = shares_to_buy * fill_price

        commission_cost = trade_value * self.commission
        total_cost = trade_value + commission_cost

        # Check minimum order value and sufficient cash
        if trade_value >= self.min_order_value and available_cash >= total_cost:
            return Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=TradeSide.BUY,
                price=fill_price,
                quantity=shares_to_buy,
                value=trade_value,
                commission=commission_cost,
                slippage_cost=(fill_price - price) * shares_to_buy
            )
        return None

    def _fillable_quantity(self, request: FillRequest, symbol: str,
                           action: str) -> float:
        """
        How many units the cost model will let this bar absorb.

        Args:
            request: The order being priced
            symbol: Traded symbol, for the log message
            action: 'BUY' or 'SELL', for the log message

        Returns:
            The cap - ``math.inf`` for models that do not limit size, and 0.0 for
            a bar that cannot absorb anything, which is logged rather than passed
            over in silence
        """
        fillable = float(self.cost_model.max_fillable_quantity(request))
        if fillable > 0:
            return fillable

        logging.warning(
            f"ORDER NOT FILLED: {action} {symbol} at {request.timestamp}: the bar "
            f"traded {request.bar_volume} and the cost model reports it cannot "
            f"absorb any part of the order."
        )
        return 0.0

    def _log_partial_fill(self, request: FillRequest, wanted: float, fillable: float,
                          symbol: str, action: str) -> None:
        """
        Report an order truncated to the liquidity of its bar.

        A capped order is reduced and still executed: a reported partial fill is
        information, a silently dropped order is a hole in the trade log.

        Args:
            request: The order being priced
            wanted: Units the engine wanted to trade
            fillable: Units the bar can absorb
            symbol: Traded symbol
            action: 'BUY' or 'SELL'
        """
        logging.warning(
            f"PARTIAL FILL: {action} {symbol} truncated from {wanted:.6f} to "
            f"{fillable:.6f} units at {request.timestamp}; the bar traded "
            f"{request.bar_volume} and the cost model caps how much of it one "
            f"order may take."
        )

    def _execute_sell_trade(self, timestamp: pd.Timestamp, symbol: str, price: float,
                            position_size: float, current_position: float,
                            bar_volume: Optional[float] = None) -> Optional[Trade]:
        """
        Execute a sell trade if conditions are met.

        Args:
            timestamp: Bar timestamp of the fill
            symbol: Traded symbol
            price: Reference price the fill is benchmarked against
            position_size: Fraction of the position to liquidate
            current_position: Units currently held
            bar_volume: Volume traded on the execution bar, for liquidity-aware
                cost models

        Returns:
            The Trade, or None if it fails the minimum order value check, or if
            the bar cannot absorb any of the order
        """
        shares_to_sell = current_position * position_size
        if shares_to_sell <= 0:
            return None

        request = FillRequest(side=-1, reference_price=price, quantity=shares_to_sell,
                              bar_volume=bar_volume, timestamp=timestamp)

        # Liquidity before price, as for a buy.
        fillable = self._fillable_quantity(request, symbol, 'SELL')
        if fillable <= 0:
            return None

        fill_price = self.cost_model.fill_price(request)
        if shares_to_sell > fillable:
            self._log_partial_fill(request, shares_to_sell, fillable, symbol, 'SELL')
            shares_to_sell = fillable

        trade_value = shares_to_sell * fill_price

        # Check minimum order value
        if trade_value >= self.min_order_value and shares_to_sell > 0:
            return Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=TradeSide.SELL,
                price=fill_price,
                quantity=shares_to_sell,
                value=trade_value,
                commission=trade_value * self.commission,
                slippage_cost=(price - fill_price) * shares_to_sell
            )
        return None

    def pair_trades(self, trades: List[Trade]) -> List[RoundTrip]:
        """
        Pair buy and sell executions into realised round trips, FIFO.

        Thin wrapper around :func:`niffler.backtesting.round_trip.pair_trades`,
        which is the single source of truth for trade-level P&L. It lives at
        module level so exporters can reconcile their own position documents with
        the engine's metrics without instantiating an engine.

        Args:
            trades: Chronological list of executions

        Returns:
            List of realised round trips (unclosed buys are simply not included)
        """
        return pair_trades(trades)

    def _calculate_trade_statistics(self, trades: List[Trade]) -> Dict[str, float]:
        """
        Derive every trade statistic from the FIFO round trips.

        Args:
            trades: Chronological list of executions

        Returns:
            Dictionary containing win_rate, profit_factor, average_win,
            average_loss, largest_win, largest_loss, num_winning_trades and
            num_losing_trades. Losses are reported as positive magnitudes.
        """
        empty = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'num_winning_trades': 0,
            'num_losing_trades': 0,
        }
        if not trades:
            return empty

        round_trips = self.pair_trades(trades)
        if not round_trips:
            return empty

        wins = [rt.pnl for rt in round_trips if rt.is_win]
        losses = [abs(rt.pnl) for rt in round_trips if rt.is_loss]

        gross_profit = sum(wins)
        gross_loss = sum(losses)

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float('inf')  # All wins, no losses
        else:
            profit_factor = 0.0  # No trades or all break-even

        return {
            'win_rate': len(wins) / len(round_trips) * 100,
            'profit_factor': profit_factor,
            'average_win': sum(wins) / len(wins) if wins else 0.0,
            'average_loss': gross_loss / len(losses) if losses else 0.0,
            'largest_win': max(wins) if wins else 0.0,
            'largest_loss': max(losses) if losses else 0.0,
            'num_winning_trades': len(wins),
            'num_losing_trades': len(losses),
        }

    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """
        Percentage of round trips that closed at a profit net of commission.

        Args:
            trades: Chronological list of executions

        Returns:
            Win rate in percent
        """
        return self._calculate_trade_statistics(trades)['win_rate']

    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """
        Gross profit divided by gross loss, both net of commission.

        Args:
            trades: Chronological list of executions

        Returns:
            Profit factor; inf when there are wins but no losses, 0.0 when there
            is nothing to measure
        """
        return self._calculate_trade_statistics(trades)['profit_factor']

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
