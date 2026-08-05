from typing import Optional

from .trade import Trade, TradeSide


class Portfolio:
    """
    Mutable portfolio state for a single-symbol backtest.

    Owns the state that used to live as five parallel locals inside
    ``BacktestEngine.run_backtest`` (cash, position, entry price, stop loss and
    position side) together with the buy/sell and mark-to-market transitions.
    Keeping them in one object is what makes it impossible to update one of them
    and forget the others.

    Attributes:
        initial_capital: Starting cash
        cash: Free cash, already net of every commission paid so far
        position: Units currently held (positive for long)
        entry_price: Fill price of the currently open position, None when flat
        stop_loss: Stop-loss price of the open position, None when flat or unset
        side: 1 for a long position, -1 for short, 0 when flat
    """

    #: Positions smaller than this are treated as dust, i.e. effectively flat.
    POSITION_TOLERANCE = 0.005

    def __init__(self, initial_capital: float) -> None:
        """
        Initialize the portfolio.

        Args:
            initial_capital: Starting cash amount

        Raises:
            ValueError: If initial_capital is not positive
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0.0
        self.entry_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.side = 0

    @property
    def is_flat(self) -> bool:
        """True when the remaining position is dust or zero."""
        return abs(self.position) < self.POSITION_TOLERANCE

    def market_value(self, price: float) -> float:
        """
        Mark the portfolio to market.

        Args:
            price: Current price of the traded symbol

        Returns:
            Cash plus the market value of the open position
        """
        return self.cash + self.position * price

    def position_fraction(self, price: float) -> float:
        """
        Position exposure as a fraction of portfolio value.

        Note this is a *value* ratio: units are converted to currency first.

        Args:
            price: Current price of the traded symbol

        Returns:
            position value / portfolio value, or 0.0 if the portfolio is worthless
        """
        portfolio_value = self.market_value(price)
        if portfolio_value <= 0:
            return 0.0
        return (self.position * price) / portfolio_value

    def unrealized_pnl(self, price: float) -> float:
        """
        Unrealised profit/loss of the open position.

        Args:
            price: Current price of the traded symbol

        Returns:
            Unrealised P&L in account currency, 0.0 when flat
        """
        if self.entry_price is None or self.position == 0.0:
            return 0.0
        direction = self.side if self.side != 0 else 1
        return (price - self.entry_price) * self.position * direction

    def apply_buy(self, trade: Trade) -> None:
        """
        Apply a buy execution: pay value plus commission, add units.

        Args:
            trade: Executed buy trade

        Raises:
            ValueError: If the trade is not a buy
        """
        if trade.side != TradeSide.BUY:
            raise ValueError(f"apply_buy requires a BUY trade, got {trade.side}")

        self.cash -= trade.value + trade.commission
        self.position += trade.quantity

    def apply_sell(self, trade: Trade) -> None:
        """
        Apply a sell execution: receive value minus commission, remove units.

        Args:
            trade: Executed sell trade

        Raises:
            ValueError: If the trade is not a sell
        """
        if trade.side != TradeSide.SELL:
            raise ValueError(f"apply_sell requires a SELL trade, got {trade.side}")

        self.cash += trade.value - trade.commission
        self.position -= trade.quantity

    def open_position(self, entry_price: float, stop_loss: Optional[float],
                      side: int = 1) -> None:
        """
        Record the risk-tracking state of a newly opened position.

        Args:
            entry_price: Fill price of the entry
            stop_loss: Stop-loss price, or None when no stop is set
            side: 1 for long, -1 for short
        """
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.side = side

    def add_to_position(self, entry_price: float, quantity: float,
                        stop_loss: Optional[float] = None) -> None:
        """
        Scale into the position already open, keeping its cost basis intact.

        Must be called *before* :meth:`apply_buy` records the new units, because
        the weighted average is taken against the quantity held so far. The entry
        price becomes the quantity-weighted average of the old and the new lot,
        so :meth:`unrealized_pnl` and every stop-distance calculation keep
        measuring from the real cost basis rather than from the last fill.

        The stop is never weakened: a ``None`` stop leaves the existing one
        armed, and a supplied stop is only adopted when it is tighter than the
        current one (higher for a long, lower for a short).

        Args:
            entry_price: Fill price of the added lot
            quantity: Units added by the new fill (must be positive)
            stop_loss: Stop-loss price of the new order, or None when it carries
                no stop

        Raises:
            ValueError: If quantity is not positive
        """
        if quantity <= 0:
            raise ValueError(f"add_to_position requires a positive quantity, got {quantity}")

        if self.entry_price is None or self.position <= 0:
            # Nothing to average against - this is really an opening fill.
            self.open_position(entry_price, stop_loss, side=self.side or 1)
            return

        total_quantity = self.position + quantity
        self.entry_price = (
            (self.entry_price * self.position + entry_price * quantity) / total_quantity
        )

        if stop_loss is None:
            return
        if self.stop_loss is None:
            self.stop_loss = stop_loss
        elif self.side >= 0:
            self.stop_loss = max(self.stop_loss, stop_loss)
        else:
            self.stop_loss = min(self.stop_loss, stop_loss)

    def close_position(self) -> None:
        """Clear the risk-tracking state after a position has been closed."""
        self.entry_price = None
        self.stop_loss = None
        self.side = 0
