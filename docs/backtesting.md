# Backtesting

## Backtest Script Usage

```bash
python scripts/backtest.py --data <data_file> --strategy <strategy_name> [--initial_capital <amount>] [--commission <rate>] [--clean]
```

**Arguments:**
- `--data`: Path to CSV file containing historical market data
- `--strategy`: Strategy to use (currently supports `simple_ma`)
- `--initial_capital`: (Optional) Starting capital amount, default: 10000
- `--commission`: (Optional) Commission rate per trade, default: 0.001 (0.1%)
- `--clean`: (Optional) Apply data cleaning pipeline before backtesting

## Examples

**Basic backtest with Simple Moving Average strategy:**
```bash
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --initial_capital 10000 --commission 0.001
```

**Backtest with automatic data cleaning:**
```bash
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --clean
```

## Backtesting Framework

### Core Components

#### BacktestEngine
The `BacktestEngine` orchestrates the entire backtesting process with realistic trading simulation:

**Key Features:**
- Portfolio management with cash and position tracking
- Commission-based trade execution
- Risk management integration
- Comprehensive input validation
- Detailed performance metrics calculation

**Configuration Parameters:**
- `initial_capital`: Starting capital (default: 10000)
- `commission`: Commission rate per trade (default: 0.001 = 0.1%)
- `min_order_value`: Minimum trade value to execute (default: 1.0)

#### Trade Execution Logic

**Buy Trade Execution:**
- Calculates maximum investment considering commission costs
- Solves for trade value where `trade_value + (trade_value * commission) = available_cash * position_size`
- Validates against minimum order value and available cash
- Updates cash and position accordingly

**Sell Trade Execution:**
- Calculates shares to sell based on position size and current holdings
- Validates against minimum order value and available position
- Handles partial position closures
- Updates cash and position accordingly

#### Performance Metrics
The `BacktestResult` provides comprehensive performance analysis:
- **Return Metrics**: Total return, total return percentage
- **Risk Metrics**: Sharpe ratio (annualized), maximum drawdown
- **Trade Statistics**: Win rate, total trades count
- **Portfolio Analytics**: Equity curve tracking

### Risk Management Integration

The backtesting engine integrates seamlessly with risk management systems:

#### Position Sizing
- Risk manager evaluates each trade signal in real-time
- Considers current portfolio state and exposure limits
- Calculates appropriate position size based on risk parameters
- Can block trades that violate risk limits

#### Stop Loss Management
- Automatic stop loss monitoring for open positions
- Risk manager determines when positions should be closed
- Stop loss triggers are checked before processing new signals
- Provides detailed logging of stop loss executions

#### Portfolio State Tracking
- Real-time tracking of portfolio value and position sizes
- Updates risk manager with current position states
- Clears position tracking when positions are fully closed
- Maintains position fraction calculations for risk assessment

### Strategy Framework

#### BaseStrategy Integration
All trading strategies inherit from `BaseStrategy` and provide:
- `generate_signals()` method that returns DataFrame with signal column
- Data validation through `validate_data()` method
- Optional risk manager integration
- Strategy-specific parameters

#### Signal Processing
The backtest engine processes signals with the following logic:
- **Signal = 1**: Buy signal (open/increase long position)
- **Signal = -1**: Sell signal (close/reduce long position) 
- **Signal = 0**: Hold signal (no action)

#### Available Strategies

##### Simple Moving Average (simple_ma)
A trend-following strategy using moving average crossovers:

**Parameters:**
- `short_window`: Fast moving average period (default: 10)
- `long_window`: Slow moving average period (default: 30)

**Signal Logic:**
- **Buy**: When short MA crosses above long MA
- **Sell**: When short MA crosses below long MA

### Realistic Trading Simulation

#### Commission Handling
- Configurable commission rates applied to all trades
- Commission calculated as percentage of trade value
- Reduces available cash for buy trades
- Reduces proceeds for sell trades

#### Minimum Order Validation
- Respects minimum order value constraints
- Prevents execution of trades below threshold
- Realistic simulation of exchange requirements

#### Position Management
- Tracks exact position quantities (shares/units)
- Validates sell orders against available position
- Handles partial position closures
- Prevents overselling scenarios

#### Data Validation
Comprehensive input validation ensures data quality:
- **Required columns**: open, high, low, close, volume
- **Data types**: All price/volume data must be numeric
- **Value constraints**: Prices > 0, volume ≥ 0
- **OHLC relationships**: High ≥ Low, Open/Close within High/Low range
- **Index requirements**: DatetimeIndex in ascending order

### Performance Calculation

#### Sharpe Ratio
- Calculated using daily returns with 252 trading days assumption
- Formula: `sqrt(252) * mean(returns) / std(returns)`
- Returns 0.0 if insufficient data or zero standard deviation

#### Maximum Drawdown
- Calculated as maximum percentage decline from running peak
- Uses expanding maximum to track all-time highs
- Expressed as negative percentage

#### Win Rate
- Based on properly paired buy/sell trades using FIFO matching
- Handles partial position closures correctly
- Only counts completed round-trip trades

### Output and Reporting

#### Console Logging
Detailed execution logging includes:
- Trade-by-trade execution with timestamps and prices
- Commission costs and cash updates
- Stop loss triggers and reasons
- Position size and portfolio value tracking

#### BacktestResult Object
Contains complete backtest results:
- Strategy and symbol identification
- Date range and capital information
- Performance metrics and trade history
- Portfolio value time series
- Trade statistics and analysis

### Integration Features

#### Data Preprocessing Integration
- Optional data cleaning via `--clean` flag
- Applies full preprocessing pipeline if requested
- Automatic column detection and datetime parsing
- Validates data format before backtesting

#### Risk Management Integration
- Optional risk manager attachment to strategies
- Real-time portfolio state evaluation
- Dynamic position sizing and stop loss management
- Risk-based trade blocking and position monitoring