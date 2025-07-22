# Risk Management

## Overview

Niffler includes a comprehensive risk management framework designed to control position sizing, manage stop losses, and enforce portfolio-level risk limits. The system is built around an abstract base class that allows for different risk management strategies.

## Risk Management Architecture

### Core Components

#### BaseRiskManager
Abstract base class defining the risk management interface:

**Key Data Classes:**
- `PositionInfo`: Tracks position details (symbol, size, entry price, stop loss, timestamp, P&L)
- `RiskDecision`: Contains evaluation results (position size, stop loss price, risk amount, trade approval, reason)

**Abstract Methods (must be implemented):**
- `calculate_position_size()`: Determines position size for new trades
- `calculate_stop_loss()`: Calculates stop loss price for positions
- `should_close_position()`: Evaluates when to close existing positions

**Portfolio Management Methods:**
- `update_position_state()`: Updates or adds position tracking
- `clear_position()`: Removes closed positions from tracking
- `get_position_info()`: Retrieves position details by symbol
- `get_total_exposure()`: Calculates total portfolio exposure

**Default Risk Limits:**
- Maximum position size: 100% (1.0) of portfolio
- Maximum risk per trade: 10% (0.1) of portfolio
- Maximum total exposure: 200% (2.0) with leverage
- Maximum concurrent positions: 10

### Available Risk Managers

#### Fixed Risk Manager ✅ FULLY IMPLEMENTED

The `FixedRiskManager` provides predictable risk management using fixed percentages:

**Constructor Parameters:**
- `position_size_pct`: 10% (0.1) - Fixed position size per trade
- `stop_loss_pct`: 5% (0.05) - Fixed stop loss from entry price
- `max_positions`: 5 - Maximum concurrent positions  
- `max_risk_per_trade`: 2% (0.02) - Maximum portfolio risk per trade

**Position Sizing Logic:**
- **Buy signals (signal=1)**: Returns `position_size_pct`
- **Sell signals (signal=-1)**: Returns current position size (full closure)
- **Hold signals (signal=0)**: Returns 0 (no action)

**Stop Loss Calculation:**
- **Long positions**: `entry_price * (1 - stop_loss_pct)`
- **Short positions**: `entry_price * (1 + stop_loss_pct)`

**Risk Validation:**
The manager performs comprehensive validation:
- Ensures `position_size_pct * stop_loss_pct ≤ max_risk_per_trade`
- Warns about very conservative settings (< 0.1% risk per trade)
- Warns about aggressive stop losses (> 20%)
- Alerts if maximum portfolio risk exceeds 50%

**Example Configuration:**
```python
from niffler.risk.fixed_risk_manager import FixedRiskManager

risk_manager = FixedRiskManager(
    position_size_pct=0.1,      # 10% of portfolio per trade
    stop_loss_pct=0.05,         # 5% stop loss from entry
    max_positions=5,            # Maximum 5 concurrent positions
    max_risk_per_trade=0.02     # Maximum 2% portfolio risk per trade
)
```

#### Kelly Risk Manager ❌ FRAMEWORK ONLY

The `KellyRiskManager` framework exists but **core calculations are not implemented**:

**Planned Parameters:**
- `lookback_periods`: 50 - Historical periods for Kelly calculation
- `max_kelly_fraction`: 25% (0.25) - Cap on Kelly-suggested position size
- `stop_loss_pct`: 5% (0.05) - Fallback fixed stop loss
- `min_trades_for_kelly`: 10 - Minimum trades needed for calculation
- `fractional_kelly`: 100% (1.0) - Fraction of full Kelly to use
- `max_positions`: 5 - Maximum concurrent positions

**Implementation Status:**
- ❌ `calculate_position_size()`: Raises `NotImplementedError`
- ❌ `calculate_stop_loss()`: Raises `NotImplementedError`  
- ❌ `should_close_position()`: Raises `NotImplementedError`

**Planned Kelly Formula:** `f* = (bp - q) / b`
- f* = optimal fraction of capital to risk
- b = odds (average win / average loss ratio)
- p = probability of winning
- q = probability of losing (1 - p)

## Risk Management Integration

### Backtesting Integration

The risk management system integrates with the backtesting engine through:

#### Trade Evaluation Process
1. **Signal Generation**: Strategy generates buy/sell/hold signals
2. **Risk Assessment**: Risk manager evaluates trade via `evaluate_trade()` method
3. **Decision Making**: Returns `RiskDecision` object with:
   - `allow_trade`: Whether trade should proceed
   - `position_size`: Calculated position size fraction
   - `stop_loss_price`: Stop loss price for the position
   - `risk_amount`: Estimated risk amount for the trade
   - `reason`: Explanation for the decision

#### Position Management
- **State Tracking**: Risk manager maintains real-time position information
- **Portfolio Monitoring**: Tracks total exposure and position count
- **Stop Loss Monitoring**: Evaluates positions for stop loss triggers via `should_close_position()`
- **Position Updates**: Updates tracking when positions change

#### Risk Controls Applied
- **Position Size Validation**: Ensures position sizes within limits
- **Exposure Limits**: Prevents excessive total portfolio exposure
- **Stop Loss Enforcement**: Automatically closes positions hitting stop losses
- **Trade Blocking**: Can prevent trades that violate risk parameters

### Strategy Integration

Risk managers attach to strategies for seamless integration:

```python
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
from niffler.risk.fixed_risk_manager import FixedRiskManager

# Create strategy with risk management
strategy = SimpleMAStrategy(short_window=10, long_window=30)
risk_manager = FixedRiskManager(
    position_size_pct=0.1,
    stop_loss_pct=0.05,
    max_positions=5
)
strategy.risk_manager = risk_manager
```

## Risk Metrics and Reporting

### Portfolio Risk Metrics

The `FixedRiskManager` provides comprehensive risk reporting:

**Position Utilization:**
- Current positions vs maximum allowed
- Position utilization rate percentage

**Risk Exposure:**
- Estimated risk per trade
- Current total risk exposure  
- Maximum theoretical portfolio risk

**Risk Efficiency:**
- Risk efficiency ratio
- Capital utilization effectiveness

**Portfolio Summary:**
Available via `get_portfolio_summary()` method:
- Number of current positions
- Total portfolio exposure
- Available position capacity
- Current risk utilization

### Risk Validation Warnings

The system provides automatic validation with warnings:

**Conservative Settings:**
- Warns when risk per trade < 0.1% (may be overly conservative)
- Suggests considering higher position sizes for meaningful profits

**Aggressive Settings:**
- Warns when stop losses > 20% (may be too wide)
- Alerts when maximum portfolio risk > 50% (high risk exposure)

**Configuration Issues:**
- Validates parameter relationships make sense
- Ensures risk calculations are mathematically sound

## Best Practices

### Fixed Risk Manager Usage

**Conservative Approach:**
- Position size: 5-10% per trade
- Stop loss: 2-5% from entry
- Max positions: 3-5 concurrent
- Risk per trade: 1-2% of portfolio

**Moderate Approach:**
- Position size: 10-15% per trade  
- Stop loss: 5-8% from entry
- Max positions: 5-8 concurrent
- Risk per trade: 2-3% of portfolio

**Aggressive Approach:**
- Position size: 15-25% per trade
- Stop loss: 8-15% from entry
- Max positions: 8-10 concurrent
- Risk per trade: 3-5% of portfolio

### Parameter Selection Guidelines

**Position Sizing Considerations:**
- Consider strategy win rate and average trade duration
- Higher win rates can support larger position sizes
- Account for correlation between positions

**Stop Loss Setting:**
- Base on asset volatility and strategy characteristics
- Tighter stops for momentum strategies
- Wider stops for mean reversion strategies

**Portfolio Limits:**
- Set maximum positions based on diversification needs
- Consider correlation between different positions
- Account for available capital and margin requirements

### Risk Management Workflow

1. **Strategy Development**: Create and test strategy without risk management
2. **Risk Manager Selection**: Choose appropriate risk management approach
3. **Parameter Optimization**: Optimize both strategy and risk parameters together
4. **Backtesting**: Test with risk management enabled
5. **Risk Analysis**: Review risk metrics and portfolio performance
6. **Parameter Refinement**: Adjust risk parameters based on results

## Future Enhancements

### Kelly Risk Manager Implementation
Priority enhancements for Kelly Criterion implementation:
- Integration with backtest trade history
- Dynamic Kelly calculation based on rolling performance
- Fractional Kelly options (quarter-Kelly, half-Kelly)
- ATR-based stop loss calculations

### Advanced Risk Features
Potential future enhancements:
- **Volatility-Based Stops**: ATR-based dynamic stop losses
- **Correlation Monitoring**: Position correlation analysis
- **Market Regime Detection**: Adaptive risk based on market conditions
- **Maximum Adverse Excursion**: MAE-based stop loss optimization