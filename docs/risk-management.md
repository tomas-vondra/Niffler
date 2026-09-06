# Risk Management

## Overview

Niffler includes a comprehensive risk management framework designed to control position sizing, manage stop losses, and enforce portfolio-level risk limits. The system is built around an abstract base class that allows for different risk management strategies.

## Risk Management Architecture

### Core Components

#### BaseRiskManager
Abstract base class defining the risk management interface:

**Key Data Classes:**
- `PortfolioSnapshot` (`niffler/risk/contract.py`): the portfolio state a manager is allowed
  to see — `open_positions`, `total_exposure`, `current_position`. Frozen, and built at the
  call site by `Portfolio.risk_snapshot(price)`
- `RiskDecision`: Contains evaluation results (position size, stop loss price, risk amount, trade approval, reason)

**Abstract Methods (must be implemented):**
- `calculate_position_size()`: Determines position size for new trades
- `calculate_stop_loss()`: Calculates stop loss price for positions
- `should_close_position()`: Evaluates when to close existing positions

**Risk managers hold no position state.** They used to keep a
`Dict[str, PositionInfo]` that the engine mutated through `update_position_state()` /
`clear_position()`. Those methods are gone, along with `get_position_info()`,
`get_total_exposure()` and `get_portfolio_summary()`. Position state belongs to
`niffler/backtesting/portfolio.py`, which already owned cash, position, entry price, stop
and side; the manager is handed a `PortfolioSnapshot` on every `evaluate_trade()` call.

That statefulness is why a risk manager could not be threaded into walk-forward or Monte
Carlo validation. Folds run in parallel and each fold is an independent hypothetical
history, so one shared manager carried the first fold's open position into the second, where
`max_positions` vetoed its first entry. `tests/test_backtesting/test_risk_state_isolation.py`
is the regression suite.

**Default Risk Limits:**
- Maximum position size: 100% (1.0) of portfolio
- Maximum risk per trade: 10% (0.1) of portfolio
- Maximum total exposure: 200% (2.0) with leverage
- Maximum concurrent positions: 10

#### The engine ↔ risk contract

`niffler/risk/contract.py` holds the two methods the backtest engine actually calls, as a
`runtime_checkable` `RiskManager` Protocol:

- `evaluate_trade(signal, current_price, portfolio_value, historical_data, portfolio)`
- `should_close_position(current_price, entry_price, stop_loss_price, signal, unrealized_pnl)`

`BacktestEngine.run_backtest()` checks the manager in force against it before the first bar
and raises `TypeError` naming the missing methods. The check is `isinstance` on a Protocol,
so it verifies that the attributes exist, not that their signatures match — a renamed
method now fails at the start of the run rather than at the first signal.

### Where the manager comes from

The engine owns one (`RunConfig.risk_manager`, set by `BacktestEngine.from_config`) and
falls back to `strategy.risk_manager` when it has none. The fallback is the original route
and still works; the engine-owned one is what lets `optimize.py`, `analyze.py`,
`compare.py` and `screen.py` run the risk layer at all, because those build their strategy
objects themselves and a manager attached to a strategy could never reach them. Configuring
both is a `ValueError` unless they are the same object — a silent preference would drop one
configuration from a run that still names it.

The dependency runs one way: `niffler/backtesting` imports `niffler/risk`, never the
reverse. That is why `PortfolioSnapshot` lives in `niffler/risk` and why the engine is
handed a snapshot rather than the `Portfolio` object itself.

### The risk manager registry

`niffler/risk/registry.py` is the single name→class map, the counterpart of
`niffler/strategies/registry.py`. Adding a risk manager is **one entry in
`RISK_MANAGER_CLASSES`**: the shared `--risk-manager` choices derive from
`get_available_risk_managers()`, and construction goes through `create_risk_manager()`, so
no script needs an `if` branch.

`'none'` is a name, not a class — it maps to `None`. The engine branches on
`risk_manager is None` to skip stop processing entirely, so a do-nothing null object would
make "no risk management" indistinguishable from "a manager that permits everything".

A parameter the chosen manager does not accept is a `ValueError` naming the ones it does,
checked with `inspect.signature`, never a silent drop:

```
Risk manager 'fixed' (FixedRiskManager) does not accept lookback_periods.
Accepted: max_positions, max_risk_per_trade, position_size_pct, stop_loss_pct
```

`backtest.py`'s four risk flags (`--max-position-size`, `--stop-loss-pct`,
`--max-positions`, `--max-risk-per-trade`) are `FixedRiskManager`-shaped and are still
passed positionally into the registry. A `--risk-params` JSON flag mirroring `--params` is
the follow-up; until it lands, a manager with a different parameter set needs those flags
generalised.

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
- **Sell signals (signal=-1)**: Returns current position size (full closure). **The backtest
  engine no longer applies this to exits** — see
  [Entries vs exits](#entries-vs-exits-behaviour-change) below
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

#### Kelly Risk Manager ❌ NOT IMPLEMENTED — DO NOT USE

`KellyRiskManager` is a **stub**. The class and its constructor parameters exist; all three
abstract methods raise `NotImplementedError`, so attaching it to a strategy will crash the
first time the engine consults it. It is deliberately **not registered** in
`RISK_MANAGER_CLASSES`, so `--risk-manager kelly` is rejected by argparse rather than
loading data and then raising at the first signal. `FixedRiskManager` is the only working
risk manager.

Registering it is now one line, so what is missing is design, not plumbing. Three questions
block it, each of which changes something outside the class (the full argument is in the
class docstring):

1. **The contract carries prices, not outcomes.** `evaluate_trade()` receives bars and a
   `PortfolioSnapshot`; `p` and `b` are properties of realised round trips, which nothing
   hands over. Supplying them needs a `TradeOutcome`-shaped type in `niffler/risk` (it
   cannot be `niffler.backtesting.round_trip.RoundTrip` without inverting the layering) plus
   the engine feeding `pair_trades()` output. The snapshot is where it would land — one
   added field, not another signature break.
2. **The bootstrap deadlock.** Below the minimum-sample gate the honest answer is to stand
   aside, so it never trades, so it never accumulates the round trips the gate waits for.
   The only non-fabricating way out is an explicit opt-in seed fraction that the console
   labels as *not* Kelly, and choosing its default is a research decision.
3. **The gate value contradicts itself.** `min_trades_for_kelly` defaults to 10 while
   `niffler.backtesting.significance.DEFAULT_MIN_TRADES` is 30 and the framework already
   refuses a verdict below 30.

The parameters below describe what it *would* take, not what it does:

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

#### Entries vs exits (behaviour change)

`RiskDecision.position_size` is an **entry** sizing target, expressed as a fraction of
portfolio *value*. It is applied to buys only.

It used to be applied to sells as well, where the engine reinterpreted it as a fraction of
the held *units*. Because `FixedRiskManager.calculate_position_size()` returns the current
position for a sell, and the engine passed a portfolio *value* fraction as
`current_position`, a risk-managed exit liquidated only a small slice of the position. Over
repeated exit signals the strategy never actually got flat — it degenerated into
buy-and-hold with exposure creep, which flattered results in a rising market.

Exits now use the strategy's own `position_size` column (a fraction of held units). A risk
manager can still **veto** an exit through `allow_trade`, but it cannot resize one.

Every risk-managed backtest with exit signals therefore changes: returns, drawdown, win rate
and trade counts all move, and positions now genuinely close.

#### Units versus value

`PortfolioSnapshot.current_position` and `.total_exposure` are **portfolio value
fractions**, computed on demand as
`Portfolio.position_fraction(price) = position * price / market_value(price)`. An all-in buy
therefore reports ~1.0, not ~0.01. Computing this on demand also removed a genuine
`UnboundLocalError` that fired when a risk manager was attached and the very first bar
produced a buy.

Exposure is valued **at the price passed in**, i.e. now. The old manager recorded the
fraction at each fill and never revalued it, so a position whose price had risen still
reported its entry-day weight and the `max_total_exposure` cap quietly stopped binding. That
is the one behaviour change a `--risk-manager fixed` run sees: the cap now binds where it
used to be evaded, which also blocks the scale-ins that were incidentally ratcheting the
stop upward (`add_to_position` only ever tightens a stop). Numbers for runs with **no** risk
manager — the default — are unchanged.

#### Position Management
- **State Ownership**: `Portfolio` holds the position; the manager is handed a snapshot
- **Portfolio Monitoring**: `max_positions` and `max_total_exposure` are checked against
  that snapshot, so they are evaluated against the run in progress and nothing else
- **Stop Loss Monitoring**: Evaluates positions for stop loss triggers via `should_close_position()`
- **Stop tightening only**: scaling into a position never weakens an existing stop. A buy
  order carrying no stop leaves the existing one armed; a supplied stop is adopted only if
  it is tighter. Previously a second buy overwrote the stop with `None` and left the
  position permanently unprotected
- **Stops are probed against the bar's traded range** (the low for a long, the high for a
  short), not just the execution price, and fill at `min(open, stop)` for a long so a
  gap-through fills at the open rather than at an unreachable stop price. A triggered stop
  that cannot execute because the residual is below `min_order_value` logs a WARNING instead
  of looking identical to "stop not hit"

#### Risk Controls Applied
- **Position Size Validation**: Ensures position sizes within limits
- **Exposure Limits**: Prevents excessive total portfolio exposure
- **Stop Loss Enforcement**: Automatically closes positions hitting stop losses
- **Trade Blocking**: Can prevent trades that violate risk parameters

### Library integration

Put the manager on the `RunConfig`. It then reaches every engine the run builds, including
the ones inside optimisation and analysis worker processes:

```python
from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.run_config import RunConfig
from niffler.risk.fixed_risk_manager import FixedRiskManager
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy

run_config = RunConfig(risk_manager=FixedRiskManager(
    position_size_pct=0.1,
    stop_loss_pct=0.05,
    max_positions=5
))
engine = BacktestEngine.from_config(run_config)
result = engine.run_backtest(SimpleMAStrategy(short_window=10, long_window=30), data)
```

Attaching one to a strategy (`strategy.risk_manager = ...`, or
`create_strategy(name, params, risk_manager=...)`) still works and is what the engine falls
back to, but it only affects backtests run through that strategy object.

## Risk Metrics and Reporting

### Portfolio Risk Metrics

The `FixedRiskManager` provides comprehensive risk reporting:

**Risk Exposure:**
- Estimated risk per trade
- Maximum theoretical portfolio risk

**Risk Efficiency:**
- Risk efficiency ratio
- Capital utilization effectiveness

`get_risk_metrics()` reports **configuration only**. Live position counts, current exposure
and utilisation percentages are gone with the position state that produced them — a
stateless manager cannot report a portfolio it does not hold. Ask `Portfolio` instead.

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

### Threading a risk manager into validation
Now unblocked by statelessness, but not done: `WalkForwardAnalyzer` and
`MonteCarloAnalyzer` still construct strategies with no risk manager, and `optimize.py`,
`analyze.py` and `compare.py` have no `--risk-manager` flag. That plumbing waits on the run
configuration work rather than being threaded through the current call sites.

### Kelly Risk Manager Implementation
Priority enhancements for Kelly Criterion implementation, in the order the blockers above
have to be cleared:
- A `TradeOutcome` type in `niffler/risk` plus realised round trips on the snapshot
- A decision on the insufficient-sample fallback (stand aside vs. an opt-in seed fraction)
- One minimum-sample answer shared with `significance.DEFAULT_MIN_TRADES`
- Fractional Kelly options (quarter-Kelly, half-Kelly) — the parameters already exist
- ATR-based stop loss calculations

### Advanced Risk Features
Potential future enhancements:
- **Volatility-Based Stops**: ATR-based dynamic stop losses
- **Correlation Monitoring**: Position correlation analysis
- **Market Regime Detection**: Adaptive risk based on market conditions
- **Maximum Adverse Excursion**: MAE-based stop loss optimization