# Backtesting

## Backtest Script Usage

```bash
python scripts/backtest.py --data <data_file> [--strategy <strategy_name>] [--capital <amount>] [--commission <rate>] [--clean] [...]
```

**Data and strategy:**
- `--data`, `-d`: Path to CSV file containing historical market data (**required**)
- `--strategy`, `-s`: Strategy to use, default: `simple_ma` (the only choice today)
- `--short-window`: Fast MA window for `simple_ma`, default: 10
- `--long-window`: Slow MA window for `simple_ma`, default: 30
- `--position-size`: Position size as a fraction of the portfolio, default: 1.0
- `--symbol`: Symbol identifier, default: extracted from the filename
- `--clean`: Apply the default data-cleaning pipeline before backtesting

**Backtest parameters:**
- `--capital`: Starting capital, default: 10000. (The flag is `--capital`, **not**
  `--initial_capital` — `analyze.py` and `optimize.py` use different spellings of their own.)
- `--commission`: Commission rate per trade, default: 0.001 (0.1%)
- `--min-order-value`: Minimum trade value to execute, default: 1.0

**Risk management:**
- `--risk-manager`: `none` (default) or `fixed`. The Kelly manager is a stub and is
  deliberately not offered
- `--max-position-size`: Fraction of portfolio per trade, default: 0.2
- `--stop-loss-pct`: Stop loss distance from entry, default: 0.05
- `--max-positions`: Maximum concurrent positions, default: 5
- `--max-risk-per-trade`: Maximum portfolio risk per trade, default: 0.02

**Transaction costs:**
- `--cost-model`: `none` (default), `fixed` or `volume` — see
  [Transaction costs](#transaction-costs)
- `--slippage-bps`: fixed model only, execution slippage in basis points, default 5
- `--half-spread-bps`: fixed and volume models, half the bid/ask spread in basis points,
  default 1
- `--impact-coefficient`: volume model only, coefficient on `sqrt(participation)` as a
  fraction of price, default 0.1
- `--max-participation`: volume model only, largest share of a bar's volume one order may
  take, default 0.1

A flag the selected model does not read is an **error**, not a silently ignored argument:
`--cost-model fixed --impact-coefficient 0.2` exits non-zero rather than charging no impact
while you believe otherwise.

**Export and logging:**
- `--exporters`: Comma-separated list — `console` (default), `csv`, `elasticsearch`
- `--csv-output-dir`: Directory for CSV output, default: current directory
- `--es-host`, `--es-port`, `--es-index-prefix`: Override the `ELASTICSEARCH_*` environment
  variables (see [Exporters](exporters.md))
- `--log-level`: `DEBUG` / `INFO` (default) / `WARNING` / `ERROR`

There is deliberately **no `--execution-timing` or `--periods-per-year` flag**. The safe
default (`next_bar_open`) is already in force and the annualisation factor is inferred from
the data, so the CLI is correct as shipped; exposing the biased `same_bar_close` mode would
be a footgun. Both are available on `BacktestEngine` for library callers.

### Exit codes

`backtest.py` returns **1** when any exporter fails, when an exporter could not be
constructed at all, or when no requested exporter name was usable — and **0** only when
every configured exporter succeeded. On failure it prints an "Export report" block naming
each exporter as `OK` or `FAILED: <reason>`, with the error on stderr.

## Examples

**Basic backtest with Simple Moving Average strategy:**
```bash
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --capital 10000 --commission 0.001
```

**Backtest with automatic data cleaning:**
```bash
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --clean
```

**Backtest with risk management and CSV export:**
```bash
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma \
  --risk-manager fixed --max-position-size 0.1 --stop-loss-pct 0.05 \
  --exporters console,csv --csv-output-dir results/
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
- `execution_timing`: When a signal is filled (default: `next_bar_open`, see below)
- `periods_per_year`: Explicit annualisation factor (default: inferred from the data)

#### Execution Timing (no look-ahead)

A signal generated from bar *i* is filled at the **open of bar i+1**. A strategy can only
act on information that was available when the bar it was computed from had closed, so a
signal can never be filled at a price from its own bar:

- `execution_timing='next_bar_open'` (**default**): signal on bar *i* → fill at
  `open` of bar *i+1*. A signal on the final bar never fills, because there is nothing
  left to execute against.
- `execution_timing='same_bar_close'`: signal on bar *i* → fill at `close` of bar *i*.
  This is **look-ahead biased** — the closing price is not knowable at the moment the
  signal is computed from that same bar — and exists only for comparison against
  historical results. Do not use it to evaluate a strategy.

Switching from the old same-bar-close behaviour changes every backtest's fill prices,
timestamps, trade count and derived metrics. That is the intended correction, not a
regression.

#### Trade Execution Logic

**Buy Trade Execution:**
- Calculates maximum investment considering commission costs
- Solves for the largest trade value where
  `trade_value + (trade_value * commission) <= available_cash * position_size` holds **in
  floating point**, not just on paper. `budget / (1 + commission)` recomposes one or two ULP
  above the budget for most balances, and the affordability check below then rejected the
  order silently; `_affordable_trade_value` steps the quotient down with `math.nextafter`
  until it genuinely fits. This can only move the order below the budget, never above it
- Validates against minimum order value and available cash
- Updates cash and position accordingly
- The engine trades **fractional units**, so a balance smaller than one share is not
  "insufficient cash" — it buys a fraction of a share

**Sell Trade Execution:**
- Calculates shares to sell based on position size and current holdings
- The exit fraction is a fraction of the **open position in units**, and it comes from the
  strategy's `position_size` column only. `RiskDecision.position_size` is an *entry* sizing
  target expressed as a fraction of portfolio *value* and is never reinterpreted as an exit
  fraction; a risk manager can still veto an exit through `allow_trade`
- Validates against minimum order value and available position
- Handles partial position closures
- Updates cash and position accordingly

**Scaling Into a Position:**
- A buy while a position is already open scales in rather than replacing it: the entry
  price becomes the quantity-weighted average of both lots, so unrealised P&L and stop
  distances keep measuring from the real cost basis
- The existing stop is never weakened - an order carrying no stop leaves it armed, and a
  supplied stop is adopted only when it is tighter

### Transaction costs

Commission was the only cost the engine ever charged. Fills happened at the exact next-bar
open, in unlimited size, which makes every "profitable" strategy profitable in a market
that does not exist. `niffler/backtesting/cost_model.py` closes that gap.

**Models**

| `--cost-model` | Class | Cost of a fill |
|----------------|-------|----------------|
| `none` (default) | `ZeroCostModel` | nothing, unlimited size |
| `fixed` | `FixedSlippageModel` | `slippage_bps + half_spread_bps`, independent of size |
| `volume` | `VolumeShareSlippageModel` | `half_spread + impact_coefficient * sqrt(participation)`, capped at `max_participation * bar_volume` |

The **square-root** impact law is used because that is what the empirical market-impact
literature reports: impact grows quickly across the first slice of the book and then
flattens. A linear term would make small orders look free and large ones implausibly
expensive. `impact_coefficient` is dimensionless and expressed as a fraction of price —
`0.1` means an order equal to a whole bar's volume pays 10% (1000 bps) of impact, and one
taking 1% of the bar pays `0.1 * sqrt(0.01)` = 1% (100 bps).

**Invariants**

- **Costs are always adverse.** A buy fills at or above the reference price, a sell at or
  below it, for every model and every parameterisation. This is structural: subclasses
  implement `adverse_fraction` (a non-negative number) and the base class applies the sign.
  A negative, non-finite or >= 100% fraction raises.
- **Fill prices stay strictly positive.** Constructors reject a configuration whose
  worst-case cost reaches 100%, and `fill_price` raises if a price still comes out
  non-positive.
- **Stops pay too.** `min(open, stop)` for a long is the *reference* price handed to the
  cost model, so a stop exit fills at or below it — costs can only make a stop worse.
- **The buy budget is solved against the slipped price**, not the reference price, so a
  full-capital buy under heavy slippage still leaves cash >= 0. The order's market footprint
  is measured at the pre-slippage size, which over-charges impact slightly and never
  under-charges.
- **A capped order becomes a partial fill.** The trade is recorded at the reduced quantity
  and a `PARTIAL FILL` warning is logged; the order is never silently dropped. A stop that
  only partly fills leaves the remainder open with its stop still armed.
- **A bar with no usable volume is unfillable** for the volume model (missing, zero or NaN
  volume all count), logging `ORDER NOT FILLED`. `ZeroCostModel` and `FixedSlippageModel`
  ignore volume entirely, so old backtests over data with zero-volume bars still run.
- **Negative basis points, negative coefficients and `max_participation <= 0` raise
  `ValueError`** in the constructor, as does `max_participation > 1`.

**Reporting**

`Trade.slippage_cost` records what each execution gave up, and `BacktestResult` carries
`total_commission` and `total_slippage`. The console exporter prints a `TRANSACTION COSTS`
block naming the model in force, the CSV trade export gains a `slippage_cost` column, and
the Elasticsearch trade/backtest documents carry the same fields.

**Honesty**

`scripts/backtest.py`, `optimize.py` and `analyze.py` print the cost model they are using,
and emit a prominent warning on stderr whenever that model charges nothing — including
`--cost-model fixed --slippage-bps 0 --half-spread-bps 0`. A frictionless number is never
presented as if it described a real market.

**Library use**

```python
from niffler.backtesting import BacktestEngine, FixedSlippageModel

engine = BacktestEngine(
    initial_capital=10_000,
    commission=0.001,
    cost_model=FixedSlippageModel(slippage_bps=5.0, half_spread_bps=1.0),
)
```

The same `cost_model` argument exists on `BaseOptimizer`, `create_optimizer`,
`WalkForwardAnalyzer` and `MonteCarloAnalyzer`, and defaults to `None` (frictionless)
everywhere.

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
- The stop is probed against the bar's **traded range** (the low for a long, the high for a
  short), not only against the bar's execution price, because a resting stop fills the
  moment price trades through it
- Fills respect gaps: a long exits at `min(open, stop)`, so a bar opening below the stop
  fills at the open rather than at the unreachable stop price
- Stop loss triggers are checked before processing new signals. What stays unmodelled is
  intra-bar ordering within the triggering bar, and the entry bar itself, which is checked
  before the entry fills
- Provides detailed logging of stop loss executions, including a WARNING when a triggered
  stop cannot be executed because the residual position is below `min_order_value`

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

Exactly one strategy ships with Niffler. The framework is extensible; the library is not
stocked.

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

#### What is *not* modelled

Be aware of these before trusting a backtest figure:

- **No slippage and no bid/ask spread.** Fills happen at the exact bar open. On illiquid
  instruments or large sizes this is optimistic
- **No shorting.** The engine is long-only; a `-1` signal with no open position is a no-op.
  `Portfolio` is the right place to add it
- **No partial-fill or liquidity risk.** An order either executes in full or is rejected for
  being below `min_order_value`
- **Single asset per run.** Portfolio-level risk limits (max concurrent positions, total
  exposure) exist in the risk framework but are not exercised by a single-symbol backtest
- **No funding, borrow, or overnight financing costs.** Commission only
- **Intra-bar ordering is unknown.** When a bar both triggers a stop and carries a signal,
  the stop is processed first; and the entry bar's stop is checked before the entry fills

#### Data Validation
Comprehensive input validation ensures data quality:
- **Required columns**: open, high, low, close, volume
- **Data types**: All price/volume data must be numeric
- **Value constraints**: Prices > 0, volume ≥ 0
- **OHLC relationships**: High ≥ Low, Open/Close within High/Low range
- **Index requirements**: DatetimeIndex in ascending order
- **Position size**: validated once up front across the whole `position_size` column
  (`Position size must be between 0 and 1, got X`), plus a check on whatever the risk
  manager returns. A strategy returning the wrong number of rows now raises instead of
  silently misaligning

#### CSV loading (shared with `analyze.py` and `optimize.py`)

All three scripts load data through `scripts/common.py::load_ohlcv_csv`, so a file is
interpreted identically everywhere:

- Headers are stripped and **lowercased**; two headers that collide after lowercasing
  (e.g. `Open` and `open`) raise instead of being silently merged
- The timestamp column is detected from `timestamp` / `date` / `datetime` / `time`, or the
  unnamed index column pandas writes. A file with no timestamp information raises
  `Could not determine the timestamp column ...` rather than being given a 1970 epoch index
- The index is parsed to datetimes and **sorted**
- **Duplicate timestamps raise a `ValueError`** rather than being backtested silently. (The
  `preprocessor.py` CLI, which cleans arbitrary files, only warns.)

### Performance Calculation

#### Sharpe Ratio
- Formula: `sqrt(periods_per_year) * mean(returns) / std(returns)`
- `periods_per_year` is **inferred from the data**, not hardcoded. The engine measures the
  median bar spacing and checks whether weekend bars are present:
  - Daily data with weekend bars (crypto) → 365
  - Daily data without weekend bars (equities) → 252
  - Coarser than daily → counted on the calendar, because a weekly or monthly bar spans
    whole calendar weeks whether or not the market trades at weekends (weekly → ~52,
    monthly → ~12)
  - Intraday data → sessions per year scaled by the observed bars per day (e.g. hourly
    crypto → 8760)
- Pass `BacktestEngine(periods_per_year=252)` to override the inference explicitly, e.g. to
  reproduce figures produced before this became automatic.
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