# Strategy Optimization

## Optimization Script Usage

```bash
python scripts/optimize.py --data <data_file> --strategy <strategy_name> --method <optimization_method> [--trials <number>] [--sort-by <metric>] [--output <output_file>] [--clean]
```

**Arguments:**
- `--data`: Path to CSV file containing historical market data
- `--strategy`: Strategy to optimize (currently supports `simple_ma`)
- `--method`: Optimization method (`grid` for grid search, `random` for random search)
- `--trials`: (Optional) Number of trials for random search, default: 50
- `--sort-by`: (Optional) Metric to sort results by (`total_return`, `sharpe_ratio`, `max_drawdown`, etc.)
- `--output`: (Optional) Output JSON file for results, defaults to auto-generated filename
- `--clean`: (Optional) Apply data cleaning pipeline before optimization

## Examples

**Grid search optimization for Simple Moving Average strategy:**
```bash
python scripts/optimize.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --method grid
```

**Random search with 100 trials, sorted by Sharpe ratio:**
```bash
python scripts/optimize.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --method random --trials 100 --sort-by sharpe_ratio
```

**Optimization with data cleaning and custom output file:**
```bash
python scripts/optimize.py --data data/BTCUSDT_binance_1d_20240101_20240105.csv --strategy simple_ma --method grid --clean --output my_optimization_results.json
```

## Optimization Framework

### Core Components

#### OptimizerFactory
The `OptimizerFactory` creates optimizers and manages parameter spaces:

**Key Features:**
- Strategy-specific parameter space definition via `PARAMETER_SPACES` mapping
- Optimizer creation based on method selection via `OPTIMIZER_CLASSES` registry
- Strategy class lookup via `STRATEGY_CLASSES` mapping
- Parameter validation and bounds checking

**Available Methods:**
- `grid`: GridSearchOptimizer - Exhaustive grid search
- `random`: RandomSearchOptimizer - Random parameter sampling

#### Parameter Space Management
Parameter spaces are defined using the `ParameterSpace` class with validation:

**Simple MA Strategy Parameters (SIMPLE_MA_PARAMETER_SPACE):**
- `short_window`: Integer range [5, 20] with step 1 - Fast moving average period
- `long_window`: Integer range [20, 100] with step 5 - Slow moving average period  
- `position_size`: Float range [0.5, 1.0] with step 0.1 - Position size fraction

**Parameter Types Supported:**
- `int`: Integer parameters with min/max/step
- `float`: Float parameters with min/max/step  
- `choice`: Discrete choice parameters with list of options

#### Optimization Methods

##### Grid Search (`GridSearchOptimizer`)
Exhaustive search through all parameter combinations:

**Features:**
- Systematic exploration of entire parameter space
- Deterministic and reproducible results
- Guarantees finding best combination within search space
- Higher computational cost for large parameter spaces

##### Random Search (`RandomSearchOptimizer`) 
Stochastic sampling of parameter space:

**Features:**
- Configurable number of trials (default: 50)
- Faster exploration for large parameter spaces
- Good for initial parameter discovery
- Efficient for high-dimensional optimization problems

### Optimization Process

#### 1. Data Loading and Validation
- Loads CSV data with required columns: timestamp, open, high, low, close, volume
- Converts timestamp to datetime index and sorts data
- Optional data preprocessing if `--clean` flag is used
- Validates data format and completeness

#### 2. Strategy and Parameter Space Setup
- Retrieves strategy class from `STRATEGY_CLASSES` mapping
- Gets corresponding parameter space from `PARAMETER_SPACES` mapping
- Validates parameter configuration and bounds
- Creates optimizer instance with specified method

#### 3. Optimization Execution
- Creates optimizer using `create_optimizer()` factory function
- Configures optimization parameters (trials, sorting metric, etc.)
- Runs optimization with specified method
- Uses same backtesting engine as standalone backtests

#### 4. Result Collection and Analysis
- Collects comprehensive performance metrics for each parameter combination
- Sorts results by specified metric (default: `total_return`)
- Generates optimization summary and statistics
- Saves results to JSON file for further analysis

### Available Sorting Metrics

Results can be sorted by any of these performance metrics:
- `total_return`: Absolute profit/loss in currency units
- `total_return_pct`: Percentage return on initial capital
- `sharpe_ratio`: Risk-adjusted return measure
- `max_drawdown`: Maximum peak-to-trough decline percentage
- `win_rate`: Percentage of profitable trades
- `total_trades`: Number of trades executed

### Output Format

#### JSON Output Structure
Results are saved in structured JSON format containing:
- **Optimization metadata**: Strategy name, method, data period
- **Parameter space definition**: Complete parameter ranges and types
- **Results array**: All parameter combinations tested with their performance metrics
- **Best results**: Top performing parameter combinations

#### Console Output
Real-time optimization progress including:
- Parameter combination being tested
- Progress indicators and completion estimates
- Performance metrics for each run
- Final summary with best results

### Integration Features

#### Data Preprocessing Integration
- Optional data cleaning via `--clean` flag using `PreprocessorManager`
- Applies full preprocessing pipeline before optimization
- Ensures consistent data quality across all optimization runs

#### Backtesting Integration
- Uses same `BacktestEngine` for consistent performance measurement
- Same commission rates and trading constraints
- Reliable comparison across parameter combinations
- Integration with risk management systems

#### Analysis Pipeline Integration
- Optimization results compatible with analysis scripts (`analyze.py`)
- JSON output can be used as `--params_file` input for robustness testing
- Seamless workflow from optimization to validation

### Parameter Space Configuration

The parameter space system supports flexible parameter definitions:

#### Parameter Types
- **Integer parameters**: `{'type': 'int', 'min': 5, 'max': 20, 'step': 1}`
- **Float parameters**: `{'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.1}`
- **Choice parameters**: `{'type': 'choice', 'choices': ['option1', 'option2']}`

#### Validation Rules
- Min value must be less than max value for numeric parameters
- Step size must be positive for numeric parameters
- Choice parameters must have non-empty choices list
- All parameters must specify valid type

### Best Practices

#### Parameter Space Design
- Start with reasonable ranges based on domain knowledge
- Use step sizes that balance resolution with computational cost
- Consider parameter interactions and constraints
- Validate parameter ranges make sense for the strategy

#### Method Selection
- **Grid Search**: Use for comprehensive search of small parameter spaces
- **Random Search**: Use for initial exploration or large parameter spaces
- Consider computational budget and available time
- Grid search guarantees finding best combination within bounds

#### Result Interpretation
- Focus on risk-adjusted metrics (Sharpe ratio) over raw returns
- Consider multiple metrics to avoid overfitting
- Validate results with out-of-sample testing
- Use optimization results as input for robustness analysis