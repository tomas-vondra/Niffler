# Niffler

Niffler is a Python-based trading application that helps you sniff out market opportunities — just like a Niffler can't resist shiny gold!

## What is Niffler?

Niffler is a personal quantitative trading framework that provides end-to-end functionality from data acquisition to strategy validation. It's designed for systematic development, testing, and validation of trading strategies with rigorous statistical analysis.

## Key Features

- **📈 Data Acquisition**: Download from cryptocurrency exchanges (CCXT) and traditional markets (Yahoo Finance)
- **🧹 Data Processing**: Comprehensive cleaning and validation pipeline
- **🎯 Strategy Framework**: Extensible strategy development with risk management integration  
- **⚡ Backtesting Engine**: Realistic simulation with commission handling and portfolio management
- **🔍 Parameter Optimization**: Grid search and random search with parallel processing
- **📊 Advanced Analysis**: Walk-forward and Monte Carlo robustness testing
- **🛡️ Risk Management**: Position sizing, stop-loss management, and portfolio controls
- **📤 Flexible Exports**: Multi-format result export (console, CSV, Elasticsearch) for analysis and visualization
- **✅ Comprehensive Testing**: 452 unit tests ensuring reliability

## Quick Start

### Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <repository-url>
cd Niffler
uv sync

# Verify installation
python -m unittest discover -s tests -p "test_*.py"
```

### Basic Usage

```bash
# 1. Download market data
python scripts/download_data.py --source ccxt --symbol BTC/USDT --timeframe 1d --start_date 2024-01-01 --end_date 2024-12-31 --exchange binance

# 2. Clean the data
python scripts/preprocessor.py --input data/BTCUSDT_binance_1d_20240101_20241231.csv

# 3. Run backtest with export to CSV
python scripts/backtest.py --data data/BTCUSDT_binance_1d_20240101_20241231_cleaned.csv --strategy simple_ma --symbol BTC/USDT --exporters console,csv --csv-output-dir results/

# 4. Optimize parameters  
python scripts/optimize.py --data data/BTCUSDT_binance_1d_20240101_20241231_cleaned.csv --strategy simple_ma --method grid --output data/optimization_results.json

# 5. Validate robustness
python scripts/analyze.py --data data/BTCUSDT_binance_1d_20240101_20241231_cleaned.csv --analysis walk_forward --strategy simple_ma --params_file data/optimization_results.json
```

## Framework Workflow

Niffler follows a systematic approach to quantitative trading strategy development:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Download  │ -> │ Data Processing │ -> │ Strategy Design │
│                 │    │                 │    │                 │
│ • Crypto (CCXT) │    │ • Clean data    │    │ • Signal logic  │  
│ • Traditional   │    │ • Validate      │    │ • Parameters    │
│   (Yahoo)       │    │ • Handle gaps   │    │ • Risk rules    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         v                        v                        v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Backtesting   │ -> │  Optimization   │ -> │  Validation     │
│                 │    │                 │    │                 │
│ • Portfolio mgmt│    │ • Grid search   │    │ • Walk-forward  │
│ • Commission    │    │ • Random search │    │ • Monte Carlo   │
│ • Risk controls │    │ • Parallel exec │    │ • Robustness    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         v                        v                        v
┌─────────────────┐
│   Export        │
│                 │
│ • Console       │
│ • CSV files     │
│ • Elasticsearch │
└─────────────────┘
```

### 1. Data Layer
**Download** market data from multiple sources, then **process** it through a comprehensive cleaning pipeline that handles missing values, validates OHLC relationships, and ensures data quality.

### 2. Strategy Development  
**Design** trading strategies using the extensible framework, implementing signal generation logic with risk management integration.

### 3. Backtesting
**Test** strategies against historical data with realistic simulation including commissions, position tracking, and portfolio management.

### 4. Optimization
**Find** optimal parameters using grid search or random search methods with parallel processing for efficiency.

### 5. Robustness Validation
**Validate** strategy robustness using advanced analysis methods:
- **Walk-Forward Analysis**: Tests temporal stability across rolling time windows
- **Monte Carlo Analysis**: Tests performance across thousands of market scenarios

### 6. Results Export
**Export** backtest results to multiple formats for analysis and monitoring:
- **Console**: Immediate human-readable feedback
- **CSV Files**: Structured data for external analysis tools
- **Elasticsearch**: Database integration for visualization dashboards

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[Installation Guide](docs/installation.md)** - Setup and dependencies
- **[Data Management](docs/data-management.md)** - Data download and preprocessing  
- **[Backtesting](docs/backtesting.md)** - Strategy testing and simulation
- **[Optimization](docs/optimization.md)** - Parameter optimization methods
- **[Analysis](docs/analysis.md)** - Advanced robustness testing
- **[Risk Management](docs/risk-management.md)** - Position sizing and risk controls
- **[Exporters](docs/exporters.md)** - Result export system and configuration

## Architecture

### Core Components

```
niffler/
├── data/           # Data acquisition and preprocessing
├── strategies/     # Trading strategy implementations  
├── backtesting/    # Portfolio simulation engine
├── optimization/   # Parameter optimization framework
├── analysis/       # Advanced validation methods
├── risk/          # Risk management systems
└── utils/         # Utilities and helpers
```

### Technology Stack

- **Python ≥3.13** with modern `uv` dependency management
- **pandas** for data manipulation and time series analysis
- **ccxt** for cryptocurrency exchange integration
- **yfinance** for traditional financial market data
- **numpy** for numerical computations
- **multiprocessing** for parallel optimization and analysis

## Project Goals

This personal project aims to:

- **Systematic Approach**: Develop a structured methodology for trading strategy development
- **Rigorous Validation**: Implement advanced statistical testing to avoid overfitting
- **Realistic Testing**: Ensure backtesting reflects real-world trading constraints  
- **Risk Management**: Integrate proper risk controls from the ground up
- **Continuous Learning**: Provide a platform for experimenting with new ideas and techniques

## Development Philosophy

### Quality First
- **Comprehensive Testing**: 60+ unit tests covering all major components
- **Data Quality**: Built-in preprocessing ensures clean, reliable data
- **Error Handling**: Robust error management and validation throughout

### Performance Focused
- **Realistic Simulation**: Commission handling, position tracking, risk management
- **Multi-core Processing**: Parallel optimization and analysis for efficiency
- **Memory Management**: Intelligent handling of large datasets and results

### Extensible Design
- **Abstract Base Classes**: Easy to add new strategies, risk managers, and analyzers
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Modern Python**: Uses latest features and best practices

---

*"Just like the magical creatures they're named after, Niffler helps discover hidden treasures in financial markets through systematic, rigorous analysis."*