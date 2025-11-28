#!/usr/bin/env python3
"""
Demo script to show the new exporter functionality.
This is a temporary demo file - not part of the core codebase.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from niffler.backtesting import BacktestEngine, BacktestResult
from niffler.strategies.simple_ma_strategy import SimpleMAStrategy
from niffler.exporters import ExporterManager


def create_test_data():
    """Create simple test data for demonstration."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'open': [100 + i * 0.5 for i in range(50)],
        'high': [102 + i * 0.5 for i in range(50)],
        'low': [98 + i * 0.5 for i in range(50)],
        'close': [101 + i * 0.5 for i in range(50)],
        'volume': [1000] * 50
    }, index=dates)
    return data


def demo_exporters():
    """Demonstrate the new exporter functionality."""
    print("🚀 Niffler Exporters Demo")
    print("=" * 50)
    
    # Create test data
    data = create_test_data()
    
    # Setup strategy
    strategy = SimpleMAStrategy(short_window=5, long_window=10, position_size=1.0)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=10000, commission=0.001)
    result = engine.run_backtest(strategy, data, symbol="DEMO")
    
    print(f"✅ Backtest completed: {result.total_trades} trades executed")
    
    # Setup exporters
    manager = ExporterManager()
    
    # Add console exporter
    manager.create_console_exporter()
    
    # Add CSV exporter to /tmp
    manager.create_csv_exporter(output_dir="/tmp/niffler_demo")
    
    # Export results
    strategy_params = {
        'short_window': 5,
        'long_window': 10,
        'position_size': 1.0
    }
    
    backtest_id = manager.export_backtest_result(
        result=result,
        strategy_params=strategy_params,
        symbol="DEMO",
        initial_capital=10000,
        commission=0.001
    )
    
    print(f"\n🆔 Backtest ID: {backtest_id}")
    print(f"📊 Exporters used: {', '.join(manager.get_exporter_names())}")
    
    return backtest_id


if __name__ == "__main__":
    demo_exporters()
    print("\n✨ Demo completed! Check /tmp/niffler_demo for CSV files.")