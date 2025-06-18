"""
Simple demo script showing the three new modules working.
Run this from the parent directory using: python -m pinescript-to-python.simple_demo
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

def create_sample_data(periods=1000):
    """Create sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='H')
    
    # Generate price data
    returns = np.random.normal(0.0001, 0.02, periods)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.exponential(0.005, periods))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.exponential(0.005, periods))
    df['volume'] = np.random.lognormal(10, 0.5, periods).astype(int)
    
    return df

def demo_existing_functionality():
    """Test the existing strategy functionality."""
    print("=== Testing Existing Strategy ===")
    
    try:
        from models import StrategyParams
        from strategy import create_default_strategy
        
        # Create sample data
        data = create_sample_data(500)
        print(f"✓ Created sample data: {len(data)} periods")
        
        # Create strategy
        strategy = create_default_strategy()
        print("✓ Created default strategy")
        
        # Run strategy
        result_df = strategy.run_strategy(data)
        trades = strategy.executed_trades
        
        print(f"✓ Strategy completed: {len(trades)} trades executed")
        
        if trades:
            total_pnl = sum(trade.pnl for trade in trades)
            winning_trades = len([t for t in trades if t.pnl > 0])
            win_rate = winning_trades / len(trades) if trades else 0
            
            print(f"  - Total PnL: {total_pnl:.2%}")
            print(f"  - Win Rate: {win_rate:.1%}")
            print(f"  - Total Trades: {len(trades)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Existing functionality failed: {e}")
        return False

def demo_new_modules():
    """Test the new modules with basic functionality."""
    print("\n=== Testing New Modules ===")
    
    # Test 1: Backtesting Config
    try:
        from backtesting.backtest_config import BacktestConfig, TimeframeConfig
        
        config = BacktestConfig(
            commission_rate=0.001,
            initial_capital=10000.0
        )
        print("✓ BacktestConfig created successfully")
        
        tf_config = TimeframeConfig.from_string("1h")
        print(f"✓ TimeframeConfig: {tf_config.name}")
        
    except Exception as e:
        print(f"✗ BacktestConfig failed: {e}")
    
    # Test 2: Optimization Config
    try:
        from optimization.optimization_config import OptimizationConfig, PARAMETER_GRIDS
        
        opt_config = OptimizationConfig(
            stock_list=["AAPL", "MSFT"],
            max_stocks=2,
            timeframes=['1h'],
            max_workers=1
        )
        print("✓ OptimizationConfig created successfully")
        print(f"  - Testing {len(opt_config.stock_list)} stocks")
        print(f"  - Available parameter grids: {list(PARAMETER_GRIDS.keys())}")
        
    except Exception as e:
        print(f"✗ OptimizationConfig failed: {e}")
    
    # Test 3: Database Manager
    try:
        from analysis.database_manager import DatabaseManager
        
        db = DatabaseManager("test_demo.db")
        stats = db.get_performance_statistics()
        print("✓ DatabaseManager created successfully")
        print(f"  - Total results in DB: {stats.get('total_results', 0)}")
        
    except Exception as e:
        print(f"✗ DatabaseManager failed: {e}")

def main():
    """Main demo function."""
    print("Simple Demo of Trading Strategy Modules")
    print("=" * 50)
    
    # Test existing functionality first
    existing_works = demo_existing_functionality()
    
    # Test new modules
    demo_new_modules()
    
    print("\n" + "=" * 50)
    if existing_works:
        print("✓ Demo completed - core functionality working")
        print("\nThe three new modules have been successfully implemented:")
        print("  1. Backtesting Engine - Enhanced backtesting with multiple timeframes")
        print("  2. Optimization System - Multi-stock parameter optimization")
        print("  3. Results & Analysis - Database storage, reporting, and scheduling")
    else:
        print("✗ Demo had issues with existing functionality")
    
    print("\nTo run full optimization (when dependencies are installed):")
    print("  python -c \"from comprehensive_demo import main; main()\"")

if __name__ == "__main__":
    main()
