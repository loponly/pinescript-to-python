"""
Demo script showing how to use the reorganized momentum strategy.

This demonstrates the clean modular structure and how to use
the strategy with the new organization.
"""

import pandas as pd
import numpy as np

# Import from the reorganized modules
from models import StrategyParams
from strategy import MomentumStrategy, create_default_strategy, create_custom_strategy
from indicators import TechnicalIndicatorCalculator
from signals import SignalGenerator
from trading import TradeSimulator


def create_sample_data(periods: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Start with a base price
    base_price = 50000
    
    # Generate random price movements
    returns = np.random.normal(0.001, 0.02, periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC from closing prices
    df = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.001, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
        'volume': np.random.randint(1000, 10000, periods)
    })
    
    # Ensure high >= close >= low
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    return df


def demo_modular_usage():
    """Demonstrate using individual components."""
    print("=== Demo: Using Individual Components ===")
    
    # Create sample data
    df = create_sample_data(500)
    print(f"Created sample data with {len(df)} rows")
    
    # Create parameters
    params = StrategyParams(
        smooth_type="EMA",
        enable_shorts=True,
        use_rsi_filter=True,
        sl_percent_long=2.0
    )
    print(f"Strategy parameters: {params.smooth_type}, RSI filter: {params.use_rsi_filter}")
    
    # Use individual components
    indicator_calc = TechnicalIndicatorCalculator()
    signal_gen = SignalGenerator()
    trade_sim = TradeSimulator()
    
    # Calculate indicators
    df_with_indicators = indicator_calc.calculate_all_indicators(df, params)
    print(f"Calculated indicators, added {len(df_with_indicators.columns) - len(df.columns)} columns")
    
    # Generate signals
    df_with_signals = signal_gen.generate_all_signals(df_with_indicators, params)
    long_signals = df_with_signals['long_signal'].sum()
    short_signals = df_with_signals['short_signal'].sum()
    print(f"Generated {long_signals} long signals and {short_signals} short signals")
    
    # Simulate trades
    final_df = trade_sim.simulate_trades(df_with_signals, params)
    trades = trade_sim.trades
    print(f"Simulated {len(trades)} trades")
    
    if trades:
        profitable_trades = sum(1 for trade in trades if trade.pnl > 0)
        print(f"Profitable trades: {profitable_trades}/{len(trades)} ({profitable_trades/len(trades)*100:.1f}%)")


def demo_strategy_usage():
    """Demonstrate using the complete strategy."""
    print("\n=== Demo: Using Complete Strategy ===")
    
    # Create sample data
    df = create_sample_data(1000)
    print(f"Created sample data with {len(df)} rows")
    
    # Method 1: Default strategy
    default_strategy = create_default_strategy()
    result_df = default_strategy.run_strategy(df)
    trades = default_strategy.executed_trades
    print(f"Default strategy executed {len(trades)} trades")
    
    # Method 2: Custom strategy
    custom_strategy = create_custom_strategy(
        smooth_type="SMA",
        enable_shorts=False,
        sl_percent_long=1.5,
        use_rsi_filter=True,
        use_adx_filter=True
    )
    result_df_custom = custom_strategy.run_strategy(df)
    trades_custom = custom_strategy.executed_trades
    print(f"Custom strategy executed {len(trades_custom)} trades")
    
    # Method 3: Direct instantiation
    params = StrategyParams(
        smooth_type="EMA",
        enable_longs=True,
        enable_shorts=True,
        use_trend_filter=True,
        sl_percent_long=2.5,
        tp_percent_short=3.0
    )
    direct_strategy = MomentumStrategy(params)
    result_df_direct = direct_strategy.run_strategy(df)
    trades_direct = direct_strategy.executed_trades
    print(f"Direct strategy executed {len(trades_direct)} trades")
    
    # Show some trade details
    if trades_direct:
        print(f"\nFirst 3 trades from direct strategy:")
        for i, trade in enumerate(trades_direct[:3]):
            print(f"  Trade {i+1}: {trade.position_type}, "
                  f"Entry: ${trade.entry_price:.2f}, "
                  f"Exit: ${trade.exit_price:.2f}, "
                  f"PnL: {trade.pnl:.2%}")


def demo_model_usage():
    """Demonstrate using the models directly."""
    print("\n=== Demo: Using Models Directly ===")
    
    # Create different parameter configurations
    conservative_params = StrategyParams(
        sl_percent_long=1.0,
        sl_percent_short=1.0,
        enable_shorts=False,
        use_rsi_filter=True,
        use_adx_filter=True
    )
    
    aggressive_params = StrategyParams(
        sl_percent_long=5.0,
        sl_percent_short=5.0,
        enable_shorts=True,
        use_rsi_filter=False,
        use_adx_filter=False
    )
    
    print(f"Conservative: SL {conservative_params.sl_percent_long}%, Shorts: {conservative_params.enable_shorts}")
    print(f"Aggressive: SL {aggressive_params.sl_percent_long}%, Shorts: {aggressive_params.enable_shorts}")
    
    # Show parameter immutability
    try:
        conservative_params.sl_percent_long = 10.0
        print("ERROR: Should not be able to modify frozen dataclass!")
    except AttributeError:
        print("✓ Parameters are properly immutable (frozen dataclass)")


if __name__ == "__main__":
    print("Pinescript-to-Python Modular Strategy Demo")
    print("=" * 50)
    
    # Run all demos
    demo_modular_usage()
    demo_strategy_usage()
    demo_model_usage()
    
    print("\n" + "=" * 50)
    print("Demo completed! The code is now properly organized into:")
    print("📁 models/     - Data classes and configuration")
    print("📁 indicators/ - Technical analysis indicators")
    print("📁 signals/    - Signal generation logic")
    print("📁 trading/    - Trade simulation and execution")
    print("📁 strategy/   - Main strategy orchestration")
    print("\nEach component can be used independently or together!")
