"""Debug script to understand why no trades are generated."""

import pandas as pd
import numpy as np
from models import StrategyParams
from strategy import MomentumStrategy
from btc_data_generator import generate_btc_usdt_data

def debug_signals():
    """Debug the signal generation process."""
    print("=== DEBUGGING SIGNAL GENERATION ===")
    
    # Generate larger test data
    data = generate_btc_usdt_data(periods=1000, base_price=30000, volatility=0.03, seed=42)
    print(f"Generated data shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Create simple strategy
    params = StrategyParams(
        smooth_type="SMA",
        smoothing_length=20,
        rsi_length_long=14,
        enable_longs=True,
        enable_shorts=False,
        sl_percent_long=2.0,
        use_rsi_filter=False,  # Disable filters to see base signals
        use_trend_filter=False,
        use_adx_filter=False,
        use_atr_filter=False
    )
    
    strategy = MomentumStrategy(params)
    
    # Run strategy step by step
    print("\n=== CALCULATING INDICATORS ===")
    df_with_indicators = strategy._indicator_calculator.calculate_all_indicators(data, params)
    
    # Check if indicators are calculated
    indicator_cols = ['ma100', 'ma500', 'rsi_long', 'rsi_long_smooth']
    for col in indicator_cols:
        if col in df_with_indicators.columns:
            non_null = df_with_indicators[col].notna().sum()
            print(f"{col}: {non_null}/{len(df_with_indicators)} non-null values")
            if non_null > 0:
                print(f"  Range: {df_with_indicators[col].min():.2f} - {df_with_indicators[col].max():.2f}")
        else:
            print(f"{col}: MISSING")
    
    print("\n=== GENERATING SIGNALS ===")
    df_with_signals = strategy._signal_generator.generate_all_signals(df_with_indicators, params)
    
    # Check signals
    long_signals = df_with_signals['long_signal'].sum()
    short_signals = df_with_signals['short_signal'].sum()
    print(f"Long signals: {long_signals}")
    print(f"Short signals: {short_signals}")
    
    # Check signal conditions manually
    print("\n=== CHECKING SIGNAL CONDITIONS ===")
    df = df_with_signals
    
    # Long signal conditions
    print(f"enable_longs: {params.enable_longs}")
    close_above_ma100 = (df['close'] > df['ma100']).sum()
    close_above_ma500 = (df['close'] > df['ma500']).sum()
    print(f"close > ma100: {close_above_ma100}/{len(df)} rows")
    print(f"close > ma500: {close_above_ma500}/{len(df)} rows")
    
    # Show sample data
    print("\n=== SAMPLE DATA ===")
    sample_df = df[['close', 'ma100', 'ma500', 'long_signal', 'short_signal']].tail(10)
    print(sample_df.to_string())
    
    # Simulate trades
    print("\n=== SIMULATING TRADES ===")
    final_df = strategy._trade_simulator.simulate_trades(df_with_signals, params)
    trades = strategy.executed_trades
    
    print(f"Executed trades: {len(trades)}")
    for i, trade in enumerate(trades[:5]):
        print(f"Trade {i+1}: {trade.position_type}, entry=${trade.entry_price:.2f}, exit=${trade.exit_price:.2f}, PnL=${trade.pnl:.2f}")

if __name__ == "__main__":
    debug_signals()
