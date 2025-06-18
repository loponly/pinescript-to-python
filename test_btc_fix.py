"""
Quick test to verify BTC data and strategy are working correctly.
"""

import pandas as pd
import numpy as np
from btc_data_generator import generate_btc_usdt_data, validate_ohlcv_data
from strategy import create_default_strategy
from models import StrategyParams

def test_btc_strategy():
    print("🧪 Testing BTC/USDT Data and Strategy...")
    
    # Generate BTC data
    print("📊 Generating BTC/USDT data...")
    btc_data = generate_btc_usdt_data(
        periods=500,
        base_price=30000,
        volatility=0.03,
        trend=0.0005,  # Slight uptrend
        seed=42
    )
    
    print(f"✅ Generated {len(btc_data)} periods of data")
    print(f"   Price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
    print(f"   Total return: {((btc_data['close'].iloc[-1] / btc_data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # Validate data
    is_valid, message = validate_ohlcv_data(btc_data)
    print(f"   Validation: {'✅' if is_valid else '❌'} {message}")
    
    if not is_valid:
        return False
    
    # Test strategy with simple parameters
    print("\n🚀 Testing strategy...")
    params = StrategyParams(
        smooth_type="EMA",
        smoothing_length=50,
        enable_longs=True,
        enable_shorts=False,  # Start with longs only
        sl_percent_long=2.0,
        use_rsi_filter=False,  # Disable complex filters initially
        use_trend_filter=False,
        use_adx_filter=False,
        use_atr_filter=False
    )
    
    try:
        strategy = create_default_strategy()
        strategy._params = params  # Override with our test parameters
        
        result = strategy.run_strategy(btc_data)
        print(f"✅ Strategy ran successfully, result shape: {result.shape}")
        
        # Check if we have the expected columns
        expected_cols = ['close', 'ma100', 'ma500', 'long_signal', 'short_signal']
        missing_cols = [col for col in expected_cols if col not in result.columns]
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
        else:
            print("✅ All expected columns present")
        
        # Check for signals
        long_signals = result['long_signal'].sum()
        short_signals = result['short_signal'].sum()
        print(f"📈 Long signals: {long_signals}")
        print(f"📉 Short signals: {short_signals}")
        
        # Check trades
        trades = strategy.executed_trades
        print(f"💼 Executed trades: {len(trades)}")
        
        if trades:
            total_pnl = sum(trade.pnl for trade in trades)
            winning_trades = len([t for t in trades if t.pnl > 0])
            win_rate = winning_trades / len(trades)
            
            print(f"   Total PnL: {total_pnl:.2f}")
            print(f"   Win Rate: {win_rate:.1%}")
            print(f"   Avg PnL per trade: {total_pnl / len(trades):.2f}")
            
            # Show first few trades
            print("   First 3 trades:")
            for i, trade in enumerate(trades[:3]):
                print(f"     {i+1}. {trade.position_type} @ {trade.entry_price:.2f} → {trade.exit_price:.2f} = {trade.pnl:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_signal_generation():
    print("\n🔍 Testing Signal Generation Logic...")
    
    # Create simple test data
    dates = pd.date_range('2023-01-01', periods=100, freq='h')
    test_data = pd.DataFrame(index=dates)
    
    # Simple uptrending price data
    prices = np.linspace(100, 120, 100)  # Simple uptrend
    test_data['close'] = prices
    test_data['open'] = prices * 0.999
    test_data['high'] = prices * 1.005
    test_data['low'] = prices * 0.995
    test_data['volume'] = 1000
    
    print(f"📊 Created simple test data: {len(test_data)} periods")
    print(f"   Price: {test_data['close'].iloc[0]:.2f} → {test_data['close'].iloc[-1]:.2f}")
    
    try:
        strategy = create_default_strategy()
        
        # Very simple parameters
        simple_params = StrategyParams(
            smooth_type="EMA",
            smoothing_length=20,
            enable_longs=True,
            enable_shorts=False,
            use_rsi_filter=False,
            use_trend_filter=False,
            use_adx_filter=False,
            use_atr_filter=False
        )
        strategy._params = simple_params
        
        result = strategy.run_strategy(test_data)
        print(f"✅ Simple strategy test passed, shape: {result.shape}")
        
        # Check signals
        long_signals = result['long_signal'].sum()
        print(f"📈 Long signals generated: {long_signals}")
        
        if long_signals > 0:
            print("✅ Signal generation is working!")
            return True
        else:
            print("⚠️ No signals generated - might be parameter issue")
            
            # Debug: show some indicator values
            print("Debug info:")
            print(f"   MA100 range: {result['ma100'].min():.2f} - {result['ma100'].max():.2f}")
            print(f"   MA500 range: {result['ma500'].min():.2f} - {result['ma500'].max():.2f}")
            print(f"   Close vs MA100: {(result['close'] > result['ma100']).sum()} periods")
            print(f"   Close vs MA500: {(result['close'] > result['ma500']).sum()} periods")
            
            return False
        
    except Exception as e:
        print(f"❌ Simple signal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 BTC Strategy Testing Suite")
    print("=" * 50)
    
    # Test 1: Simple signal generation
    simple_test = test_simple_signal_generation()
    
    # Test 2: BTC data and strategy
    btc_test = test_btc_strategy()
    
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS:")
    print(f"   Simple Signal Test: {'✅ PASS' if simple_test else '❌ FAIL'}")
    print(f"   BTC Strategy Test: {'✅ PASS' if btc_test else '❌ FAIL'}")
    
    if simple_test and btc_test:
        print("🎉 All tests passed! BTC strategy is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the issues above.")
