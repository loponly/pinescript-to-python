"""
BTC/USDT data generator for testing trading strategies.
Creates realistic cryptocurrency price data with typical volatility patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def generate_btc_usdt_data(
    start_date: str = "2023-01-01",
    periods: int = 2000,
    freq: str = "1h",
    base_price: float = 25000.0,
    volatility: float = 0.03,
    trend: float = 0.0002,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate realistic BTC/USDT OHLCV data for testing.
    
    Args:
        start_date: Start date for the data
        periods: Number of periods to generate
        freq: Frequency ('1h', '4h', '1d')
        base_price: Starting price in USDT
        volatility: Daily volatility (0.03 = 3%)
        trend: Daily trend (0.0002 = 0.02% per period)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with OHLCV columns and datetime index
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create datetime index
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate price movements with crypto-like characteristics
    # Higher volatility during certain periods (simulate market cycles)
    volatility_multiplier = 1 + 0.5 * np.sin(np.arange(periods) / 50) 
    
    # Generate returns with fat tails (common in crypto)
    returns = np.random.normal(trend, volatility * volatility_multiplier, periods)
    
    # Add occasional large moves (simulate news events)
    shock_probability = 0.02  # 2% chance of shock per period
    shocks = np.random.binomial(1, shock_probability, periods)
    shock_magnitude = np.random.normal(0, volatility * 3, periods)
    returns += shocks * shock_magnitude
    
    # Calculate cumulative prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    
    # Generate open prices (previous close + small gap)
    gaps = np.random.normal(0, volatility * 0.1, periods)
    df['open'] = df['close'].shift(1) * (1 + gaps)
    df['open'].iloc[0] = base_price
    
    # Generate high/low with realistic spreads
    hl_range = np.random.exponential(volatility * 0.5, periods) + volatility * 0.1
    
    # High is max of open/close plus upward movement
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + hl_range * np.random.random(periods))
    
    # Low is min of open/close minus downward movement  
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - hl_range * np.random.random(periods))
    
    # Generate volume (higher volume during big moves)
    price_change_pct = np.abs(returns)
    base_volume = 1000000  # Base volume in USDT
    volume_multiplier = 1 + 5 * price_change_pct  # Higher volume with bigger moves
    volume_noise = np.random.lognormal(0, 0.8, periods)
    df['volume'] = (base_volume * volume_multiplier * volume_noise).astype(int)
    
    # Ensure OHLC relationships are maintained
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    return df


def get_btc_usdt_test_scenarios() -> dict:
    """
    Get different BTC/USDT market scenarios for comprehensive testing.
    
    Returns:
        Dictionary of scenario name -> DataFrame
    """
    scenarios = {}
    
    # Bull market scenario
    scenarios['bull_market'] = generate_btc_usdt_data(
        start_date="2023-01-01",
        periods=1000,
        base_price=20000,
        volatility=0.025,
        trend=0.0008,  # Strong uptrend
        seed=42
    )
    
    # Bear market scenario  
    scenarios['bear_market'] = generate_btc_usdt_data(
        start_date="2022-06-01", 
        periods=1000,
        base_price=35000,
        volatility=0.035,
        trend=-0.0005,  # Downtrend
        seed=123
    )
    
    # Sideways market scenario
    scenarios['sideways_market'] = generate_btc_usdt_data(
        start_date="2023-06-01",
        periods=1000, 
        base_price=28000,
        volatility=0.02,
        trend=0.0001,  # Minimal trend
        seed=456
    )
    
    # High volatility scenario
    scenarios['high_volatility'] = generate_btc_usdt_data(
        start_date="2023-03-01",
        periods=1000,
        base_price=25000,
        volatility=0.05,  # Very high volatility
        trend=0.0003,
        seed=789
    )
    
    return scenarios


def validate_ohlcv_data(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that the OHLCV data is properly formatted.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if (df[col] <= 0).any():
            return False, f"Found non-positive prices in {col}"
    
    # Check OHLC relationships
    if (df['high'] < df['low']).any():
        return False, "High prices are lower than low prices"
    
    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        return False, "High prices are lower than open/close prices"
    
    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        return False, "Low prices are higher than open/close prices"
    
    # Check volume
    if (df['volume'] < 0).any():
        return False, "Found negative volume"
    
    # Check for NaN values
    if df[required_columns].isnull().any().any():
        return False, "Found NaN values in OHLCV data"
    
    return True, "Data validation passed"


if __name__ == "__main__":
    # Demo usage
    print("Generating BTC/USDT test data...")
    
    # Generate sample data
    btc_data = generate_btc_usdt_data(periods=500)
    print(f"Generated {len(btc_data)} periods of BTC/USDT data")
    print(f"Price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
    print(f"Data period: {btc_data.index[0]} to {btc_data.index[-1]}")
    
    # Validate data
    is_valid, message = validate_ohlcv_data(btc_data)
    print(f"Data validation: {message}")
    
    # Show scenarios
    scenarios = get_btc_usdt_test_scenarios()
    print(f"\nAvailable test scenarios: {list(scenarios.keys())}")
    
    # Display sample data
    print("\nSample BTC/USDT data:")
    print(btc_data.head())
