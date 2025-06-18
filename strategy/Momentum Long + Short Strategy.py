"""
Momentum Long + Short Strategy Implementation.

This module provides a trading strategy based on momentum indicators
including EMA/SMA, RSI, ADX, ATR, and Bollinger Bands.

Follows SOLID principles with clean separation of concerns:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Strategy can be extended without modification
- Liskov Substitution: Interfaces can be substituted
- Interface Segregation: Clean, focused interfaces
- Dependency Inversion: Depends on abstractions, not concretions
"""

from typing import Dict, Optional, Union

import pandas as pd

from ..models import StrategyParams, TradeResult
from ..indicators import TechnicalIndicatorCalculator
from ..signals import SignalGenerator
from ..trading import TradeSimulator

class MomentumStrategy:
    """Main strategy class that orchestrates the momentum trading strategy."""
    
    def __init__(self, params: StrategyParams):
        """
        Initialize the momentum strategy.
        
        Args:
            params: Strategy configuration parameters
        """
        self._params = params
        self._indicator_calculator = TechnicalIndicatorCalculator()
        self._signal_generator = SignalGenerator()
        self._trade_simulator = TradeSimulator()
    
    def run_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete momentum strategy.
        
        Args:
            df: OHLC data DataFrame with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with indicators, signals, and trade simulation results
            
        Raises:
            ValueError: If required columns are missing from input DataFrame
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate technical indicators
        df_with_indicators = self._indicator_calculator.calculate_all_indicators(
            df, self._params
        )
        
        # Generate trading signals
        df_with_signals = self._signal_generator.generate_all_signals(
            df_with_indicators, self._params
        )
        
        # Simulate trades
        final_df = self._trade_simulator.simulate_trades(
            df_with_signals, self._params
        )
        
        return final_df
    
    @property
    def executed_trades(self) -> list[TradeResult]:
        """Get list of executed trades from the last strategy run."""
        return self._trade_simulator.trades


# Example usage and configuration
def create_default_strategy() -> MomentumStrategy:
    """Create strategy with default parameters."""
    params = StrategyParams()
    return MomentumStrategy(params)


def create_custom_strategy(**kwargs) -> MomentumStrategy:
    """
    Create strategy with custom parameters.
    
    Args:
        **kwargs: Custom parameter values
        
    Returns:
        Configured MomentumStrategy instance
    """
    params = StrategyParams(**kwargs)
    return MomentumStrategy(params)


# Example usage
if __name__ == "__main__":
    # Example of using the refactored momentum strategy
    
    # Create strategy with default parameters
    default_strategy = create_default_strategy()
    
    # Create strategy with custom parameters
    custom_strategy = create_custom_strategy(
        smooth_type="SMA",
        enable_shorts=False,
        sl_percent_long=2.5,
        use_rsi_filter=True
    )
    
    # Example data loading and strategy execution
    # df = pd.read_csv("btc_3h_data.csv")  # OHLCV data required
    # result_df = custom_strategy.run_strategy(df)
    # trades = custom_strategy.executed_trades
    # 
    # print(f"Total trades executed: {len(trades)}")
    # for trade in trades[:5]:  # Show first 5 trades
    #     print(f"Trade: {trade.position_type}, "
    #           f"Entry: {trade.entry_price:.2f}, "
    #           f"Exit: {trade.exit_price:.2f}, "
    #           f"PnL: {trade.pnl:.2%}")
