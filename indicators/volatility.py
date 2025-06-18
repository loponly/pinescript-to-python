"""Volatility indicators implementation."""

import numpy as np
import pandas as pd


class AverageTrueRange:
    """Average True Range (ATR) calculator."""
    
    def calculate(self, df: pd.DataFrame, length: int) -> pd.Series:
        """
        Calculate ATR for given OHLC data and length.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            length: Period length for ATR calculation
            
        Returns:
            ATR values as pandas Series
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(length).mean()
