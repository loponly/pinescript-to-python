"""Trend indicators implementation."""

import numpy as np
import pandas as pd

from .volatility import AverageTrueRange


class AverageDirectionalIndex:
    """Average Directional Index (ADX) calculator."""
    
    def __init__(self, atr_calculator: AverageTrueRange):
        """Initialize with ATR calculator dependency."""
        self._atr_calculator = atr_calculator
    
    def calculate(self, df: pd.DataFrame, length: int) -> pd.Series:
        """
        Calculate ADX for given OHLC data and length.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            length: Period length for ADX calculation
            
        Returns:
            ADX values as pandas Series
        """
        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        tr = self._atr_calculator.calculate(df, length)
        plus_di = 100 * pd.Series(plus_dm).rolling(length).mean() / tr
        minus_di = 100 * pd.Series(minus_dm).rolling(length).mean() / tr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(length).mean()
