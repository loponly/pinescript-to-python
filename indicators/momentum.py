"""Momentum indicators implementation."""

import numpy as np
import pandas as pd


class RelativeStrengthIndex:
    """Relative Strength Index (RSI) calculator."""
    
    def calculate(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculate RSI for given series and length.
        
        Args:
            series: Price series to calculate RSI for
            length: Period length for RSI calculation
            
        Returns:
            RSI values as pandas Series
        """
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain_avg = pd.Series(gain).rolling(length).mean()
        loss_avg = pd.Series(loss).rolling(length).mean()
        rs = gain_avg / loss_avg
        return 100 - (100 / (1 + rs))
