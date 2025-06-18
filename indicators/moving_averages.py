"""Moving average indicators implementation."""

import pandas as pd

from .base import MovingAverageIndicator


class ExponentialMovingAverage(MovingAverageIndicator):
    """Exponential Moving Average implementation."""
    
    def calculate(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculate EMA for given series and length.
        
        Args:
            series: Price series to calculate EMA for
            length: Period length for EMA calculation
            
        Returns:
            EMA values as pandas Series
        """
        return series.ewm(span=length, adjust=False).mean()


class SimpleMovingAverage(MovingAverageIndicator):
    """Simple Moving Average implementation."""
    
    def calculate(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculate SMA for given series and length.
        
        Args:
            series: Price series to calculate SMA for
            length: Period length for SMA calculation
            
        Returns:
            SMA values as pandas Series
        """
        return series.rolling(window=length).mean()


class MovingAverageFactory:
    """Factory for creating moving average calculators."""
    
    @staticmethod
    def create(ma_type: str) -> MovingAverageIndicator:
        """
        Create moving average calculator based on type.
        
        Args:
            ma_type: Type of moving average ("EMA" or "SMA")
            
        Returns:
            Moving average calculator instance
            
        Raises:
            ValueError: If unsupported MA type is provided
        """
        if ma_type.upper() == "EMA":
            return ExponentialMovingAverage()
        elif ma_type.upper() == "SMA":
            return SimpleMovingAverage()
        else:
            raise ValueError(f"Unsupported moving average type: {ma_type}")
