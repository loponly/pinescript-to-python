"""Base classes and protocols for technical indicators."""

from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd


class TechnicalIndicator(Protocol):
    """Protocol for technical indicators."""
    
    def calculate(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate the indicator value."""
        ...


class MovingAverageIndicator(ABC):
    """Abstract base class for moving average indicators."""
    
    @abstractmethod
    def calculate(self, series: pd.Series, length: int) -> pd.Series:
        """Calculate moving average for given series and length."""
        pass
