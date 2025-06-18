"""Technical indicator calculator service."""

import pandas as pd

from models import StrategyParams
from .moving_averages import MovingAverageFactory, SimpleMovingAverage
from .momentum import RelativeStrengthIndex
from .volatility import AverageTrueRange
from .trend import AverageDirectionalIndex


class TechnicalIndicatorCalculator:
    """Service class for calculating technical indicators."""
    
    def __init__(self):
        """Initialize with technical indicator calculators."""
        self._rsi = RelativeStrengthIndex()
        self._atr = AverageTrueRange()
        self._adx = AverageDirectionalIndex(self._atr)
        self._sma = SimpleMovingAverage()
    
    def calculate_all_indicators(
        self, 
        df: pd.DataFrame, 
        params: StrategyParams
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators for the strategy.
        
        Args:
            df: OHLC data DataFrame
            params: Strategy parameters
            
        Returns:
            DataFrame with calculated indicators
        """
        result_df = df.copy()
        ma_calculator = MovingAverageFactory.create(params.smooth_type)
        
        # Moving averages
        result_df['ma100'] = ma_calculator.calculate(df['close'], 100)
        result_df['ma500'] = ma_calculator.calculate(df['close'], 500)
        
        # RSI indicators
        result_df['rsi_long'] = self._rsi.calculate(
            df['close'], params.rsi_length_long
        )
        result_df['rsi_long_smooth'] = ma_calculator.calculate(
            result_df['rsi_long'], params.smoothing_length
        )
        result_df['rsi_short'] = self._rsi.calculate(
            df['close'], params.rsi_length_short
        )
        
        # ADX indicator
        result_df['adx'] = self._adx.calculate(df, params.adx_length)
        result_df['adx_smooth'] = ma_calculator.calculate(
            result_df['adx'], params.smoothing_length
        )
        
        # ATR indicators
        result_df['atr'] = self._atr.calculate(df, params.atr_length)
        result_df['atr_smooth'] = ma_calculator.calculate(
            result_df['atr'], params.smoothing_length
        )
        result_df['atr_short'] = self._atr.calculate(df, params.atr_length)
        result_df['atr_short_smooth'] = ma_calculator.calculate(
            result_df['atr_short'], params.smoothing_length
        )
        
        # Bollinger Bands
        result_df['bb_basis'] = self._sma.calculate(df['close'], params.bb_length)
        result_df['bb_dev'] = df['close'].rolling(params.bb_length).std()
        result_df['bb_lower'] = result_df['bb_basis'] - 2 * result_df['bb_dev']
        
        # EMA gap percentage
        result_df['ema_gap_pct'] = (
            (result_df['ma100'] - result_df['ma500']) / result_df['ma500'] * 100
        )
        
        return result_df
