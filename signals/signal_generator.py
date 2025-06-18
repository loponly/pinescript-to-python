"""Signal generation logic for trading strategies."""

import numpy as np
import pandas as pd

from models import StrategyParams


class SignalGenerator:
    """Generates trading signals based on technical indicators."""
    
    def generate_long_signals(
        self, 
        df: pd.DataFrame, 
        params: StrategyParams
    ) -> pd.Series:
        """
        Generate long trading signals.
        
        Args:
            df: DataFrame with technical indicators
            params: Strategy parameters
            
        Returns:
            Boolean series indicating long signals
        """
        conditions = [
            params.enable_longs,
            df['close'] > df['ma100'],
            df['close'] > df['ma500']
        ]
        
        if params.use_trend_filter:
            conditions.append(df['ma100'] > df['ma500'])
        
        if params.use_rsi_filter:
            conditions.append(df['rsi_long'] > df['rsi_long_smooth'])
        
        if params.use_adx_filter:
            conditions.append(df['adx'] > df['adx_smooth'])
        
        if params.use_atr_filter:
            conditions.append(df['atr'] > df['atr_smooth'])
        
        # Convert all conditions to pandas Series and combine them
        result = pd.Series(True, index=df.index) if params.enable_longs else pd.Series(False, index=df.index)
        
        for condition in conditions[1:]:  # Skip the enable_longs boolean
            if isinstance(condition, bool):
                result = result & condition
            else:
                result = result & condition.fillna(False)
                
        return result
    
    def generate_short_signals(
        self, 
        df: pd.DataFrame, 
        params: StrategyParams
    ) -> pd.Series:
        """
        Generate short trading signals.
        
        Args:
            df: DataFrame with technical indicators
            params: Strategy parameters
            
        Returns:
            Boolean series indicating short signals
        """
        conditions = [
            params.enable_shorts,
            df['close'] < df['ma100'],
            df['close'] < df['ma500'],
            df['close'] < df['bb_lower'],
            df['rsi_short'] < params.rsi_threshold_short
        ]
        
        if params.use_atr_filter_short:
            conditions.append(df['atr_short'] > df['atr_short_smooth'])
        
        if params.use_strong_uptrend_block:
            conditions.append(df['ema_gap_pct'] <= params.short_trend_gap_pct)
        
        # Convert all conditions to pandas Series and combine them
        result = pd.Series(True, index=df.index) if params.enable_shorts else pd.Series(False, index=df.index)
        
        for condition in conditions[1:]:  # Skip the enable_shorts boolean
            if isinstance(condition, bool):
                result = result & condition
            else:
                result = result & condition.fillna(False)
                
        return result
    
    def generate_all_signals(
        self, 
        df: pd.DataFrame, 
        params: StrategyParams
    ) -> pd.DataFrame:
        """
        Generate all trading signals.
        
        Args:
            df: DataFrame with technical indicators
            params: Strategy parameters
            
        Returns:
            DataFrame with trading signals
        """
        result_df = df.copy()
        result_df['long_signal'] = self.generate_long_signals(df, params)
        result_df['short_signal'] = self.generate_short_signals(df, params)
        return result_df
