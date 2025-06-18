"""Trade simulation and execution logic."""

from typing import Optional

import numpy as np
import pandas as pd

from models import StrategyParams, TradeResult


class TradeSimulator:
    """Simulates trade execution with stop losses and take profits."""
    
    def __init__(self):
        """Initialize trade simulator."""
        self._trades: list[TradeResult] = []
    
    def simulate_trades(
        self, 
        df: pd.DataFrame, 
        params: StrategyParams
    ) -> pd.DataFrame:
        """
        Simulate trade execution based on signals.
        
        Args:
            df: DataFrame with trading signals
            params: Strategy parameters
            
        Returns:
            DataFrame with trade simulation results
        """
        result_df = df.copy()
        result_df['position'] = 0
        result_df['entry_price'] = np.nan
        result_df['exit_price'] = np.nan
        
        in_trade = False
        long_trade = False
        entry_price = 0.0
        entry_index = 0
        
        for i in range(1, len(result_df)):
            if not in_trade:
                if result_df.iloc[i]['long_signal']:
                    result_df.iloc[i, result_df.columns.get_loc('position')] = 1
                    entry_price = result_df.iloc[i]['close']
                    result_df.iloc[i, result_df.columns.get_loc('entry_price')] = entry_price
                    in_trade = True
                    long_trade = True
                    entry_index = i
                elif result_df.iloc[i]['short_signal']:
                    result_df.iloc[i, result_df.columns.get_loc('position')] = -1
                    entry_price = result_df.iloc[i]['close']
                    result_df.iloc[i, result_df.columns.get_loc('entry_price')] = entry_price
                    in_trade = True
                    long_trade = False
                    entry_index = i
            else:
                exit_price = self._check_exit_conditions(
                    result_df.iloc[i], entry_price, long_trade, params
                )
                
                if exit_price is not None:
                    result_df.iloc[i, result_df.columns.get_loc('exit_price')] = exit_price
                    in_trade = False
                    
                    # Record trade result
                    trade = TradeResult(
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_type='long' if long_trade else 'short',
                        entry_index=entry_index,
                        exit_index=i
                    )
                    self._trades.append(trade)
            
            if not in_trade:
                result_df.iloc[i, result_df.columns.get_loc('position')] = 0
        
        return result_df
    
    def _check_exit_conditions(
        self, 
        row: pd.Series, 
        entry_price: float, 
        is_long: bool, 
        params: StrategyParams
    ) -> Optional[float]:
        """
        Check if exit conditions are met.
        
        Args:
            row: Current row data
            entry_price: Entry price of the trade
            is_long: Whether it's a long trade
            params: Strategy parameters
            
        Returns:
            Exit price if conditions are met, None otherwise
        """
        if is_long:
            stop_price = entry_price * (1 - params.sl_percent_long / 100)
            if row['low'] <= stop_price:
                return stop_price
            elif row['close'] < row['ma500']:
                return row['close']
        else:  # short trade
            stop_price = entry_price * (1 + params.sl_percent_short / 100)
            tp_price = entry_price * (1 - params.tp_percent_short / 100)
            if row['high'] >= stop_price:
                return stop_price
            elif row['low'] <= tp_price:
                return tp_price
        
        return None
    
    @property
    def trades(self) -> list[TradeResult]:
        """Get list of executed trades."""
        return self._trades.copy()
