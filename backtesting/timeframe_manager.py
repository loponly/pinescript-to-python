"""Timeframe management for backtesting multiple timeframes."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from .backtest_config import TimeframeConfig, ALL_TIMEFRAMES


class TimeframeManager:
    """Manages multiple timeframes for backtesting."""
    
    def __init__(self, base_timeframe: str = '1h'):
        """
        Initialize timeframe manager.
        
        Args:
            base_timeframe: Base timeframe for data (e.g., '1h')
        """
        self.base_timeframe = base_timeframe
        self.timeframes: Dict[str, TimeframeConfig] = {}
        
        # Add base timeframe
        if base_timeframe in ALL_TIMEFRAMES:
            self.timeframes[base_timeframe] = ALL_TIMEFRAMES[base_timeframe]
        else:
            self.timeframes[base_timeframe] = TimeframeConfig.from_string(base_timeframe)
    
    def add_timeframe(self, timeframe: str) -> None:
        """
        Add a timeframe for backtesting.
        
        Args:
            timeframe: Timeframe string (e.g., '5m', '1h', '1d')
        """
        if timeframe in ALL_TIMEFRAMES:
            self.timeframes[timeframe] = ALL_TIMEFRAMES[timeframe]
        else:
            self.timeframes[timeframe] = TimeframeConfig.from_string(timeframe)
    
    def add_multiple_timeframes(self, timeframes: List[str]) -> None:
        """Add multiple timeframes at once."""
        for tf in timeframes:
            self.add_timeframe(tf)
    
    def add_standard_timeframes(self) -> None:
        """Add all standard timeframes (5m, 15m, 1h, 1d)."""
        standard = ['5m', '15m', '1h', '1d']
        self.add_multiple_timeframes(standard)
    
    def add_custom_timeframes(self) -> None:
        """Add custom timeframes (13m, 45m, etc.)."""
        custom = ['13m', '45m', '2h', '6h', '12h']
        self.add_multiple_timeframes(custom)
    
    def resample_data(
        self, 
        df: pd.DataFrame, 
        target_timeframe: str,
        ohlcv_columns: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe.
        
        Args:
            df: Input DataFrame with OHLCV data
            target_timeframe: Target timeframe to resample to
            ohlcv_columns: Column mapping for OHLCV data
            
        Returns:
            Resampled DataFrame
        """
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        # Get target timeframe config
        if target_timeframe not in self.timeframes:
            self.add_timeframe(target_timeframe)
        
        target_config = self.timeframes[target_timeframe]
        
        # Convert timeframe to pandas frequency
        freq = self._timeframe_to_pandas_freq(target_config.timeframe)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")
        
        # Resample OHLCV data
        resampled = df.resample(freq).agg({
            ohlcv_columns['open']: 'first',
            ohlcv_columns['high']: 'max',
            ohlcv_columns['low']: 'min',
            ohlcv_columns['close']: 'last',
            ohlcv_columns['volume']: 'sum'
        }).dropna()
        
        # Rename columns to standard names
        resampled.columns = ['open', 'high', 'low', 'close', 'volume']
        
        return resampled
    
    def get_all_timeframes(self) -> List[str]:
        """Get list of all configured timeframes."""
        return list(self.timeframes.keys())
    
    def get_timeframe_multiplier(self, timeframe: str) -> int:
        """Get multiplier for timeframe relative to minutes."""
        if timeframe not in self.timeframes:
            self.add_timeframe(timeframe)
        return self.timeframes[timeframe].multiplier
    
    def can_resample(self, from_tf: str, to_tf: str) -> bool:
        """
        Check if we can resample from one timeframe to another.
        Can only resample to higher timeframes (e.g., 1h -> 1d).
        """
        from_mult = self.get_timeframe_multiplier(from_tf)
        to_mult = self.get_timeframe_multiplier(to_tf)
        return to_mult >= from_mult and to_mult % from_mult == 0
    
    def _timeframe_to_pandas_freq(self, timeframe: str) -> str:
        """Convert timeframe string to pandas frequency string."""
        import re
        
        match = re.match(r'(\d+)([mhd])', timeframe.lower())
        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
        
        number, unit = match.group(1), match.group(2)
        
        # Map to pandas frequency codes
        unit_map = {
            'm': 'min',  # minutes  
            'h': 'H',    # hours
            'd': 'D'     # days
        }
        
        return f"{number}{unit_map[unit]}"
    
    def validate_data_coverage(
        self, 
        df: pd.DataFrame, 
        timeframe: str, 
        min_periods: int = 1000
    ) -> Tuple[bool, str]:
        """
        Validate that we have enough data for the timeframe.
        
        Args:
            df: Input data
            timeframe: Target timeframe
            min_periods: Minimum required periods
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            resampled = self.resample_data(df, timeframe)
            
            if len(resampled) < min_periods:
                return False, f"Insufficient data: {len(resampled)} periods, need {min_periods}"
            
            # Check for gaps in data
            expected_periods = self._calculate_expected_periods(
                resampled.index[0], 
                resampled.index[-1], 
                timeframe
            )
            
            coverage = len(resampled) / expected_periods
            if coverage < 0.9:  # Less than 90% coverage
                return False, f"Data coverage too low: {coverage:.1%}"
            
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _calculate_expected_periods(
        self, 
        start: pd.Timestamp, 
        end: pd.Timestamp, 
        timeframe: str
    ) -> int:
        """Calculate expected number of periods between start and end."""
        multiplier = self.get_timeframe_multiplier(timeframe)
        total_minutes = (end - start).total_seconds() / 60
        return int(total_minutes / multiplier)
    
    def get_supported_timeframes(self) -> Dict[str, str]:
        """Get all supported timeframes with descriptions."""
        return {
            '1m': '1 Minute',
            '5m': '5 Minutes', 
            '15m': '15 Minutes',
            '30m': '30 Minutes',
            '1h': '1 Hour',
            '4h': '4 Hours',
            '1d': '1 Day',
            '13m': '13 Minutes (Custom)',
            '45m': '45 Minutes (Custom)',
            '2h': '2 Hours (Custom)',
            '6h': '6 Hours (Custom)', 
            '12h': '12 Hours (Custom)'
        }
