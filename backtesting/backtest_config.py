"""Backtest configuration and settings."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from decimal import Decimal


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    # Commission and slippage settings
    commission_rate: float = 0.001  # 0.1% commission
    slippage_bps: float = 1.0  # 1 basis point slippage
    
    # Position sizing
    initial_capital: float = 10000.0
    position_sizing: str = "fixed"  # "fixed", "percent_of_equity", "kelly"
    position_size: float = 1000.0  # Fixed size or percentage
    max_position_count: int = 1  # Maximum simultaneous positions
    
    # Risk management
    max_drawdown_limit: Optional[float] = None  # Stop trading if DD exceeds this
    max_daily_loss: Optional[float] = None  # Daily loss limit
    
    # Timeframe settings
    base_timeframe: str = "1h"  # Base timeframe for data
    signal_timeframes: List[str] = None  # Timeframes to test
    
    # Data validation
    min_data_points: int = 1000  # Minimum data points required
    warmup_period: int = 500  # Periods needed for indicators to stabilize
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.signal_timeframes is None:
            object.__setattr__(self, 'signal_timeframes', ['5m', '15m', '1h', '1d'])
            
        if self.commission_rate < 0 or self.commission_rate > 0.1:
            raise ValueError("Commission rate must be between 0 and 10%")
            
        if self.slippage_bps < 0 or self.slippage_bps > 100:
            raise ValueError("Slippage must be between 0 and 100 basis points")
            
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")


@dataclass(frozen=True) 
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    
    timeframe: str
    multiplier: int  # How many base periods make this timeframe
    name: str
    
    @classmethod
    def from_string(cls, timeframe_str: str) -> 'TimeframeConfig':
        """Create TimeframeConfig from string like '5m', '1h', '1d'."""
        import re
        
        # Parse timeframe string
        match = re.match(r'(\d+)([mhd])', timeframe_str.lower())
        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe_str}")
            
        number, unit = int(match.group(1)), match.group(2)
        
        # Convert to minutes for standardization
        unit_multipliers = {'m': 1, 'h': 60, 'd': 1440}
        minutes = number * unit_multipliers[unit]
        
        return cls(
            timeframe=timeframe_str,
            multiplier=minutes,
            name=f"{number}{unit.upper()}"
        )


# Predefined timeframe configurations
STANDARD_TIMEFRAMES = {
    '1m': TimeframeConfig('1m', 1, '1M'),
    '5m': TimeframeConfig('5m', 5, '5M'),
    '15m': TimeframeConfig('15m', 15, '15M'),
    '30m': TimeframeConfig('30m', 30, '30M'),
    '1h': TimeframeConfig('1h', 60, '1H'),
    '4h': TimeframeConfig('4h', 240, '4H'),
    '1d': TimeframeConfig('1d', 1440, '1D')
}

# Custom timeframes as requested
CUSTOM_TIMEFRAMES = {
    '13m': TimeframeConfig('13m', 13, '13M'),
    '45m': TimeframeConfig('45m', 45, '45M'),
    '2h': TimeframeConfig('2h', 120, '2H'),
    '6h': TimeframeConfig('6h', 360, '6H'),
    '12h': TimeframeConfig('12h', 720, '12H')
}

ALL_TIMEFRAMES = {**STANDARD_TIMEFRAMES, **CUSTOM_TIMEFRAMES}
