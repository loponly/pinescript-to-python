"""Strategy parameters model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    """Immutable configuration parameters for the momentum strategy."""
    
    smooth_type: str = "EMA"
    smoothing_length: int = 100
    rsi_length_long: int = 14
    rsi_length_short: int = 14
    rsi_threshold_short: float = 33.0
    adx_length: int = 14
    atr_length: int = 14
    bb_length: int = 20
    sl_percent_long: float = 3.0
    sl_percent_short: float = 3.0
    tp_percent_short: float = 4.0
    short_trend_gap_pct: float = 2.0
    enable_longs: bool = True
    enable_shorts: bool = True
    use_rsi_filter: bool = False
    use_adx_filter: bool = False
    use_atr_filter: bool = False
    use_trend_filter: bool = True
    use_atr_filter_short: bool = True
    use_strong_uptrend_block: bool = True
