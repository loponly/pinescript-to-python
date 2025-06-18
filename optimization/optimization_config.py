"""Configuration for optimization system."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from ..backtesting.backtest_config import BacktestConfig


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for optimization system."""
    
    # Stock selection
    stock_list: List[str] = None  # List of stock symbols to test
    stock_universe: str = "SP500"  # Stock universe to test
    min_market_cap: Optional[float] = None  # Minimum market cap filter
    max_stocks: int = 50  # Maximum number of stocks to test
    
    # Timeframe settings
    timeframes: List[str] = None  # Timeframes to test
    base_timeframe: str = "1h"  # Base timeframe for data
    
    # Optimization parameters
    parameter_grid: Optional[Dict[str, List[Any]]] = None
    optimization_metric: str = "profit_factor"  # Metric to optimize
    min_trades: int = 10  # Minimum trades required for valid result
    
    # Data requirements
    min_data_points: int = 1000  # Minimum data points per stock
    data_start_date: Optional[str] = None  # Start date for data
    data_end_date: Optional[str] = None  # End date for data
    
    # Processing settings
    max_workers: int = 4  # Number of parallel workers
    batch_size: int = 10  # Batch size for processing
    memory_limit_gb: float = 8.0  # Memory limit in GB
    
    # Output settings
    save_results: bool = True
    output_dir: str = "optimization_results"
    save_equity_curves: bool = False  # Whether to save equity curves
    
    # Backtesting configuration
    backtest_config: Optional[BacktestConfig] = None
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.stock_list is None:
            object.__setattr__(self, 'stock_list', self._get_default_stocks())
        
        if self.timeframes is None:
            object.__setattr__(self, 'timeframes', ['15m', '1h', '4h', '1d'])
        
        if self.backtest_config is None:
            object.__setattr__(self, 'backtest_config', BacktestConfig())
        
        # Validate parameters
        if self.max_stocks <= 0:
            raise ValueError("max_stocks must be positive")
        
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if not self.timeframes:
            raise ValueError("At least one timeframe must be specified")
    
    def _get_default_stocks(self) -> List[str]:
        """Get default stock list based on universe."""
        if self.stock_universe == "SP500":
            return self._get_sp500_sample()
        elif self.stock_universe == "NASDAQ":
            return self._get_nasdaq_sample()
        elif self.stock_universe == "CRYPTO":
            return self._get_crypto_sample()
        else:
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Default fallback
    
    def _get_sp500_sample(self) -> List[str]:
        """Get sample of S&P 500 stocks."""
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "META", "BRK-B",
            "UNH", "JNJ", "JPM", "V", "PG", "HD", "MA", "DIS", "BAC", "ADBE",
            "CRM", "NFLX", "XOM", "TMO", "ABBV", "CVX", "KO", "PEP", "AVGO",
            "WMT", "LLY", "COST", "ABT", "MRK", "ACN", "DHR", "VZ", "TXN",
            "NKE", "PM", "NEE", "ORCL", "T", "LOW", "UNP", "RTX", "QCOM",
            "LIN", "HON", "SBUX", "AMD", "SPGI", "INTU"
        ]
    
    def _get_nasdaq_sample(self) -> List[str]:
        """Get sample of NASDAQ stocks."""
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "META", "ADBE",
            "NFLX", "CRM", "PYPL", "INTC", "CSCO", "CMCSA", "PEP", "AVGO",
            "TXN", "QCOM", "COST", "TMUS", "AMAT", "INTU", "AMD", "ISRG",
            "BKNG", "MU", "LRCX", "ADI", "GILD", "MRVL", "KLAC", "MDLZ",
            "REGN", "ADP", "SNPS", "CDNS", "FTNT", "CSX", "ORLY", "WDAY",
            "NXPI", "MCHP", "MNST", "CRWD", "ADSK", "DXCM", "BIIB", "KHC",
            "MRNA", "DOCU"
        ]
    
    def _get_crypto_sample(self) -> List[str]:
        """Get sample of crypto symbols."""
        return [
            "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD",
            "DOGE-USD", "DOT-USD", "AVAX-USD", "SHIB-USD", "MATIC-USD", "TRX-USD",
            "DAI-USD", "WBTC-USD", "UNI-USD", "ATOM-USD", "ETC-USD", "LTC-USD",
            "BCH-USD", "LINK-USD", "XLM-USD", "NEAR-USD", "ALGO-USD", "VET-USD",
            "ICP-USD", "FIL-USD", "HBAR-USD", "APE-USD", "MANA-USD", "SAND-USD"
        ]


@dataclass
class OptimizationJob:
    """Configuration for a single optimization job."""
    
    symbol: str
    timeframe: str
    parameter_grid: Dict[str, List[Any]]
    priority: int = 0  # Higher priority jobs run first
    
    def __str__(self) -> str:
        return f"{self.symbol}_{self.timeframe}"
    
    def __hash__(self) -> int:
        return hash((self.symbol, self.timeframe))


# Predefined parameter grids for different optimization strategies
PARAMETER_GRIDS = {
    "quick": {
        'smooth_type': ['EMA', 'SMA'],
        'sl_percent_long': [2.0, 3.0, 4.0],
        'use_rsi_filter': [True, False],
        'enable_shorts': [True, False]
    },
    
    "comprehensive": {
        'smooth_type': ['EMA', 'SMA'],
        'sl_percent_long': [1.0, 2.0, 3.0, 4.0, 5.0],
        'sl_percent_short': [1.0, 2.0, 3.0, 4.0, 5.0],
        'use_rsi_filter': [True, False],
        'use_adx_filter': [True, False],
        'use_trend_filter': [True, False],
        'enable_shorts': [True, False],
        'rsi_length_long': [10, 14, 21],
        'smoothing_length': [50, 100, 150]
    },
    
    "risk_focused": {
        'smooth_type': ['EMA'],
        'sl_percent_long': [1.0, 1.5, 2.0, 2.5, 3.0],
        'sl_percent_short': [1.0, 1.5, 2.0, 2.5, 3.0],
        'use_rsi_filter': [True],
        'use_adx_filter': [True],
        'use_trend_filter': [True],
        'enable_shorts': [False]  # Long only for risk management
    },
    
    "momentum_focused": {
        'smooth_type': ['EMA'],
        'rsi_length_long': [7, 10, 14, 21, 28],
        'smoothing_length': [20, 50, 100, 200],
        'use_rsi_filter': [True],
        'use_adx_filter': [True],
        'sl_percent_long': [3.0, 4.0, 5.0]
    }
}
