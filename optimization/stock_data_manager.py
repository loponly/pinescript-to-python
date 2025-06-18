"""Stock data management for optimization system."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yfinance as yf
except ImportError:
    yf = None


class StockDataManager:
    """Manages stock data for optimization system."""
    
    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize stock data manager.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Data cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    def download_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Download stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            
        Returns:
            OHLCV DataFrame or None if failed
        """
        if yf is None:
            self.logger.error("yfinance not installed. Run: pip install yfinance")
            return None
        
        try:
            # Set default dates
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            
            # Check cache first
            cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
            if cache_key in self._data_cache:
                return self._data_cache[cache_key]
            
            # Check file cache
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.csv")
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    self._data_cache[cache_key] = df
                    return df
                except Exception as e:
                    self.logger.warning(f"Failed to load cached data: {e}")
            
            # Download data
            self.logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
            
            # Map timeframe to yfinance interval
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            interval = interval_map.get(timeframe, '1h')
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                self.logger.warning(f"No data downloaded for {symbol}")
                return None
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            if 'adj close' in df.columns:
                df = df.drop('adj close', axis=1)
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns for {symbol}")
                return None
            
            # Cache the data
            self._data_cache[cache_key] = df
            df.to_csv(cache_file)
            
            self.logger.info(f"Downloaded {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download {symbol}: {str(e)}")
            return None
    
    def download_multiple_stocks(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1h",
        max_workers: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple stocks in parallel.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary mapping symbol to OHLCV DataFrame
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download jobs
            futures = {
                executor.submit(
                    self.download_stock_data, 
                    symbol, start_date, end_date, timeframe
                ): symbol 
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        results[symbol] = data
                    else:
                        self.logger.warning(f"No data for {symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to download {symbol}: {str(e)}")
        
        self.logger.info(f"Successfully downloaded {len(results)}/{len(symbols)} stocks")
        return results
    
    def validate_data_quality(
        self,
        data: pd.DataFrame,
        symbol: str,
        min_points: int = 1000
    ) -> Tuple[bool, str]:
        """
        Validate data quality for backtesting.
        
        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol
            min_points: Minimum required data points
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check minimum data points
            if len(data) < min_points:
                return False, f"Insufficient data: {len(data)} < {min_points}"
            
            # Check for missing values
            missing_pct = data.isnull().sum().max() / len(data)
            if missing_pct > 0.05:  # More than 5% missing
                return False, f"Too many missing values: {missing_pct:.1%}"
            
            # Check for zero prices
            if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
                return False, "Contains zero or negative prices"
            
            # Check OHLC logic
            invalid_ohlc = (
                (data['high'] < data[['open', 'close']].max(axis=1)) |
                (data['low'] > data[['open', 'close']].min(axis=1))
            ).any()
            
            if invalid_ohlc:
                return False, "Invalid OHLC relationships"
            
            # Check for extreme outliers (price changes > 50% in one period)
            returns = data['close'].pct_change().abs()
            extreme_moves = (returns > 0.5).sum()
            if extreme_moves > len(data) * 0.01:  # More than 1% of data
                return False, f"Too many extreme price moves: {extreme_moves}"
            
            # Check data continuity (gaps in time series)
            if hasattr(data.index, 'freq') and data.index.freq:
                expected_periods = len(pd.date_range(
                    data.index[0], data.index[-1], freq=data.index.freq
                ))
                coverage = len(data) / expected_periods
                if coverage < 0.9:  # Less than 90% coverage
                    return False, f"Poor time series coverage: {coverage:.1%}"
            
            return True, f"Data quality OK ({len(data)} points)"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get basic stock information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock info or None
        """
        if yf is None:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'country': info.get('country', 'Unknown')
            }
        except Exception as e:
            self.logger.error(f"Failed to get info for {symbol}: {str(e)}")
            return None
    
    def filter_stocks_by_criteria(
        self,
        symbols: List[str],
        min_market_cap: Optional[float] = None,
        sectors: Optional[List[str]] = None,
        max_workers: int = 4
    ) -> List[str]:
        """
        Filter stocks based on criteria.
        
        Args:
            symbols: List of stock symbols
            min_market_cap: Minimum market cap filter
            sectors: Allowed sectors
            max_workers: Number of parallel workers
            
        Returns:
            Filtered list of symbols
        """
        if not min_market_cap and not sectors:
            return symbols
        
        filtered_symbols = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_stock_info, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    info = future.result()
                    if info is None:
                        continue
                    
                    # Apply filters
                    if min_market_cap and info.get('market_cap', 0) < min_market_cap:
                        continue
                    
                    if sectors and info.get('sector') not in sectors:
                        continue
                    
                    filtered_symbols.append(symbol)
                    
                except Exception as e:
                    self.logger.error(f"Failed to filter {symbol}: {str(e)}")
        
        self.logger.info(f"Filtered {len(filtered_symbols)}/{len(symbols)} stocks")
        return filtered_symbols
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for data."""
        return {
            'periods': len(data),
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'avg_volume': data['volume'].mean(),
            'price_range': f"${data['low'].min():.2f} - ${data['high'].max():.2f}",
            'missing_values': data.isnull().sum().sum()
        }
    
    def clear_cache(self):
        """Clear data cache."""
        self._data_cache.clear()
        
        # Optionally clear file cache
        for file in os.listdir(self.cache_dir):
            if file.endswith('.csv'):
                os.remove(os.path.join(self.cache_dir, file))
        
        self.logger.info("Data cache cleared")
