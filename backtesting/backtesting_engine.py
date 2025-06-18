"""Main backtesting engine for testing indicator combinations and strategies."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import logging

from ..models import StrategyParams, TradeResult
from ..strategy import MomentumStrategy
from .backtest_config import BacktestConfig
from .timeframe_manager import TimeframeManager
from .performance_metrics import PerformanceCalculator, PerformanceMetrics


class BacktestResult:
    """Container for backtesting results."""
    
    def __init__(
        self,
        strategy_config: StrategyParams,
        timeframe: str,
        performance: PerformanceMetrics,
        trades: List[TradeResult],
        equity_curve: Optional[pd.Series] = None,
        symbol: Optional[str] = None
    ):
        self.strategy_config = strategy_config
        self.timeframe = timeframe
        self.performance = performance
        self.trades = trades
        self.equity_curve = equity_curve
        self.symbol = symbol
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert result to summary dictionary."""
        return {
            'symbol': self.symbol or 'N/A',
            'timeframe': self.timeframe,
            'strategy_config': str(self.strategy_config),
            'total_return': self.performance.total_return,
            'win_rate': self.performance.win_rate,
            'profit_factor': self.performance.profit_factor,
            'max_drawdown': self.performance.max_drawdown,
            'sharpe_ratio': self.performance.sharpe_ratio,
            'total_trades': self.performance.total_trades,
            'avg_trade_duration': self.performance.avg_trade_duration
        }


class BacktestingEngine:
    """
    Comprehensive backtesting engine for testing indicator combinations.
    
    Features:
    - Multiple timeframes (5m, 15m, 1h, 1d)
    - Custom timeframes (13m, 45m, etc.)
    - Commission and slippage settings
    - Performance metrics calculation
    - Parallel processing support
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.timeframe_manager = TimeframeManager()
        self.logger = logging.getLogger(__name__)
        
        # Setup default timeframes
        self.timeframe_manager.add_standard_timeframes()
        self.timeframe_manager.add_custom_timeframes()
    
    def single_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: StrategyParams,
        timeframe: str,
        symbol: Optional[str] = None
    ) -> BacktestResult:
        """
        Run a single backtest with given parameters.
        
        Args:
            data: OHLCV data
            strategy_params: Strategy configuration
            timeframe: Timeframe to test
            symbol: Optional symbol name
            
        Returns:
            BacktestResult with performance metrics
        """
        try:
            # Validate and resample data
            if timeframe != self.timeframe_manager.base_timeframe:
                data = self.timeframe_manager.resample_data(data, timeframe)
            
            # Validate data coverage
            is_valid, message = self.timeframe_manager.validate_data_coverage(
                data, timeframe, self.config.min_data_points
            )
            if not is_valid:
                self.logger.warning(f"Data validation failed for {symbol} {timeframe}: {message}")
                return self._empty_result(strategy_params, timeframe, symbol)
            
            # Run strategy
            strategy = MomentumStrategy(strategy_params)
            result_df = strategy.run_strategy(data)
            trades = strategy.executed_trades
            
            # Calculate equity curve with commission and slippage
            equity_curve = self._calculate_equity_curve(
                result_df, trades, self.config.initial_capital
            )
            
            # Calculate performance metrics
            performance = PerformanceCalculator.calculate_metrics(
                trades, equity_curve
            )
            
            return BacktestResult(
                strategy_config=strategy_params,
                timeframe=timeframe,
                performance=performance,
                trades=trades,
                equity_curve=equity_curve,
                symbol=symbol
            )
            
        except Exception as e:
            self.logger.error(f"Backtest failed for {symbol} {timeframe}: {str(e)}")
            return self._empty_result(strategy_params, timeframe, symbol)
    
    def multi_timeframe_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: StrategyParams,
        timeframes: Optional[List[str]] = None,
        symbol: Optional[str] = None
    ) -> List[BacktestResult]:
        """
        Run backtests across multiple timeframes.
        
        Args:
            data: OHLCV data
            strategy_params: Strategy configuration
            timeframes: List of timeframes to test
            symbol: Optional symbol name
            
        Returns:
            List of BacktestResult objects
        """
        if timeframes is None:
            timeframes = self.config.signal_timeframes
        
        results = []
        for timeframe in timeframes:
            result = self.single_backtest(data, strategy_params, timeframe, symbol)
            results.append(result)
        
        return results
    
    def parameter_optimization(
        self,
        data: pd.DataFrame,
        parameter_grid: Dict[str, List[Any]],
        timeframe: str,
        symbol: Optional[str] = None,
        max_workers: int = 4
    ) -> List[BacktestResult]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: OHLCV data
            parameter_grid: Dictionary of parameters to test
            timeframe: Timeframe to test
            symbol: Optional symbol name
            max_workers: Number of parallel workers
            
        Returns:
            List of BacktestResult objects sorted by performance
        """
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(product(*param_values))
        
        self.logger.info(f"Testing {len(combinations)} parameter combinations")
        
        results = []
        
        if max_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for combo in combinations:
                    params_dict = dict(zip(param_names, combo))
                    strategy_params = StrategyParams(**params_dict)
                    
                    future = executor.submit(
                        self.single_backtest,
                        data.copy(),
                        strategy_params,
                        timeframe,
                        symbol
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Parameter optimization failed: {str(e)}")
        else:
            # Sequential processing
            for combo in combinations:
                params_dict = dict(zip(param_names, combo))
                strategy_params = StrategyParams(**params_dict)
                
                result = self.single_backtest(data, strategy_params, timeframe, symbol)
                results.append(result)
        
        # Sort by profit factor (or another metric)
        results.sort(key=lambda x: x.performance.profit_factor, reverse=True)
        return results
    
    def get_default_parameter_grid(self) -> Dict[str, List[Any]]:
        """Get default parameter grid for optimization."""
        return {
            'smooth_type': ['EMA', 'SMA'],
            'sl_percent_long': [1.0, 2.0, 3.0, 4.0, 5.0],
            'sl_percent_short': [1.0, 2.0, 3.0, 4.0, 5.0],
            'use_rsi_filter': [True, False],
            'use_adx_filter': [True, False],
            'use_trend_filter': [True, False],
            'enable_shorts': [True, False],
            'rsi_length_long': [10, 14, 21],
            'smoothing_length': [50, 100, 150]
        }
    
    def _calculate_equity_curve(
        self,
        result_df: pd.DataFrame,
        trades: List[TradeResult],
        initial_capital: float
    ) -> pd.Series:
        """
        Calculate equity curve including commission and slippage.
        
        Args:
            result_df: Strategy result DataFrame
            trades: List of executed trades
            initial_capital: Starting capital
            
        Returns:
            Equity curve as pandas Series
        """
        equity = [initial_capital]
        current_capital = initial_capital
        
        for trade in trades:
            # Calculate trade PnL
            gross_pnl = trade.pnl * current_capital
            
            # Apply commission (both entry and exit)
            commission = current_capital * self.config.commission_rate * 2
            
            # Apply slippage (estimated as percentage of trade value)
            slippage = current_capital * (self.config.slippage_bps / 10000)
            
            # Net PnL after costs
            net_pnl = gross_pnl - commission - slippage
            current_capital += net_pnl
            
            equity.append(current_capital)
        
        # Create time series (simplified - would need proper timestamps)
        if len(equity) > len(result_df):
            equity = equity[:len(result_df)]
        elif len(equity) < len(result_df):
            # Pad with last value
            equity.extend([equity[-1]] * (len(result_df) - len(equity)))
        
        return pd.Series(equity, index=result_df.index)
    
    def _empty_result(
        self,
        strategy_params: StrategyParams,
        timeframe: str,
        symbol: Optional[str]
    ) -> BacktestResult:
        """Create empty result for failed backtests."""
        empty_metrics = PerformanceMetrics(
            total_return=0, annualized_return=0, win_rate=0, profit_factor=0,
            max_drawdown=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            var_95=0, total_trades=0, winning_trades=0, losing_trades=0,
            avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
            time_in_market=0, avg_trade_duration=0
        )
        
        return BacktestResult(
            strategy_config=strategy_params,
            timeframe=timeframe,
            performance=empty_metrics,
            trades=[],
            symbol=symbol
        )
    
    def get_best_strategy_per_timeframe(
        self,
        results: List[BacktestResult],
        metric: str = 'profit_factor'
    ) -> Dict[str, BacktestResult]:
        """
        Get best strategy for each timeframe based on specified metric.
        
        Args:
            results: List of backtest results
            metric: Metric to optimize for
            
        Returns:
            Dictionary mapping timeframe to best result
        """
        timeframe_results = {}
        
        for result in results:
            tf = result.timeframe
            if tf not in timeframe_results:
                timeframe_results[tf] = []
            timeframe_results[tf].append(result)
        
        best_per_timeframe = {}
        for tf, tf_results in timeframe_results.items():
            best_result = max(
                tf_results,
                key=lambda x: getattr(x.performance, metric, 0)
            )
            best_per_timeframe[tf] = best_result
        
        return best_per_timeframe
