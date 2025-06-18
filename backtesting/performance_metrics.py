"""Performance metrics calculation for backtesting results."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from models import TradeResult


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting results."""
    
    # Basic metrics
    total_return: float
    annualized_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    
    # Advanced metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_95: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Time-based metrics
    time_in_market: float  # Percentage of time with open positions
    avg_trade_duration: float  # In hours
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary for easy export."""
        return {
            'Total Return (%)': round(self.total_return * 100, 2),
            'Annualized Return (%)': round(self.annualized_return * 100, 2),
            'Win Rate (%)': round(self.win_rate * 100, 2),
            'Profit Factor': round(self.profit_factor, 2),
            'Max Drawdown (%)': round(self.max_drawdown * 100, 2),
            'Sharpe Ratio': round(self.sharpe_ratio, 2),
            'Sortino Ratio': round(self.sortino_ratio, 2),
            'Calmar Ratio': round(self.calmar_ratio, 2),
            'VaR 95% (%)': round(self.var_95 * 100, 2),
            'Total Trades': self.total_trades,
            'Winning Trades': self.winning_trades,
            'Losing Trades': self.losing_trades,
            'Avg Win (%)': round(self.avg_win * 100, 2),
            'Avg Loss (%)': round(self.avg_loss * 100, 2),
            'Largest Win (%)': round(self.largest_win * 100, 2),
            'Largest Loss (%)': round(self.largest_loss * 100, 2),
            'Time in Market (%)': round(self.time_in_market * 100, 2),
            'Avg Trade Duration (hrs)': round(self.avg_trade_duration, 1)
        }


class PerformanceCalculator:
    """Calculator for backtesting performance metrics."""
    
    @staticmethod
    def calculate_metrics(
        trades: List[TradeResult],
        equity_curve: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades: List of executed trades
            equity_curve: Time series of portfolio value
            benchmark_returns: Optional benchmark for comparison
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if not trades or equity_curve.empty:
            return PerformanceCalculator._empty_metrics()
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = PerformanceCalculator._annualized_return(equity_curve)
        
        # Trade metrics
        trade_returns = [trade.pnl for trade in trades]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        profit_factor = (
            sum(winning_trades) / abs(sum(losing_trades))
            if losing_trades else float('inf') if winning_trades else 0
        )
        
        # Drawdown
        max_drawdown = PerformanceCalculator._max_drawdown(equity_curve)
        
        # Risk metrics
        sharpe_ratio = PerformanceCalculator._sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = PerformanceCalculator._sortino_ratio(returns, risk_free_rate)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # Trade statistics
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        largest_win = max(winning_trades) if winning_trades else 0
        largest_loss = min(losing_trades) if losing_trades else 0
        
        # Time-based metrics
        time_in_market = PerformanceCalculator._time_in_market(trades, equity_curve)
        avg_trade_duration = PerformanceCalculator._avg_trade_duration(trades)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            time_in_market=time_in_market,
            avg_trade_duration=avg_trade_duration
        )
    
    @staticmethod
    def _empty_metrics() -> PerformanceMetrics:
        """Return empty metrics for invalid data."""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, win_rate=0, profit_factor=0,
            max_drawdown=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            var_95=0, total_trades=0, winning_trades=0, losing_trades=0,
            avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
            time_in_market=0, avg_trade_duration=0
        )
    
    @staticmethod
    def _annualized_return(equity_curve: pd.Series) -> float:
        """Calculate annualized return."""
        if len(equity_curve) < 2:
            return 0
            
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Estimate time period (assume daily data if no index)
        if hasattr(equity_curve.index, 'freq') and equity_curve.index.freq:
            periods = len(equity_curve)
            freq = equity_curve.index.freq
        else:
            # Default assumption: daily data
            periods = len(equity_curve)
            years = periods / 252  # Trading days per year
            return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Handle different frequencies
        if 'D' in str(freq):
            years = periods / 252
        elif 'H' in str(freq):
            years = periods / (252 * 24)
        elif 'min' in str(freq):
            years = periods / (252 * 24 * 60)
        else:
            years = periods / 252  # Default to daily
            
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if equity_curve.empty:
            return 0
            
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def _sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0
            
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def _sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if returns.empty:
            return 0
            
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
            
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    @staticmethod
    def _time_in_market(trades: List[TradeResult], equity_curve: pd.Series) -> float:
        """Calculate percentage of time with open positions."""
        if not trades or equity_curve.empty:
            return 0
            
        total_periods = len(equity_curve)
        periods_in_trade = sum(
            trade.exit_index - trade.entry_index 
            for trade in trades 
            if hasattr(trade, 'exit_index') and hasattr(trade, 'entry_index')
        )
        
        return min(periods_in_trade / total_periods, 1.0) if total_periods > 0 else 0
    
    @staticmethod
    def _avg_trade_duration(trades: List[TradeResult]) -> float:
        """Calculate average trade duration in hours."""
        if not trades:
            return 0
            
        durations = []
        for trade in trades:
            if hasattr(trade, 'exit_index') and hasattr(trade, 'entry_index'):
                duration = trade.exit_index - trade.entry_index
                durations.append(duration)
        
        return np.mean(durations) if durations else 0
