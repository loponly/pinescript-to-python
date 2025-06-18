"""Results management for optimization system."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import json
import csv
import os
from datetime import datetime
import logging

from ..backtesting.backtesting_engine import BacktestResult


@dataclass
class OptimizationSummary:
    """Summary of optimization results for a single stock/timeframe combination."""
    
    symbol: str
    timeframe: str
    best_strategy_config: str
    profit_factor: float
    win_rate: float
    max_drawdown: float
    total_return: float
    sharpe_ratio: float
    total_trades: int
    avg_trade_duration: float
    
    # Additional metrics
    tested_combinations: int
    optimization_time: Optional[float] = None
    data_quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return asdict(self)


class OptimizationResults:
    """Container and manager for optimization results."""
    
    def __init__(self, output_dir: str = "optimization_results"):
        """
        Initialize optimization results manager.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for results
        self.results: List[BacktestResult] = []
        self.summaries: List[OptimizationSummary] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_result(self, result: BacktestResult) -> None:
        """Add a backtest result."""
        self.results.append(result)
    
    def add_results(self, results: List[BacktestResult]) -> None:
        """Add multiple backtest results."""
        self.results.extend(results)
    
    def generate_summaries(self) -> List[OptimizationSummary]:
        """Generate summary statistics from results."""
        summaries = []
        
        # Group results by symbol and timeframe
        grouped = self._group_results_by_symbol_timeframe()
        
        for (symbol, timeframe), group_results in grouped.items():
            if not group_results:
                continue
            
            # Find best result by profit factor
            best_result = max(group_results, key=lambda x: x.performance.profit_factor)
            
            # Create summary
            summary = OptimizationSummary(
                symbol=symbol,
                timeframe=timeframe,
                best_strategy_config=str(best_result.strategy_config),
                profit_factor=best_result.performance.profit_factor,
                win_rate=best_result.performance.win_rate,
                max_drawdown=best_result.performance.max_drawdown,
                total_return=best_result.performance.total_return,
                sharpe_ratio=best_result.performance.sharpe_ratio,
                total_trades=best_result.performance.total_trades,
                avg_trade_duration=best_result.performance.avg_trade_duration,
                tested_combinations=len(group_results)
            )
            
            summaries.append(summary)
        
        self.summaries = summaries
        return summaries
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export optimization results to CSV.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Generate summaries if not already done
        if not self.summaries:
            self.generate_summaries()
        
        # Convert to DataFrame
        df = pd.DataFrame([summary.to_dict() for summary in self.summaries])
        
        # Sort by profit factor descending
        df = df.sort_values('profit_factor', ascending=False)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Results exported to {filepath}")
        return filepath
    
    def export_detailed_results(self, filename: Optional[str] = None) -> str:
        """
        Export detailed results including all tested combinations.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert all results to list of dictionaries
        detailed_data = []
        for result in self.results:
            row = result.to_summary_dict()
            detailed_data.append(row)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(detailed_data)
        df = df.sort_values(['symbol', 'timeframe', 'profit_factor'], 
                           ascending=[True, True, False])
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Detailed results exported to {filepath}")
        return filepath
    
    def export_to_json(self, filename: Optional[str] = None) -> str:
        """
        Export results to JSON format.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data for JSON export
        export_data = {
            'metadata': self.metadata,
            'summary_count': len(self.summaries),
            'total_results': len(self.results),
            'generated_at': datetime.now().isoformat(),
            'summaries': [summary.to_dict() for summary in self.summaries]
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {filepath}")
        return filepath
    
    def get_best_strategies_per_stock(self) -> Dict[str, OptimizationSummary]:
        """
        Get best strategy for each stock across all timeframes.
        
        Returns:
            Dictionary mapping stock symbol to best strategy
        """
        if not self.summaries:
            self.generate_summaries()
        
        best_per_stock = {}
        
        # Group by symbol
        for summary in self.summaries:
            symbol = summary.symbol
            
            if (symbol not in best_per_stock or 
                summary.profit_factor > best_per_stock[symbol].profit_factor):
                best_per_stock[symbol] = summary
        
        return best_per_stock
    
    def get_best_strategies_per_timeframe(self) -> Dict[str, OptimizationSummary]:
        """
        Get best strategy for each timeframe across all stocks.
        
        Returns:
            Dictionary mapping timeframe to best strategy
        """
        if not self.summaries:
            self.generate_summaries()
        
        best_per_timeframe = {}
        
        # Group by timeframe
        for summary in self.summaries:
            timeframe = summary.timeframe
            
            if (timeframe not in best_per_timeframe or 
                summary.profit_factor > best_per_timeframe[timeframe].profit_factor):
                best_per_timeframe[timeframe] = summary
        
        return best_per_timeframe
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.summaries:
            self.generate_summaries()
        
        if not self.summaries:
            return {}
        
        profit_factors = [s.profit_factor for s in self.summaries]
        win_rates = [s.win_rate for s in self.summaries]
        max_drawdowns = [s.max_drawdown for s in self.summaries]
        
        return {
            'total_combinations_tested': len(self.results),
            'unique_stock_timeframe_pairs': len(self.summaries),
            'profitable_strategies': len([pf for pf in profit_factors if pf > 1.0]),
            'avg_profit_factor': sum(profit_factors) / len(profit_factors),
            'avg_win_rate': sum(win_rates) / len(win_rates),
            'avg_max_drawdown': sum(max_drawdowns) / len(max_drawdowns),
            'best_profit_factor': max(profit_factors),
            'worst_profit_factor': min(profit_factors),
            'best_win_rate': max(win_rates),
            'worst_max_drawdown': min(max_drawdowns)
        }
    
    def filter_results(
        self,
        min_profit_factor: Optional[float] = None,
        min_win_rate: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_trades: Optional[int] = None,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None
    ) -> List[OptimizationSummary]:
        """
        Filter results based on criteria.
        
        Args:
            min_profit_factor: Minimum profit factor
            min_win_rate: Minimum win rate
            max_drawdown: Maximum drawdown
            min_trades: Minimum number of trades
            symbols: Filter by symbols
            timeframes: Filter by timeframes
            
        Returns:
            Filtered list of summaries
        """
        if not self.summaries:
            self.generate_summaries()
        
        filtered = []
        
        for summary in self.summaries:
            # Apply filters
            if min_profit_factor and summary.profit_factor < min_profit_factor:
                continue
            if min_win_rate and summary.win_rate < min_win_rate:
                continue
            if max_drawdown and abs(summary.max_drawdown) > max_drawdown:
                continue
            if min_trades and summary.total_trades < min_trades:
                continue
            if symbols and summary.symbol not in symbols:
                continue
            if timeframes and summary.timeframe not in timeframes:
                continue
            
            filtered.append(summary)
        
        return filtered
    
    def _group_results_by_symbol_timeframe(self) -> Dict[Tuple[str, str], List[BacktestResult]]:
        """Group results by symbol and timeframe."""
        grouped = {}
        
        for result in self.results:
            key = (result.symbol or 'Unknown', result.timeframe)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        return grouped
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to results."""
        self.metadata[key] = value
    
    def save_equity_curves(self, symbol: str, timeframe: str) -> Optional[str]:
        """Save equity curves for a specific symbol/timeframe."""
        # Find results for this symbol/timeframe
        matching_results = [
            r for r in self.results 
            if r.symbol == symbol and r.timeframe == timeframe and r.equity_curve is not None
        ]
        
        if not matching_results:
            return None
        
        # Save best equity curve
        best_result = max(matching_results, key=lambda x: x.performance.profit_factor)
        
        filename = f"equity_curve_{symbol}_{timeframe}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        best_result.equity_curve.to_csv(filepath)
        
        self.logger.info(f"Equity curve saved to {filepath}")
        return filepath
