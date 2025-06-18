"""Report generator for optimization results."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import json
import os
from datetime import datetime
import logging

from .database_manager import DatabaseManager
from ..optimization.optimization_results import OptimizationResults, OptimizationSummary


class ReportGenerator:
    """
    Generate various reports from optimization results.
    
    Features:
    - Summary reports
    - Detailed analysis
    - Performance comparisons
    - Export to multiple formats
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize report generator.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logging.getLogger(__name__)
    
    def generate_summary_report(
        self,
        output_file: str = "optimization_summary.txt",
        min_profit_factor: float = 1.0
    ) -> str:
        """
        Generate a text summary report.
        
        Args:
            output_file: Output file path
            min_profit_factor: Minimum profit factor for filtering
            
        Returns:
            Path to generated report
        """
        # Get data from database
        stats = self.db_manager.get_performance_statistics()
        best_per_stock = self.db_manager.get_best_strategies_per_stock(
            limit=10, min_profit_factor=min_profit_factor
        )
        best_per_timeframe = self.db_manager.get_best_strategies_per_timeframe(
            min_profit_factor=min_profit_factor
        )
        
        # Generate report content
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TRADING STRATEGY OPTIMIZATION SUMMARY REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Total Results: {stats['total_results']:,}")
        report_lines.append(f"Unique Stocks: {stats['unique_stocks']}")
        report_lines.append(f"Unique Timeframes: {stats['unique_timeframes']}")
        report_lines.append(f"Profitable Strategies: {stats['profitable_strategies']:,}")
        report_lines.append(f"Profitability Rate: {stats['profitability_rate']:.1%}")
        report_lines.append(f"Average Profit Factor: {stats['avg_profit_factor']:.2f}")
        report_lines.append(f"Best Profit Factor: {stats['max_profit_factor']:.2f}")
        report_lines.append(f"Average Win Rate: {stats['avg_win_rate']:.1%}")
        report_lines.append(f"Average Max Drawdown: {stats['avg_max_drawdown']:.1%}")
        report_lines.append("")
        
        # Best strategies per stock
        report_lines.append("TOP 10 BEST STRATEGIES BY STOCK")
        report_lines.append("-" * 40)
        if not best_per_stock.empty:
            for _, row in best_per_stock.head(10).iterrows():
                report_lines.append(
                    f"{row['symbol']:8} | {row['timeframe']:4} | "
                    f"PF: {row['profit_factor']:6.2f} | "
                    f"WR: {row['win_rate']:6.1%} | "
                    f"DD: {row['max_drawdown']:6.1%} | "
                    f"Trades: {row['total_trades']:3.0f}"
                )
        else:
            report_lines.append("No profitable strategies found")
        report_lines.append("")
        
        # Best strategies per timeframe
        report_lines.append("BEST STRATEGIES BY TIMEFRAME")
        report_lines.append("-" * 35)
        if not best_per_timeframe.empty:
            for _, row in best_per_timeframe.iterrows():
                report_lines.append(
                    f"{row['timeframe']:4} | {row['symbol']:8} | "
                    f"PF: {row['profit_factor']:6.2f} | "
                    f"WR: {row['win_rate']:6.1%} | "
                    f"DD: {row['max_drawdown']:6.1%}"
                )
        else:
            report_lines.append("No profitable strategies found")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 20)
        
        if stats['profitability_rate'] > 0.3:
            report_lines.append("✓ Good overall profitability rate (>30%)")
        else:
            report_lines.append("⚠ Low profitability rate - consider parameter adjustments")
        
        if stats['avg_profit_factor'] > 1.2:
            report_lines.append("✓ Good average profit factor (>1.2)")
        else:
            report_lines.append("⚠ Low average profit factor - review strategy logic")
        
        if stats['avg_win_rate'] > 0.5:
            report_lines.append("✓ Good average win rate (>50%)")
        else:
            report_lines.append("⚠ Low win rate - consider entry/exit criteria")
        
        if abs(stats['avg_max_drawdown']) < 0.15:
            report_lines.append("✓ Acceptable drawdown levels (<15%)")
        else:
            report_lines.append("⚠ High drawdown - review risk management")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Summary report generated: {output_file}")
        return output_file
    
    def generate_detailed_analysis(
        self,
        output_file: str = "detailed_analysis.csv",
        include_filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate detailed analysis CSV.
        
        Args:
            output_file: Output CSV file path
            include_filters: Optional filters for data
            
        Returns:
            Path to generated file
        """
        # Get filtered data
        filters = include_filters or {}
        df = self.db_manager.query_results(**filters)
        
        if df.empty:
            self.logger.warning("No data found for detailed analysis")
            return output_file
        
        # Add calculated columns
        df['profitability'] = (df['profit_factor'] > 1.0).astype(int)
        df['risk_adjusted_return'] = df['total_return'] / abs(df['max_drawdown'])
        df['risk_adjusted_return'] = df['risk_adjusted_return'].replace([float('inf')], 0)
        
        # Sort by multiple criteria
        df = df.sort_values(['profit_factor', 'win_rate', 'total_trades'], ascending=[False, False, False])
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Detailed analysis generated: {output_file} ({len(df)} records)")
        return output_file
    
    def generate_timeframe_comparison(self, output_file: str = "timeframe_comparison.csv") -> str:
        """Generate comparison across timeframes."""
        df = self.db_manager.query_results()
        
        if df.empty:
            return output_file
        
        # Group by timeframe and calculate statistics
        comparison = df.groupby('timeframe').agg({
            'profit_factor': ['mean', 'median', 'max', 'std', 'count'],
            'win_rate': ['mean', 'median', 'max'],
            'max_drawdown': ['mean', 'median', 'min'],
            'total_return': ['mean', 'median', 'max'],
            'total_trades': ['mean', 'median'],
            'sharpe_ratio': ['mean', 'median', 'max']
        }).round(4)
        
        # Flatten column names
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
        
        # Add profitability rate
        profitable = df[df['profit_factor'] > 1.0].groupby('timeframe').size()
        total = df.groupby('timeframe').size()
        comparison['profitability_rate'] = (profitable / total).fillna(0).round(4)
        
        # Sort by average profit factor
        comparison = comparison.sort_values('profit_factor_mean', ascending=False)
        
        # Save to CSV
        comparison.to_csv(output_file)
        
        self.logger.info(f"Timeframe comparison generated: {output_file}")
        return output_file
    
    def generate_stock_ranking(
        self,
        output_file: str = "stock_ranking.csv",
        min_timeframes: int = 2
    ) -> str:
        """
        Generate ranking of stocks by performance.
        
        Args:
            output_file: Output file path
            min_timeframes: Minimum timeframes required per stock
            
        Returns:
            Path to generated file
        """
        df = self.db_manager.query_results()
        
        if df.empty:
            return output_file
        
        # Group by stock and calculate metrics
        stock_stats = df.groupby('symbol').agg({
            'profit_factor': ['mean', 'max', 'count'],
            'win_rate': ['mean', 'max'],
            'max_drawdown': ['mean', 'min'],
            'total_return': ['mean', 'max'],
            'total_trades': ['mean', 'sum'],
            'timeframe': 'nunique'
        }).round(4)
        
        # Flatten column names
        stock_stats.columns = ['_'.join(col).strip() for col in stock_stats.columns.values]
        
        # Filter by minimum timeframes
        stock_stats = stock_stats[stock_stats['timeframe_nunique'] >= min_timeframes]
        
        # Calculate profitability rate per stock
        profitable = df[df['profit_factor'] > 1.0].groupby('symbol').size()
        total = df.groupby('symbol').size()
        stock_stats['profitability_rate'] = (profitable / total).fillna(0).round(4)
        
        # Calculate composite score
        stock_stats['composite_score'] = (
            stock_stats['profit_factor_mean'] * 0.4 +
            stock_stats['win_rate_mean'] * 0.3 +
            stock_stats['profitability_rate'] * 0.3
        ).round(4)
        
        # Sort by composite score
        stock_stats = stock_stats.sort_values('composite_score', ascending=False)
        
        # Save to CSV
        stock_stats.to_csv(output_file)
        
        self.logger.info(f"Stock ranking generated: {output_file}")
        return output_file
    
    def generate_parameter_analysis(self, output_file: str = "parameter_analysis.csv") -> str:
        """Analyze which parameters work best."""
        df = self.db_manager.query_results()
        
        if df.empty:
            return output_file
        
        # Extract parameters from strategy_config string
        # This is a simplified version - would need more sophisticated parsing
        parameter_performance = []
        
        for _, row in df.iterrows():
            config_str = row['strategy_config']
            
            # Simple parameter extraction (would need improvement for real use)
            if 'EMA' in config_str:
                ma_type = 'EMA'
            elif 'SMA' in config_str:
                ma_type = 'SMA'
            else:
                ma_type = 'Unknown'
            
            parameter_performance.append({
                'symbol': row['symbol'],
                'timeframe': row['timeframe'],
                'ma_type': ma_type,
                'profit_factor': row['profit_factor'],
                'win_rate': row['win_rate'],
                'max_drawdown': row['max_drawdown']
            })
        
        param_df = pd.DataFrame(parameter_performance)
        
        # Analyze by parameter
        ma_analysis = param_df.groupby('ma_type').agg({
            'profit_factor': ['mean', 'count'],
            'win_rate': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        
        ma_analysis.to_csv(output_file)
        
        self.logger.info(f"Parameter analysis generated: {output_file}")
        return output_file
    
    def generate_executive_summary(
        self,
        output_file: str = "executive_summary.json"
    ) -> str:
        """Generate executive summary in JSON format."""
        stats = self.db_manager.get_performance_statistics()
        best_per_stock = self.db_manager.get_best_strategies_per_stock(limit=5)
        best_per_timeframe = self.db_manager.get_best_strategies_per_timeframe()
        
        # Create executive summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'overview': {
                'total_strategies_tested': stats['total_results'],
                'unique_stocks': stats['unique_stocks'],
                'unique_timeframes': stats['unique_timeframes'],
                'profitability_rate': stats['profitability_rate'],
                'avg_profit_factor': stats['avg_profit_factor'],
                'best_profit_factor': stats['max_profit_factor']
            },
            'key_findings': {
                'most_profitable_stock': best_per_stock.iloc[0]['symbol'] if not best_per_stock.empty else None,
                'best_timeframe': best_per_timeframe.iloc[0]['timeframe'] if not best_per_timeframe.empty else None,
                'avg_win_rate': stats['avg_win_rate'],
                'avg_drawdown': stats['avg_max_drawdown']
            },
            'recommendations': self._generate_recommendations(stats)
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Executive summary generated: {output_file}")
        return output_file
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on statistics."""
        recommendations = []
        
        if stats['profitability_rate'] > 0.4:
            recommendations.append("Strong profitability rate indicates robust strategy")
        elif stats['profitability_rate'] < 0.2:
            recommendations.append("Low profitability rate - consider strategy refinement")
        
        if stats['avg_profit_factor'] > 1.5:
            recommendations.append("Excellent profit factor suggests good risk/reward")
        elif stats['avg_profit_factor'] < 1.1:
            recommendations.append("Low profit factor - review entry/exit criteria")
        
        if abs(stats['avg_max_drawdown']) > 0.2:
            recommendations.append("High drawdown - implement stronger risk management")
        
        if stats['unique_timeframes'] < 3:
            recommendations.append("Test additional timeframes for better diversification")
        
        return recommendations
    
    def generate_all_reports(self, output_dir: str = "reports") -> Dict[str, str]:
        """
        Generate all available reports.
        
        Args:
            output_dir: Directory to save reports
            
        Returns:
            Dictionary mapping report type to file path
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        reports = {}
        
        try:
            reports['summary'] = self.generate_summary_report(
                os.path.join(output_dir, f"summary_{timestamp}.txt")
            )
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
        
        try:
            reports['detailed'] = self.generate_detailed_analysis(
                os.path.join(output_dir, f"detailed_{timestamp}.csv")
            )
        except Exception as e:
            self.logger.error(f"Failed to generate detailed analysis: {e}")
        
        try:
            reports['timeframe'] = self.generate_timeframe_comparison(
                os.path.join(output_dir, f"timeframe_comparison_{timestamp}.csv")
            )
        except Exception as e:
            self.logger.error(f"Failed to generate timeframe comparison: {e}")
        
        try:
            reports['stocks'] = self.generate_stock_ranking(
                os.path.join(output_dir, f"stock_ranking_{timestamp}.csv")
            )
        except Exception as e:
            self.logger.error(f"Failed to generate stock ranking: {e}")
        
        try:
            reports['executive'] = self.generate_executive_summary(
                os.path.join(output_dir, f"executive_summary_{timestamp}.json")
            )
        except Exception as e:
            self.logger.error(f"Failed to generate executive summary: {e}")
        
        self.logger.info(f"Generated {len(reports)} reports in {output_dir}")
        return reports
