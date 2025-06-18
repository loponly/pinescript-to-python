"""Database manager for storing and retrieving optimization results."""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from datetime import datetime
import os

from ..optimization.optimization_results import OptimizationSummary
from ..backtesting.backtesting_engine import BacktestResult


class DatabaseManager:
    """
    Database manager for optimization results storage and retrieval.
    
    Features:
    - SQLite database for results storage
    - Efficient querying and filtering
    - Historical data tracking
    - Export capabilities
    """
    
    def __init__(self, db_path: str = "optimization_results.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Create database and tables
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Optimization results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_config TEXT NOT NULL,
                    profit_factor REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    total_return REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    avg_trade_duration REAL NOT NULL,
                    tested_combinations INTEGER,
                    optimization_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, strategy_config)
                )
            ''')
            
            # Detailed backtest results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_config TEXT NOT NULL,
                    profit_factor REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    total_return REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    sortino_ratio REAL NOT NULL,
                    calmar_ratio REAL NOT NULL,
                    var_95 REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    avg_win REAL NOT NULL,
                    avg_loss REAL NOT NULL,
                    largest_win REAL NOT NULL,
                    largest_loss REAL NOT NULL,
                    time_in_market REAL NOT NULL,
                    avg_trade_duration REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Optimization runs metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    total_stocks INTEGER,
                    total_timeframes INTEGER,
                    parameter_combinations INTEGER,
                    status TEXT DEFAULT 'running',
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON optimization_results(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframe ON optimization_results(timeframe)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_profit_factor ON optimization_results(profit_factor)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON optimization_results(created_at)')
            
            conn.commit()
    
    def save_optimization_summary(self, summary: OptimizationSummary) -> int:
        """
        Save optimization summary to database.
        
        Args:
            summary: OptimizationSummary object
            
        Returns:
            Row ID of inserted record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO optimization_results (
                    symbol, timeframe, strategy_config, profit_factor, win_rate,
                    max_drawdown, total_return, sharpe_ratio, total_trades,
                    avg_trade_duration, tested_combinations, optimization_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                summary.symbol,
                summary.timeframe,
                summary.best_strategy_config,
                summary.profit_factor,
                summary.win_rate,
                summary.max_drawdown,
                summary.total_return,
                summary.sharpe_ratio,
                summary.total_trades,
                summary.avg_trade_duration,
                summary.tested_combinations,
                summary.optimization_time
            ))
            
            return cursor.lastrowid
    
    def save_backtest_result(self, result: BacktestResult) -> int:
        """
        Save detailed backtest result to database.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Row ID of inserted record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            perf = result.performance
            cursor.execute('''
                INSERT INTO backtest_results (
                    symbol, timeframe, strategy_config, profit_factor, win_rate,
                    max_drawdown, total_return, sharpe_ratio, sortino_ratio,
                    calmar_ratio, var_95, total_trades, winning_trades,
                    losing_trades, avg_win, avg_loss, largest_win, largest_loss,
                    time_in_market, avg_trade_duration
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.symbol or 'Unknown',
                result.timeframe,
                str(result.strategy_config),
                perf.profit_factor,
                perf.win_rate,
                perf.max_drawdown,
                perf.total_return,
                perf.sharpe_ratio,
                perf.sortino_ratio,
                perf.calmar_ratio,
                perf.var_95,
                perf.total_trades,
                perf.winning_trades,
                perf.losing_trades,
                perf.avg_win,
                perf.avg_loss,
                perf.largest_win,
                perf.largest_loss,
                perf.time_in_market,
                perf.avg_trade_duration
            ))
            
            return cursor.lastrowid
    
    def create_optimization_run(
        self,
        run_name: str,
        total_stocks: int,
        total_timeframes: int,
        parameter_combinations: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Create new optimization run record.
        
        Args:
            run_name: Name/description of the run
            total_stocks: Number of stocks being tested
            total_timeframes: Number of timeframes being tested
            parameter_combinations: Number of parameter combinations
            metadata: Additional metadata
            
        Returns:
            Run ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO optimization_runs (
                    run_name, start_time, total_stocks, total_timeframes,
                    parameter_combinations, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                run_name,
                datetime.now().isoformat(),
                total_stocks,
                total_timeframes,
                parameter_combinations,
                json.dumps(metadata) if metadata else None
            ))
            
            return cursor.lastrowid
    
    def complete_optimization_run(self, run_id: int, status: str = 'completed') -> None:
        """Mark optimization run as completed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE optimization_runs 
                SET end_time = ?, status = ? 
                WHERE id = ?
            ''', (datetime.now().isoformat(), status, run_id))
    
    def get_best_strategies_per_stock(
        self,
        limit: Optional[int] = None,
        min_profit_factor: float = 1.0
    ) -> pd.DataFrame:
        """
        Get best strategy for each stock.
        
        Args:
            limit: Maximum number of results
            min_profit_factor: Minimum profit factor filter
            
        Returns:
            DataFrame with best strategies per stock
        """
        query = '''
            SELECT symbol, timeframe, strategy_config, profit_factor, win_rate,
                   max_drawdown, total_return, sharpe_ratio, total_trades,
                   avg_trade_duration, created_at
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY profit_factor DESC) as rn
                FROM optimization_results
                WHERE profit_factor >= ?
            ) ranked
            WHERE rn = 1
            ORDER BY profit_factor DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=[min_profit_factor])
    
    def get_best_strategies_per_timeframe(
        self,
        limit: Optional[int] = None,
        min_profit_factor: float = 1.0
    ) -> pd.DataFrame:
        """
        Get best strategy for each timeframe.
        
        Args:
            limit: Maximum number of results
            min_profit_factor: Minimum profit factor filter
            
        Returns:
            DataFrame with best strategies per timeframe
        """
        query = '''
            SELECT symbol, timeframe, strategy_config, profit_factor, win_rate,
                   max_drawdown, total_return, sharpe_ratio, total_trades,
                   avg_trade_duration, created_at
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY timeframe ORDER BY profit_factor DESC) as rn
                FROM optimization_results
                WHERE profit_factor >= ?
            ) ranked
            WHERE rn = 1
            ORDER BY profit_factor DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=[min_profit_factor])
    
    def query_results(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        min_profit_factor: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_trades: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: str = 'profit_factor DESC'
    ) -> pd.DataFrame:
        """
        Query optimization results with filters.
        
        Args:
            symbols: Filter by symbols
            timeframes: Filter by timeframes
            min_profit_factor: Minimum profit factor
            max_drawdown: Maximum drawdown (absolute value)
            min_trades: Minimum number of trades
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of results
            order_by: Order by clause
            
        Returns:
            Filtered DataFrame
        """
        where_clauses = []
        params = []
        
        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            where_clauses.append(f'symbol IN ({placeholders})')
            params.extend(symbols)
        
        if timeframes:
            placeholders = ','.join(['?' for _ in timeframes])
            where_clauses.append(f'timeframe IN ({placeholders})')
            params.extend(timeframes)
        
        if min_profit_factor is not None:
            where_clauses.append('profit_factor >= ?')
            params.append(min_profit_factor)
        
        if max_drawdown is not None:
            where_clauses.append('ABS(max_drawdown) <= ?')
            params.append(max_drawdown)
        
        if min_trades is not None:
            where_clauses.append('total_trades >= ?')
            params.append(min_trades)
        
        if start_date:
            where_clauses.append('created_at >= ?')
            params.append(start_date)
        
        if end_date:
            where_clauses.append('created_at <= ?')
            params.append(end_date)
        
        query = '''
            SELECT symbol, timeframe, strategy_config, profit_factor, win_rate,
                   max_drawdown, total_return, sharpe_ratio, total_trades,
                   avg_trade_duration, tested_combinations, optimization_time,
                   created_at
            FROM optimization_results
        '''
        
        if where_clauses:
            query += ' WHERE ' + ' AND '.join(where_clauses)
        
        query += f' ORDER BY {order_by}'
        
        if limit:
            query += f' LIMIT {limit}'
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute('SELECT COUNT(*) FROM optimization_results')
            total_results = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT symbol) FROM optimization_results')
            unique_stocks = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT timeframe) FROM optimization_results')
            unique_timeframes = cursor.fetchone()[0]
            
            # Performance metrics
            cursor.execute('''
                SELECT 
                    AVG(profit_factor) as avg_pf,
                    MAX(profit_factor) as max_pf,
                    MIN(profit_factor) as min_pf,
                    AVG(win_rate) as avg_wr,
                    AVG(max_drawdown) as avg_dd,
                    COUNT(*) FILTER (WHERE profit_factor > 1.0) as profitable_count
                FROM optimization_results
            ''')
            
            stats = cursor.fetchone()
            
            return {
                'total_results': total_results,
                'unique_stocks': unique_stocks,
                'unique_timeframes': unique_timeframes,
                'avg_profit_factor': stats[0] or 0,
                'max_profit_factor': stats[1] or 0,
                'min_profit_factor': stats[2] or 0,
                'avg_win_rate': stats[3] or 0,
                'avg_max_drawdown': stats[4] or 0,
                'profitable_strategies': stats[5] or 0,
                'profitability_rate': (stats[5] or 0) / max(total_results, 1)
            }
    
    def export_to_csv(self, filename: str, query_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Export results to CSV file.
        
        Args:
            filename: Output filename
            query_params: Optional query parameters for filtering
            
        Returns:
            Path to exported file
        """
        if query_params:
            df = self.query_results(**query_params)
        else:
            df = self.query_results()
        
        df.to_csv(filename, index=False)
        self.logger.info(f"Exported {len(df)} records to {filename}")
        return filename
    
    def get_optimization_runs_history(self) -> pd.DataFrame:
        """Get history of optimization runs."""
        query = '''
            SELECT id, run_name, start_time, end_time, total_stocks,
                   total_timeframes, parameter_combinations, status,
                   created_at
            FROM optimization_runs
            ORDER BY created_at DESC
        '''
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def cleanup_old_results(self, days_to_keep: int = 90) -> int:
        """
        Clean up old results to manage database size.
        
        Args:
            days_to_keep: Number of days to keep
            
        Returns:
            Number of deleted records
        """
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old optimization results
            cursor.execute('''
                DELETE FROM optimization_results 
                WHERE created_at < ?
            ''', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            
            # Delete old backtest results
            cursor.execute('''
                DELETE FROM backtest_results 
                WHERE created_at < ?
            ''', (cutoff_date,))
            
            deleted_count += cursor.rowcount
            
            conn.commit()
        
        self.logger.info(f"Deleted {deleted_count} old records")
        return deleted_count
    
    def vacuum_database(self) -> None:
        """Optimize database performance."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('VACUUM')
        
        self.logger.info("Database vacuumed and optimized")
