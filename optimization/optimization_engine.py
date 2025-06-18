"""Main optimization engine for testing strategies across multiple stocks."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import gc

from ..backtesting.backtesting_engine import BacktestingEngine, BacktestResult
from ..models import StrategyParams
from .optimization_config import OptimizationConfig, OptimizationJob, PARAMETER_GRIDS
from .stock_data_manager import StockDataManager
from .optimization_results import OptimizationResults


class OptimizationEngine:
    """
    Main optimization engine for testing strategies across multiple stocks.
    
    Features:
    - Test all indicator combinations across 50+ stocks
    - Efficient parallel processing
    - Output: Ticker | Timeframe | Strategy Config | PF | WR | DD
    - Memory management for large-scale optimization
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimization engine.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.data_manager = StockDataManager()
        self.backtesting_engine = BacktestingEngine(self.config.backtest_config)
        self.results = OptimizationResults(self.config.output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run_full_optimization(
        self,
        parameter_grid: Optional[Dict[str, List[Any]]] = None
    ) -> OptimizationResults:
        """
        Run full optimization across all stocks and timeframes.
        
        Args:
            parameter_grid: Custom parameter grid to test
            
        Returns:
            OptimizationResults with all results
        """
        start_time = time.time()
        
        # Use provided grid or get from config
        if parameter_grid is None:
            parameter_grid = self.config.parameter_grid or PARAMETER_GRIDS["comprehensive"]
        
        self.logger.info(f"Starting optimization with {len(self.config.stock_list)} stocks")
        self.logger.info(f"Timeframes: {self.config.timeframes}")
        self.logger.info(f"Parameter combinations: {self._count_combinations(parameter_grid)}")
        
        # Download data for all stocks
        self.logger.info("Downloading stock data...")
        stock_data = self._download_all_data()
        
        if not stock_data:
            self.logger.error("No stock data available")
            return self.results
        
        # Create optimization jobs
        jobs = self._create_optimization_jobs(stock_data.keys(), parameter_grid)
        self.logger.info(f"Created {len(jobs)} optimization jobs")
        
        # Run optimization jobs in parallel
        self._run_optimization_jobs(jobs, stock_data)
        
        # Generate summaries
        self.results.generate_summaries()
        
        # Add metadata
        elapsed_time = time.time() - start_time
        self.results.add_metadata('optimization_time', elapsed_time)
        self.results.add_metadata('total_stocks', len(stock_data))
        self.results.add_metadata('total_timeframes', len(self.config.timeframes))
        self.results.add_metadata('parameter_combinations', self._count_combinations(parameter_grid))
        
        self.logger.info(f"Optimization completed in {elapsed_time:.1f} seconds")
        self.logger.info(f"Total results: {len(self.results.results)}")
        
        # Export results if configured
        if self.config.save_results:
            self.results.export_to_csv()
            self.results.export_detailed_results()
        
        return self.results
    
    def run_single_stock_optimization(
        self,
        symbol: str,
        parameter_grid: Optional[Dict[str, List[Any]]] = None,
        timeframes: Optional[List[str]] = None
    ) -> List[BacktestResult]:
        """
        Run optimization for a single stock across timeframes.
        
        Args:
            symbol: Stock symbol
            parameter_grid: Parameter grid to test
            timeframes: Timeframes to test
            
        Returns:
            List of BacktestResult objects
        """
        if timeframes is None:
            timeframes = self.config.timeframes
        
        if parameter_grid is None:
            parameter_grid = PARAMETER_GRIDS["quick"]
        
        self.logger.info(f"Optimizing {symbol} across {len(timeframes)} timeframes")
        
        # Download data
        base_data = self.data_manager.download_stock_data(
            symbol, 
            self.config.data_start_date,
            self.config.data_end_date,
            self.config.base_timeframe
        )
        
        if base_data is None:
            self.logger.error(f"Failed to download data for {symbol}")
            return []
        
        # Validate data quality
        is_valid, message = self.data_manager.validate_data_quality(
            base_data, symbol, self.config.min_data_points
        )
        
        if not is_valid:
            self.logger.warning(f"Data quality issues for {symbol}: {message}")
            return []
        
        # Run optimization for each timeframe
        all_results = []
        
        for timeframe in timeframes:
            self.logger.info(f"Optimizing {symbol} on {timeframe}")
            
            # Run parameter optimization
            results = self.backtesting_engine.parameter_optimization(
                base_data.copy(),
                parameter_grid,
                timeframe,
                symbol,
                max_workers=1  # Sequential for single stock
            )
            
            all_results.extend(results)
            self.logger.info(f"Completed {len(results)} tests for {symbol} {timeframe}")
        
        return all_results
    
    def get_best_strategies_summary(self) -> Dict[str, Any]:
        """Get summary of best strategies found."""
        if not self.results.summaries:
            self.results.generate_summaries()
        
        best_per_stock = self.results.get_best_strategies_per_stock()
        best_per_timeframe = self.results.get_best_strategies_per_timeframe()
        stats = self.results.get_performance_statistics()
        
        return {
            'best_per_stock': {k: v.to_dict() for k, v in best_per_stock.items()},
            'best_per_timeframe': {k: v.to_dict() for k, v in best_per_timeframe.items()},
            'overall_statistics': stats
        }
    
    def _download_all_data(self) -> Dict[str, pd.DataFrame]:
        """Download data for all stocks."""
        # Filter stocks if criteria specified
        stocks_to_download = self.config.stock_list[:self.config.max_stocks]
        
        if self.config.min_market_cap:
            stocks_to_download = self.data_manager.filter_stocks_by_criteria(
                stocks_to_download,
                min_market_cap=self.config.min_market_cap,
                max_workers=self.config.max_workers
            )
        
        # Download data
        stock_data = self.data_manager.download_multiple_stocks(
            stocks_to_download,
            self.config.data_start_date,
            self.config.data_end_date,
            self.config.base_timeframe,
            max_workers=self.config.max_workers
        )
        
        # Validate data quality
        valid_stocks = {}
        for symbol, data in stock_data.items():
            is_valid, message = self.data_manager.validate_data_quality(
                data, symbol, self.config.min_data_points
            )
            
            if is_valid:
                valid_stocks[symbol] = data
            else:
                self.logger.warning(f"Excluding {symbol}: {message}")
        
        self.logger.info(f"Valid stocks for optimization: {len(valid_stocks)}")
        return valid_stocks
    
    def _create_optimization_jobs(
        self,
        symbols: List[str],
        parameter_grid: Dict[str, List[Any]]
    ) -> List[OptimizationJob]:
        """Create optimization jobs for all symbol/timeframe combinations."""
        jobs = []
        
        for symbol in symbols:
            for timeframe in self.config.timeframes:
                job = OptimizationJob(
                    symbol=symbol,
                    timeframe=timeframe,
                    parameter_grid=parameter_grid.copy()
                )
                jobs.append(job)
        
        return jobs
    
    def _run_optimization_jobs(
        self,
        jobs: List[OptimizationJob],
        stock_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Run optimization jobs in parallel."""
        if self.config.max_workers > 1:
            self._run_parallel_optimization(jobs, stock_data)
        else:
            self._run_sequential_optimization(jobs, stock_data)
    
    def _run_parallel_optimization(
        self,
        jobs: List[OptimizationJob],
        stock_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Run optimization jobs in parallel."""
        completed_jobs = 0
        total_jobs = len(jobs)
        
        # Process jobs in batches to manage memory
        batch_size = self.config.batch_size
        
        for i in range(0, len(jobs), batch_size):
            batch_jobs = jobs[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(jobs)-1)//batch_size + 1}")
            
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                
                for job in batch_jobs:
                    data = stock_data[job.symbol].copy()
                    future = executor.submit(
                        self._run_single_optimization_job,
                        job,
                        data
                    )
                    futures[future] = job
                
                # Collect results
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        results = future.result()
                        if results:
                            self.results.add_results(results)
                        completed_jobs += 1
                        
                        if completed_jobs % 10 == 0:
                            progress = completed_jobs / total_jobs * 100
                            self.logger.info(f"Progress: {progress:.1f}% ({completed_jobs}/{total_jobs})")
                    
                    except Exception as e:
                        self.logger.error(f"Job {job} failed: {str(e)}")
                        completed_jobs += 1
            
            # Force garbage collection after each batch
            gc.collect()
    
    def _run_sequential_optimization(
        self,
        jobs: List[OptimizationJob],
        stock_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Run optimization jobs sequentially."""
        for i, job in enumerate(jobs):
            self.logger.info(f"Processing job {i+1}/{len(jobs)}: {job}")
            
            try:
                data = stock_data[job.symbol].copy()
                results = self._run_single_optimization_job(job, data)
                
                if results:
                    self.results.add_results(results)
                
                # Progress update
                if (i + 1) % 10 == 0:
                    progress = (i + 1) / len(jobs) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(jobs)})")
            
            except Exception as e:
                self.logger.error(f"Job {job} failed: {str(e)}")
    
    def _run_single_optimization_job(
        self,
        job: OptimizationJob,
        data: pd.DataFrame
    ) -> List[BacktestResult]:
        """Run a single optimization job."""
        try:
            # Create fresh backtesting engine for this job
            backtesting_engine = BacktestingEngine(self.config.backtest_config)
            
            # Run parameter optimization
            results = backtesting_engine.parameter_optimization(
                data,
                job.parameter_grid,
                job.timeframe,
                job.symbol,
                max_workers=1  # Use single worker per job
            )
            
            # Filter results based on minimum trades
            filtered_results = [
                r for r in results 
                if r.performance.total_trades >= self.config.min_trades
            ]
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Failed to run job {job}: {str(e)}")
            return []
    
    def _count_combinations(self, parameter_grid: Dict[str, List[Any]]) -> int:
        """Count total parameter combinations."""
        count = 1
        for values in parameter_grid.values():
            count *= len(values)
        return count
    
    def export_results(
        self,
        format: str = "csv",
        filename: Optional[str] = None
    ) -> str:
        """
        Export optimization results.
        
        Args:
            format: Export format ('csv' or 'json')
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if format.lower() == "csv":
            return self.results.export_to_csv(filename)
        elif format.lower() == "json":
            return self.results.export_to_json(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get current optimization configuration."""
        return {
            'max_stocks': self.config.max_stocks,
            'timeframes': self.config.timeframes,
            'max_workers': self.config.max_workers,
            'min_trades': self.config.min_trades,
            'optimization_metric': self.config.optimization_metric,
            'output_dir': self.config.output_dir
        }
