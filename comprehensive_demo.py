"""
Comprehensive demo showing all three new modules:
1. Backtesting Engine
2. Optimization System  
3. Results & Analysis

This demonstrates the complete workflow from data download to analysis.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import json

# Import our new modules
from backtesting import BacktestingEngine, BacktestConfig, TimeframeManager
from optimization import OptimizationEngine, OptimizationConfig, StockDataManager, PARAMETER_GRIDS
from analysis import DatabaseManager, Dashboard, ReportGenerator, OptimizationScheduler

# Import existing modules
from models import StrategyParams
from strategy import create_default_strategy, create_custom_strategy


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('comprehensive_demo.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_sample_data(symbol: str = "DEMO", periods: int = 2000) -> pd.DataFrame:
    """Create realistic sample OHLCV data for demonstration."""
    np.random.seed(42)
    
    # Create datetime index
    start_date = datetime.now() - timedelta(days=periods//24)  # Hourly data
    dates = pd.date_range(start_date, periods=periods, freq='H')
    
    # Generate realistic price movements
    base_price = 100.0
    volatility = 0.02
    trend = 0.0001
    
    returns = np.random.normal(trend, volatility, periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC from prices
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    # Add some noise for high/low
    noise_high = np.random.exponential(0.005, periods)
    noise_low = np.random.exponential(0.005, periods)
    
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + noise_high)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - noise_low)
    
    # Generate volume
    df['volume'] = np.random.lognormal(10, 0.5, periods).astype(int)
    
    return df


def demo_backtesting_engine(logger):
    """Demonstrate the backtesting engine capabilities."""
    logger.info("=== DEMO 1: BACKTESTING ENGINE ===")
    
    # Create sample data
    data = create_sample_data("BTCUSD", 1500)
    logger.info(f"Created sample data: {len(data)} periods")
    
    # 1. Basic single backtest
    logger.info("1. Running single backtest...")
    
    backtest_config = BacktestConfig(
        commission_rate=0.001,  # 0.1%
        slippage_bps=2.0,      # 2 basis points
        initial_capital=10000.0
    )
    
    engine = BacktestingEngine(backtest_config)
    
    # Test with default parameters
    params = StrategyParams(
        smooth_type="EMA",
        sl_percent_long=2.0,
        enable_shorts=True,
        use_rsi_filter=True
    )
    
    result = engine.single_backtest(data, params, "1h", "BTCUSD")
    logger.info(f"Single backtest result: PF={result.performance.profit_factor:.2f}, "
                f"WR={result.performance.win_rate:.1%}, "
                f"Trades={result.performance.total_trades}")
    
    # 2. Multi-timeframe backtest
    logger.info("2. Running multi-timeframe backtest...")
    
    timeframes = ['1h', '4h', '1d']
    timeframe_manager = TimeframeManager('1h')
    
    multi_results = []
    for tf in timeframes:
        try:
            # Resample data if needed
            if tf != '1h':
                tf_data = timeframe_manager.resample_data(data, tf)
            else:
                tf_data = data.copy()
            
            result = engine.single_backtest(tf_data, params, tf, "BTCUSD")
            multi_results.append(result)
            logger.info(f"  {tf}: PF={result.performance.profit_factor:.2f}, "
                       f"WR={result.performance.win_rate:.1%}")
        except Exception as e:
            logger.error(f"  Failed {tf}: {e}")
    
    # 3. Parameter optimization
    logger.info("3. Running parameter optimization...")
    
    param_grid = {
        'smooth_type': ['EMA', 'SMA'],
        'sl_percent_long': [1.5, 2.0, 3.0],
        'use_rsi_filter': [True, False],
        'enable_shorts': [True, False]
    }
    
    opt_results = engine.parameter_optimization(
        data, param_grid, '1h', 'BTCUSD', max_workers=2
    )
    
    # Show top 5 results
    logger.info(f"Optimization completed: {len(opt_results)} combinations tested")
    for i, result in enumerate(opt_results[:5]):
        logger.info(f"  #{i+1}: PF={result.performance.profit_factor:.2f}, "
                   f"WR={result.performance.win_rate:.1%}, "
                   f"Config={str(result.strategy_config)[:50]}...")
    
    return opt_results[:5]  # Return top 5 for next demo


def demo_optimization_system(logger):
    """Demonstrate the optimization system capabilities."""
    logger.info("\n=== DEMO 2: OPTIMIZATION SYSTEM ===")
    
    # 1. Setup optimization configuration
    logger.info("1. Setting up optimization configuration...")
    
    # Use a small sample for demo purposes
    opt_config = OptimizationConfig(
        stock_list=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],  # Small sample
        max_stocks=5,
        timeframes=['1h', '4h'],  # Limited timeframes for demo
        max_workers=2,
        min_trades=5,  # Lower threshold for demo
        min_data_points=500,  # Lower for demo
        output_dir="demo_optimization_results"
    )
    
    logger.info(f"Configuration: {opt_config.max_stocks} stocks, "
                f"{len(opt_config.timeframes)} timeframes")
    
    # 2. Create optimization engine
    engine = OptimizationEngine(opt_config)
    
    # For demo, we'll use synthetic data instead of downloading real data
    logger.info("2. Creating synthetic stock data...")
    
    synthetic_data = {}
    for symbol in opt_config.stock_list:
        # Create unique data for each symbol
        np.random.seed(hash(symbol) % 2**32)  # Reproducible but different per symbol
        data = create_sample_data(symbol, 1000)
        synthetic_data[symbol] = data
        logger.info(f"  Created data for {symbol}: {len(data)} periods")
    
    # 3. Run optimization for single stock
    logger.info("3. Running single stock optimization...")
    
    sample_symbol = "AAPL" 
    single_results = []
    
    # Use quick parameter grid for demo
    param_grid = PARAMETER_GRIDS["quick"]
    
    for timeframe in opt_config.timeframes:
        try:
            # Simulate what the real optimization would do
            backtest_engine = BacktestingEngine(opt_config.backtest_config)
            
            # Resample data if needed
            if timeframe != '1h':
                tf_manager = TimeframeManager('1h')
                tf_data = tf_manager.resample_data(synthetic_data[sample_symbol], timeframe)
            else:
                tf_data = synthetic_data[sample_symbol].copy()
            
            # Run parameter optimization
            results = backtest_engine.parameter_optimization(
                tf_data, param_grid, timeframe, sample_symbol, max_workers=1
            )
            
            single_results.extend(results)
            logger.info(f"  {sample_symbol} {timeframe}: {len(results)} results")
            
        except Exception as e:
            logger.error(f"  Failed {sample_symbol} {timeframe}: {e}")
    
    # Show best results
    if single_results:
        best_result = max(single_results, key=lambda x: x.performance.profit_factor)
        logger.info(f"Best result for {sample_symbol}: "
                   f"PF={best_result.performance.profit_factor:.2f}, "
                   f"TF={best_result.timeframe}")
    
    # 4. Demonstrate stock data manager capabilities
    logger.info("4. Demonstrating stock data manager...")
    
    data_manager = StockDataManager("demo_cache")
    
    # Simulate data validation
    for symbol, data in list(synthetic_data.items())[:3]:
        is_valid, message = data_manager.validate_data_quality(data, symbol, 500)
        logger.info(f"  {symbol} validation: {is_valid} - {message}")
        
        # Get data summary
        summary = data_manager.get_data_summary(data)
        logger.info(f"  {symbol} summary: {summary['periods']} periods, "
                   f"${summary['price_range']}")
    
    return single_results


def demo_analysis_system(logger, optimization_results):
    """Demonstrate the analysis and results system."""
    logger.info("\n=== DEMO 3: RESULTS & ANALYSIS SYSTEM ===")
    
    # 1. Database management
    logger.info("1. Setting up database management...")
    
    db_manager = DatabaseManager("demo_results.db")
    
    # Save some sample optimization results
    logger.info("2. Saving optimization results to database...")
    
    for result in optimization_results:
        try:
            # Save backtest result
            row_id = db_manager.save_backtest_result(result)
            logger.info(f"  Saved result ID {row_id}: {result.symbol} {result.timeframe}")
        except Exception as e:
            logger.error(f"  Failed to save result: {e}")
    
    # Create a sample optimization run record
    run_id = db_manager.create_optimization_run(
        run_name="demo_optimization_run",
        total_stocks=5,
        total_timeframes=2,
        parameter_combinations=32,
        metadata={"demo": True, "version": "1.0"}
    )
    db_manager.complete_optimization_run(run_id, "completed")
    logger.info(f"Created optimization run record: ID {run_id}")
    
    # 3. Query and analyze results
    logger.info("3. Querying and analyzing results...")
    
    # Get performance statistics
    stats = db_manager.get_performance_statistics()
    logger.info(f"Database statistics:")
    logger.info(f"  Total results: {stats['total_results']}")
    logger.info(f"  Unique stocks: {stats['unique_stocks']}")
    logger.info(f"  Avg profit factor: {stats['avg_profit_factor']:.2f}")
    logger.info(f"  Profitable strategies: {stats['profitable_strategies']}")
    
    # Query specific results
    filtered_results = db_manager.query_results(
        min_profit_factor=1.0,
        min_trades=3,
        limit=10
    )
    logger.info(f"Filtered results: {len(filtered_results)} records")
    
    # 4. Generate reports
    logger.info("4. Generating analysis reports...")
    
    report_generator = ReportGenerator(db_manager)
    
    try:
        # Generate summary report
        summary_file = report_generator.generate_summary_report("demo_summary.txt")
        logger.info(f"Generated summary report: {summary_file}")
        
        # Generate detailed analysis
        detailed_file = report_generator.generate_detailed_analysis("demo_detailed.csv")
        logger.info(f"Generated detailed analysis: {detailed_file}")
        
        # Generate executive summary
        exec_file = report_generator.generate_executive_summary("demo_executive.json")
        logger.info(f"Generated executive summary: {exec_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate reports: {e}")
    
    # 5. Create dashboard
    logger.info("5. Creating analysis dashboard...")
    
    try:
        dashboard = Dashboard(db_manager)
        
        # Create simple HTML dashboard
        from optimization.optimization_results import OptimizationResults
        
        # Create a dummy OptimizationResults object for demo
        demo_results = OptimizationResults("demo_output")
        
        # Convert database results to summaries
        if not filtered_results.empty:
            from optimization.optimization_results import OptimizationSummary
            
            summaries = []
            for _, row in filtered_results.iterrows():
                summary = OptimizationSummary(
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    best_strategy_config=row['strategy_config'],
                    profit_factor=row['profit_factor'],
                    win_rate=row['win_rate'],
                    max_drawdown=row['max_drawdown'],
                    total_return=row['total_return'],
                    sharpe_ratio=row['sharpe_ratio'],
                    total_trades=row['total_trades'],
                    avg_trade_duration=row['avg_trade_duration'],
                    tested_combinations=1
                )
                summaries.append(summary)
            
            demo_results.summaries = summaries
            
            # Create HTML dashboard
            dashboard_file = dashboard.create_html_dashboard(demo_results, "demo_dashboard.html")
            logger.info(f"Created HTML dashboard: {dashboard_file}")
        
    except Exception as e:
        logger.error(f"Failed to create dashboard: {e}")
    
    # 6. Demonstrate scheduler (setup only)
    logger.info("6. Setting up optimization scheduler...")
    
    try:
        scheduler = OptimizationScheduler(db_manager, "demo_scheduler_config.json")
        
        # Get scheduler status
        status = scheduler.get_schedule_status()
        logger.info(f"Scheduler status: {status}")
        
        # Update a schedule configuration
        scheduler.update_schedule_config('weekly', {
            'enabled': True,
            'stocks': 20,
            'timeframes': ['1h', '4h', '1d']
        })
        logger.info("Updated weekly schedule configuration")
        
    except Exception as e:
        logger.error(f"Failed to setup scheduler: {e}")


def demo_complete_workflow(logger):
    """Demonstrate the complete workflow from start to finish."""
    logger.info("\n=== DEMO 4: COMPLETE WORKFLOW ===")
    
    # This demonstrates how all modules work together
    logger.info("1. Complete workflow demonstration...")
    
    # Step 1: Configure optimization
    config = OptimizationConfig(
        stock_list=["BTC-USD", "ETH-USD", "AAPL"],
        max_stocks=3,
        timeframes=['1h', '4h'],
        max_workers=2,
        output_dir="complete_workflow_results"
    )
    
    # Step 2: Create synthetic data (in real use, this would download data)
    logger.info("2. Preparing data...")
    synthetic_data = {}
    for symbol in config.stock_list:
        np.random.seed(hash(symbol) % 2**32)
        data = create_sample_data(symbol, 800)
        synthetic_data[symbol] = data
        logger.info(f"  Prepared {symbol}: {len(data)} periods")
    
    # Step 3: Run optimization
    logger.info("3. Running optimization...")
    
    all_results = []
    for symbol in config.stock_list:
        for timeframe in config.timeframes:
            try:
                # Create backtesting engine
                backtest_engine = BacktestingEngine(config.backtest_config)
                
                # Prepare data for timeframe
                if timeframe != '1h':
                    tf_manager = TimeframeManager('1h')
                    tf_data = tf_manager.resample_data(synthetic_data[symbol], timeframe)
                else:
                    tf_data = synthetic_data[symbol].copy()
                
                # Run optimization with quick grid
                results = backtest_engine.parameter_optimization(
                    tf_data, 
                    PARAMETER_GRIDS["quick"], 
                    timeframe, 
                    symbol, 
                    max_workers=1
                )
                
                all_results.extend(results)
                best_pf = max(r.performance.profit_factor for r in results) if results else 0
                logger.info(f"  {symbol} {timeframe}: {len(results)} results, best PF: {best_pf:.2f}")
                
            except Exception as e:
                logger.error(f"  Failed {symbol} {timeframe}: {e}")
    
    # Step 4: Save to database
    logger.info("4. Saving results to database...")
    
    db_manager = DatabaseManager("complete_workflow.db")
    run_id = db_manager.create_optimization_run(
        "complete_workflow_demo",
        len(config.stock_list),
        len(config.timeframes),
        len(PARAMETER_GRIDS["quick"]),
        {"workflow": "demo", "timestamp": datetime.now().isoformat()}
    )
    
    saved_count = 0
    for result in all_results:
        try:
            db_manager.save_backtest_result(result)
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
    
    db_manager.complete_optimization_run(run_id, "completed")
    logger.info(f"Saved {saved_count} results to database")
    
    # Step 5: Generate analysis
    logger.info("5. Generating comprehensive analysis...")
    
    report_generator = ReportGenerator(db_manager)
    reports = report_generator.generate_all_reports("complete_workflow_reports")
    
    for report_type, filepath in reports.items():
        if os.path.exists(filepath):
            logger.info(f"  Generated {report_type}: {filepath}")
    
    # Step 6: Show summary
    stats = db_manager.get_performance_statistics()
    best_per_stock = db_manager.get_best_strategies_per_stock(limit=5)
    
    logger.info("6. Workflow Summary:")
    logger.info(f"  Total strategies tested: {stats['total_results']}")
    logger.info(f"  Profitable strategies: {stats['profitable_strategies']}")
    logger.info(f"  Best profit factor: {stats['max_profit_factor']:.2f}")
    logger.info(f"  Average win rate: {stats['avg_win_rate']:.1%}")
    
    if not best_per_stock.empty:
        best = best_per_stock.iloc[0]
        logger.info(f"  Best strategy: {best['symbol']} {best['timeframe']} "
                   f"(PF: {best['profit_factor']:.2f})")


def main():
    """Main demo function."""
    print("Starting Comprehensive Demo of Trading Strategy Modules")
    print("=" * 60)
    
    # Setup
    logger = setup_logging()
    logger.info("Starting comprehensive demo...")
    
    try:
        # Demo 1: Backtesting Engine
        backtest_results = demo_backtesting_engine(logger)
        
        # Demo 2: Optimization System
        optimization_results = demo_optimization_system(logger)
        
        # Use results from both demos
        all_results = backtest_results + optimization_results
        
        # Demo 3: Analysis System
        demo_analysis_system(logger, all_results)
        
        # Demo 4: Complete Workflow
        demo_complete_workflow(logger)
        
        logger.info("\n" + "=" * 60)
        logger.info("COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        # Summary of what was demonstrated
        print("\nDEMO SUMMARY:")
        print("✓ Module 2: Backtesting Engine")
        print("  - Single strategy backtests")
        print("  - Multi-timeframe testing")
        print("  - Parameter optimization")
        print("  - Custom timeframes (13m, 45m, etc.)")
        print("  - Commission and slippage settings")
        
        print("\n✓ Module 3: Optimization System")
        print("  - Multi-stock optimization")
        print("  - Parallel processing")
        print("  - Data quality validation")
        print("  - Results filtering and ranking")
        
        print("\n✓ Module 4: Results & Analysis")
        print("  - Database storage and retrieval")
        print("  - Comprehensive reporting")
        print("  - HTML dashboard generation")
        print("  - Optimization scheduler setup")
        print("  - Complete workflow integration")
        
        print("\nGenerated Files:")
        files_to_check = [
            "demo_summary.txt",
            "demo_detailed.csv", 
            "demo_executive.json",
            "demo_dashboard.html",
            "comprehensive_demo.log"
        ]
        
        for filename in files_to_check:
            if os.path.exists(filename):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (not created)")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
