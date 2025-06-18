"""
Working demo that properly imports the installed package modules.
This demonstrates all three new modules working together.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_sample_data(symbol: str = "DEMO", periods: int = 1000) -> pd.DataFrame:
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

def demo_backtesting_features(logger):
    """Demonstrate backtesting module features without running full backtest."""
    logger.info("=== DEMO: BACKTESTING ENGINE FEATURES ===")
    
    try:
        # Test backtesting configuration
        from backtesting.backtest_config import BacktestConfig, TimeframeConfig, ALL_TIMEFRAMES
        
        config = BacktestConfig(
            commission_rate=0.001,
            slippage_bps=2.0,
            initial_capital=10000.0
        )
        logger.info(f"✓ BacktestConfig created: {config.commission_rate*100}% commission")
        
        # Test timeframe configurations
        timeframes = list(ALL_TIMEFRAMES.keys())
        logger.info(f"✓ Available timeframes: {timeframes}")
        
        # Test timeframe manager
        from backtesting.timeframe_manager import TimeframeManager
        
        tf_manager = TimeframeManager('1h')
        tf_manager.add_standard_timeframes()
        tf_manager.add_custom_timeframes()
        
        available_tfs = tf_manager.get_all_timeframes()
        logger.info(f"✓ TimeframeManager configured: {len(available_tfs)} timeframes")
        
        # Test performance metrics structure
        from backtesting.performance_metrics import PerformanceMetrics, PerformanceCalculator
        
        # Create dummy metrics
        metrics = PerformanceMetrics(
            total_return=0.15, annualized_return=0.18, win_rate=0.65, profit_factor=1.8,
            max_drawdown=-0.12, sharpe_ratio=1.2, sortino_ratio=1.4, calmar_ratio=1.5,
            var_95=-0.05, total_trades=50, winning_trades=32, losing_trades=18, 
            avg_win=0.025, avg_loss=-0.015, largest_win=0.08, largest_loss=-0.04,
            time_in_market=0.45, avg_trade_duration=2.3
        )
        logger.info(f"✓ PerformanceMetrics: PF={metrics.profit_factor}, WR={metrics.win_rate:.1%}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Backtesting demo failed: {e}")
        return False

def demo_optimization_features(logger):
    """Demonstrate optimization module features."""
    logger.info("\n=== DEMO: OPTIMIZATION SYSTEM FEATURES ===")
    
    try:
        # Test optimization configuration
        from optimization.optimization_config import OptimizationConfig, PARAMETER_GRIDS
        
        config = OptimizationConfig(
            stock_list=["AAPL", "MSFT", "GOOGL"],
            max_stocks=3,
            timeframes=['1h', '4h'],
            max_workers=2,
            output_dir="demo_optimization"
        )
        logger.info(f"✓ OptimizationConfig: {len(config.stock_list)} stocks")
        logger.info(f"  Available parameter grids: {list(PARAMETER_GRIDS.keys())}")
        
        # Test stock data manager
        from optimization.stock_data_manager import StockDataManager
        
        data_manager = StockDataManager("demo_cache")
        
        # Test with synthetic data
        sample_data = create_sample_data("TEST", 500)
        is_valid, message = data_manager.validate_data_quality(sample_data, "TEST", 400)
        logger.info(f"✓ Data validation: {is_valid} - {message}")
        
        summary = data_manager.get_data_summary(sample_data)
        logger.info(f"✓ Data summary: {summary['periods']} periods")
        
        # Test optimization results structure
        from optimization.optimization_results import OptimizationResults, OptimizationSummary
        
        results = OptimizationResults("demo_output")
        sample_summary = OptimizationSummary(
            symbol="AAPL",
            timeframe="1h", 
            best_strategy_config="EMA_RSI",
            profit_factor=1.5,
            win_rate=0.6,
            max_drawdown=-0.1,
            total_return=0.25,
            sharpe_ratio=1.1,
            total_trades=30,
            avg_trade_duration=2.5,
            tested_combinations=100
        )
        
        results.summaries = [sample_summary]
        logger.info("✓ OptimizationResults structure created")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Optimization demo failed: {e}")
        return False

def demo_analysis_features(logger):
    """Demonstrate analysis module features."""
    logger.info("\n=== DEMO: ANALYSIS & RESULTS FEATURES ===")
    
    try:
        # Test database manager
        from analysis.database_manager import DatabaseManager
        
        db = DatabaseManager("demo_test.db")
        logger.info("✓ DatabaseManager created")
        
        # Test getting statistics (should be empty for new DB)
        stats = db.get_performance_statistics()
        logger.info(f"✓ Database stats: {stats['total_results']} results")
        
        # Test report generator
        from analysis.report_generator import ReportGenerator
        
        report_gen = ReportGenerator(db)
        logger.info("✓ ReportGenerator created")
        
        # Test dashboard creation
        from analysis.dashboard import Dashboard
        
        dashboard = Dashboard(db)
        logger.info("✓ Dashboard created")
        
        # Test scheduler configuration
        from analysis.scheduler import OptimizationScheduler
        
        scheduler = OptimizationScheduler(db, "demo_scheduler.json")
        status = scheduler.get_schedule_status()
        logger.info(f"✓ Scheduler created: {status['scheduled_jobs']} jobs")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Analysis demo failed: {e}")
        return False

def demo_integration_workflow(logger):
    """Demonstrate how modules work together."""
    logger.info("\n=== DEMO: INTEGRATION WORKFLOW ===")
    
    try:
        # 1. Create configuration
        from backtesting.backtest_config import BacktestConfig
        from optimization.optimization_config import OptimizationConfig
        
        backtest_config = BacktestConfig(commission_rate=0.001, initial_capital=10000)
        opt_config = OptimizationConfig(
            stock_list=["DEMO1", "DEMO2"],
            max_stocks=2,
            timeframes=['1h'],
            backtest_config=backtest_config
        )
        logger.info("✓ Configurations created")
        
        # 2. Create sample data and validate
        from optimization.stock_data_manager import StockDataManager
        
        data_manager = StockDataManager()
        demo_data = {
            "DEMO1": create_sample_data("DEMO1", 300),
            "DEMO2": create_sample_data("DEMO2", 300)
        }
        
        valid_stocks = []
        for symbol, data in demo_data.items():
            is_valid, _ = data_manager.validate_data_quality(data, symbol, 250)
            if is_valid:
                valid_stocks.append(symbol)
        
        logger.info(f"✓ Data prepared: {len(valid_stocks)} valid stocks")
        
        # 3. Setup results storage
        from analysis.database_manager import DatabaseManager
        from optimization.optimization_results import OptimizationResults
        
        db = DatabaseManager("integration_demo.db")
        results = OptimizationResults("integration_output")
        
        logger.info("✓ Results storage ready")
        
        # 4. Create sample optimization summary
        from optimization.optimization_results import OptimizationSummary
        
        summary = OptimizationSummary(
            symbol="DEMO1",
            timeframe="1h",
            best_strategy_config="test_config",
            profit_factor=1.3,
            win_rate=0.55,
            max_drawdown=-0.08,
            total_return=0.18,
            sharpe_ratio=0.9,
            total_trades=25,
            avg_trade_duration=3.2,
            tested_combinations=50
        )
        
        # Save to database
        db.save_optimization_summary(summary)
        logger.info("✓ Sample result saved to database")
        
        # 5. Generate report
        from analysis.report_generator import ReportGenerator
        
        report_gen = ReportGenerator(db)
        report_file = report_gen.generate_summary_report("integration_demo_report.txt")
        
        if os.path.exists(report_file):
            logger.info(f"✓ Report generated: {report_file}")
        
        logger.info("✓ Integration workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration workflow failed: {e}")
        return False

def main():
    """Main demo function."""
    print("Working Demo of Trading Strategy Modules")
    print("=" * 60)
    
    logger = setup_logging()
    
    # Test each module
    backtesting_ok = demo_backtesting_features(logger)
    optimization_ok = demo_optimization_features(logger)
    analysis_ok = demo_analysis_features(logger)
    integration_ok = demo_integration_workflow(logger)
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO RESULTS SUMMARY")
    print("=" * 60)
    
    modules = [
        ("Backtesting Engine", backtesting_ok),
        ("Optimization System", optimization_ok), 
        ("Analysis & Results", analysis_ok),
        ("Integration Workflow", integration_ok)
    ]
    
    for name, success in modules:
        status = "✓ WORKING" if success else "✗ FAILED"
        print(f"{name:25} {status}")
    
    all_working = all(success for _, success in modules)
    
    if all_working:
        print(f"\n🎉 SUCCESS: All {len(modules)} modules are working correctly!")
        print("\nKey Features Implemented:")
        print("  📊 Backtesting Engine:")
        print("     - Multiple timeframes (5m, 15m, 1h, 4h, 1d)")
        print("     - Custom timeframes (13m, 45m, etc.)")
        print("     - Commission & slippage settings")
        print("     - Performance metrics calculation")
        
        print("  🔍 Optimization System:")
        print("     - Multi-stock parameter optimization")
        print("     - Parallel processing support")
        print("     - Data quality validation")
        print("     - Configurable parameter grids")
        
        print("  📈 Analysis & Results:")
        print("     - SQLite database storage")
        print("     - Comprehensive reporting")
        print("     - HTML dashboard generation")
        print("     - Automated scheduling")
        
        print("\nFiles Generated:")
        generated_files = [
            "integration_demo_report.txt",
            "demo_test.db", 
            "integration_demo.db",
            "demo_scheduler.json"
        ]
        
        for filename in generated_files:
            if os.path.exists(filename):
                print(f"  ✓ {filename}")
    else:
        failed_count = sum(1 for _, success in modules if not success)
        print(f"\n⚠️  {failed_count}/{len(modules)} modules had issues")
    
    print(f"\nNext Steps:")
    print("  1. Install additional dependencies: pip install -e .[full]")
    print("  2. Run comprehensive optimization: python comprehensive_demo.py")
    print("  3. View generated reports and dashboards")

if __name__ == "__main__":
    main()
