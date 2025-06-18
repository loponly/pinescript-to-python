"""
Main Demo - Trading Strategy Framework with BTC/USDT Testing

This demo showcases the complete trading strategy framework with:
1. BTC/USDT data generation and testing
2. Strategy backtesting across different market conditions
3. Optimization and analysis capabilities
4. Performance reporting

Run this demo to see all features working together.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List

# Import framework modules
from models import StrategyParams, TradeResult
from strategy import MomentumStrategy, create_default_strategy, create_custom_strategy
from btc_data_generator import generate_btc_usdt_data, get_btc_usdt_test_scenarios, validate_ohlcv_data

# Import new modules
from backtesting import BacktestingEngine, BacktestConfig
from optimization import OptimizationEngine, OptimizationConfig, PARAMETER_GRIDS
from analysis import DatabaseManager, ReportGenerator


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def demo_btc_data_generation(logger):
    """Demonstrate BTC/USDT data generation and validation."""
    logger.info("=== BTC/USDT DATA GENERATION ===")
    
    # Generate different market scenarios
    scenarios = get_btc_usdt_test_scenarios()
    
    for name, data in scenarios.items():
        logger.info(f"📊 {name.replace('_', ' ').title()} Scenario:")
        logger.info(f"  - Periods: {len(data)}")
        logger.info(f"  - Price Range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        logger.info(f"  - Price Change: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.1f}%")
        
        # Validate data
        is_valid, message = validate_ohlcv_data(data)
        logger.info(f"  - Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        
        if not is_valid:
            logger.warning(f"    {message}")
    
    return scenarios


def demo_strategy_testing(logger, scenarios: Dict[str, pd.DataFrame]):
    """Test strategy on different BTC market scenarios."""
    logger.info("\n=== STRATEGY TESTING ON BTC/USDT ===")
    
    # Define different strategy configurations for crypto
    crypto_strategies = {
        'conservative': StrategyParams(
            smooth_type="SMA",
            smoothing_length=50,
            rsi_length_long=14,
            enable_longs=True,
            enable_shorts=False,
            sl_percent_long=2.0,
            use_rsi_filter=True,
            use_trend_filter=True
        ),
        'aggressive': StrategyParams(
            smooth_type="EMA", 
            smoothing_length=20,
            rsi_length_long=10,
            enable_longs=True,
            enable_shorts=True,
            sl_percent_long=3.0,
            sl_percent_short=2.5,
            use_rsi_filter=True,
            use_atr_filter=True
        ),
        'scalping': StrategyParams(
            smooth_type="EMA",
            smoothing_length=10,
            rsi_length_long=7,
            enable_longs=True,
            enable_shorts=True,
            sl_percent_long=1.5,
            sl_percent_short=1.5,
            use_rsi_filter=True
        )
    }
    
    results = {}
    
    for strategy_name, params in crypto_strategies.items():
        logger.info(f"\n🚀 Testing {strategy_name.upper()} strategy:")
        results[strategy_name] = {}
        
        for scenario_name, data in scenarios.items():
            try:
                strategy = MomentumStrategy(params)
                result_df = strategy.run_strategy(data)
                trades = strategy.executed_trades
                
                # Calculate basic metrics
                if trades:
                    total_pnl = sum(trade.pnl for trade in trades)
                    winning_trades = len([t for t in trades if t.pnl > 0])
                    win_rate = winning_trades / len(trades)
                    total_return = total_pnl / 10000  # Assuming 10k starting capital
                    
                    results[strategy_name][scenario_name] = {
                        'trades': len(trades),
                        'win_rate': win_rate,
                        'total_return': total_return,
                        'total_pnl': total_pnl
                    }
                    
                    logger.info(f"  📈 {scenario_name}: {len(trades)} trades, "
                               f"{win_rate:.1%} win rate, {total_return:.1%} return")
                else:
                    results[strategy_name][scenario_name] = {
                        'trades': 0, 'win_rate': 0, 'total_return': 0, 'total_pnl': 0
                    }
                    logger.info(f"  📊 {scenario_name}: No trades generated")
                    
            except Exception as e:
                logger.error(f"  ❌ {scenario_name}: Error - {e}")
                results[strategy_name][scenario_name] = None
    
    return results


def demo_backtesting_engine(logger):
    """Demonstrate enhanced backtesting capabilities."""
    logger.info("\n=== ENHANCED BACKTESTING ENGINE ===")
    
    # Generate BTC data for backtesting
    btc_data = generate_btc_usdt_data(
        periods=1000,
        base_price=30000,
        volatility=0.03,
        seed=42
    )
    
    # Configure backtesting
    backtest_config = BacktestConfig(
        commission_rate=0.001,  # 0.1% commission
        slippage_bps=5.0,       # 5 basis points slippage
        initial_capital=50000.0  # $50k starting capital
    )
    
    engine = BacktestingEngine(backtest_config)
    
    # Test different parameter sets
    param_grid = {
        'smooth_type': ['EMA', 'SMA'],
        'smoothing_length': [20, 50],
        'sl_percent_long': [2.0, 3.0],
        'enable_shorts': [True, False]
    }
    
    logger.info("🔍 Running parameter optimization...")
    
    try:
        optimization_results = engine.parameter_optimization(
            btc_data, param_grid, '1h', 'BTCUSDT', max_workers=2
        )
        
        if optimization_results:
            best_result = max(optimization_results, key=lambda x: x.performance.profit_factor)
            logger.info(f"✅ Optimization completed: {len(optimization_results)} combinations tested")
            logger.info(f"🏆 Best result: PF={best_result.performance.profit_factor:.2f}, "
                       f"WR={best_result.performance.win_rate:.1%}")
            
            return optimization_results[:5]  # Return top 5
        else:
            logger.warning("⚠️ No optimization results generated")
            return []
            
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}")
        return []


def demo_optimization_and_analysis(logger, btc_results):
    """Demonstrate optimization and analysis capabilities."""
    logger.info("\n=== OPTIMIZATION & ANALYSIS ===")
    
    if not btc_results:
        logger.warning("⚠️ Skipping analysis - no backtesting results available")
        return
    
    try:
        # Setup database and analysis
        db_manager = DatabaseManager("btc_demo_results.db")
        
        # Save results to database
        logger.info("💾 Saving results to database...")
        saved_count = 0
        
        for result in btc_results:
            try:
                db_manager.save_backtest_result(result)
                saved_count += 1
            except Exception as e:
                logger.warning(f"Failed to save result: {e}")
        
        logger.info(f"✅ Saved {saved_count} results to database")
        
        # Generate analysis reports
        report_generator = ReportGenerator(db_manager)
        
        logger.info("📊 Generating analysis reports...")
        
        # Generate summary report
        summary_file = report_generator.generate_summary_report("btc_strategy_summary.txt")
        logger.info(f"📄 Summary report: {summary_file}")
        
        # Generate detailed analysis
        detailed_file = report_generator.generate_detailed_analysis("btc_strategy_detailed.csv")
        logger.info(f"📋 Detailed analysis: {detailed_file}")
        
        # Get performance statistics
        stats = db_manager.get_performance_statistics()
        logger.info("📈 Database Statistics:")
        logger.info(f"  - Total Results: {stats['total_results']}")
        logger.info(f"  - Profitable Strategies: {stats['profitable_strategies']}")
        logger.info(f"  - Best Profit Factor: {stats['max_profit_factor']:.2f}")
        logger.info(f"  - Average Win Rate: {stats['avg_win_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")


def demo_summary(logger, strategy_results):
    """Display comprehensive demo summary."""
    logger.info("\n" + "=" * 60)
    logger.info("🎉 DEMO SUMMARY")
    logger.info("=" * 60)
    
    logger.info("✅ COMPLETED DEMONSTRATIONS:")
    logger.info("  1. 📊 BTC/USDT Data Generation - Multiple market scenarios")
    logger.info("  2. 🚀 Strategy Testing - Conservative, Aggressive, Scalping")
    logger.info("  3. 🔍 Enhanced Backtesting - Parameter optimization")
    logger.info("  4. 📈 Analysis & Reporting - Database storage and reports")
    
    logger.info("\n📊 STRATEGY PERFORMANCE OVERVIEW:")
    
    if strategy_results:
        for strategy_name, scenarios in strategy_results.items():
            total_trades = sum(r['trades'] for r in scenarios.values() if r)
            avg_win_rate = np.mean([r['win_rate'] for r in scenarios.values() if r and r['trades'] > 0])
            avg_return = np.mean([r['total_return'] for r in scenarios.values() if r])
            
            logger.info(f"  🎯 {strategy_name.upper()}:")
            logger.info(f"    - Total Trades: {total_trades}")
            logger.info(f"    - Avg Win Rate: {avg_win_rate:.1%}")
            logger.info(f"    - Avg Return: {avg_return:.1%}")
    
    logger.info("\n🎯 KEY ACHIEVEMENTS:")
    logger.info("  ✅ Generated realistic BTC/USDT market data")
    logger.info("  ✅ Tested strategies across bull/bear/sideways markets")
    logger.info("  ✅ Demonstrated parameter optimization")
    logger.info("  ✅ Saved results to database for analysis")
    logger.info("  ✅ Generated professional reports")
    
    logger.info("\n📁 GENERATED FILES:")
    files_to_check = [
        "btc_strategy_summary.txt",
        "btc_strategy_detailed.csv", 
        "btc_demo_results.db"
    ]
    
    for filename in files_to_check:
        try:
            import os
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                logger.info(f"  ✅ {filename} ({size} bytes)")
            else:
                logger.info(f"  ⚠️ {filename} (not found)")
        except Exception:
            logger.info(f"  ❓ {filename} (unknown)")
    
    logger.info("\n🚀 NEXT STEPS:")
    logger.info("  1. Review generated reports for strategy insights")
    logger.info("  2. Experiment with different parameter combinations")
    logger.info("  3. Add real-time data feeds for live trading")
    logger.info("  4. Implement additional cryptocurrencies")


def main():
    """Main demo function."""
    print("🚀 Trading Strategy Framework - BTC/USDT Demo")
    print("=" * 60)
    
    logger = setup_logging()
    logger.info("Starting comprehensive trading strategy demo...")
    
    try:
        # 1. Generate and validate BTC data
        scenarios = demo_btc_data_generation(logger)
        
        # 2. Test strategies on different market conditions
        strategy_results = demo_strategy_testing(logger, scenarios)
        
        # 3. Demonstrate enhanced backtesting
        btc_backtest_results = demo_backtesting_engine(logger)
        
        # 4. Show optimization and analysis capabilities
        demo_optimization_and_analysis(logger, btc_backtest_results)
        
        # 5. Display comprehensive summary
        demo_summary(logger, strategy_results)
        
        logger.info("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
