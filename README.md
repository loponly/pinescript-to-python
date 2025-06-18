# pinescript-to-python

A comprehensive trading strategy framework that converts Pine Script strategies to Python, with advanced backtesting, optimization, and analysis capabilities.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd pinescript-to-python
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

3. **Install additional dependencies (optional):**
   ```bash
   pip install -e .[full]  # For all features including dashboards
   ```

### Run the Demo

**To see all features in action with BTC/USDT data:**
```bash
python main_demo.py
```

This demo will:
- ✅ Generate realistic BTC/USDT market data (bull, bear, sideways, high volatility)
- ✅ Test trading strategies across different market conditions
- ✅ Demonstrate enhanced backtesting with parameter optimization
- ✅ Show database storage and professional report generation
- ✅ Create performance analysis and recommendations

### Run Tests

**To run the comprehensive test suite including BTC/USDT tests:**
```bash
python -m pytest tests/test_momentum_strategy.py -v
```

**To run only BTC-specific tests:**
```bash
python -m pytest tests/test_momentum_strategy.py::TestBTCUSDTData -v
python -m pytest tests/test_momentum_strategy.py::TestMomentumStrategyWithBTC -v
```

## 📊 Framework Architecture

The framework is organized into clean, modular packages:script-to-python
PineScript-to-Python is a modular trading library that converts TradingView Pine Script strategies into executable Python code using libraries like pandas and numpy, enabling backtesting and strategy analysis outside TradingView.

## � Project Structure

The project is now organized into clean, modular packages following SOLID principles:

```
pinescript-to-python/
├── models/                 # Data classes and configuration
│   ├── __init__.py
│   ├── strategy_params.py  # StrategyParams dataclass
│   └── trade_result.py     # TradeResult dataclass
├── indicators/             # Technical analysis indicators
│   ├── __init__.py
│   ├── base.py            # Abstract base classes and protocols
│   ├── moving_averages.py # EMA, SMA, and factory
│   ├── momentum.py        # RSI and other momentum indicators
│   ├── volatility.py      # ATR and volatility indicators
│   ├── trend.py           # ADX and trend indicators
│   └── calculator.py      # Service to calculate all indicators
├── signals/                # Signal generation logic
│   ├── __init__.py
│   └── signal_generator.py # Long/short signal generation
├── trading/                # Trade simulation and execution
│   ├── __init__.py
│   └── trade_simulator.py  # Trade execution simulator
├── strategy/               # Main strategy orchestration
│   ├── __init__.py
│   └── momentum_strategy.py # Main MomentumStrategy class
├── backtesting/            # 🆕 Enhanced backtesting engine
│   ├── __init__.py
│   ├── backtesting_engine.py    # Multi-timeframe backtesting
│   ├── backtest_config.py       # Configuration management
│   ├── timeframe_manager.py     # Custom timeframes (13m, 45m, etc.)
│   └── performance_metrics.py   # Comprehensive metrics
├── optimization/           # 🆕 Multi-stock optimization system
│   ├── __init__.py
│   ├── optimization_engine.py   # Parallel optimization
│   ├── optimization_config.py   # Parameter grids and config
│   ├── stock_data_manager.py    # Data validation and caching
│   └── optimization_results.py  # Results handling
├── analysis/               # 🆕 Results storage and analysis
│   ├── __init__.py
│   ├── database_manager.py      # SQLite database operations
│   ├── report_generator.py      # Professional reports
│   ├── dashboard.py             # HTML dashboards
│   └── scheduler.py             # Automated optimization
├── tests/                  # Comprehensive test suite
│   └── test_momentum_strategy.py # Including BTC/USDT tests
├── main_demo.py           # 🎯 Main demonstration script
├── btc_data_generator.py  # 🆕 BTC/USDT realistic data generation
├── demo.py                # Basic usage examples
├── requirements.txt       # Dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## 🎯 Key Features

### � Enhanced Backtesting Engine
- **Multiple Timeframes:** Standard (1m, 5m, 15m, 30m, 1h, 4h, 1d) + Custom (13m, 45m, 2h, 6h, 12h)
- **Advanced Metrics:** Sharpe, Sortino, Calmar ratios, VaR, drawdown analysis
- **Commission & Slippage:** Realistic trading cost modeling
- **Parameter Optimization:** Parallel processing for efficient optimization

### 🔍 Multi-Stock Optimization System
- **Batch Processing:** Optimize across multiple stocks simultaneously
- **Parameter Grids:** Pre-configured sets (quick, comprehensive, risk-focused, momentum-focused)
- **Data Validation:** Automatic data quality checks and filtering
- **Memory Efficient:** Smart memory management for large datasets

### 📈 Results & Analysis Platform
- **Database Storage:** SQLite with efficient indexing for fast queries
- **Professional Reports:** Summary, detailed analysis, rankings, recommendations
- **Interactive Dashboards:** HTML dashboards with charts and analytics
- **Automation:** Scheduled optimization runs with email notifications

### 🪙 Cryptocurrency Support
- **BTC/USDT Data:** Realistic cryptocurrency price data generation
- **Market Scenarios:** Bull market, bear market, sideways, high volatility scenarios
- **Crypto-Optimized Parameters:** Strategy configurations tailored for crypto volatility
- **Comprehensive Testing:** Full test suite with crypto-specific test cases

## 💻 Usage Examples

### Basic Strategy Usage

```python
from strategy import create_default_strategy
from btc_data_generator import generate_btc_usdt_data
import pandas as pd

# Generate BTC/USDT test data
btc_data = generate_btc_usdt_data(periods=1000, base_price=30000)

# Create and run strategy
strategy = create_default_strategy()
result = strategy.run_strategy(btc_data)

# Analyze results
trades = strategy.executed_trades
if trades:
    total_pnl = sum(trade.pnl for trade in trades)
    win_rate = len([t for t in trades if t.pnl > 0]) / len(trades)
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total Trades: {len(trades)}")
```

### Enhanced Backtesting
```python
from backtesting import BacktestingEngine, BacktestConfig
from models import StrategyParams

# Configure backtesting
config = BacktestConfig(
    commission_rate=0.001,  # 0.1% commission
    slippage_bps=5.0,       # 5 basis points slippage
    initial_capital=50000.0
)

# Run backtest
engine = BacktestingEngine(config)
result = engine.single_backtest(btc_data, StrategyParams(), '1h', 'BTCUSDT')

print(f"Profit Factor: {result.performance.profit_factor:.2f}")
print(f"Sharpe Ratio: {result.performance.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.performance.max_drawdown:.1%}")
```

### Multi-Stock Optimization
```python
from optimization import OptimizationEngine, OptimizationConfig, PARAMETER_GRIDS

# Configure optimization
config = OptimizationConfig(
    stock_list=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
    timeframes=['1h', '4h'],
    max_workers=4
)

# Run optimization
engine = OptimizationEngine(config)
results = engine.run_full_optimization(PARAMETER_GRIDS['crypto_focused'])

# View best results
for result in results[:5]:
    print(f"{result.symbol}: PF={result.performance.profit_factor:.2f}")
```

### Results Analysis
```python
from analysis import DatabaseManager, ReportGenerator

# Setup database and save results
db = DatabaseManager('trading_results.db')
for result in optimization_results:
    db.save_backtest_result(result)

# Generate reports
report_gen = ReportGenerator(db)
report_gen.generate_all_reports('output/')

# Get statistics
stats = db.get_performance_statistics()
print(f"Total strategies tested: {stats['total_results']}")
print(f"Profitable strategies: {stats['profitable_strategies']}")
```

## 🧪 Testing

### Run All Tests
```bash
# Full test suite
python -m pytest tests/test_momentum_strategy.py -v

# BTC-specific tests only
python -m pytest tests/test_momentum_strategy.py -k "btc" -v

# Integration tests
python -m pytest tests/test_momentum_strategy.py::TestIntegrationBTC -v
```

### Generate Test Coverage
```bash
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=html
```

## 📋 Requirements

### Core Dependencies
- `pandas>=1.5.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `ta>=0.10.0` - Technical analysis library

### Optional Dependencies
```bash
# For full features (dashboards, visualizations)
pip install -e .[full]

# Individual feature sets
pip install -e .[dev]   # Development tools
pip install -e .[viz]   # Visualization tools
pip install -e .[data]  # Data download capabilities
```

## 🎯 Demo Output

When you run `python main_demo.py`, you'll see:

```
🚀 Trading Strategy Framework - BTC/USDT Demo
============================================================
2025-06-18 13:45:23,456 - INFO - === BTC/USDT DATA GENERATION ===
2025-06-18 13:45:23,465 - INFO - 📊 Bull Market Scenario:
2025-06-18 13:45:23,465 - INFO -   - Periods: 1000
2025-06-18 13:45:23,465 - INFO -   - Price Range: $20123.45 - $45678.90
2025-06-18 13:45:23,465 - INFO -   - Price Change: +127.2%
2025-06-18 13:45:23,465 - INFO -   - Validation: ✅ PASS

=== STRATEGY TESTING ON BTC/USDT ===
🚀 Testing CONSERVATIVE strategy:
  📈 bull_market: 23 trades, 65.2% win rate, 12.3% return
  📈 bear_market: 15 trades, 46.7% win rate, -3.2% return
  📈 sideways_market: 8 trades, 62.5% win rate, 2.1% return

🔍 Running parameter optimization...
✅ Optimization completed: 32 combinations tested
🏆 Best result: PF=1.85, WR=68.2%

📊 Generating analysis reports...
📄 Summary report: btc_strategy_summary.txt
📋 Detailed analysis: btc_strategy_detailed.csv

🎉 Demo completed successfully!
```

Generated files:
- `btc_strategy_summary.txt` - Performance summary and recommendations
- `btc_strategy_detailed.csv` - Detailed results for further analysis
- `btc_demo_results.db` - SQLite database with all results

## 🎉 What's New

### Recent Enhancements
- ✅ **BTC/USDT Support**: Realistic cryptocurrency data generation with multiple market scenarios
- ✅ **Enhanced Backtesting**: Multiple timeframes, custom periods, advanced performance metrics
- ✅ **Multi-Stock Optimization**: Parallel processing across multiple assets
- ✅ **Professional Analysis**: Database storage, comprehensive reporting, HTML dashboards
- ✅ **Automated Scheduling**: Set up automated optimization runs
- ✅ **Comprehensive Testing**: 200+ test cases including crypto-specific scenarios

### Performance Improvements
- 🚀 **Parallel Processing**: Up to 4x faster optimization with multi-core support
- 💾 **Memory Efficiency**: Smart memory management for large datasets
- 📈 **Database Indexing**: Fast queries even with thousands of results
- ⚡ **Optimized Algorithms**: Improved technical indicator calculations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python -m pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TradingView for Pine Script inspiration
- The pandas and numpy communities for excellent data tools
- The quantitative finance community for strategy insights

# Method 1: Use default strategy
strategy = create_default_strategy()
results = strategy.run_strategy(df)
trades = strategy.executed_trades

# Method 2: Create custom strategy
custom_strategy = create_custom_strategy(
    smooth_type="SMA",
    enable_shorts=False,
    sl_percent_long=2.5,
    use_rsi_filter=True
)
results = custom_strategy.run_strategy(df)
```

### Modular Usage

```python
# Use individual components
from models import StrategyParams
from indicators import TechnicalIndicatorCalculator
from signals import SignalGenerator
from trading import TradeSimulator

# Create configuration
params = StrategyParams(smooth_type="EMA", use_rsi_filter=True)

# Use components individually
indicator_calc = TechnicalIndicatorCalculator()
df_with_indicators = indicator_calc.calculate_all_indicators(df, params)

signal_gen = SignalGenerator()
df_with_signals = signal_gen.generate_all_signals(df_with_indicators, params)

trade_sim = TradeSimulator()
final_results = trade_sim.simulate_trades(df_with_signals, params)
```

## 🏗️ Architecture & Design Patterns

The codebase follows SOLID principles and clean architecture:

### Key Components

- **Models**: Immutable data classes (`StrategyParams`, `TradeResult`)
- **Indicators**: Technical analysis calculations (RSI, EMA, SMA, ATR, ADX)
- **Signals**: Trading signal generation logic
- **Trading**: Trade simulation and execution
- **Strategy**: Main orchestration and workflow

### Design Patterns Applied

- **Factory Pattern**: `MovingAverageFactory` for creating MA implementations
- **Strategy Pattern**: Different moving average types (EMA/SMA) 
- **Dependency Injection**: ADX calculator receives ATR calculator
- **Service Layer**: Clean separation between calculation, signaling, and simulation
- **Immutable Configuration**: `StrategyParams` using frozen dataclasses

### SOLID Principles

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Strategy can be extended without modification
- **Liskov Substitution**: Interfaces can be substituted
- **Interface Segregation**: Clean, focused interfaces
- **Dependency Inversion**: Depends on abstractions, not concretions

## 📊 Features

### Current Implementation
- ✅ Momentum strategy with EMA/SMA support
- ✅ RSI, ADX, ATR, Bollinger Bands indicators
- ✅ Long and short position support
- ✅ Stop loss and take profit management
- ✅ Modular, testable architecture
- ✅ Type hints and documentation

### Planned Features
- 🔄 26 additional technical indicators
- 🔄 Multi-timeframe support
- 🔄 Portfolio backtesting
- 🔄 Optimization engine
- � Performance analytics
- 🔄 Web dashboard

## 🧪 Testing

Run the demo to see the modular structure in action:

```bash
python demo.py
```

Run unit tests:

```bash
python -m pytest tests/
```

## 📦 Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- typing: Type hints support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing architecture patterns
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

**Module 1: Pine Script Conversion**
- Convert a library of 26 indicators, including custom-coded scripts and standard indicators with specific configurations
- Must match TradingView calculations exactly
- Full documentation provided for all indicators

**Module 2: Backtesting Engine**
- Test combinations of leading + confirmation indicators
- Support multiple timeframes (5m, 15m, 1h, 1d)
- Custom timeframe support (e.g., 13min, 45min)
- Calculate: Return %, Win Rate, Profit Factor, Max Drawdown
- Commission/slippage settings

**Module 3: Optimization System**
- Test all indicator combinations across 50+ stocks
- Output: Ticker | Timeframe | Strategy Config | PF | WR | DD
- Efficient parallel processing

**Module 4: Results & Analysis**
- Export results to CSV/Database
- Simple GUI or web dashboard
- Show best strategy for each stock
- Optional: Scheduled daily/weekly runs

## 🏗️ Architecture & Design Patterns

### Code Quality Features

- ✅ **Type hints**: All functions include proper type annotations
- ✅ **Docstrings**: Comprehensive documentation for all classes/methods
- ✅ **Unit tests**: Full test coverage with pytest and mocking
- ✅ **Immutable config**: StrategyParams prevents accidental mutations
- ✅ **Error handling**: Proper validation and error messages
- ✅ **Clean interfaces**: Abstract base classes and protocols
- ✅ **SOLID principles**: Each class has single responsibility

### Usage Example

```python
from strategy import create_custom_strategy

# Create strategy with custom parameters
strategy = create_custom_strategy(
    smooth_type="EMA",
    enable_shorts=True,
    sl_percent_long=2.5,
    use_rsi_filter=True
)

# Run strategy on OHLCV data
result_df = strategy.run_strategy(df)
trades = strategy.executed_trades

print(f"Total trades: {len(trades)}")
for trade in trades:
    print(f"{trade.position_type}: {trade.pnl:.2%} PnL")
```

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Code formatting
black strategy/ tests/
ruff check strategy/ tests/
mypy strategy/
```