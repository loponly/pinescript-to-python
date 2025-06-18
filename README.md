# pinescript-to-python
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
├── tests/                  # Unit tests
│   └── test_momentum_strategy.py
├── pinescripts/            # Original Pine Script files
│   └── Momentum Long + Short Strategy.pinescript
├── demo.py                 # Usage examples
├── requirements.txt        # Dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🚀 Quick Start

### Basic Usage

```python
from strategy import create_default_strategy, create_custom_strategy
import pandas as pd

# Load your OHLCV data
df = pd.read_csv("your_data.csv")  # Must have: open, high, low, close, volume

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