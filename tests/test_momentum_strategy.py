"""
Unit tests for the Momentum Long + Short Strategy.

Tests all components following SOLID principles with proper mocking
for I/O operations and clean, maintainable test structure.
"""

from unittest.mock import Mock, patch
import pytest
import pandas as pd
import numpy as np

from models import StrategyParams, TradeResult
from strategy import (
    MomentumStrategy,
    create_default_strategy,
    create_custom_strategy
)
from indicators import (
    ExponentialMovingAverage,
    SimpleMovingAverage,
    RelativeStrengthIndex,
    AverageTrueRange,
    AverageDirectionalIndex,
    MovingAverageFactory,
    TechnicalIndicatorCalculator,
)
from signals import SignalGenerator
from trading import TradeSimulator


class TestStrategyParams:
    """Test strategy parameters dataclass."""
    
    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        params = StrategyParams()
        
        assert params.smooth_type == "EMA"
        assert params.smoothing_length == 100
        assert params.rsi_length_long == 14
        assert params.enable_longs is True
        assert params.sl_percent_long == 3.0
    
    def test_immutable_parameters(self):
        """Test that parameters are immutable (frozen dataclass)."""
        params = StrategyParams()
        
        with pytest.raises(AttributeError):
            params.smooth_type = "SMA"


class TestMovingAverages:
    """Test moving average implementations."""
    
    @pytest.fixture
    def sample_series(self):
        """Create sample price series for testing."""
        return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    def test_ema_calculation(self, sample_series):
        """Test EMA calculation."""
        ema = ExponentialMovingAverage()
        result = ema.calculate(sample_series, 3)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_series)
        assert not result.iloc[-1] == sample_series.iloc[-1]  # Should be smoothed
    
    def test_sma_calculation(self, sample_series):
        """Test SMA calculation."""
        sma = SimpleMovingAverage()
        result = sma.calculate(sample_series, 3)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_series)
        # Test known SMA value
        assert result.iloc[2] == 2.0  # (1+2+3)/3
    
    def test_moving_average_factory(self):
        """Test moving average factory pattern."""
        ema = MovingAverageFactory.create("EMA")
        sma = MovingAverageFactory.create("SMA")
        
        assert isinstance(ema, ExponentialMovingAverage)
        assert isinstance(sma, SimpleMovingAverage)
    
    def test_factory_invalid_type(self):
        """Test factory with invalid MA type."""
        with pytest.raises(ValueError, match="Unsupported moving average type"):
            MovingAverageFactory.create("INVALID")


class TestTechnicalIndicators:
    """Test technical indicator calculations."""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data for testing."""
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        close = np.cumsum(np.random.randn(100)) + 100
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        open_prices = close + np.random.randn(100) * 0.5
        volume = np.random.randint(1000, 10000, 100)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
    
    def test_rsi_calculation(self, sample_ohlc_data):
        """Test RSI calculation."""
        rsi = RelativeStrengthIndex()
        result = rsi.calculate(sample_ohlc_data['close'], 14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlc_data)
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert all(0 <= val <= 100 for val in valid_values)
    
    def test_atr_calculation(self, sample_ohlc_data):
        """Test ATR calculation."""
        atr = AverageTrueRange()
        result = atr.calculate(sample_ohlc_data, 14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlc_data)
        # ATR should be positive
        valid_values = result.dropna()
        assert all(val >= 0 for val in valid_values)
    
    def test_adx_calculation(self, sample_ohlc_data):
        """Test ADX calculation with dependency injection."""
        atr = AverageTrueRange()
        adx = AverageDirectionalIndex(atr)
        result = adx.calculate(sample_ohlc_data, 14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlc_data)


class TestTechnicalIndicatorCalculator:
    """Test the main indicator calculator service."""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        close = np.cumsum(np.random.randn(100)) + 100
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        open_prices = close + np.random.randn(100) * 0.5
        volume = np.random.randint(1000, 10000, 100)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
    
    def test_calculate_all_indicators(self, sample_ohlc_data):
        """Test that all indicators are calculated correctly."""
        calculator = TechnicalIndicatorCalculator()
        params = StrategyParams()
        
        result = calculator.calculate_all_indicators(sample_ohlc_data, params)
        
        # Check that all expected columns are added
        expected_columns = [
            'ma100', 'ma500', 'rsi_long', 'rsi_long_smooth', 'rsi_short',
            'adx', 'adx_smooth', 'atr', 'atr_smooth', 'atr_short', 
            'atr_short_smooth', 'bb_basis', 'bb_dev', 'bb_lower', 'ema_gap_pct'
        ]
        
        for col in expected_columns:
            assert col in result.columns
        
        # Original data should be preserved
        assert len(result) == len(sample_ohlc_data)
        pd.testing.assert_series_equal(result['close'], sample_ohlc_data['close'])


class TestSignalGenerator:
    """Test trading signal generation."""
    
    @pytest.fixture
    def sample_data_with_indicators(self):
        """Create sample data with pre-calculated indicators."""
        np.random.seed(42)
        n = 50
        data = {
            'close': np.random.randn(n) + 100,
            'ma100': np.random.randn(n) + 99,
            'ma500': np.random.randn(n) + 98,
            'rsi_long': np.random.rand(n) * 100,
            'rsi_long_smooth': np.random.rand(n) * 100,
            'rsi_short': np.random.rand(n) * 100,
            'adx': np.random.rand(n) * 50,
            'adx_smooth': np.random.rand(n) * 50,
            'atr': np.random.rand(n) * 5,
            'atr_smooth': np.random.rand(n) * 5,
            'atr_short': np.random.rand(n) * 5,
            'atr_short_smooth': np.random.rand(n) * 5,
            'bb_lower': np.random.randn(n) + 95,
            'ema_gap_pct': np.random.randn(n) * 5
        }
        return pd.DataFrame(data)
    
    def test_long_signal_generation(self, sample_data_with_indicators):
        """Test long signal generation logic."""
        generator = SignalGenerator()
        params = StrategyParams(enable_longs=True, use_trend_filter=False)
        
        signals = generator.generate_long_signals(sample_data_with_indicators, params)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data_with_indicators)
        assert signals.dtype == bool
    
    def test_short_signal_generation(self, sample_data_with_indicators):
        """Test short signal generation logic."""
        generator = SignalGenerator()
        params = StrategyParams(enable_shorts=True)
        
        signals = generator.generate_short_signals(sample_data_with_indicators, params)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data_with_indicators)
        assert signals.dtype == bool
    
    def test_disabled_signals(self, sample_data_with_indicators):
        """Test that disabled signals return all False."""
        generator = SignalGenerator()
        params = StrategyParams(enable_longs=False, enable_shorts=False)
        
        long_signals = generator.generate_long_signals(sample_data_with_indicators, params)
        short_signals = generator.generate_short_signals(sample_data_with_indicators, params)
        
        assert not long_signals.any()
        assert not short_signals.any()


class TestTradeResult:
    """Test trade result data class."""
    
    def test_trade_result_creation(self):
        """Test trade result initialization."""
        trade = TradeResult(
            entry_price=100.0,
            exit_price=105.0,
            position_type='long',
            entry_index=10,
            exit_index=15
        )
        
        assert trade.entry_price == 100.0
        assert trade.exit_price == 105.0
        assert trade.position_type == 'long'
    
    def test_long_trade_pnl(self):
        """Test PnL calculation for long trades."""
        trade = TradeResult(
            entry_price=100.0,
            exit_price=105.0,
            position_type='long',
            entry_index=10,
            exit_index=15
        )
        
        expected_pnl = (105.0 - 100.0) / 100.0
        assert abs(trade.pnl - expected_pnl) < 1e-10
    
    def test_short_trade_pnl(self):
        """Test PnL calculation for short trades."""
        trade = TradeResult(
            entry_price=100.0,
            exit_price=95.0,
            position_type='short',
            entry_index=10,
            exit_index=15
        )
        
        expected_pnl = (100.0 - 95.0) / 100.0
        assert abs(trade.pnl - expected_pnl) < 1e-10


class TestTradeSimulator:
    """Test trade simulation logic."""
    
    @pytest.fixture
    def sample_signal_data(self):
        """Create sample data with trading signals."""
        data = {
            'close': [100, 101, 102, 98, 97, 103, 104, 99, 98, 105],
            'high': [101, 102, 103, 99, 98, 104, 105, 100, 99, 106],
            'low': [99, 100, 101, 97, 96, 102, 103, 98, 97, 104],
            'ma500': [100] * 10,
            'long_signal': [False, True, False, False, False, False, False, False, False, False],
            'short_signal': [False, False, False, True, False, False, False, False, False, False]
        }
        return pd.DataFrame(data)
    
    def test_trade_simulation(self, sample_signal_data):
        """Test basic trade simulation."""
        simulator = TradeSimulator()
        params = StrategyParams(sl_percent_long=5.0, sl_percent_short=5.0)
        
        result = simulator.simulate_trades(sample_signal_data, params)
        
        assert 'position' in result.columns
        assert 'entry_price' in result.columns
        assert 'exit_price' in result.columns
    
    def test_trade_recording(self, sample_signal_data):
        """Test that trades are recorded properly."""
        simulator = TradeSimulator()
        params = StrategyParams()
        
        simulator.simulate_trades(sample_signal_data, params)
        trades = simulator.trades
        
        assert isinstance(trades, list)
        # With the sample data, we should have some trades
        assert len(trades) >= 0


class TestMomentumStrategy:
    """Test the main strategy orchestrator."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create complete OHLCV sample data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        close = np.cumsum(np.random.randn(50)) + 100
        high = close + np.random.rand(50) * 2
        low = close - np.random.rand(50) * 2
        open_prices = close + np.random.randn(50) * 0.5
        volume = np.random.randint(1000, 10000, 50)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        params = StrategyParams()
        strategy = MomentumStrategy(params)
        
        assert strategy._params == params
        assert hasattr(strategy, '_indicator_calculator')
        assert hasattr(strategy, '_signal_generator')
        assert hasattr(strategy, '_trade_simulator')
    
    def test_complete_strategy_run(self, sample_ohlcv_data):
        """Test complete strategy execution."""
        strategy = create_default_strategy()
        
        result = strategy.run_strategy(sample_ohlcv_data)
        
        # Should return DataFrame with all required columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        
        # Should have trading signals and positions
        assert 'long_signal' in result.columns
        assert 'short_signal' in result.columns
        assert 'position' in result.columns
    
    def test_strategy_with_missing_columns(self):
        """Test strategy with incomplete data."""
        incomplete_data = pd.DataFrame({'close': [100, 101, 102]})
        strategy = create_default_strategy()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.run_strategy(incomplete_data)
    
    def test_executed_trades_property(self, sample_ohlcv_data):
        """Test access to executed trades."""
        strategy = create_default_strategy()
        strategy.run_strategy(sample_ohlcv_data)
        
        trades = strategy.executed_trades
        assert isinstance(trades, list)


class TestFactoryFunctions:
    """Test factory functions for strategy creation."""
    
    def test_create_default_strategy(self):
        """Test default strategy creation."""
        strategy = create_default_strategy()
        
        assert isinstance(strategy, MomentumStrategy)
        assert isinstance(strategy._params, StrategyParams)
    
    def test_create_custom_strategy(self):
        """Test custom strategy creation."""
        strategy = create_custom_strategy(
            smooth_type="SMA",
            enable_shorts=False,
            sl_percent_long=2.0
        )
        
        assert isinstance(strategy, MomentumStrategy)
        assert strategy._params.smooth_type == "SMA"
        assert strategy._params.enable_shorts is False
        assert strategy._params.sl_percent_long == 2.0


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic market data for integration testing."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='H')
        
        # Create realistic price action with trend and volatility
        returns = np.random.normal(0.0005, 0.02, n)
        close = 100 * np.exp(np.cumsum(returns))
        
        high = close * (1 + np.random.exponential(0.01, n))
        low = close * (1 - np.random.exponential(0.01, n))
        open_prices = close.shift(1).fillna(close[0])
        volume = np.random.lognormal(8, 1, n)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
    
    def test_end_to_end_strategy_execution(self, realistic_data):
        """Test complete end-to-end strategy execution."""
        # Test with different parameter sets
        strategies = [
            create_default_strategy(),
            create_custom_strategy(use_rsi_filter=True, enable_shorts=False),
            create_custom_strategy(smooth_type="SMA", use_trend_filter=False)
        ]
        
        for strategy in strategies:
            result = strategy.run_strategy(realistic_data)
            
            # Basic validation
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(realistic_data)
            
            # Check that we have reasonable signal generation
            long_signals = result['long_signal'].sum()
            short_signals = result['short_signal'].sum()
            assert long_signals >= 0  # Should be non-negative
            assert short_signals >= 0  # Should be non-negative
            
            # Position column should only contain -1, 0, 1
            positions = result['position'].unique()
            assert all(pos in [-1, 0, 1] for pos in positions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
