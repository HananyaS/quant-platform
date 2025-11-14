"""
Unit tests for trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
from quant_framework.models import (
    MomentumStrategy,
    MeanReversionStrategy,
    PairsTradingStrategy,
    MLVolatilityModel
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Generate random walk prices
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + np.abs(np.random.randn(100)) * 1,
        'low': prices - np.abs(np.random.randn(100)) * 1,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return data


class TestMomentumStrategy:
    """Test momentum strategy."""
    
    def test_momentum_init(self):
        """Test momentum strategy initialization."""
        strategy = MomentumStrategy(short_window=10, long_window=20)
        assert strategy.short_window == 10
        assert strategy.long_window == 20
        assert strategy.config.name == "MomentumStrategy"
    
    def test_momentum_signals(self, sample_data):
        """Test signal generation."""
        strategy = MomentumStrategy(short_window=10, long_window=20)
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert all(signals.isin([-1, 0, 1]))


class TestMeanReversionStrategy:
    """Test mean reversion strategy."""
    
    def test_mean_reversion_init(self):
        """Test mean reversion strategy initialization."""
        strategy = MeanReversionStrategy(window=20, num_std=2.0)
        assert strategy.window == 20
        assert strategy.num_std == 2.0
    
    def test_mean_reversion_signals(self, sample_data):
        """Test signal generation."""
        strategy = MeanReversionStrategy(window=20, num_std=2.0)
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert all(signals.isin([-1, 0, 1]))


class TestMLVolatilityModel:
    """Test ML volatility model."""
    
    def test_ml_model_init(self):
        """Test ML model initialization."""
        model = MLVolatilityModel(lookback_window=30)
        assert model.lookback_window == 30
    
    def test_feature_creation(self, sample_data):
        """Test feature engineering."""
        model = MLVolatilityModel(lookback_window=20)
        features = model.create_features(sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert 'returns' in features.columns
        assert 'volatility' in features.columns


if __name__ == "__main__":
    pytest.main([__file__])

