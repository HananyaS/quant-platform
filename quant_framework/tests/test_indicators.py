"""
Unit tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np
from quant_framework.data.indicators import TechnicalIndicators


@pytest.fixture
def sample_prices():
    """Create sample price series."""
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(100)))
    return prices


class TestTechnicalIndicators:
    """Test technical indicators."""
    
    def test_sma(self, sample_prices):
        """Test Simple Moving Average."""
        sma = TechnicalIndicators.sma(sample_prices, window=20)
        
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_prices)
        # First values should be NaN
        assert pd.isna(sma.iloc[0])
    
    def test_ema(self, sample_prices):
        """Test Exponential Moving Average."""
        ema = TechnicalIndicators.ema(sample_prices, span=20)
        
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_prices)
    
    def test_rsi(self, sample_prices):
        """Test RSI."""
        rsi = TechnicalIndicators.rsi(sample_prices, window=14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_prices)
        # RSI should be between 0 and 100
        assert all(rsi.dropna() >= 0)
        assert all(rsi.dropna() <= 100)
    
    def test_bollinger_bands(self, sample_prices):
        """Test Bollinger Bands."""
        middle, upper, lower = TechnicalIndicators.bollinger_bands(
            sample_prices, window=20, num_std=2.0
        )
        
        assert isinstance(middle, pd.Series)
        assert isinstance(upper, pd.Series)
        assert isinstance(lower, pd.Series)
        
        # Upper should be above middle, middle above lower
        valid_idx = ~middle.isna()
        assert all(upper[valid_idx] >= middle[valid_idx])
        assert all(middle[valid_idx] >= lower[valid_idx])
    
    def test_macd(self, sample_prices):
        """Test MACD."""
        macd, signal, histogram = TechnicalIndicators.macd(sample_prices)
        
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd) == len(sample_prices)
    
    def test_volatility(self, sample_prices):
        """Test volatility calculation."""
        vol = TechnicalIndicators.volatility(
            sample_prices, window=20, annualize=True
        )
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(sample_prices)
        assert all(vol.dropna() >= 0)


if __name__ == "__main__":
    pytest.main([__file__])

