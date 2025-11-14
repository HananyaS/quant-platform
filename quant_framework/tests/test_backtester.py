"""
Unit tests for backtesting engine.
"""

import pytest
import pandas as pd
import numpy as np
from quant_framework.backtest import Backtester
from quant_framework.backtest.metrics import (
    calc_sharpe_ratio,
    calc_max_drawdown,
    calc_win_rate
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100))
    
    data = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    return data


@pytest.fixture
def sample_signals():
    """Create sample signals."""
    signals = pd.Series(
        [0] * 10 + [1] * 30 + [0] * 20 + [-1] * 30 + [0] * 10,
        index=pd.date_range(start='2020-01-01', periods=100, freq='D')
    )
    return signals


class TestBacktester:
    """Test backtesting engine."""
    
    def test_backtester_init(self):
        """Test backtester initialization."""
        bt = Backtester(initial_capital=100000, fee_perc=0.001)
        assert bt.config.initial_capital == 100000
        assert bt.config.fee_perc == 0.001
    
    def test_backtester_run(self, sample_data, sample_signals):
        """Test running backtest."""
        bt = Backtester(initial_capital=100000)
        results = bt.run(sample_data, sample_signals)
        
        assert 'equity_curve' in results
        assert 'metrics' in results
        assert 'final_equity' in results
        assert isinstance(results['equity_curve'], pd.Series)
    
    def test_equity_curve_length(self, sample_data, sample_signals):
        """Test equity curve has correct length."""
        bt = Backtester(initial_capital=100000)
        results = bt.run(sample_data, sample_signals)
        
        assert len(results['equity_curve']) == len(sample_data)


class TestMetrics:
    """Test performance metrics."""
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)
        sharpe = calc_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        equity = pd.Series([100, 110, 105, 115, 90, 95, 120])
        max_dd = calc_max_drawdown(equity)
        
        assert isinstance(max_dd, float)
        assert 0 <= max_dd <= 1
    
    def test_win_rate(self):
        """Test win rate calculation."""
        returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        win_rate = calc_win_rate(returns)
        
        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1
        assert win_rate == 0.6  # 3 wins out of 5


if __name__ == "__main__":
    pytest.main([__file__])

