"""
Unit tests for trading pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from quant_framework.infra import TradingPipeline
from quant_framework.data.loaders import BaseDataLoader, DataConfig
from quant_framework.models import MomentumStrategy
from quant_framework.backtest import Backtester


class MockDataLoader(BaseDataLoader):
    """Mock data loader for testing."""
    
    def __init__(self):
        super().__init__(DataConfig(symbol="TEST"))
    
    def load(self) -> pd.DataFrame:
        """Load mock data."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100))
        
        data = pd.DataFrame({
            'Open': prices + 1,
            'High': prices + 2,
            'Low': prices - 1,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        return data


class TestTradingPipeline:
    """Test trading pipeline."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        data_loader = MockDataLoader()
        strategy = MomentumStrategy(short_window=10, long_window=20)
        backtester = Backtester(initial_capital=100000)
        
        pipeline = TradingPipeline(
            data_loader=data_loader,
            strategy=strategy,
            backtester=backtester,
            verbose=False
        )
        
        assert pipeline.data_loader is not None
        assert pipeline.strategy is not None
        assert pipeline.backtester is not None
    
    def test_pipeline_run(self):
        """Test running pipeline."""
        data_loader = MockDataLoader()
        strategy = MomentumStrategy(short_window=10, long_window=20)
        backtester = Backtester(initial_capital=100000)
        
        pipeline = TradingPipeline(
            data_loader=data_loader,
            strategy=strategy,
            backtester=backtester,
            verbose=False,
            save_results=False
        )
        
        results = pipeline.run()
        
        assert results is not None
        assert 'metrics' in results
        assert 'equity_curve' in results


if __name__ == "__main__":
    pytest.main([__file__])

