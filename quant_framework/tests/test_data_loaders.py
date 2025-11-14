"""
Unit tests for data loaders.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from quant_framework.data.loaders import (
    CSVDataLoader,
    YahooDataLoader,
    DataConfig,
    MultiTickerLoader
)


class TestCSVDataLoader:
    """Test CSV data loader."""
    
    def test_csv_loader_init(self):
        """Test initialization of CSV loader."""
        loader = CSVDataLoader("data/test.csv")
        assert loader.filepath == Path("data/test.csv")
        assert loader.date_column == 'Date'
    
    def test_csv_loader_with_config(self):
        """Test CSV loader with config."""
        config = DataConfig(
            symbol="TEST",
            start_date="2020-01-01",
            end_date="2021-01-01"
        )
        loader = CSVDataLoader("data/test.csv", config=config)
        assert loader.config.symbol == "TEST"


class TestYahooDataLoader:
    """Test Yahoo Finance data loader."""
    
    def test_yahoo_loader_init(self):
        """Test initialization of Yahoo loader."""
        loader = YahooDataLoader("AAPL", start="2020-01-01", end="2021-01-01")
        assert loader.symbol == "AAPL"
        assert loader.start == "2020-01-01"
    
    @pytest.mark.skip(reason="Requires network connection")
    def test_yahoo_loader_load(self):
        """Test loading data from Yahoo Finance."""
        loader = YahooDataLoader("AAPL", start="2020-01-01", end="2020-01-31")
        data = loader.load()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'Close' in data.columns or 'close' in data.columns


class TestDataConfig:
    """Test data configuration."""
    
    def test_data_config_creation(self):
        """Test creating data config."""
        config = DataConfig(
            symbol="AAPL",
            start_date="2020-01-01",
            end_date="2021-01-01",
            dropna=True
        )
        
        assert config.symbol == "AAPL"
        assert config.start_date == "2020-01-01"
        assert config.dropna is True


if __name__ == "__main__":
    pytest.main([__file__])

