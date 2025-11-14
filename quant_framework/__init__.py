"""
Quantitative Trading Framework

A modular framework for algorithmic trading that supports:
- Multiple data sources (CSV, Yahoo Finance, APIs)
- Pluggable trading strategies (classical quant and ML-based)
- Backtesting with comprehensive performance metrics
- Live trading execution layer (future)
- Config-driven experiments

Author: Quantitative Trading Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Quantitative Trading Team"

from quant_framework.data.loaders import CSVDataLoader, YahooDataLoader
from quant_framework.models.base_strategy import BaseStrategy
from quant_framework.backtest.backtester import Backtester
from quant_framework.infra.pipeline import TradingPipeline

__all__ = [
    "CSVDataLoader",
    "YahooDataLoader",
    "BaseStrategy",
    "Backtester",
    "TradingPipeline",
]

