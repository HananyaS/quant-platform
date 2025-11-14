"""
Triple Moving Average Strategy.
"""

from dataclasses import dataclass
import pandas as pd
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig
from quant_framework.data.indicators import TechnicalIndicators


@dataclass
class TripleMAStrategyConfig(StrategyConfig):
    """Configuration for Triple MA Strategy."""
    name: str = "TripleMAStrategy"
    description: str = "Three moving average trend following"
    fast_period: int = 10
    medium_period: int = 20
    slow_period: int = 50


class TripleMAStrategy(BaseStrategy):
    """
    Triple Moving Average Strategy.
    
    Buy when fast > medium > slow (all aligned).
    Sell when fast < medium < slow.
    """
    
    def __init__(self, fast_period: int = 10, medium_period: int = 20, slow_period: int = 50, allow_short: bool = False):
        config = TripleMAStrategyConfig(
            fast_period=fast_period,
            medium_period=medium_period,
            slow_period=slow_period,
            parameters={
                'fast_period': fast_period,
                'medium_period': medium_period,
                'slow_period': slow_period,
                'allow_short': allow_short
            }
        )
        super().__init__(config)
        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period
        self.allow_short = allow_short
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        
        self.validate_data(df, ['close'])
        
        fast_ma = TechnicalIndicators.sma(df['close'], self.fast_period)
        medium_ma = TechnicalIndicators.sma(df['close'], self.medium_period)
        slow_ma = TechnicalIndicators.sma(df['close'], self.slow_period)
        
        signals = pd.Series(0, index=df.index)
        
        for i in range(self.slow_period, len(df)):
            if pd.isna(fast_ma.iloc[i]) or pd.isna(medium_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                continue
            
            if fast_ma.iloc[i] > medium_ma.iloc[i] > slow_ma.iloc[i]:
                signals.iloc[i] = 1
            elif self.allow_short and fast_ma.iloc[i] < medium_ma.iloc[i] < slow_ma.iloc[i]:
                signals.iloc[i] = -1
        
        self.signals = signals
        return signals

