"""
Breakout strategy using Donchian Channels.
"""

from dataclasses import dataclass
import pandas as pd
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig


@dataclass
class BreakoutStrategyConfig(StrategyConfig):
    """Configuration for Breakout Strategy."""
    name: str = "BreakoutStrategy"
    description: str = "Donchian channel breakout strategy"
    lookback_period: int = 20


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy using Donchian Channels.
    
    Buy on breakout above highest high.
    Sell on breakout below lowest low.
    """
    
    def __init__(self, lookback_period: int = 20, allow_short: bool = False):
        config = BreakoutStrategyConfig(
            lookback_period=lookback_period,
            parameters={'lookback_period': lookback_period, 'allow_short': allow_short}
        )
        super().__init__(config)
        self.lookback_period = lookback_period
        self.allow_short = allow_short
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        if 'high' not in df.columns and 'High' in df.columns:
            df['high'] = df['High']
        if 'low' not in df.columns and 'Low' in df.columns:
            df['low'] = df['Low']
        
        self.validate_data(df, ['close', 'high', 'low'])
        
        upper_band = df['high'].rolling(window=self.lookback_period).max()
        lower_band = df['low'].rolling(window=self.lookback_period).min()
        
        signals = pd.Series(0, index=df.index)
        
        for i in range(self.lookback_period, len(df)):
            if df['close'].iloc[i] >= upper_band.iloc[i-1]:
                signals.iloc[i:] = 1
            elif self.allow_short and df['close'].iloc[i] <= lower_band.iloc[i-1]:
                signals.iloc[i:] = -1
        
        self.signals = signals
        return signals

