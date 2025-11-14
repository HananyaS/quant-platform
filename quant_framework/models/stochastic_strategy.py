"""
Stochastic Oscillator Strategy.
"""

from dataclasses import dataclass
import pandas as pd
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig
from quant_framework.data.indicators import TechnicalIndicators


@dataclass
class StochasticStrategyConfig(StrategyConfig):
    """Configuration for Stochastic Strategy."""
    name: str = "StochasticStrategy"
    description: str = "Stochastic oscillator overbought/oversold"
    window: int = 14
    oversold: float = 20
    overbought: float = 80


class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator Strategy.
    
    Buy when %K crosses above oversold level.
    Sell when %K crosses below overbought level.
    """
    
    def __init__(self, window: int = 14, oversold: float = 20, overbought: float = 80, allow_short: bool = False):
        config = StochasticStrategyConfig(
            window=window,
            oversold=oversold,
            overbought=overbought,
            parameters={'window': window, 'oversold': oversold, 'overbought': overbought, 'allow_short': allow_short}
        )
        super().__init__(config)
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        
        if 'high' not in df.columns and 'High' in df.columns:
            df['high'] = df['High']
        if 'low' not in df.columns and 'Low' in df.columns:
            df['low'] = df['Low']
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        
        self.validate_data(df, ['high', 'low', 'close'])
        
        k, d = TechnicalIndicators.stochastic_oscillator(
            df['high'], df['low'], df['close'], self.window
        )
        
        signals = pd.Series(0, index=df.index)
        position = 0
        
        for i in range(1, len(df)):
            if pd.isna(k.iloc[i]):
                signals.iloc[i] = position
                continue
            
            if position == 0:
                if k.iloc[i] < self.oversold and k.iloc[i] > k.iloc[i-1]:
                    position = 1
                elif self.allow_short and k.iloc[i] > self.overbought and k.iloc[i] < k.iloc[i-1]:
                    position = -1
            elif position == 1:
                if k.iloc[i] > self.overbought:
                    position = 0
            elif position == -1:
                if k.iloc[i] < self.oversold:
                    position = 0
            
            signals.iloc[i] = position
        
        self.signals = signals
        return signals

