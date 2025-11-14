"""
Turtle Trading Strategy.
"""

from dataclasses import dataclass
import pandas as pd
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig
from quant_framework.data.indicators import TechnicalIndicators


@dataclass
class TurtleStrategyConfig(StrategyConfig):
    """Configuration for Turtle Strategy."""
    name: str = "TurtleStrategy"
    description: str = "Famous Turtle Trading System"
    entry_period: int = 20
    exit_period: int = 10


class TurtleStrategy(BaseStrategy):
    """
    Turtle Trading Strategy.
    
    Entry: 20-day high/low breakout
    Exit: 10-day high/low in opposite direction
    """
    
    def __init__(self, entry_period: int = 20, exit_period: int = 10, allow_short: bool = False):
        config = TurtleStrategyConfig(
            entry_period=entry_period,
            exit_period=exit_period,
            parameters={'entry_period': entry_period, 'exit_period': exit_period, 'allow_short': allow_short}
        )
        super().__init__(config)
        self.entry_period = entry_period
        self.exit_period = exit_period
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
        
        entry_high = df['high'].rolling(window=self.entry_period).max()
        entry_low = df['low'].rolling(window=self.entry_period).min()
        exit_high = df['high'].rolling(window=self.exit_period).max()
        exit_low = df['low'].rolling(window=self.exit_period).min()
        
        signals = pd.Series(0, index=df.index)
        position = 0
        
        for i in range(max(self.entry_period, self.exit_period), len(df)):
            if position == 0:
                if df['close'].iloc[i] >= entry_high.iloc[i-1]:
                    position = 1
                elif self.allow_short and df['close'].iloc[i] <= entry_low.iloc[i-1]:
                    position = -1
            elif position == 1:
                if df['close'].iloc[i] <= exit_low.iloc[i-1]:
                    position = 0
            elif position == -1:
                if df['close'].iloc[i] >= exit_high.iloc[i-1]:
                    position = 0
            
            signals.iloc[i] = position
        
        self.signals = signals
        return signals

