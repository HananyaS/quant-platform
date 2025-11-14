"""
MACD crossover strategy.
"""

from dataclasses import dataclass
import pandas as pd
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig
from quant_framework.data.indicators import TechnicalIndicators


@dataclass
class MACDStrategyConfig(StrategyConfig):
    """Configuration for MACD Strategy."""
    name: str = "MACDStrategy"
    description: str = "MACD crossover strategy"
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9


class MACDStrategy(BaseStrategy):
    """
    MACD crossover strategy.
    
    Buy when MACD crosses above signal line.
    Sell when MACD crosses below signal line.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        allow_short: bool = False
    ):
        config = MACDStrategyConfig(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period,
                'allow_short': allow_short
            }
        )
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.allow_short = allow_short
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        
        self.validate_data(df, ['close'])
        
        macd, signal, histogram = TechnicalIndicators.macd(
            df['close'], self.fast_period, self.slow_period, self.signal_period
        )
        
        signals = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            if pd.isna(macd.iloc[i]) or pd.isna(signal.iloc[i]):
                continue
            
            if macd.iloc[i-1] <= signal.iloc[i-1] and macd.iloc[i] > signal.iloc[i]:
                signals.iloc[i:] = 1
            elif macd.iloc[i-1] >= signal.iloc[i-1] and macd.iloc[i] < signal.iloc[i]:
                signals.iloc[i:] = -1 if self.allow_short else 0
        
        self.signals = signals
        return signals

