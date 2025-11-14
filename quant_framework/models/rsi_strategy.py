"""
RSI-based mean reversion strategy.
"""

from dataclasses import dataclass
import pandas as pd
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig
from quant_framework.data.indicators import TechnicalIndicators


@dataclass
class RSIStrategyConfig(StrategyConfig):
    """Configuration for RSI Strategy."""
    name: str = "RSIStrategy"
    description: str = "RSI overbought/oversold strategy"
    rsi_window: int = 14
    oversold_threshold: float = 30
    overbought_threshold: float = 70


class RSIStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy.
    
    Buy when RSI is oversold, sell when overbought.
    """
    
    def __init__(
        self,
        rsi_window: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
        allow_short: bool = False
    ):
        config = RSIStrategyConfig(
            rsi_window=rsi_window,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold,
            parameters={
                'rsi_window': rsi_window,
                'oversold_threshold': oversold_threshold,
                'overbought_threshold': overbought_threshold,
                'allow_short': allow_short
            }
        )
        super().__init__(config)
        self.rsi_window = rsi_window
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.allow_short = allow_short
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        
        self.validate_data(df, ['close'])
        
        rsi = TechnicalIndicators.rsi(df['close'], self.rsi_window)
        
        signals = pd.Series(0, index=df.index)
        position = 0
        
        for i in range(1, len(df)):
            current_rsi = rsi.iloc[i]
            
            if pd.isna(current_rsi):
                signals.iloc[i] = position
                continue
            
            if position == 0:
                if current_rsi < self.oversold_threshold:
                    position = 1
                elif self.allow_short and current_rsi > self.overbought_threshold:
                    position = -1
            elif position != 0:
                if abs(current_rsi - 50) < 10:
                    position = 0
            
            signals.iloc[i] = position
        
        self.signals = signals
        return signals

