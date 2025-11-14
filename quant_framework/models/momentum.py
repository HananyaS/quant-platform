"""
Momentum Strategy using Moving Average Crossover.

Strategy Logic:
- Buy when short-term MA crosses above long-term MA (golden cross)
- Sell when short-term MA crosses below long-term MA (death cross)
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig
from quant_framework.data.indicators import TechnicalIndicators


@dataclass
class MomentumConfig(StrategyConfig):
    """Configuration for Momentum Strategy."""
    name: str = "MomentumStrategy"
    description: str = "Moving average crossover momentum strategy"
    short_window: int = 20
    long_window: int = 50
    use_ema: bool = False


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy based on moving average crossover.
    
    Classic trend-following strategy that buys on bullish crossovers
    and sells on bearish crossovers.
    
    Example:
        strategy = MomentumStrategy(short_window=20, long_window=50)
        signals = strategy.generate_signals(data)
    """
    
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        use_ema: bool = False,
        allow_short: bool = False
    ):
        """
        Initialize momentum strategy.
        
        Args:
            short_window: Short moving average period
            long_window: Long moving average period
            use_ema: Use EMA instead of SMA
            allow_short: Allow short positions (default: False, uses cash instead)
        """
        config = MomentumConfig(
            short_window=short_window,
            long_window=long_window,
            use_ema=use_ema,
            parameters={
                'short_window': short_window,
                'long_window': long_window,
                'use_ema': use_ema,
                'allow_short': allow_short
            }
        )
        super().__init__(config)
        self.short_window = short_window
        self.long_window = long_window
        self.use_ema = use_ema
        self.allow_short = allow_short
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals based on MA crossover.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with signals (+1 long, 0 neutral, -1 short)
        """
        df = data.copy()
        
        # Ensure we have close prices
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        
        self.validate_data(df, ['close'])
        
        # Calculate moving averages
        if self.use_ema:
            short_ma = TechnicalIndicators.ema(df['close'], self.short_window)
            long_ma = TechnicalIndicators.ema(df['close'], self.long_window)
        else:
            short_ma = TechnicalIndicators.sma(df['close'], self.short_window)
            long_ma = TechnicalIndicators.sma(df['close'], self.long_window)
        
        # Initialize signals
        signals = pd.Series(0, index=df.index)
        
        # Generate signals based on crossover
        for i in range(1, len(df)):
            if pd.isna(short_ma.iloc[i]) or pd.isna(long_ma.iloc[i]):
                continue
            
            # Golden cross: short MA crosses above long MA (bullish)
            if short_ma.iloc[i-1] <= long_ma.iloc[i-1] and short_ma.iloc[i] > long_ma.iloc[i]:
                signals.iloc[i:] = 1  # Go long
            
            # Death cross: short MA crosses below long MA (bearish)
            elif short_ma.iloc[i-1] >= long_ma.iloc[i-1] and short_ma.iloc[i] < long_ma.iloc[i]:
                signals.iloc[i:] = -1  # Go short (or exit)
        
        # Alternative: continuous signal based on current position
        signals_continuous = pd.Series(0, index=df.index)
        signals_continuous[short_ma > long_ma] = 1
        
        if self.allow_short:
            # Allow short positions when fast MA < slow MA
            signals_continuous[short_ma < long_ma] = -1
        else:
            # Stay in cash (0) when fast MA < slow MA
            signals_continuous[short_ma < long_ma] = 0
        
        self.signals = signals_continuous
        return signals_continuous

