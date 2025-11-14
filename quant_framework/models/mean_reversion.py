"""
Mean Reversion Strategy using Bollinger Bands.

Strategy Logic:
- Buy signal when price crosses below lower Bollinger Band
- Sell signal when price crosses above upper Bollinger Band
- Exit when price returns to middle band
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig
from quant_framework.data.indicators import TechnicalIndicators


@dataclass
class MeanReversionConfig(StrategyConfig):
    """Configuration for Mean Reversion Strategy."""
    name: str = "MeanReversionStrategy"
    description: str = "Bollinger Bands mean reversion strategy"
    window: int = 20
    num_std: float = 2.0
    exit_on_middle: bool = True


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy based on Bollinger Bands.
    
    Assumes prices revert to the mean after extreme deviations.
    
    Example:
        strategy = MeanReversionStrategy(window=20, num_std=2)
        signals = strategy.generate_signals(data)
    """
    
    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        exit_on_middle: bool = True,
        allow_short: bool = False
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            window: Bollinger Bands window period
            num_std: Number of standard deviations for bands
            exit_on_middle: Whether to exit when price reaches middle band
            allow_short: Allow short positions (default: False, uses cash instead)
        """
        config = MeanReversionConfig(
            window=window,
            num_std=num_std,
            exit_on_middle=exit_on_middle,
            parameters={
                'window': window,
                'num_std': num_std,
                'exit_on_middle': exit_on_middle,
                'allow_short': allow_short
            }
        )
        super().__init__(config)
        self.window = window
        self.num_std = num_std
        self.exit_on_middle = exit_on_middle
        self.allow_short = allow_short
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion signals.
        
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
        
        # Calculate Bollinger Bands
        middle, upper, lower = TechnicalIndicators.bollinger_bands(
            df['close'], self.window, self.num_std
        )
        
        # Initialize signals
        signals = pd.Series(0, index=df.index)
        position = 0  # Track current position
        
        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            prev_price = df['close'].iloc[i-1]
            
            # Check for valid band values
            if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                signals.iloc[i] = position
                continue
            
            # Entry signals
            if position == 0:
                # Long signal: price crosses below lower band
                if prev_price > lower.iloc[i-1] and price <= lower.iloc[i]:
                    position = 1
                # Short signal: price crosses above upper band (only if allowed)
                elif self.allow_short and prev_price < upper.iloc[i-1] and price >= upper.iloc[i]:
                    position = -1
            
            # Exit signals
            elif position == 1:  # Currently long
                if self.exit_on_middle:
                    # Exit when price reaches middle band
                    if price >= middle.iloc[i]:
                        position = 0
                else:
                    # Exit when price reaches upper band
                    if price >= upper.iloc[i]:
                        position = 0
            
            elif position == -1:  # Currently short
                if self.exit_on_middle:
                    # Exit when price reaches middle band
                    if price <= middle.iloc[i]:
                        position = 0
                else:
                    # Exit when price reaches lower band
                    if price <= lower.iloc[i]:
                        position = 0
            
            signals.iloc[i] = position
        
        self.signals = signals
        return signals

