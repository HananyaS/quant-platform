"""
Fibonacci Retracement Strategy

Trades based on Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%).
Identifies potential support/resistance levels from recent price swings.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from .base_strategy import BaseStrategy, StrategyConfig


@dataclass
class FibonacciStrategyConfig(StrategyConfig):
    """Configuration for Fibonacci Retracement Strategy."""
    name: str = "FibonacciStrategy"
    description: str = "Fibonacci retracement levels strategy"
    lookback_period: int = 20  # Period to find swing high/low
    retracement_level: float = 0.618  # Which Fibonacci level to trade
    tolerance: float = 0.02  # Price tolerance around Fibonacci level (2%)
    allow_short: bool = False


class FibonacciStrategy(BaseStrategy):
    """
    Fibonacci Retracement Strategy.
    
    Logic:
    - Identifies swing highs and lows over lookback period
    - Calculates Fibonacci retracement levels
    - Enters long when price bounces off key Fibonacci support
    - Exits when price reaches swing high or falls below tolerance
    
    Fibonacci Levels:
    - 0.236 (23.6%)
    - 0.382 (38.2%)
    - 0.500 (50.0%)
    - 0.618 (61.8%) - Golden ratio
    - 0.786 (78.6%)
    
    Parameters:
        lookback_period: Period to identify swing high/low (default: 20)
        retracement_level: Fibonacci level to trade (default: 0.618)
        tolerance: Price tolerance around level (default: 0.02 = 2%)
        allow_short: Allow short positions (default: False)
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        retracement_level: float = 0.618,
        tolerance: float = 0.02,
        allow_short: bool = False
    ):
        config = FibonacciStrategyConfig(
            lookback_period=lookback_period,
            retracement_level=retracement_level,
            tolerance=tolerance,
            allow_short=allow_short,
            parameters={
                'lookback_period': lookback_period,
                'retracement_level': retracement_level,
                'tolerance': tolerance,
                'allow_short': allow_short
            }
        )
        super().__init__(config)
        
        self.lookback_period = lookback_period
        self.retracement_level = retracement_level
        self.tolerance = tolerance
        self.allow_short = allow_short
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Fibonacci retracements.
        
        Returns:
            pd.Series with values:
            +1 = LONG (price at Fibonacci support, potential bounce)
            0 = CASH/NEUTRAL
            -1 = SHORT (price at Fibonacci resistance, if allow_short)
        """
        df = data.copy()
        
        # Ensure required columns exist
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        if 'high' not in df.columns and 'High' in df.columns:
            df['high'] = df['High']
        if 'low' not in df.columns and 'Low' in df.columns:
            df['low'] = df['Low']
        
        self.validate_data(df, ['close', 'high', 'low'])
        
        # Initialize signals
        signals = pd.Series(0, index=df.index)
        
        # Calculate rolling swing highs and lows
        swing_high = df['high'].rolling(window=self.lookback_period).max()
        swing_low = df['low'].rolling(window=self.lookback_period).min()
        
        # Calculate Fibonacci retracement level
        # Fib level = swing_low + (swing_high - swing_low) * retracement_level
        price_range = swing_high - swing_low
        fib_level = swing_low + (price_range * self.retracement_level)
        
        # Calculate tolerance bands
        tolerance_amount = fib_level * self.tolerance
        fib_upper = fib_level + tolerance_amount
        fib_lower = fib_level - tolerance_amount
        
        # Determine trend direction (for context)
        # Uptrend: recent prices above midpoint
        midpoint = (swing_high + swing_low) / 2
        is_uptrend = df['close'] > midpoint
        is_downtrend = df['close'] < midpoint
        
        # Generate signals
        for i in range(self.lookback_period, len(df)):
            current_price = df['close'].iloc[i]
            prev_price = df['close'].iloc[i-1]
            
            # Long signal: Price bounces off Fibonacci support in uptrend
            if is_uptrend.iloc[i]:
                # Price was below/at Fibonacci, now at or above it (bounce)
                if (prev_price <= fib_level.iloc[i] and 
                    fib_lower.iloc[i] <= current_price <= fib_upper.iloc[i]):
                    signals.iloc[i] = 1
                
                # Continue holding if price above Fibonacci
                elif signals.iloc[i-1] == 1 and current_price > fib_lower.iloc[i]:
                    signals.iloc[i] = 1
                
                # Exit if price falls significantly below Fibonacci
                elif signals.iloc[i-1] == 1 and current_price < swing_low.iloc[i]:
                    signals.iloc[i] = 0
            
            # Short signal: Price rejected at Fibonacci resistance in downtrend
            if self.allow_short and is_downtrend.iloc[i]:
                # Price was above/at Fibonacci, now at or below it (rejection)
                if (prev_price >= fib_level.iloc[i] and 
                    fib_lower.iloc[i] <= current_price <= fib_upper.iloc[i]):
                    signals.iloc[i] = -1
                
                # Continue holding short if price below Fibonacci
                elif signals.iloc[i-1] == -1 and current_price < fib_upper.iloc[i]:
                    signals.iloc[i] = -1
                
                # Exit short if price rises significantly above Fibonacci
                elif signals.iloc[i-1] == -1 and current_price > swing_high.iloc[i]:
                    signals.iloc[i] = 0
        
        self.signals = signals
        return signals
    
    def get_fibonacci_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Fibonacci retracement levels for analysis.
        
        Returns:
            DataFrame with all Fibonacci levels and swing points
        """
        df = data.copy()
        
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        if 'high' not in df.columns and 'High' in df.columns:
            df['high'] = df['High']
        if 'low' not in df.columns and 'Low' in df.columns:
            df['low'] = df['Low']
        
        # Calculate swing points
        swing_high = df['high'].rolling(window=self.lookback_period).max()
        swing_low = df['low'].rolling(window=self.lookback_period).min()
        price_range = swing_high - swing_low
        
        # All Fibonacci levels
        fib_levels = pd.DataFrame(index=df.index)
        fib_levels['swing_high'] = swing_high
        fib_levels['swing_low'] = swing_low
        fib_levels['fib_0.0'] = swing_low  # 0% retracement
        fib_levels['fib_23.6'] = swing_low + (price_range * 0.236)
        fib_levels['fib_38.2'] = swing_low + (price_range * 0.382)
        fib_levels['fib_50.0'] = swing_low + (price_range * 0.500)
        fib_levels['fib_61.8'] = swing_low + (price_range * 0.618)  # Golden ratio
        fib_levels['fib_78.6'] = swing_low + (price_range * 0.786)
        fib_levels['fib_100.0'] = swing_high  # 100% retracement
        fib_levels['price'] = df['close']
        
        return fib_levels

