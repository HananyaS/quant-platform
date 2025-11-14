"""
Buy and Hold Strategy - Baseline for comparison.

This strategy simply buys at the start and holds forever.
It serves as a benchmark to compare active strategies against.
"""

from dataclasses import dataclass
import pandas as pd
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig


@dataclass
class BuyHoldConfig(StrategyConfig):
    """Configuration for Buy and Hold Strategy."""
    name: str = "BuyAndHold"
    description: str = "Buy at start and hold forever - baseline strategy"


class BuyHoldStrategy(BaseStrategy):
    """
    Buy and Hold Strategy.
    
    The simplest strategy - buy at the beginning and hold forever.
    This serves as a baseline to compare other strategies against.
    
    If a strategy can't beat buy & hold, it's not adding value!
    
    Example:
        strategy = BuyHoldStrategy()
        signals = strategy.generate_signals(data)
        # Returns: [0, 0, 0, 1, 1, 1, 1, ...]
        #           ^wait  ^buy and hold forever
    """
    
    def __init__(self):
        """
        Initialize Buy and Hold strategy.
        
        No parameters needed - it's that simple!
        """
        config = BuyHoldConfig(
            parameters={}
        )
        super().__init__(config)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate buy and hold signals.
        
        Strategy:
        - Wait for first valid price
        - Buy (signal = +1)
        - Hold forever (signal = +1 for all remaining days)
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with +1 signals (buy and hold)
        """
        df = data.copy()
        
        # Ensure 'close' column exists
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        
        self.validate_data(df, ['close'])
        
        # Initialize all signals to 0
        signals = pd.Series(0, index=df.index)
        
        # Find first valid (non-NaN) price
        first_valid_idx = df['close'].first_valid_index()
        
        if first_valid_idx is not None:
            # Buy at first valid price and hold forever
            signals.loc[first_valid_idx:] = 1
        
        self.signals = signals
        return signals
    
    def get_description(self) -> str:
        """Get strategy description."""
        return (
            "Buy and Hold Strategy\n"
            "=====================\n"
            "• Buy at the start\n"
            "• Hold forever\n"
            "• No selling, no timing\n"
            "• Pure market exposure\n"
            "\n"
            "This is the baseline strategy that all other strategies\n"
            "should try to beat. If a strategy can't outperform buy & hold,\n"
            "it's not adding value (and costing you in fees!)."
        )

