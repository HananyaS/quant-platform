"""
Pairs Trading Strategy.

Strategy Logic:
- Identify cointegrated pairs of assets
- Trade the spread when it deviates from mean
- Long undervalued asset, short overvalued asset
"""

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
from quant_framework.models.base_strategy import BaseStrategy, StrategyConfig


@dataclass
class PairsTradingConfig(StrategyConfig):
    """Configuration for Pairs Trading Strategy."""
    name: str = "PairsTradingStrategy"
    description: str = "Statistical arbitrage on cointegrated pairs"
    lookback_window: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs trading strategy for two cointegrated assets.
    
    Trades the mean-reverting spread between two correlated assets.
    
    Example:
        strategy = PairsTradingStrategy(lookback_window=60, entry_threshold=2.0)
        signals = strategy.generate_signals(data)
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ):
        """
        Initialize pairs trading strategy.
        
        Args:
            lookback_window: Period for calculating spread statistics
            entry_threshold: Z-score threshold for entry (in std devs)
            exit_threshold: Z-score threshold for exit (in std devs)
        """
        config = PairsTradingConfig(
            lookback_window=lookback_window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            parameters={
                'lookback_window': lookback_window,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold
            }
        )
        super().__init__(config)
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def calculate_spread(
        self,
        asset1: pd.Series,
        asset2: pd.Series
    ) -> Tuple[pd.Series, float]:
        """
        Calculate the spread between two assets using linear regression.
        
        Args:
            asset1: Price series of first asset
            asset2: Price series of second asset
            
        Returns:
            Tuple of (spread series, hedge ratio)
        """
        # Calculate hedge ratio using OLS
        # spread = asset1 - hedge_ratio * asset2
        
        # Simple approach: use correlation-based hedge ratio
        hedge_ratio = asset1.cov(asset2) / asset2.var()
        spread = asset1 - hedge_ratio * asset2
        
        return spread, hedge_ratio
    
    def calculate_zscore(self, spread: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling z-score of the spread.
        
        Args:
            spread: Spread series
            window: Rolling window period
            
        Returns:
            Z-score series
        """
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
        zscore = (spread - mean) / std
        
        return zscore
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate pairs trading signals.
        
        Expects DataFrame with two asset columns (e.g., 'asset1', 'asset2')
        or 'close_1' and 'close_2'.
        
        Args:
            data: DataFrame with prices for both assets
            
        Returns:
            Series with signals (+1 long spread, 0 neutral, -1 short spread)
        """
        df = data.copy()
        
        # Try to identify the two asset columns
        price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
        
        if len(price_cols) < 2:
            # Fallback: assume first two numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                raise ValueError(
                    "Pairs trading requires at least two price series. "
                    f"Found columns: {df.columns.tolist()}"
                )
            asset1 = df[numeric_cols[0]]
            asset2 = df[numeric_cols[1]]
        else:
            asset1 = df[price_cols[0]]
            asset2 = df[price_cols[1]]
        
        # Calculate spread and z-score
        spread, hedge_ratio = self.calculate_spread(asset1, asset2)
        zscore = self.calculate_zscore(spread, self.lookback_window)
        
        # Initialize signals
        signals = pd.Series(0, index=df.index)
        position = 0  # Track current position
        
        for i in range(self.lookback_window, len(df)):
            z = zscore.iloc[i]
            
            if pd.isna(z):
                signals.iloc[i] = position
                continue
            
            # Entry signals
            if position == 0:
                # Spread too high: short spread (short asset1, long asset2)
                if z > self.entry_threshold:
                    position = -1
                # Spread too low: long spread (long asset1, short asset2)
                elif z < -self.entry_threshold:
                    position = 1
            
            # Exit signals
            elif position == 1:  # Currently long spread
                if z > -self.exit_threshold:
                    position = 0
            
            elif position == -1:  # Currently short spread
                if z < self.exit_threshold:
                    position = 0
            
            signals.iloc[i] = position
        
        self.signals = signals
        return signals

