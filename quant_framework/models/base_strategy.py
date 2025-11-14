"""
Base strategy class for all trading strategies.

All strategies should inherit from BaseStrategy and implement
the generate_signals() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class StrategyConfig:
    """Base configuration for strategies."""
    name: str = "BaseStrategy"
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement the generate_signals() method which
    returns a pandas Series with values:
        +1: Long position
         0: No position / Neutral
        -1: Short position
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize the strategy.
        
        Args:
            config: StrategyConfig object with strategy parameters
        """
        self.config = config or StrategyConfig()
        self.signals: Optional[pd.Series] = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            
        Returns:
            Series with trading signals (+1, 0, -1)
        """
        pass
    
    def validate_data(self, data: pd.DataFrame, required_columns: list) -> None:
        """
        Validate that the input data has required columns.
        
        Args:
            data: Input DataFrame
            required_columns: List of required column names
            
        Raises:
            ValueError: If required columns are missing
        """
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {data.columns.tolist()}"
            )
    
    def get_position_changes(self, signals: pd.Series) -> pd.Series:
        """
        Calculate position changes from signals.
        
        Args:
            signals: Signal series (+1, 0, -1)
            
        Returns:
            Series with position changes
        """
        return signals.diff().fillna(signals)
    
    def get_trade_count(self, signals: pd.Series) -> int:
        """
        Count the number of trades (position changes).
        
        Args:
            signals: Signal series
            
        Returns:
            Number of trades
        """
        position_changes = self.get_position_changes(signals)
        return (position_changes != 0).sum()
    
    def summary(self) -> Dict[str, Any]:
        """
        Get strategy summary information.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            'name': self.config.name,
            'description': self.config.description,
            'parameters': self.config.parameters,
            'signals_generated': len(self.signals) if self.signals is not None else 0
        }
    
    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(config={self.config})"

