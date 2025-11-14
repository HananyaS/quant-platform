"""
Example of creating a custom trading strategy.
"""

import pandas as pd
from quant_framework.models import BaseStrategy, StrategyConfig
from quant_framework.data import YahooDataLoader, TechnicalIndicators
from quant_framework.backtest import Backtester
from quant_framework.infra import TradingPipeline


class RSIStrategy(BaseStrategy):
    """
    Custom RSI-based strategy.
    
    Strategy logic:
    - Long when RSI < oversold_threshold (e.g., 30)
    - Short when RSI > overbought_threshold (e.g., 70)
    - Exit to neutral when RSI crosses 50
    """
    
    def __init__(
        self,
        rsi_window: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70
    ):
        """Initialize RSI strategy."""
        config = StrategyConfig(
            name="RSIStrategy",
            description=f"RSI strategy with oversold={oversold_threshold}, overbought={overbought_threshold}",
            parameters={
                'rsi_window': rsi_window,
                'oversold_threshold': oversold_threshold,
                'overbought_threshold': overbought_threshold
            }
        )
        super().__init__(config)
        self.rsi_window = rsi_window
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on RSI."""
        df = data.copy()
        
        # Ensure we have close prices
        if 'close' not in df.columns and 'Close' in df.columns:
            df['close'] = df['Close']
        
        self.validate_data(df, ['close'])
        
        # Calculate RSI
        rsi = TechnicalIndicators.rsi(df['close'], self.rsi_window)
        
        # Initialize signals
        signals = pd.Series(0, index=df.index)
        position = 0
        
        for i in range(1, len(df)):
            current_rsi = rsi.iloc[i]
            
            if pd.isna(current_rsi):
                signals.iloc[i] = position
                continue
            
            # Entry signals
            if position == 0:
                if current_rsi < self.oversold_threshold:
                    position = 1  # Long when oversold
                elif current_rsi > self.overbought_threshold:
                    position = -1  # Short when overbought
            
            # Exit signals
            elif position != 0:
                if abs(current_rsi - 50) < 5:  # Near neutral
                    position = 0
            
            signals.iloc[i] = position
        
        self.signals = signals
        return signals


def main():
    """Run custom RSI strategy."""
    
    print("="*70)
    print("CUSTOM RSI STRATEGY EXAMPLE")
    print("="*70)
    
    # Load data
    data_loader = YahooDataLoader(
        symbol="AAPL",
        start="2020-01-01",
        end="2024-01-01"
    )
    
    # Create custom strategy
    strategy = RSIStrategy(
        rsi_window=14,
        oversold_threshold=30,
        overbought_threshold=70
    )
    
    # Create backtester
    backtester = Backtester(
        initial_capital=100000,
        fee_perc=0.001
    )
    
    # Run pipeline
    pipeline = TradingPipeline(
        data_loader=data_loader,
        strategy=strategy,
        backtester=backtester
    )
    
    results = pipeline.run()
    
    return results


if __name__ == "__main__":
    main()

