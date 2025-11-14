"""
Multi-strategy comparison example.

Compare performance of different strategies on the same data.
"""

from quant_framework.data import YahooDataLoader
from quant_framework.models import (
    MomentumStrategy,
    MeanReversionStrategy,
    MLVolatilityModel
)
from quant_framework.infra.pipeline import MultiStrategyPipeline


def main():
    """Compare multiple strategies."""
    
    print("="*70)
    print("MULTI-STRATEGY COMPARISON")
    print("="*70)
    
    # Define strategies to compare
    strategies = [
        MomentumStrategy(
            short_window=20,
            long_window=50,
            use_ema=False
        ),
        MomentumStrategy(
            short_window=10,
            long_window=30,
            use_ema=True
        ),
        MeanReversionStrategy(
            window=20,
            num_std=2.0,
            exit_on_middle=True
        ),
        MLVolatilityModel(
            lookback_window=30,
            volatility_threshold=0.02
        )
    ]
    
    # Load data
    data_loader = YahooDataLoader(
        symbol="AAPL",
        start="2020-01-01",
        end="2024-01-01"
    )
    
    # Create multi-strategy pipeline
    multi_pipeline = MultiStrategyPipeline(
        data_loader=data_loader,
        strategies=strategies,
        backtester_config={
            'initial_capital': 100000,
            'fee_perc': 0.001,
            'slippage_perc': 0.0005
        }
    )
    
    # Run all strategies
    results = multi_pipeline.run_all()
    
    return results


if __name__ == "__main__":
    main()

