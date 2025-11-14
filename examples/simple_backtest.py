"""
Simple backtest example.

Demonstrates basic usage of the quantitative trading framework.
"""

from quant_framework.data import YahooDataLoader
from quant_framework.models import MomentumStrategy
from quant_framework.backtest import Backtester
from quant_framework.infra import TradingPipeline


def main():
    """Run a simple momentum strategy backtest."""
    
    # Step 1: Load data
    print("Loading data...")
    data_loader = YahooDataLoader(
        symbol="AAPL",
        start="2020-01-01",
        end="2024-01-01"
    )
    
    # Step 2: Create strategy
    print("Creating strategy...")
    strategy = MomentumStrategy(
        short_window=20,
        long_window=50
    )
    
    # Step 3: Create backtester
    print("Setting up backtester...")
    backtester = Backtester(
        initial_capital=100000,
        fee_perc=0.001,  # 0.1% fee
        slippage_perc=0.0005  # 0.05% slippage
    )
    
    # Step 4: Run pipeline
    print("\nRunning backtest pipeline...")
    pipeline = TradingPipeline(
        data_loader=data_loader,
        strategy=strategy,
        backtester=backtester,
        verbose=True
    )
    
    results = pipeline.run()
    
    # Step 5: Access results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    metrics = results['metrics']
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    return results


if __name__ == "__main__":
    main()

