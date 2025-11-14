"""
Main entry point for the quantitative trading framework.

Run various trading strategies with different configurations.
"""

import argparse
from pathlib import Path

from quant_framework.data.loaders import YahooDataLoader, CSVDataLoader
from quant_framework.models import (
    MomentumStrategy,
    MeanReversionStrategy,
    PairsTradingStrategy,
    MLVolatilityModel
)
from quant_framework.backtest import Backtester
from quant_framework.infra import TradingPipeline
from quant_framework.utils import load_config, setup_logger


def run_from_config(config_path: str) -> None:
    """
    Run backtest from configuration file.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
    """
    # Load configuration
    config = load_config(config_path)
    
    logger = setup_logger("main")
    logger.info(f"Loaded configuration from {config_path}")
    
    # Initialize data loader
    data_config = config.get('data', {})
    source = data_config.get('source', 'yahoo')
    
    if source == 'yahoo':
        data_loader = YahooDataLoader(
            symbol=data_config.get('symbol', 'AAPL'),
            start=data_config.get('start_date', '2020-01-01'),
            end=data_config.get('end_date', '2024-01-01'),
            interval=data_config.get('interval', '1d')
        )
    elif source == 'csv':
        data_loader = CSVDataLoader(
            filepath=data_config.get('filepath'),
            date_column=data_config.get('date_column', 'Date')
        )
    else:
        raise ValueError(f"Unsupported data source: {source}")
    
    # Initialize strategy
    strategy_config = config.get('strategy', {})
    strategy_type = strategy_config.get('type')
    params = strategy_config.get('parameters', {})
    
    if strategy_type == 'momentum':
        strategy = MomentumStrategy(**params)
    elif strategy_type == 'mean_reversion':
        strategy = MeanReversionStrategy(**params)
    elif strategy_type == 'pairs_trading':
        strategy = PairsTradingStrategy(**params)
    elif strategy_type == 'ml_volatility':
        strategy = MLVolatilityModel(**params)
    else:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
    # Initialize backtester
    backtest_config = config.get('backtest', {})
    backtester = Backtester(**backtest_config)
    
    # Initialize pipeline
    output_config = config.get('output', {})
    pipeline = TradingPipeline(
        data_loader=data_loader,
        strategy=strategy,
        backtester=backtester,
        verbose=True,
        save_results=output_config.get('save_results', True),
        output_dir=output_config.get('output_dir', 'results')
    )
    
    # Run pipeline
    logger.info("Starting pipeline execution...")
    results = pipeline.run()
    
    logger.info("Pipeline completed successfully!")
    
    return results


def run_example_momentum():
    """Run example momentum strategy."""
    print("\n" + "="*70)
    print("EXAMPLE: Momentum Strategy on AAPL")
    print("="*70 + "\n")
    
    # Load data
    data_loader = YahooDataLoader("AAPL", start="2020-01-01", end="2024-01-01")
    
    # Create strategy
    strategy = MomentumStrategy(short_window=20, long_window=50)
    
    # Create backtester
    backtester = Backtester(initial_capital=100000, fee_perc=0.001)
    
    # Run pipeline
    pipeline = TradingPipeline(data_loader, strategy, backtester)
    results = pipeline.run()
    
    return results


def run_example_mean_reversion():
    """Run example mean reversion strategy."""
    print("\n" + "="*70)
    print("EXAMPLE: Mean Reversion Strategy on SPY")
    print("="*70 + "\n")
    
    # Load data
    data_loader = YahooDataLoader("SPY", start="2020-01-01", end="2024-01-01")
    
    # Create strategy
    strategy = MeanReversionStrategy(window=20, num_std=2.0)
    
    # Create backtester
    backtester = Backtester(initial_capital=100000, fee_perc=0.001)
    
    # Run pipeline
    pipeline = TradingPipeline(data_loader, strategy, backtester)
    results = pipeline.run()
    
    return results


def run_example_comparison():
    """Run comparison of multiple strategies."""
    print("\n" + "="*70)
    print("EXAMPLE: Strategy Comparison")
    print("="*70 + "\n")
    
    from quant_framework.infra.pipeline import MultiStrategyPipeline
    
    # Define strategies
    strategies = [
        MomentumStrategy(short_window=20, long_window=50),
        MomentumStrategy(short_window=10, long_window=30, use_ema=True),
        MeanReversionStrategy(window=20, num_std=2.0),
    ]
    
    # Create multi-strategy pipeline
    data_loader = YahooDataLoader("AAPL", start="2020-01-01", end="2024-01-01")
    
    multi_pipeline = MultiStrategyPipeline(
        data_loader=data_loader,
        strategies=strategies,
        backtester_config={'initial_capital': 100000, 'fee_perc': 0.001}
    )
    
    # Run all strategies
    results = multi_pipeline.run_all()
    
    return results


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Quantitative Trading Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config configs/example_momentum.yaml
  python main.py --example momentum
  python main.py --example mean_reversion
  python main.py --example comparison
  python main.py --symbol AAPL --start 2020-01-01 --end 2024-01-01
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--example',
        type=str,
        choices=['momentum', 'mean_reversion', 'comparison'],
        help='Run built-in example'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        help='Stock ticker symbol (e.g., AAPL)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['momentum', 'mean_reversion', 'rsi', 'macd', 'breakout', 'turtle'],
        help='Strategy to run'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital (default: 100000)'
    )
    
    parser.add_argument(
        '--allow-short',
        action='store_true',
        help='Allow short positions (default: False, long-only)'
    )
    
    args = parser.parse_args()
    
    if args.config:
        # Run from config file
        run_from_config(args.config)
    
    elif args.example:
        # Run example
        if args.example == 'momentum':
            run_example_momentum()
        elif args.example == 'mean_reversion':
            run_example_mean_reversion()
        elif args.example == 'comparison':
            run_example_comparison()
    
    elif args.symbol and args.start and args.end:
        # Run with command-line parameters
        print(f"\n🚀 Running backtest for {args.symbol}")
        print(f"📅 Date range: {args.start} to {args.end}")
        print(f"💰 Initial capital: ${args.capital:,.2f}\n")
        
        # Load data
        data_loader = YahooDataLoader(args.symbol, start=args.start, end=args.end)
        
        # Select strategy
        allow_short = args.allow_short
        if args.strategy == 'momentum':
            strategy = MomentumStrategy(short_window=20, long_window=50, allow_short=allow_short)
        elif args.strategy == 'mean_reversion':
            strategy = MeanReversionStrategy(window=20, num_std=2.0, allow_short=allow_short)
        elif args.strategy == 'rsi':
            from quant_framework.models import RSIStrategy
            strategy = RSIStrategy(rsi_window=14, oversold_threshold=30, overbought_threshold=70, allow_short=allow_short)
        elif args.strategy == 'macd':
            from quant_framework.models import MACDStrategy
            strategy = MACDStrategy(allow_short=allow_short)
        elif args.strategy == 'breakout':
            from quant_framework.models import BreakoutStrategy
            strategy = BreakoutStrategy(lookback_period=20, allow_short=allow_short)
        elif args.strategy == 'turtle':
            from quant_framework.models import TurtleStrategy
            strategy = TurtleStrategy(allow_short=allow_short)
        else:
            # Default to momentum
            strategy = MomentumStrategy(short_window=20, long_window=50, allow_short=allow_short)
        
        # Run backtest
        backtester = Backtester(initial_capital=args.capital, fee_perc=0.001)
        pipeline = TradingPipeline(data_loader, strategy, backtester)
        results = pipeline.run()
    
    else:
        # Default: run momentum example
        print("No arguments provided. Running default momentum example...")
        print("Use --help for more options.\n")
        run_example_momentum()


if __name__ == "__main__":
    main()

