"""
Example: Using CSV data to avoid Yahoo Finance rate limits.

This script shows how to download data once and save it as CSV,
then use it for multiple backtests without hitting rate limits.
"""

import pandas as pd
from pathlib import Path


def download_and_save_data(symbol: str, start: str, end: str, output_dir: str = "data"):
    """
    Download data from Yahoo Finance and save as CSV.
    
    Args:
        symbol: Stock ticker
        start: Start date
        end: End date
        output_dir: Directory to save CSV files
    """
    try:
        import yfinance as yf
    except ImportError:
        print("Please install yfinance: pip install yfinance")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Download data
    print(f"Downloading {symbol} data...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start, end=end)
    
    if data.empty:
        print(f"No data found for {symbol}")
        return
    
    # Save to CSV
    csv_file = output_path / f"{symbol}.csv"
    data.to_csv(csv_file)
    print(f"✓ Saved data to {csv_file}")
    print(f"  Records: {len(data)}")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    
    return csv_file


def run_backtest_from_csv(csv_file: str):
    """
    Run backtest using CSV data.
    
    Args:
        csv_file: Path to CSV file
    """
    from quant_framework.data import CSVDataLoader
    from quant_framework.models import MomentumStrategy
    from quant_framework.backtest import Backtester
    from quant_framework.infra import TradingPipeline
    
    print(f"\nRunning backtest with {csv_file}...")
    
    # Load data from CSV
    data_loader = CSVDataLoader(
        filepath=csv_file,
        date_column='Date'  # or 'index' if Date is the index
    )
    
    # Create strategy
    strategy = MomentumStrategy(short_window=20, long_window=50)
    
    # Create backtester
    backtester = Backtester(initial_capital=100000, fee_perc=0.001)
    
    # Run pipeline
    pipeline = TradingPipeline(
        data_loader=data_loader,
        strategy=strategy,
        backtester=backtester,
        verbose=True,
        save_results=False
    )
    
    results = pipeline.run()
    return results


def main():
    """Main function."""
    symbols = ["AAPL", "GOOGL", "MSFT"]
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    print("=" * 70)
    print("CSV DATA WORKFLOW - Avoid Rate Limits")
    print("=" * 70)
    
    # Step 1: Download and save data (do this once)
    print("\nStep 1: Download and save data to CSV files")
    print("-" * 70)
    
    csv_files = {}
    for symbol in symbols:
        try:
            csv_file = download_and_save_data(symbol, start_date, end_date)
            if csv_file:
                csv_files[symbol] = csv_file
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
    
    if not csv_files:
        print("\n⚠ No data downloaded. You can:")
        print("1. Wait and try again (if rate limited)")
        print("2. Download manually from Yahoo Finance")
        print("3. Use existing CSV files")
        return
    
    # Step 2: Run backtests using CSV files (no API calls!)
    print("\n\nStep 2: Run backtests using saved CSV files")
    print("-" * 70)
    print("✓ No API calls needed - using local CSV files!\n")
    
    for symbol, csv_file in csv_files.items():
        print(f"\n{'='*70}")
        print(f"Backtesting {symbol}")
        print('='*70)
        try:
            results = run_backtest_from_csv(csv_file)
            print(f"\n✓ {symbol} backtest completed successfully!")
        except Exception as e:
            print(f"⚠ Error backtesting {symbol}: {e}")
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print("\n💡 Next time, you can skip Step 1 and directly use the CSV files!")
    print("This avoids rate limits and is much faster.")


if __name__ == "__main__":
    main()

