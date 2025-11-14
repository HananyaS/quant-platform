"""
Trading Pipeline - Orchestrates the entire trading workflow.

Coordinates data loading, signal generation, backtesting, and reporting.
"""

from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime

from quant_framework.data.loaders import BaseDataLoader
from quant_framework.models.base_strategy import BaseStrategy
from quant_framework.backtest.backtester import Backtester
from quant_framework.utils.logger import setup_logger
from quant_framework.utils.plotting import plot_equity_curve, plot_drawdown, plot_signals
from quant_framework.utils.performance_report import PerformanceReport


class TradingPipeline:
    """
    Orchestrates the complete trading workflow.
    
    Pipeline stages:
    1. Load and preprocess data
    2. Generate trading signals
    3. Run backtest
    4. Generate performance report
    5. Visualize results
    
    Example:
        pipeline = TradingPipeline(
            data_loader=YahooDataLoader("AAPL", start="2020-01-01"),
            strategy=MomentumStrategy(short_window=20, long_window=50),
            backtester=Backtester(initial_capital=100000)
        )
        pipeline.run()
    """
    
    def __init__(
        self,
        data_loader: BaseDataLoader,
        strategy: BaseStrategy,
        backtester: Backtester,
        verbose: bool = True,
        save_results: bool = True,
        output_dir: str = "results"
    ):
        """
        Initialize trading pipeline.
        
        Args:
            data_loader: Data loader instance
            strategy: Trading strategy instance
            backtester: Backtester instance
            verbose: Whether to print progress and results
            save_results: Whether to save results to disk
            output_dir: Directory for saving results
        """
        self.data_loader = data_loader
        self.strategy = strategy
        self.backtester = backtester
        self.verbose = verbose
        self.save_results = save_results
        self.output_dir = output_dir
        
        self.logger = setup_logger("TradingPipeline", verbose=verbose)
        
        self.data: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.Series] = None
        self.results: Optional[Dict[str, Any]] = None
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete trading pipeline.
        
        Returns:
            Dictionary with all results
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRADING PIPELINE")
        self.logger.info("=" * 60)
        
        # Step 1: Load and preprocess data
        self.logger.info("Step 1: Loading data...")
        self.data = self._load_data()
        self.logger.info(f"Loaded {len(self.data)} data points")
        self.logger.info(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        
        # Step 2: Generate signals
        self.logger.info("\nStep 2: Generating trading signals...")
        self.signals = self._generate_signals()
        num_long = (self.signals == 1).sum()
        num_short = (self.signals == -1).sum()
        num_neutral = (self.signals == 0).sum()
        self.logger.info(f"Signals - Long: {num_long}, Short: {num_short}, Neutral: {num_neutral}")
        
        # Step 3: Run backtest
        self.logger.info("\nStep 3: Running backtest...")
        self.results = self._run_backtest()
        
        # Step 4: Generate report
        self.logger.info("\nStep 4: Generating performance report...")
        self._generate_report()
        
        # Step 5: Create visualizations
        if self.verbose:
            self.logger.info("\nStep 5: Creating visualizations...")
            self._create_visualizations()
        
        # Step 6: Save results
        if self.save_results:
            self.logger.info("\nStep 6: Saving results...")
            self._save_results()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 60)
        
        return self.results
    
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess data."""
        data = self.data_loader.get_data()
        
        # Add technical indicators if needed
        from quant_framework.data.indicators import TechnicalIndicators
        data = TechnicalIndicators.add_all_indicators(data)
        
        return data
    
    def _generate_signals(self) -> pd.Series:
        """Generate trading signals using the strategy."""
        signals = self.strategy.generate_signals(self.data)
        return signals
    
    def _run_backtest(self) -> Dict[str, Any]:
        """Run backtest simulation."""
        results = self.backtester.run(self.data, self.signals)
        # Add data and signals to results for visualization
        results['data'] = self.data
        results['signals'] = self.signals
        return results
    
    def _generate_report(self) -> None:
        """Generate and print performance report."""
        if self.verbose:
            self.backtester.print_summary()
    
    def _create_visualizations(self) -> None:
        """Create performance visualizations."""
        try:
            # Plot equity curve
            plot_equity_curve(
                self.backtester.get_equity_curve(),
                title=f"{self.strategy.config.name} - Equity Curve"
            )
            
            # Plot drawdown
            plot_drawdown(
                self.backtester.get_equity_curve(),
                title=f"{self.strategy.config.name} - Drawdown"
            )
            
            # Plot signals overlaid on price
            price_col = 'close' if 'close' in self.data.columns else 'Close'
            plot_signals(
                self.data[price_col],
                self.signals,
                title=f"{self.strategy.config.name} - Trading Signals"
            )
            
        except Exception as e:
            self.logger.warning(f"Could not create visualizations: {e}")
    
    def _save_results(self) -> None:
        """Save results to disk."""
        from pathlib import Path
        import json
        
        # Create output directory
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = self.strategy.config.name
        
        try:
            # Save equity curve
            equity_file = output_path / f"{strategy_name}_equity_{timestamp}.csv"
            self.backtester.get_equity_curve().to_csv(equity_file)
            self.logger.info(f"Saved equity curve to {equity_file}")
            
            # Save trades
            if not self.backtester.get_trades().empty:
                trades_file = output_path / f"{strategy_name}_trades_{timestamp}.csv"
                self.backtester.get_trades().to_csv(trades_file, index=False)
                self.logger.info(f"Saved trades to {trades_file}")
            
            # Save metrics
            metrics_file = output_path / f"{strategy_name}_metrics_{timestamp}.json"
            metrics = self.backtester.get_metrics()
            # Convert numpy types to Python types for JSON serialization
            metrics_serializable = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in metrics.items()
            }
            with open(metrics_file, 'w') as f:
                json.dump(metrics_serializable, f, indent=4)
            self.logger.info(f"Saved metrics to {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get pipeline results.
        
        Returns:
            Dictionary with all results
        """
        if self.results is None:
            raise ValueError("No results available. Run pipeline first.")
        
        return self.results
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve from backtest."""
        return self.backtester.get_equity_curve()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return self.backtester.get_metrics()


class MultiStrategyPipeline:
    """
    Run multiple strategies in parallel for comparison.
    
    Example:
        strategies = [
            MomentumStrategy(20, 50),
            MeanReversionStrategy(20, 2),
        ]
        multi_pipeline = MultiStrategyPipeline(
            data_loader=YahooDataLoader("AAPL"),
            strategies=strategies,
            backtester_config={'initial_capital': 100000}
        )
        results = multi_pipeline.run_all()
    """
    
    def __init__(
        self,
        data_loader: BaseDataLoader,
        strategies: list,
        backtester_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-strategy pipeline.
        
        Args:
            data_loader: Data loader instance
            strategies: List of strategy instances
            backtester_config: Configuration dict for backtester
        """
        self.data_loader = data_loader
        self.strategies = strategies
        self.backtester_config = backtester_config or {}
        
        self.logger = setup_logger("MultiStrategyPipeline")
        self.results: Dict[str, Any] = {}
    
    def run_all(self) -> Dict[str, Any]:
        """
        Run all strategies and compare results.
        
        Returns:
            Dictionary with results for each strategy
        """
        self.logger.info(f"Running {len(self.strategies)} strategies...")
        
        for strategy in self.strategies:
            self.logger.info(f"\nRunning {strategy.config.name}...")
            
            backtester = Backtester(**self.backtester_config)
            
            pipeline = TradingPipeline(
                data_loader=self.data_loader,
                strategy=strategy,
                backtester=backtester,
                verbose=False,
                save_results=False
            )
            
            result = pipeline.run()
            self.results[strategy.config.name] = result
        
        # Compare strategies
        self._compare_strategies()
        
        return self.results
    
    def _compare_strategies(self) -> None:
        """Print comparison of all strategies."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STRATEGY COMPARISON")
        self.logger.info("=" * 80)
        
        comparison_data = []
        
        for name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Strategy': name,
                'Total Return': f"{metrics['total_return']*100:.2f}%",
                'Sharpe': f"{metrics['sharpe_ratio']:.3f}",
                'Max DD': f"{metrics['max_drawdown']*100:.2f}%",
                'Win Rate': f"{metrics['win_rate']*100:.2f}%"
            })
        
        # Print as table
        df = pd.DataFrame(comparison_data)
        self.logger.info("\n" + df.to_string(index=False))
        self.logger.info("=" * 80)

