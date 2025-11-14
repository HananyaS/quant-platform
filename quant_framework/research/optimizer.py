"""
Strategy optimization tools.

Provides parameter optimization using various methods:
- Grid search
- Random search  
- Bayesian optimization
- Genetic algorithms
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Optional
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class StrategyOptimizer:
    """
    Optimize strategy parameters.
    
    Example:
        optimizer = StrategyOptimizer(strategy_class=MomentumStrategy)
        param_grid = {
            'short_window': [5, 10, 20],
            'long_window': [20, 50, 100]
        }
        best_params, results = optimizer.grid_search(data, param_grid)
    """
    
    def __init__(
        self,
        strategy_class,
        data_loader=None,
        backtester_config: Optional[Dict] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            data_loader: Data loader instance
            backtester_config: Backtester configuration
        """
        self.strategy_class = strategy_class
        self.data_loader = data_loader
        self.backtester_config = backtester_config or {}
        self.results = []
    
    def grid_search(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        metric: str = 'sharpe_ratio',
        verbose: bool = True
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Perform grid search over parameter space.
        
        Args:
            data: Price data
            param_grid: Dictionary of parameters to search
            metric: Metric to optimize
            verbose: Print progress
            
        Returns:
            Tuple of (best_params, results_df)
        """
        from quant_framework.backtest.backtester import Backtester
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        
        if verbose:
            print(f"Grid Search: Testing {total_combinations} parameter combinations")
            print(f"Optimizing for: {metric}")
            print()
        
        results = []
        
        for i, params_tuple in enumerate(param_combinations, 1):
            params = dict(zip(param_names, params_tuple))
            
            if verbose and i % max(1, total_combinations // 10) == 0:
                print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
            
            try:
                # Create strategy with these parameters
                strategy = self.strategy_class(**params)
                
                # Generate signals
                signals = strategy.generate_signals(data)
                
                # Backtest
                backtester = Backtester(**self.backtester_config)
                bt_results = backtester.run(data, signals)
                
                # Store results
                result = {
                    **params,
                    **bt_results['metrics']
                }
                results.append(result)
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed for params {params}: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            raise ValueError("No valid parameter combinations found")
        
        # Find best parameters
        best_idx = results_df[metric].idxmax()
        best_params = {k: results_df.loc[best_idx, k] 
                      for k in param_names}
        best_score = results_df.loc[best_idx, metric]
        
        if verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZATION COMPLETE")
            print('='*60)
            print(f"Best {metric}: {best_score:.4f}")
            print(f"Best parameters:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
        
        self.results = results_df
        return best_params, results_df
    
    def random_search(
        self,
        data: pd.DataFrame,
        param_distributions: Dict[str, Tuple],
        n_iter: int = 100,
        metric: str = 'sharpe_ratio',
        random_state: int = 42,
        verbose: bool = True
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Perform random search over parameter space.
        
        Args:
            data: Price data
            param_distributions: Dict of (min, max) tuples for each param
            n_iter: Number of iterations
            metric: Metric to optimize
            random_state: Random seed
            verbose: Print progress
            
        Returns:
            Tuple of (best_params, results_df)
        """
        from quant_framework.backtest.backtester import Backtester
        
        np.random.seed(random_state)
        
        if verbose:
            print(f"Random Search: Testing {n_iter} random parameter combinations")
            print(f"Optimizing for: {metric}")
            print()
        
        results = []
        
        for i in range(n_iter):
            if verbose and (i + 1) % max(1, n_iter // 10) == 0:
                print(f"Progress: {i+1}/{n_iter} ({(i+1)/n_iter*100:.1f}%)")
            
            # Sample random parameters
            params = {}
            for param_name, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            try:
                # Create strategy
                strategy = self.strategy_class(**params)
                signals = strategy.generate_signals(data)
                
                # Backtest
                backtester = Backtester(**self.backtester_config)
                bt_results = backtester.run(data, signals)
                
                # Store results
                result = {
                    **params,
                    **bt_results['metrics']
                }
                results.append(result)
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed for params {params}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            raise ValueError("No valid parameter combinations found")
        
        # Find best
        best_idx = results_df[metric].idxmax()
        best_params = {k: results_df.loc[best_idx, k] 
                      for k in param_distributions.keys()}
        best_score = results_df.loc[best_idx, metric]
        
        if verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZATION COMPLETE")
            print('='*60)
            print(f"Best {metric}: {best_score:.4f}")
            print(f"Best parameters:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
        
        self.results = results_df
        return best_params, results_df
    
    def walk_forward_optimization(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 21,
        metric: str = 'sharpe_ratio',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.
        
        Optimizes parameters on train window, tests on next window,
        then steps forward and repeats.
        
        Args:
            data: Price data
            param_grid: Parameters to optimize
            train_size: Training window size
            test_size: Testing window size
            step_size: Step size for rolling window
            metric: Metric to optimize
            verbose: Print progress
            
        Returns:
            Dictionary with walk-forward results
        """
        from quant_framework.backtest.backtester import Backtester
        
        n = len(data)
        windows = []
        
        # Generate windows
        start = 0
        while start + train_size + test_size <= n:
            train_end = start + train_size
            test_end = train_end + test_size
            
            windows.append({
                'train_start': start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })
            
            start += step_size
        
        if verbose:
            print(f"Walk-Forward Optimization: {len(windows)} windows")
            print(f"  Train size: {train_size}")
            print(f"  Test size: {test_size}")
            print(f"  Step size: {step_size}")
            print()
        
        window_results = []
        
        for i, window in enumerate(windows, 1):
            if verbose:
                print(f"Window {i}/{len(windows)}")
            
            # Get train and test data
            train_data = data.iloc[window['train_start']:window['train_end']]
            test_data = data.iloc[window['test_start']:window['test_end']]
            
            # Optimize on train window
            best_params, train_results = self.grid_search(
                train_data, param_grid, metric, verbose=False
            )
            
            # Test on test window with best params
            strategy = self.strategy_class(**best_params)
            signals = strategy.generate_signals(test_data)
            
            backtester = Backtester(**self.backtester_config)
            test_results = backtester.run(test_data, signals)
            
            window_results.append({
                'window': i,
                'train_dates': (train_data.index[0], train_data.index[-1]),
                'test_dates': (test_data.index[0], test_data.index[-1]),
                'best_params': best_params,
                'test_metrics': test_results['metrics']
            })
            
            if verbose:
                test_metric = test_results['metrics'][metric]
                print(f"  Test {metric}: {test_metric:.4f}")
                print(f"  Best params: {best_params}")
                print()
        
        # Calculate average test performance
        avg_test_metrics = {}
        if window_results:
            metric_keys = window_results[0]['test_metrics'].keys()
            for key in metric_keys:
                values = [w['test_metrics'][key] for w in window_results]
                avg_test_metrics[key] = np.mean(values)
        
        if verbose:
            print(f"{'='*60}")
            print("WALK-FORWARD OPTIMIZATION COMPLETE")
            print('='*60)
            print(f"Average test {metric}: {avg_test_metrics[metric]:.4f}")
        
        return {
            'windows': window_results,
            'avg_test_metrics': avg_test_metrics,
            'n_windows': len(windows)
        }
    
    def plot_optimization_surface(
        self,
        param1: str,
        param2: str,
        metric: str = 'sharpe_ratio'
    ):
        """
        Plot 3D optimization surface.
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to plot
        """
        if not hasattr(self, 'results') or len(self.results) == 0:
            raise ValueError("No optimization results available. Run optimization first.")
        
        import plotly.graph_objects as go
        
        # Pivot data for surface plot
        pivot = self.results.pivot_table(
            values=metric,
            index=param1,
            columns=param2,
            aggfunc='mean'
        )
        
        fig = go.Figure(data=[go.Surface(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis'
        )])
        
        fig.update_layout(
            title=f'Optimization Surface: {metric}',
            scene=dict(
                xaxis_title=param2,
                yaxis_title=param1,
                zaxis_title=metric
            ),
            width=800,
            height=700
        )
        
        return fig


