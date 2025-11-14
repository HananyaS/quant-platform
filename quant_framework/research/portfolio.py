"""
Portfolio analysis and multi-symbol backtesting.

Provides:
- Multi-asset portfolio construction
- Portfolio optimization
- Risk analysis
- Asset correlation analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PortfolioAnalyzer:
    """
    Analyze and optimize multi-asset portfolios.
    
    Example:
        analyzer = PortfolioAnalyzer()
        weights = analyzer.optimize_weights(returns_df, method='sharpe')
        metrics = analyzer.analyze_portfolio(returns_df, weights)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
    
    def analyze_portfolio(
        self,
        returns: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Analyze portfolio performance.
        
        Args:
            returns: DataFrame with returns for each asset
            weights: Dictionary of asset weights (equal weight if None)
            
        Returns:
            Dictionary with portfolio metrics
        """
        if weights is None:
            # Equal weights
            weights = {col: 1.0 / len(returns.columns) for col in returns.columns}
        
        # Convert weights to array
        weight_array = np.array([weights[col] for col in returns.columns])
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weight_array).sum(axis=1)
        
        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'weights': weights
        }
    
    def optimize_weights(
        self,
        returns: pd.DataFrame,
        method: str = 'sharpe',
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights.
        
        Args:
            returns: DataFrame with returns for each asset
            method: Optimization method ('sharpe', 'min_vol', 'max_return', 'equal')
            constraints: Optional constraints dict
            
        Returns:
            Dictionary with optimized weights
        """
        if method == 'equal':
            # Equal weighting
            n_assets = len(returns.columns)
            return {col: 1.0 / n_assets for col in returns.columns}
        
        elif method == 'sharpe':
            # Maximize Sharpe ratio
            return self._optimize_sharpe(returns, constraints)
        
        elif method == 'min_vol':
            # Minimize volatility
            return self._optimize_min_vol(returns, constraints)
        
        elif method == 'max_return':
            # Maximize return
            return self._optimize_max_return(returns, constraints)
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_sharpe(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize for maximum Sharpe ratio."""
        from scipy.optimize import minimize
        
        n_assets = len(returns.columns)
        
        def neg_sharpe(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            if portfolio_vol == 0:
                return 0
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Negative because we minimize
        
        # Constraints: weights sum to 1
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds: 0 <= weight <= 1
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = {col: w for col, w in zip(returns.columns, result.x)}
            return weights
        else:
            # Return equal weights if optimization fails
            return {col: 1.0 / n_assets for col in returns.columns}
    
    def _optimize_min_vol(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize for minimum volatility."""
        from scipy.optimize import minimize
        
        n_assets = len(returns.columns)
        
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        result = minimize(
            portfolio_vol,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = {col: w for col, w in zip(returns.columns, result.x)}
            return weights
        else:
            return {col: 1.0 / n_assets for col in returns.columns}
    
    def _optimize_max_return(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize for maximum return."""
        from scipy.optimize import minimize
        
        n_assets = len(returns.columns)
        
        def neg_return(weights):
            return -np.sum(returns.mean() * weights) * 252
        
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        result = minimize(
            neg_return,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = {col: w for col, w in zip(returns.columns, result.x)}
            return weights
        else:
            return {col: 1.0 / n_assets for col in returns.columns}
    
    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            returns: DataFrame with returns for each asset
            n_points: Number of points on frontier
            
        Returns:
            DataFrame with risk/return points
        """
        from scipy.optimize import minimize
        
        n_assets = len(returns.columns)
        
        # Calculate range of returns
        min_ret = returns.mean().min() * 252
        max_ret = returns.mean().max() * 252
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_vols = []
        frontier_weights = []
        
        for target_return in target_returns:
            def portfolio_vol(weights):
                return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return}
            ]
            bounds = tuple((0, 1) for _ in range(n_assets))
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            result = minimize(
                portfolio_vol,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                frontier_vols.append(result.fun)
                frontier_weights.append(result.x)
            else:
                frontier_vols.append(np.nan)
                frontier_weights.append([np.nan] * n_assets)
        
        frontier_df = pd.DataFrame({
            'return': target_returns,
            'volatility': frontier_vols,
            'sharpe': [(r - self.risk_free_rate) / v if v > 0 else 0 
                      for r, v in zip(target_returns, frontier_vols)]
        })
        
        # Add weights
        for i, col in enumerate(returns.columns):
            frontier_df[f'weight_{col}'] = [w[i] for w in frontier_weights]
        
        return frontier_df.dropna()
    
    def correlation_analysis(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix.
        
        Args:
            returns: DataFrame with returns for each asset
            
        Returns:
            Correlation matrix
        """
        return returns.corr()
    
    def risk_contribution(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Calculate risk contribution of each asset.
        
        Args:
            returns: DataFrame with returns
            weights: Asset weights
            
        Returns:
            DataFrame with risk contributions
        """
        weight_array = np.array([weights[col] for col in returns.columns])
        cov_matrix = returns.cov() * 252
        
        portfolio_vol = np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix, weight_array)))
        
        marginal_contrib = np.dot(cov_matrix, weight_array) / portfolio_vol
        risk_contrib = weight_array * marginal_contrib
        
        df = pd.DataFrame({
            'asset': returns.columns,
            'weight': weight_array,
            'risk_contribution': risk_contrib,
            'risk_contribution_pct': risk_contrib / portfolio_vol * 100
        })
        
        return df.sort_values('risk_contribution_pct', ascending=False)


class MultiSymbolBacktester:
    """
    Backtest strategies across multiple symbols.
    
    Example:
        backtester = MultiSymbolBacktester()
        results = backtester.run_multi_symbol(
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            strategy_class=MomentumStrategy,
            start='2020-01-01',
            end='2024-01-01'
        )
    """
    
    def __init__(self):
        """Initialize multi-symbol backtester."""
        self.results = {}
    
    def run_multi_symbol(
        self,
        symbols: List[str],
        strategy_class,
        start: str,
        end: str,
        strategy_params: Optional[Dict] = None,
        backtester_config: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """
        Run backtest across multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            strategy_class: Strategy class to use
            start: Start date
            end: End date
            strategy_params: Strategy parameters
            backtester_config: Backtester configuration
            
        Returns:
            Dictionary with results for each symbol
        """
        from quant_framework.data.loaders import YahooDataLoader
        from quant_framework.backtest.backtester import Backtester
        
        strategy_params = strategy_params or {}
        backtester_config = backtester_config or {}
        
        results = {}
        
        print(f"Running backtest for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"  [{i}/{len(symbols)}] {symbol}...", end=' ')
            
            try:
                # Load data
                loader = YahooDataLoader(symbol, start, end)
                data = loader.get_data()
                
                # Create strategy
                strategy = strategy_class(**strategy_params)
                signals = strategy.generate_signals(data)
                
                # Backtest
                backtester = Backtester(**backtester_config)
                result = backtester.run(data, signals)
                
                results[symbol] = result
                print(f"✓ (Sharpe: {result['metrics']['sharpe_ratio']:.2f})")
                
            except Exception as e:
                print(f"✗ ({str(e)})")
                results[symbol] = None
        
        self.results = results
        return results
    
    def compare_results(self) -> pd.DataFrame:
        """
        Compare results across symbols.
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.results:
            raise ValueError("No results available. Run backtest first.")
        
        comparison_data = []
        
        for symbol, result in self.results.items():
            if result is not None:
                metrics = result['metrics']
                comparison_data.append({
                    'Symbol': symbol,
                    'Total Return (%)': metrics['total_return'] * 100,
                    'Annual Return (%)': metrics['annual_return'] * 100,
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max DD (%)': metrics['max_drawdown'] * 100,
                    'Win Rate (%)': metrics['win_rate'] * 100
                })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Sharpe Ratio', ascending=False)


