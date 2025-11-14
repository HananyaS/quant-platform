"""
Performance metrics for backtesting.

Provides comprehensive performance analysis functions including:
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Win rate
- And more...
"""

import pandas as pd
import numpy as np
from typing import Optional


def calc_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate returns from equity curve.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        Series of returns
    """
    return equity_curve.pct_change()


def calc_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of trading periods per year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
    
    return sharpe


def calc_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of std dev).
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of trading periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_std)
    
    return sortino


def calc_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        Maximum drawdown (as positive decimal, e.g., 0.25 for 25% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Return maximum drawdown as positive value
    max_dd = abs(drawdown.min())
    
    return max_dd


def calc_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series over time.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        Series of drawdowns at each point
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    
    return drawdown


def calc_calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Series of returns
        equity_curve: Series of portfolio values
        periods_per_year: Number of trading periods per year
        
    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0
    
    annualized_return = returns.mean() * periods_per_year
    max_dd = calc_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return 0.0
    
    calmar = annualized_return / max_dd
    
    return calmar


def calc_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of positive returns).
    
    Args:
        returns: Series of returns
        
    Returns:
        Win rate (0 to 1)
    """
    if len(returns) == 0:
        return 0.0
    
    winning_periods = (returns > 0).sum()
    total_periods = len(returns[returns != 0])  # Exclude zero returns
    
    if total_periods == 0:
        return 0.0
    
    return winning_periods / total_periods


def calc_profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        returns: Series of returns
        
    Returns:
        Profit factor
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calc_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR at specified confidence level
    """
    if len(returns) == 0:
        return 0.0
    
    return np.percentile(returns, (1 - confidence_level) * 100)


def calc_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        CVaR at specified confidence level
    """
    if len(returns) == 0:
        return 0.0
    
    var = calc_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    
    return cvar


def calc_annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of trading periods per year
        
    Returns:
        Annualized return
    """
    if len(returns) == 0:
        return 0.0
    
    return returns.mean() * periods_per_year


def calc_annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of trading periods per year
        
    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


def calc_turnover(signals: pd.Series) -> float:
    """
    Calculate portfolio turnover (number of position changes).
    
    Args:
        signals: Series of position signals
        
    Returns:
        Average turnover per period
    """
    if len(signals) == 0:
        return 0.0
    
    position_changes = signals.diff().abs()
    turnover = position_changes.sum() / len(signals)
    
    return turnover


def calc_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of trading periods per year
        
    Returns:
        Information ratio
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()
    
    if tracking_error == 0:
        return 0.0
    
    ir = np.sqrt(periods_per_year) * (active_returns.mean() / tracking_error)
    
    return ir


def calc_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    signals: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate all performance metrics.
    
    Args:
        returns: Series of returns
        equity_curve: Series of portfolio values
        signals: Optional series of position signals
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of trading periods per year
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0.0,
        'annual_return': calc_annual_return(returns, periods_per_year),
        'annual_volatility': calc_annual_volatility(returns, periods_per_year),
        'sharpe_ratio': calc_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calc_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': calc_max_drawdown(equity_curve),
        'calmar_ratio': calc_calmar_ratio(returns, equity_curve, periods_per_year),
        'win_rate': calc_win_rate(returns),
        'profit_factor': calc_profit_factor(returns),
        'var_95': calc_var(returns, 0.95),
        'cvar_95': calc_cvar(returns, 0.95),
    }
    
    if signals is not None:
        metrics['turnover'] = calc_turnover(signals)
        metrics['num_trades'] = (signals.diff() != 0).sum()
    
    return metrics

