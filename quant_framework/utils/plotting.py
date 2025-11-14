"""
Plotting utilities for visualizing trading results.
"""

import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from quant_framework.backtest.metrics import calc_drawdown_series


def plot_equity_curve(
    equity_curve: pd.Series,
    title: str = "Equity Curve",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot equity curve over time.
    
    Args:
        equity_curve: Series with portfolio values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(equity_curve.index, equity_curve.values, linewidth=2, color='#2E86AB')
    ax.fill_between(equity_curve.index, equity_curve.values,
                     alpha=0.3, color='#2E86AB')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Equity curve saved to {save_path}")
    
    plt.show()


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot drawdown over time.
    
    Args:
        equity_curve: Series with portfolio values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    drawdown = calc_drawdown_series(equity_curve)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                     alpha=0.6, color='#A23B72')
    ax.plot(drawdown.index, drawdown.values * 100,
            linewidth=2, color='#A23B72')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Drawdown plot saved to {save_path}")
    
    plt.show()


def plot_signals(
    prices: pd.Series,
    signals: pd.Series,
    title: str = "Trading Signals",
    figsize: tuple = (14, 7),
    save_path: Optional[str] = None
) -> None:
    """
    Plot price with ALL trading signals (entries, exits, transitions).
    
    Args:
        prices: Price series
        signals: Signal series (+1, 0, -1)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot prices
    ax1.plot(prices.index, prices.values, linewidth=1.5,
             color='#333333', label='Price', zorder=1)
    
    # Add position background shading
    for i in range(len(signals) - 1):
        if signals.iloc[i] == 1:  # Long position
            ax1.axvspan(prices.index[i], prices.index[i+1], 
                       alpha=0.1, color='#06A77D', zorder=0)
        elif signals.iloc[i] == -1:  # Short position
            ax1.axvspan(prices.index[i], prices.index[i+1], 
                       alpha=0.1, color='#D62828', zorder=0)
    
    # Detect signal changes (entries, exits, reversals)
    signal_changes = signals.diff()
    
    # Long ENTRIES (0 -> 1 or -1 -> 1)
    long_entries = (signals == 1) & (signal_changes != 0)
    if long_entries.any():
        ax1.scatter(prices.index[long_entries], prices[long_entries],
                   marker='^', color='#06A77D', s=150, label='Long Entry',
                   edgecolors='darkgreen', linewidths=2, zorder=5)
    
    # Short ENTRIES (0 -> -1 or 1 -> -1)
    short_entries = (signals == -1) & (signal_changes != 0)
    if short_entries.any():
        ax1.scatter(prices.index[short_entries], prices[short_entries],
                   marker='v', color='#D62828', s=150, label='Short Entry',
                   edgecolors='darkred', linewidths=2, zorder=5)
    
    # EXIT signals (any position -> 0)
    exits = (signals == 0) & (signal_changes != 0)
    if exits.any():
        ax1.scatter(prices.index[exits], prices[exits],
                   marker='x', color='#F18F01', s=100, label='Exit',
                   linewidths=2, zorder=5)
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot signal timeline (bar chart)
    colors = ['#D62828' if s < 0 else '#06A77D' if s > 0 else '#CCCCCC'
              for s in signals]
    ax2.bar(signals.index, signals.values, color=colors, alpha=0.6, width=1.0)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Signal', fontsize=12)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short\n(-1)', 'Cash\n(0)', 'Long\n(+1)'])
    ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Signals plot saved to {save_path}")
    
    plt.show()


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    figsize: tuple = (10, 6),
    bins: int = 50,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of returns.
    
    Args:
        returns: Returns series
        title: Plot title
        figsize: Figure size
        bins: Number of histogram bins
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(returns.dropna(), bins=bins, alpha=0.7,
            color='#2E86AB', edgecolor='black')
    
    # Add vertical line for mean
    mean_return = returns.mean()
    ax.axvline(mean_return, color='#D62828', linestyle='--',
               linewidth=2, label=f'Mean: {mean_return*100:.2f}%')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Returns', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Returns distribution saved to {save_path}")
    
    plt.show()


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    title: str = "Rolling Sharpe Ratio",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot rolling Sharpe ratio.
    
    Args:
        returns: Returns series
        window: Rolling window size
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    rolling_sharpe = (
        returns.rolling(window=window).mean() /
        returns.rolling(window=window).std() *
        np.sqrt(252)
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            linewidth=2, color='#2E86AB')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(1, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rolling Sharpe plot saved to {save_path}")
    
    plt.show()


def plot_strategy_comparison(
    equity_curves: dict,
    title: str = "Strategy Comparison",
    figsize: tuple = (14, 7),
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple equity curves for strategy comparison.
    
    Args:
        equity_curves: Dict mapping strategy names to equity curves
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2E86AB', '#A23B72', '#06A77D', '#F18F01', '#D62828']
    
    for i, (name, curve) in enumerate(equity_curves.items()):
        # Normalize to starting value of 100
        normalized = (curve / curve.iloc[0]) * 100
        ax.plot(normalized.index, normalized.values,
                linewidth=2, label=name, color=colors[i % len(colors)])
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Value (Base=100)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Strategy comparison saved to {save_path}")
    
    plt.show()

