"""
Shared helper functions for app tabs.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quant_framework.data.loaders import YahooDataLoader
from quant_framework.backtest.backtester import Backtester
from quant_framework.backtest.fast_backtester import FastBacktester
from quant_framework.infra import TradingPipeline


def render_strategy_params(strategy_name, strategies_dict):
    """Render strategy parameter inputs."""
    params = {}
    strategy_info = strategies_dict[strategy_name]

    if not strategy_info["params"]:
        return params

    st.subheader("Strategy Parameters")

    for param_name, param_info in strategy_info["params"].items():
        if param_info["type"] == "number":
            params[param_name] = st.number_input(
                param_name.replace('_', ' ').title(),
                min_value=param_info.get("min", 0),
                max_value=param_info.get("max", 100),
                value=param_info["default"],
                step=param_info.get("step", 1),
                help=param_info.get("help", "")
            )
        elif param_info["type"] == "checkbox":
            params[param_name] = st.checkbox(
                param_name.replace('_', ' ').title(),
                value=param_info["default"],
                help=param_info.get("help", "")
            )

    return params


@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_data(symbol, start_date, end_date):
    """Load and cache market data."""
    loader = YahooDataLoader(symbol, start_date, end_date)
    return loader.get_data()


def run_backtest(config, strategy_name, strategy_params, strategies_dict):
    """Run backtest and return results."""
    try:
        # Load data with caching
        with st.spinner(f"📊 Loading {config['symbol']} data..."):
            data = load_cached_data(config["symbol"], config["start_date"], config["end_date"])
        
        # Create strategy
        strategy_class = strategies_dict[strategy_name]["class"]
        if strategy_params:
            strategy = strategy_class(**strategy_params)
        else:
            strategy = strategy_class()

        # Generate signals
        with st.spinner(f"🔄 Generating signals for {strategy_name}..."):
            signals = strategy.generate_signals(data)

        # Override for Buy & Hold
        is_buy_hold = strategy_name == "📊 Buy & Hold (Baseline)"
        if is_buy_hold:
            position_size = 1.0
            max_position_pct = 1.0
        else:
            position_size = config["position_size"]
            max_position_pct = config["max_position_pct"]

        # Choose backtester (Fast by default, ~10-50x faster)
        use_fast = config.get("use_fast_backtester", True)
        BacktesterClass = FastBacktester if use_fast else Backtester
        
        backtester = BacktesterClass(
            initial_capital=config["initial_capital"],
            fee_per_share=config.get("fee_per_share", 0.005),
            fee_minimum=config.get("fee_minimum", 1.0),
            slippage_perc=config["slippage_perc"],
            use_per_share_fee=config.get("use_per_share_fee", True),
            position_size=position_size,
            use_fixed_trade_value=config.get("use_fixed_trade_value", False),
            fixed_trade_value=config.get("fixed_trade_value", 10000),
            max_position_pct=max_position_pct
        )

        # Run backtest
        mode_label = "⚡ Fast Mode" if use_fast else "🐢 Accurate Mode"
        with st.spinner(f"{mode_label} - Running backtest simulation..."):
            results = backtester.run(data, signals)
        
        # Add data and signals to results
        results['data'] = data
        results['signals'] = signals
        
        return results, None

    except Exception as e:
        return None, str(e)


def plot_equity_curve(equity_curve, title="Equity Curve"):
    """Plot equity curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Equity',
        line=dict(color='#2E86AB', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        height=500
    )

    return fig


def plot_drawdown(equity_curve):
    """Plot drawdown."""
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#D62828', width=2)
    ))

    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=500
    )

    return fig


def plot_signals(data, signals, symbol):
    """Plot price with trading signals (optimized)."""
    # Downsample if too large
    if len(data) > 1500:
        step = max(1, len(data) // 1000)
        data = data.iloc[::step].copy()
        signals = signals.iloc[::step].copy()
    
    price = data['close'] if 'close' in data.columns else data['Close']
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=data.index, y=price,
        mode='lines', name='Price',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # Signal markers (only changes)
    signal_changes = signals.diff().fillna(signals.iloc[0])
    change_mask = signal_changes != 0
    
    if change_mask.any():
        change_idx = data.index[change_mask]
        change_prices = price[change_mask]
        change_signals = signals[change_mask]
        
        colors = ['green' if s == 1 else 'red' if s == -1 else 'orange' for s in change_signals]
        symbols = ['triangle-up' if s == 1 else 'triangle-down' if s == -1 else 'x' for s in change_signals]
        
        fig.add_trace(go.Scatter(
            x=change_idx, y=change_prices,
            mode='markers', name='Signals',
            marker=dict(color=colors, size=10, symbol=symbols)
        ))
    
    fig.update_layout(
        title=f"{symbol} - Trading Signals",
        xaxis_title="Date", yaxis_title="Price ($)",
        height=500, hovermode='x unified'
    )
    
    return fig

