"""
Shared helper functions for app tabs.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quant_framework.data.loaders import YahooDataLoader
from quant_framework.backtest.backtester import Backtester
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


def run_backtest(config, strategy_name, strategy_params, strategies_dict):
    """Run backtest and return results."""
    try:
        # Load data
        loader = YahooDataLoader(config["symbol"], config["start_date"], config["end_date"])

        # Create strategy
        strategy_class = strategies_dict[strategy_name]["class"]
        if strategy_params:
            strategy = strategy_class(**strategy_params)
        else:
            strategy = strategy_class()

        # Override for Buy & Hold
        is_buy_hold = strategy_name == "📊 Buy & Hold (Baseline)"
        if is_buy_hold:
            position_size = 1.0
            max_position_pct = 1.0
        else:
            position_size = config["position_size"]
            max_position_pct = config["max_position_pct"]

        # Create backtester
        backtester = Backtester(
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

        # Run pipeline (includes data and signals in results)
        pipeline = TradingPipeline(
            data_loader=loader,
            strategy=strategy,
            backtester=backtester,
            save_results=False
        )

        results = pipeline.run()

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
    """Plot price with ALL trading signals (entries, exits, transitions)."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{symbol} - Price with Trading Signals", "Signal Timeline"),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Get price data
    price = data['close'] if 'close' in data.columns else data['Close']

    # Plot price
    fig.add_trace(go.Scatter(
        x=data.index,
        y=price,
        mode='lines',
        name='Price',
        line=dict(color='gray', width=1.5),
        showlegend=True
    ), row=1, col=1)

    # Detect signal changes (entries, exits, reversals)
    signal_changes = signals.diff()

    # Long ENTRIES (0 -> 1 or -1 -> 1)
    long_entries = (signals == 1) & (signal_changes != 0)
    if long_entries.any():
        fig.add_trace(go.Scatter(
            x=data.index[long_entries],
            y=price[long_entries],
            mode='markers',
            name='Long Entry',
            marker=dict(color='green', size=12, symbol='triangle-up', line=dict(width=2, color='darkgreen')),
            showlegend=True
        ), row=1, col=1)

    # Short ENTRIES (0 -> -1 or 1 -> -1)
    short_entries = (signals == -1) & (signal_changes != 0)
    if short_entries.any():
        fig.add_trace(go.Scatter(
            x=data.index[short_entries],
            y=price[short_entries],
            mode='markers',
            name='Short Entry',
            marker=dict(color='red', size=12, symbol='triangle-down', line=dict(width=2, color='darkred')),
            showlegend=True
        ), row=1, col=1)

    # EXIT signals (any position -> 0)
    exits = (signals == 0) & (signal_changes != 0)
    if exits.any():
        fig.add_trace(go.Scatter(
            x=data.index[exits],
            y=price[exits],
            mode='markers',
            name='Exit',
            marker=dict(color='orange', size=10, symbol='x', line=dict(width=2, color='darkorange')),
            showlegend=True
        ), row=1, col=1)

    # Position background shading
    for i in range(len(signals) - 1):
        if signals.iloc[i] == 1:  # Long position
            fig.add_vrect(
                x0=data.index[i], x1=data.index[i + 1],
                fillcolor="green", opacity=0.1,
                layer="below", line_width=0,
                row=1, col=1
            )
        elif signals.iloc[i] == -1:  # Short position
            fig.add_vrect(
                x0=data.index[i], x1=data.index[i + 1],
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0,
                row=1, col=1
            )

    # Plot signal timeline (bar chart)
    colors = ['red' if s < 0 else 'green' if s > 0 else 'lightgray' for s in signals]

    fig.add_trace(go.Bar(
        x=data.index,
        y=signals,
        marker_color=colors,
        name='Signal',
        showlegend=False,
        hovertemplate='Date: %{x}<br>Signal: %{y}<br><extra></extra>'
    ), row=2, col=1)

    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Signal", row=2, col=1, tickvals=[-1, 0, 1],
                     ticktext=['Short (-1)', 'Cash (0)', 'Long (+1)'])

    fig.update_layout(
        hovermode='x unified',
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

