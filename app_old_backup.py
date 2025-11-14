"""
Unified Quantitative Trading Platform

Complete platform for:
- Strategy backtesting
- ML/DL model training
- Portfolio optimization
- Research tools
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Quant Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import framework modules
from quant_framework.data.loaders import YahooDataLoader
from quant_framework.models import *
from quant_framework.backtest.backtester import Backtester
from quant_framework.infra import TradingPipeline
from plotly.subplots import make_subplots
from quant_framework.models.fibonacci_strategy import FibonacciStrategy

# Title
st.title("📈 Quantitative Trading Platform")
st.markdown("*Complete platform for strategy backtesting, ML/DL training, and portfolio optimization*")

# Strategy registry
STRATEGIES = {
    "📊 Buy & Hold (Baseline)": {
        "class": BuyHoldStrategy,
        "params": {},
        "description": "Simple buy and hold - the baseline that all strategies should beat!"
    },
    "Momentum (MA Crossover)": {
        "class": MomentumStrategy,
        "params": {
            "short_window": {"type": "number", "default": 20, "min": 5, "max": 100},
            "long_window": {"type": "number", "default": 50, "min": 10, "max": 200},
            "use_ema": {"type": "checkbox", "default": False},
            "allow_short": {"type": "checkbox", "default": False, "help": "Allow short positions"}
        }
    },
    "Mean Reversion (Bollinger)": {
        "class": MeanReversionStrategy,
        "params": {
            "window": {"type": "number", "default": 20, "min": 5, "max": 100},
            "num_std": {"type": "number", "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1},
            "exit_on_middle": {"type": "checkbox", "default": True},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "RSI Strategy": {
        "class": RSIStrategy,
        "params": {
            "rsi_window": {"type": "number", "default": 14, "min": 5, "max": 30},
            "oversold_threshold": {"type": "number", "default": 30, "min": 10, "max": 40},
            "overbought_threshold": {"type": "number", "default": 70, "min": 60, "max": 90},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "MACD Strategy": {
        "class": MACDStrategy,
        "params": {
            "fast_period": {"type": "number", "default": 12, "min": 5, "max": 30},
            "slow_period": {"type": "number", "default": 26, "min": 15, "max": 50},
            "signal_period": {"type": "number", "default": 9, "min": 5, "max": 20},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "Breakout Strategy": {
        "class": BreakoutStrategy,
        "params": {
            "lookback_period": {"type": "number", "default": 20, "min": 10, "max": 100},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "Turtle Trading": {
        "class": TurtleStrategy,
        "params": {
            "entry_period": {"type": "number", "default": 20, "min": 10, "max": 50},
            "exit_period": {"type": "number", "default": 10, "min": 5, "max": 30},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "Fibonacci Retracement": {
        "class": FibonacciStrategy,
        "params": {
            "lookback_period": {"type": "number", "default": 20, "min": 10, "max": 100},
            "retracement_level": {"type": "number", "default": 0.618, "min": 0.236, "max": 0.786, "step": 0.001},
            "tolerance": {"type": "number", "default": 0.02, "min": 0.01, "max": 0.10, "step": 0.01},
            "allow_short": {"type": "checkbox", "default": False, "help": "Allow short positions"}
        }
    }
}

# Sidebar - Global Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Date range
    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End", value=datetime.now())
    
    # Symbol
    st.markdown("### 📊 Symbol")
    symbol = st.text_input("Ticker", value="SPY")
    
    # Backtesting settings
    st.markdown("### 💰 Backtest Settings")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=0,
        value=10000,
        step=1000
    )
    
    # Fee structure
    fee_type = st.radio("Fee Structure", ["Per Share (Realistic)", "Percentage"], index=0)
    
    if fee_type == "Per Share (Realistic)":
        col_a, col_b = st.columns(2)
        fee_per_share = col_a.number_input("Per Share (¢)", min_value=0.0, value=0.07, step=0.01) / 100
        fee_minimum = col_b.number_input("Min Fee ($)", min_value=0.0, value=2.5, step=0.1)
        use_per_share_fee = True
        fee_perc = 0.001
    else:
        fee_perc = st.number_input("Fee (%)", min_value=0.0, value=0.1, step=0.01) / 100
        fee_per_share = 0.0007
        fee_minimum = 2.5
        use_per_share_fee = False
    
    slippage_perc = st.number_input("Slippage (%)", min_value=0.0, value=0.05, step=0.01) / 100
    
    # Position sizing
    st.markdown("### 📏 Position Sizing")
    position_mode = st.radio("Mode", ["Percentage of Portfolio", "Fixed Dollar Amount"], index=0)

    if position_mode == "Percentage of Portfolio":
        position_size = st.slider("Position Size (%)", 0, 100, 100, 1) / 100
        use_fixed_trade_value = False
        fixed_trade_value = 10000
    else:
        fixed_trade_value = st.number_input("Trade Size ($)", min_value=0, value=10000, step=100)
        use_fixed_trade_value = True
        position_size = 1.0
    
    max_position_pct = st.slider("Max Position (%)", 0, 100, 95, 1) / 100
    
    # Store config
    config = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "symbol": symbol.upper(),
        "initial_capital": initial_capital,
        "fee_perc": fee_perc,
        "fee_per_share": fee_per_share,
        "fee_minimum": fee_minimum,
        "use_per_share_fee": use_per_share_fee,
        "slippage_perc": slippage_perc,
        "use_fixed_trade_value": use_fixed_trade_value,
        "fixed_trade_value": fixed_trade_value,
        "position_size": position_size,
        "max_position_pct": max_position_pct
    }

# Main tabs
tabs = st.tabs([
    "🎯 Strategy Backtest",
    "⚖️ Compare Strategies", 
    "🤖 ML Models",
    "🧠 Deep Learning",
    "⚙️ Optimization",
    "💼 Portfolio",
    "🔧 Custom Strategy"
])


# Helper function for strategy parameter rendering
def render_strategy_params(strategy_name):
    """Render parameter inputs for a strategy."""
    params = {}
    strategy_info = STRATEGIES[strategy_name]
    
    if not strategy_info["params"]:
        return params
    
    st.subheader("Strategy Parameters")
    
    for param_name, param_info in strategy_info["params"].items():
        if param_info["type"] == "number":
            params[param_name] = st.number_input(
                param_name.replace("_", " ").title(),
                min_value=param_info.get("min", 1),
                max_value=param_info.get("max", 1000),
                value=param_info["default"],
                step=param_info.get("step", 1),
                help=param_info.get("help", "")
            )
        elif param_info["type"] == "checkbox":
            params[param_name] = st.checkbox(
                param_name.replace("_", " ").title(),
                value=param_info["default"],
                help=param_info.get("help", "")
            )
    
    return params


# Helper function to run backtest
def run_backtest(config, strategy_name, strategy_params):
    """Run a backtest with given configuration."""
    try:
        # Load data
        loader = YahooDataLoader(config["symbol"], config["start_date"], config["end_date"])
        
        # Create strategy
        strategy_class = STRATEGIES[strategy_name]["class"]
        if strategy_params:
            strategy = strategy_class(**strategy_params)
        else:
            strategy = strategy_class()
        
        # Override for Buy & Hold
        is_buy_hold = strategy_name == "📊 Buy & Hold (Baseline)"
        if is_buy_hold:
            use_fixed_trade_value = False
            position_size = 1.0
            max_position_pct = 1.0
        else:
            use_fixed_trade_value = config["use_fixed_trade_value"]
            position_size = config["position_size"]
            max_position_pct = config["max_position_pct"]
        
        # Create backtester
        backtester = Backtester(
            initial_capital=config["initial_capital"],
            fee_perc=config["fee_perc"],
            fee_per_share=config["fee_per_share"],
            fee_minimum=config["fee_minimum"],
            use_per_share_fee=config["use_per_share_fee"],
            slippage_perc=config["slippage_perc"],
            use_fixed_trade_value=use_fixed_trade_value,
            fixed_trade_value=config["fixed_trade_value"],
            position_size=position_size,
            max_position_pct=max_position_pct
        )
        
        # Run pipeline (includes data and signals in results)
        pipeline = TradingPipeline(
            data_loader=loader,
            strategy=strategy,
            backtester=backtester,
            verbose=False,
            save_results=False
        )

        results = pipeline.run()
        
        return results, None
        
    except Exception as e:
        return None, str(e)


# Helper function to plot results
def plot_equity_curve(equity_curve, title="Equity Curve"):
    """Plot equity curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Equity',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400
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
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    return fig


def plot_signals(data, signals, symbol):
    """Plot price with ALL trading signals (entries, exits, transitions)."""
    # Create subplots: price chart on top, signal timeline on bottom
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
            name='Exit (to Cash)',
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
        name='Signal Value',
        marker=dict(color=colors),
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
        template='plotly_white',
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


# TAB 1: Strategy Backtest
with tabs[0]:
    st.header("🎯 Single Strategy Backtest")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Strategy Selection")
        strategy_name = st.selectbox("Choose Strategy", list(STRATEGIES.keys()))
        
        st.markdown("---")
        strategy_params = render_strategy_params(strategy_name)
        
        if strategy_name == "📊 Buy & Hold (Baseline)":
            st.info(
                "📊 This strategy invests 100% of capital at start and holds until end. Position sizing settings are automatically overridden.")
    
    with col2:
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
        
        if run_button:
            with st.spinner("Running backtest..."):
                results, error = run_backtest(config, strategy_name, strategy_params)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.success("✅ Backtest completed!")
                    
                    metrics = results['metrics']
                    equity_curve = results['equity_curve']
                    
                    # Metrics
                    st.markdown("## 📊 Performance Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{metrics['total_return'] * 100:.2f}%")
                    with col2:
                        st.metric("Annual Return", f"{metrics['annual_return'] * 100:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                    with col4:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
                    
                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        st.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
                    with col6:
                        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    with col7:
                        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
                    with col8:
                        st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}")
                    
                    # Charts
                    st.markdown("## 📈 Performance Charts")
                    tab1, tab2, tab3 = st.tabs(["Equity Curve", "Drawdown", "Trading Signals"])
                    
                    with tab1:
                        fig = plot_equity_curve(equity_curve, f"{strategy_name} - Equity Curve")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        fig = plot_drawdown(equity_curve)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Use data and signals from pipeline results
                        data = results.get('data')
                        signals = results.get('signals')

                        if data is not None and signals is not None:
                            fig = plot_signals(data, signals, config['symbol'])
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ Trading signals data not available")
                    
                    # Trade log
                    if 'trades' in results and not results['trades'].empty:
                        with st.expander("📋 Trade Log"):
                            st.dataframe(results['trades'], use_container_width=True)

# TAB 2: Compare Strategies
with tabs[1]:
    st.header("⚖️ Compare Multiple Strategies")
    
    st.markdown("### Select Strategies to Compare")
    selected_strategies = st.multiselect(
        "Choose strategies",
        options=list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys())[:3]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        include_baseline = st.checkbox("Include Buy & Hold Baseline", value=True)
    with col2:
        allow_short_comparison = st.checkbox("Allow Short Positions", value=False)
    
    if st.button("🔄 Run Comparison", type="primary"):
        if not selected_strategies and not include_baseline:
            st.warning("Please select at least one strategy.")
        else:
            strategies_to_compare = list(selected_strategies)
            baseline_name = "📊 Buy & Hold (Baseline)"
            if include_baseline and baseline_name not in strategies_to_compare:
                strategies_to_compare.insert(0, baseline_name)
            
            comparison_results = {}
            progress_bar = st.progress(0)
            
            for i, strat_name in enumerate(strategies_to_compare):
                default_params = {k: v["default"] for k, v in STRATEGIES[strat_name]["params"].items()}
                if "allow_short" in default_params:
                    default_params["allow_short"] = allow_short_comparison
                
                results, error = run_backtest(config, strat_name, default_params)
                if not error:
                    comparison_results[strat_name] = results
                
                progress_bar.progress((i + 1) / len(strategies_to_compare))
            
            if comparison_results:
                st.markdown("### 📊 Performance Comparison")
                
                comparison_data = []
                for name, result in comparison_results.items():
                    metrics = result['metrics']
                    comparison_data.append({
                        'Strategy': name,
                        'Total Return (%)': metrics['total_return'] * 100,
                        'Annual Return (%)': metrics['annual_return'] * 100,
                        'Sharpe Ratio': metrics['sharpe_ratio'],
                        'Max DD (%)': metrics['max_drawdown'] * 100,
                        'Win Rate (%)': metrics['win_rate'] * 100
                    })
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(
                    df.style.format({
                        'Total Return (%)': '{:.2f}',
                        'Annual Return (%)': '{:.2f}',
                        'Sharpe Ratio': '{:.3f}',
                        'Max DD (%)': '{:.2f}',
                        'Win Rate (%)': '{:.1f}'
                    }).background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # Equity curves
                st.markdown("### 📈 Equity Curves")
                fig = go.Figure()
                for name, result in comparison_results.items():
                    equity = result['equity_curve']
                    normalized = (equity / equity.iloc[0]) * 100
                    fig.add_trace(go.Scatter(x=normalized.index, y=normalized.values, mode='lines', name=name))
                
                fig.update_layout(
                    title="Normalized Equity Curves (Base=100)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Equity",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

# TAB 3: ML Models
with tabs[2]:
    st.header("🤖 Machine Learning Models")
    st.markdown("Train classical ML models with full control over features, data splits, and labels.")
    
    try:
        from quant_framework.ml.features import FeatureEngineering
        from quant_framework.ml.classifiers import (RandomForestClassifier, XGBoostClassifier,
                                                    LightGBMClassifier, SVMClassifier, GradientBoostingClassifier)
        from quant_framework.ml.trainer import ModelTrainer
        from quant_framework.ml.preprocessing import DataPreprocessor
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Data & Features")

            st.info(
                "💡 **Tip**: Larger lookback periods (100, 200) create more features but lose more samples. For datasets <500 days, stick to [5, 10, 20] for best results.")

            # Feature configuration
            with st.expander("⚙️ Feature Configuration", expanded=True):
                include_technical = st.checkbox("Technical Indicators", value=True,
                                                help="SMA, EMA, RSI, MACD, BB, ATR, etc.")
                include_statistical = st.checkbox("Statistical Features", value=True,
                                                  help="Rolling stats, volatility, skewness")
                include_time = st.checkbox("Time Features", value=True, help="Day of week, month, quarter")
                include_lagged = st.checkbox("Lagged Features", value=True, help="Previous n-period values")

                lookback_periods = st.multiselect(
                    "Lookback Periods",
                    options=[5, 10, 14, 20, 50, 100, 200],
                    default=[5, 10, 20],
                    help="Periods for rolling calculations. Larger periods = fewer valid samples (e.g., 200-day MA loses first 200 rows)"
                )

            # Label configuration
            with st.expander("🎯 Label Definition", expanded=True):
                target_type = st.radio("Target Type", ["classification", "regression"])

                if target_type == "classification":
                    label_method = st.selectbox(
                        "Classification Method",
                        ["return_threshold", "volatility_adjusted", "future_high_low"]
                    )

                    if label_method == "return_threshold":
                        threshold = st.number_input(
                            "Return Threshold (%)",
                            min_value=-10.0, max_value=10.0, value=0.5, step=0.1,
                            help="Min return % for positive label"
                        ) / 100
                    elif label_method == "volatility_adjusted":
                        vol_mult = st.number_input(
                            "Volatility Multiplier",
                            min_value=0.1, max_value=5.0, value=1.0, step=0.1
                        )
                    else:
                        horizon = st.number_input(
                            "Future Horizon (days)",
                            min_value=1, max_value=30, value=5
                        )
                else:
                    forecast_horizon = st.number_input(
                        "Forecast Horizon (days)",
                        min_value=1, max_value=30, value=5
                    )

            # Train/test split
            with st.expander("📏 Train/Test Split", expanded=True):
                split_method = st.radio(
                    "Split Method",
                    ["holdout", "time_series_cv", "walk_forward"],
                    help="How to split data for validation"
                )

                if split_method == "holdout":
                    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
                    shuffle = st.checkbox("Shuffle", value=False, help="Not recommended for time series")
                elif split_method == "time_series_cv":
                    n_splits = st.slider("Number of Splits", 2, 10, 5, 1)
                else:  # walk_forward
                    train_window = st.number_input("Train Window (days)", 100, 1000, 252, 21)
                    test_window = st.number_input("Test Window (days)", 20, 200, 63, 21)
                    step_size = st.number_input("Step Size (days)", 5, 100, 21, 7)

            # Create features button
            create_features = st.button("🔧 Create Features & Labels", type="primary")

        with col2:
            if create_features:
                with st.spinner("Creating features and labels..."):
                    try:
                        # Load data
                    loader = YahooDataLoader(config["symbol"], config["start_date"], config["end_date"])
                    data = loader.get_data()
                    
                        initial_rows = len(data)
                        st.info(
                            f"📥 Loaded {initial_rows} rows of data from {config['start_date']} to {config['end_date']}")

                        # Feature engineering
                        selected_periods = lookback_periods if lookback_periods else [5, 10, 20]
                        max_lookback = max(selected_periods) if selected_periods else 20

                        fe = FeatureEngineering(
                            include_technical=include_technical,
                            include_statistical=include_statistical,
                            include_time=include_time,
                            include_lagged=include_lagged,
                            lookback_periods=selected_periods
                        )

                    features_df = fe.create_features(data)

                        # Create labels based on configuration
                        threshold_val = threshold if target_type == "classification" else 0.0
                        X, y = fe.create_training_data(features_df, target_type, threshold_val)

                        # Calculate sample loss
                        samples_lost = initial_rows - len(X)
                        loss_pct = (samples_lost / initial_rows) * 100

                        if loss_pct > 50:
                            st.warning(
                                f"⚠️ {loss_pct:.1f}% of samples lost due to NaN values (first {max_lookback}+ rows). Consider using smaller lookback periods for more samples.")
                        elif loss_pct > 30:
                            st.info(
                                f"ℹ️ {loss_pct:.1f}% of samples lost to NaN values. This is normal with rolling features.")
                        else:
                            st.success(f"✓ {loss_pct:.1f}% sample loss - good data retention!")

                        st.session_state['X_ml'] = X
                        st.session_state['y_ml'] = y
                        st.session_state['ml_data'] = data
                        st.session_state['ml_config'] = {
                            'split_method': split_method,
                            'test_size': test_size if split_method == "holdout" else None,
                            'n_splits': n_splits if split_method == "time_series_cv" else None,
                            'train_window': train_window if split_method == "walk_forward" else None,
                            'test_window': test_window if split_method == "walk_forward" else None,
                            'step_size': step_size if split_method == "walk_forward" else None
                        }

                        st.success(
                            f"✅ Created {len(X.columns)} features with {len(X)} valid samples ({initial_rows} → {len(X)} rows)")

                        # Show feature summary
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("📊 Features", len(X.columns))
                        with col_b:
                            st.metric("📈 Valid Samples", len(X))
                        with col_c:
                            st.metric("🗑️ Dropped Rows", samples_lost, delta=f"-{loss_pct:.1f}%", delta_color="inverse")
                        with col_d:
                            if target_type == "classification":
                                pos_pct = (y == 1).sum() / len(y) * 100
                                st.metric("🎯 Positive %", f"{pos_pct:.1f}%")
                            else:
                                st.metric("🎯 Target Mean", f"{y.mean():.4f}")

                        # Show sample
                        st.subheader("Sample Features")
                        st.dataframe(X.head(10), use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback

                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())

            elif 'X_ml' in st.session_state:
                st.subheader("🚀 Train Model")

                # Model selection
                model_type = st.selectbox(
                    "Model Type",
                    ["Random Forest", "XGBoost", "LightGBM", "SVM", "Gradient Boosting"]
                )

                # Model parameters
                with st.expander("🎛️ Model Parameters"):
                    if model_type in ["Random Forest", "Gradient Boosting"]:
                        n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
                        max_depth = st.slider("Max Depth", 2, 30, 10, 1)
                        min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1)
                        model_params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                                        'min_samples_split': min_samples_split}
                    elif model_type in ["XGBoost", "LightGBM"]:
                        n_estimators = st.slider("Number of Estimators", 10, 500, 100, 10)
                        max_depth = st.slider("Max Depth", 2, 15, 6, 1)
                        learning_rate = st.slider("Learning Rate", 0.001, 0.3, 0.1, 0.01)
                        model_params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                                        'learning_rate': learning_rate}
                    else:  # SVM
                        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
                        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                        model_params = {'C': C, 'kernel': kernel}

                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner(f"Training {model_type}..."):
                        try:
                            # Create model
                        if model_type == "Random Forest":
                                model = RandomForestClassifier(**model_params)
                            elif model_type == "XGBoost":
                                model = XGBoostClassifier(**model_params)
                            elif model_type == "LightGBM":
                                model = LightGBMClassifier(**model_params)
                            elif model_type == "SVM":
                                model = SVMClassifier(**model_params)
                        else:
                                model = GradientBoostingClassifier(**model_params)

                            # Train
                            preprocessor = DataPreprocessor(scaler_type='standard')
                            trainer = ModelTrainer(model, preprocessor)

                            # Get split config
                            ml_config = st.session_state['ml_config']
                            train_kwargs = {'validation_method': ml_config['split_method']}
                            if ml_config['split_method'] == 'holdout':
                                train_kwargs['test_size'] = ml_config['test_size']
                            elif ml_config['split_method'] == 'time_series_cv':
                                train_kwargs['n_splits'] = ml_config['n_splits']
                            else:
                                train_kwargs['train_size'] = ml_config['train_window']
                                train_kwargs['test_size'] = ml_config['test_window']
                                train_kwargs['step_size'] = ml_config['step_size']

                            results = trainer.train(st.session_state['X_ml'], st.session_state['y_ml'], **train_kwargs)

                            st.session_state['ml_model'] = model
                            st.session_state['ml_results'] = results
                        
                        st.success("✓ Training complete!")

                            # Show metrics
                            metrics = results.get('test_metrics', results.get('avg_test_metrics', {}))
                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                            with col_b:
                                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                            with col_c:
                                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                            with col_d:
                                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")

                            # Feature importance
                            if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
                                st.subheader("Top 20 Feature Importance")
                                importance_df = model.get_feature_importance(top_n=20)
                                fig = go.Figure(data=[go.Bar(
                                    x=importance_df['importance'],
                                    y=importance_df['feature'],
                                    orientation='h',
                                    marker=dict(color=importance_df['importance'], colorscale='Viridis')
                                )])
                                fig.update_layout(
                                    xaxis_title="Importance",
                                    yaxis_title="Feature",
                                    height=600,
                                    yaxis=dict(autorange='reversed')
                                )
                                st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Training error: {str(e)}")
                            import traceback

                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())
            else:
                st.info("👈 Configure features and labels in the left panel, then click 'Create Features & Labels'")

    except ImportError as e:
        st.warning("⚠️ ML features require additional packages. Install with: `pip install scikit-learn xgboost lightgbm`")
        st.code(str(e))

# TAB 4: Deep Learning (PyTorch)
with tabs[3]:
    st.header("🧠 Deep Learning Models (PyTorch)")
    st.markdown("Train neural networks with PyTorch for time series prediction.")

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🔧 Model Configuration")

            # Reuse ML features if available
            if 'X_ml' in st.session_state:
                st.info("✓ Using features from ML tab")
                use_ml_features = True
            else:
                use_ml_features = False
                st.warning("Create features in ML tab first, or configure below")

            # Architecture
            with st.expander("🏗️ Model Architecture", expanded=True):
                model_arch = st.selectbox(
                    "Architecture",
                    ["LSTM", "GRU", "Transformer", "CNN", "MLP"]
                )

                sequence_length = st.slider("Sequence Length", 5, 60, 20, 5, help="Time steps to look back")

                if model_arch in ["LSTM", "GRU"]:
                    hidden_size = st.slider("Hidden Size", 16, 256, 64, 8)
                    num_layers = st.slider("Number of Layers", 1, 5, 2, 1)
                    dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
                    bidirectional = st.checkbox("Bidirectional", value=False)
                elif model_arch == "Transformer":
                    d_model = st.slider("Model Dimension", 32, 256, 64, 8)
                    nhead = st.selectbox("Number of Heads", [2, 4, 8], index=1)
                    num_layers = st.slider("Number of Layers", 1, 6, 2, 1)
                    dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
                elif model_arch == "CNN":
                    num_filters = st.slider("Number of Filters", 16, 128, 64, 8)
                    kernel_size = st.slider("Kernel Size", 2, 7, 3, 1)
                    num_conv_layers = st.slider("Conv Layers", 1, 5, 2, 1)
                    dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
                else:  # MLP
                    hidden_sizes = st.text_input("Hidden Layers (comma-separated)", "128,64,32")
                    dropout = st.slider("Dropout", 0.0, 0.5, 0.3, 0.05)

            # Training config
            with st.expander("⚙️ Training Configuration", expanded=True):
                batch_size = st.slider("Batch Size", 16, 256, 32, 16)
                epochs = st.slider("Epochs", 10, 300, 50, 10)
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                    value=0.001
                )
                optimizer_type = st.selectbox("Optimizer", ["Adam", "SGD", "AdamW", "RMSprop"])
                weight_decay = st.number_input("Weight Decay (L2)", 0.0, 0.01, 0.0001, 0.0001, format="%.5f")
                early_stopping = st.checkbox("Early Stopping", value=True)
                if early_stopping:
                    patience = st.slider("Patience", 5, 50, 10, 5)

                val_split = st.slider("Validation Split (%)", 10, 30, 20, 5) / 100

            train_dl_button = st.button("🚀 Train Deep Learning Model", type="primary")

        with col2:
            if train_dl_button:
                with st.spinner(f"Training {model_arch} model..."):
                    try:
                        if not use_ml_features:
                            st.error("Please create features in the ML Models tab first!")
                        else:
                            # Get data and ensure numeric types
                            X_df = st.session_state['X_ml'].copy()
                            y_series = st.session_state['y_ml'].copy()

                            # Convert all columns to numeric, drop any that can't be converted
                            numeric_cols = []
                            for col in X_df.columns:
                                try:
                                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
                                    if X_df[col].notna().sum() > 0:  # Keep if has valid values
                                        numeric_cols.append(col)
                                except:
                                    pass  # Skip columns that can't be converted

                            X_df = X_df[numeric_cols]

                            # Drop rows with NaN
                            valid_idx = X_df.notna().all(axis=1) & y_series.notna()
                            X_df = X_df[valid_idx]
                            y_series = y_series[valid_idx]

                            if len(numeric_cols) < len(st.session_state['X_ml'].columns):
                                dropped = len(st.session_state['X_ml'].columns) - len(numeric_cols)
                                st.info(
                                    f"ℹ️ Dropped {dropped} non-numeric feature(s). Using {len(numeric_cols)} numeric features.")

                            X = X_df.values.astype(np.float32)
                            y = y_series.values.astype(np.int64)

                            st.info(f"📊 Prepared {X.shape[0]} samples with {X.shape[1]} features (dtype: {X.dtype})")


                            # Create sequences
                            def create_sequences(X, y, seq_length):
                                X_seq, y_seq = [], []
                                for i in range(len(X) - seq_length):
                                    X_seq.append(X[i:i + seq_length])
                                    y_seq.append(y[i + seq_length])
                                return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


                            X_seq, y_seq = create_sequences(X, y, sequence_length)

                            if len(X_seq) < 50:
                                st.error(
                                    f"g Not enough sequences created ({len(X_seq)}). Need at least 50. Try:\n- Using a smaller sequence length\n- Creating more ML features (go to ML tab)")
                                st.stop()

                            st.success(f"✅ Created {len(X_seq)} sequences (shape: {X_seq.shape})")

                            # Train/val split
                            split_idx = int(len(X_seq) * (1 - val_split))
                            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
                            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

                            st.info(f"📊 Split: {len(X_train)} train samples, {len(X_val)} validation samples")

                            # Convert to PyTorch tensors
                            X_train_t = torch.FloatTensor(X_train)
                            y_train_t = torch.LongTensor(y_train)
                            X_val_t = torch.FloatTensor(X_val)
                            y_val_t = torch.LongTensor(y_val)


                            # DataLoaders
                            class TimeSeriesDataset(Dataset):
                                def __init__(self, X, y):
                                    self.X = X
                                    self.y = y

                                def __len__(self):
                                    return len(self.X)

                                def __getitem__(self, idx):
                                    return self.X[idx], self.y[idx]


                            train_dataset = TimeSeriesDataset(X_train_t, y_train_t)
                            val_dataset = TimeSeriesDataset(X_val_t, y_val_t)
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                            # Build model
                            input_size = X_train.shape[2]
                            num_classes = len(np.unique(y))


                            class TimeSeriesModel(nn.Module):
                                def __init__(self):
                                    super(TimeSeriesModel, self).__init__()

                                    if model_arch == "LSTM":
                                        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                                                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                                           bidirectional=bidirectional)
                                        rnn_output_size = hidden_size * (2 if bidirectional else 1)
                                    elif model_arch == "GRU":
                                        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                                                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                                          bidirectional=bidirectional)
                                        rnn_output_size = hidden_size * (2 if bidirectional else 1)
                                    elif model_arch == "Transformer":
                                        encoder_layer = nn.TransformerEncoderLayer(
                                            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
                                        )
                                        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                                        self.embedding = nn.Linear(input_size, d_model)
                                        rnn_output_size = d_model
                                    elif model_arch == "CNN":
                                        layers = []
                                        in_channels = input_size
                                        for _ in range(num_conv_layers):
                                            layers.extend([
                                                nn.Conv1d(in_channels, num_filters, kernel_size, padding='same'),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(num_filters),
                                                nn.Dropout(dropout)
                                            ])
                                            in_channels = num_filters
                                        self.conv_layers = nn.Sequential(*layers)
                                        self.pool = nn.AdaptiveAvgPool1d(1)
                                        rnn_output_size = num_filters
                                    else:  # MLP
                                        layers = []
                                        sizes = [input_size * sequence_length] + [int(x) for x in
                                                                                  hidden_sizes.split(',')]
                                        for i in range(len(sizes) - 1):
                                            layers.extend([
                                                nn.Linear(sizes[i], sizes[i + 1]),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(sizes[i + 1]),
                                                nn.Dropout(dropout)
                                            ])
                                        self.mlp_layers = nn.Sequential(*layers)
                                        rnn_output_size = sizes[-1]

                                    self.fc = nn.Linear(rnn_output_size, num_classes)
                                    self.dropout = nn.Dropout(dropout)

                                def forward(self, x):
                                    if model_arch in ["LSTM", "GRU"]:
                                        out, _ = self.rnn(x)
                                        out = out[:, -1, :]
                                    elif model_arch == "Transformer":
                                        x = self.embedding(x)
                                        out = self.transformer(x)
                                        out = out.mean(dim=1)
                                    elif model_arch == "CNN":
                                        x = x.transpose(1, 2)
                                        out = self.conv_layers(x)
                                        out = self.pool(out).squeeze(-1)
                                    else:  # MLP
                                        out = x.reshape(x.size(0), -1)
                                        out = self.mlp_layers(out)

                                    out = self.dropout(out)
                                    out = self.fc(out)
                                    return out


                            model = TimeSeriesModel()
                            criterion = nn.CrossEntropyLoss()

                            if optimizer_type == "Adam":
                                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                            elif optimizer_type == "SGD":
                                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                                                      weight_decay=weight_decay)
                            elif optimizer_type == "AdamW":
                                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                            else:  # RMSprop
                                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate,
                                                          weight_decay=weight_decay)

                            # Training loop
                            train_losses, val_losses, val_accs = [], [], []
                            best_val_loss = float('inf')
                            patience_counter = 0

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for epoch in range(epochs):
                                # Training
                                model.train()
                                train_loss = 0
                                for X_batch, y_batch in train_loader:
                                    optimizer.zero_grad()
                                    outputs = model(X_batch)
                                    loss = criterion(outputs, y_batch)
                                    loss.backward()
                                    optimizer.step()
                                    train_loss += loss.item()

                                train_loss /= len(train_loader)
                                train_losses.append(train_loss)

                                # Validation
                                model.eval()
                                val_loss = 0
                                correct = 0
                                total = 0
                                with torch.no_grad():
                                    for X_batch, y_batch in val_loader:
                                        outputs = model(X_batch)
                                        loss = criterion(outputs, y_batch)
                                        val_loss += loss.item()
                                        _, predicted = torch.max(outputs.data, 1)
                                        total += y_batch.size(0)
                                        correct += (predicted == y_batch).sum().item()

                                val_loss /= len(val_loader)
                                val_acc = 100 * correct / total
                                val_losses.append(val_loss)
                                val_accs.append(val_acc)

                                # Update progress
                                progress_bar.progress((epoch + 1) / epochs)
                                status_text.text(
                                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                                # Early stopping
                                if early_stopping:
                                    if val_loss < best_val_loss:
                                        best_val_loss = val_loss
                                        patience_counter = 0
                                        best_model_state = model.state_dict()
                                    else:
                                        patience_counter += 1
                                        if patience_counter >= patience:
                                            st.info(f"Early stopping at epoch {epoch + 1}")
                                            model.load_state_dict(best_model_state)
                                            break

                            st.success("✓ Training complete!")

                            # Store model
                            st.session_state['dl_model'] = model
                            st.session_state['dl_history'] = {
                                'train_loss': train_losses,
                                'val_loss': val_losses,
                                'val_acc': val_accs
                            }

                            # Show final metrics
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Final Train Loss", f"{train_losses[-1]:.4f}")
                            with col_b:
                                st.metric("Final Val Loss", f"{val_losses[-1]:.4f}")
                            with col_c:
                                st.metric("Final Val Accuracy", f"{val_accs[-1]:.2f}%")

                            # Plot training history
                            st.subheader("Training History")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))
                            fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Val Loss'))
                            fig.update_layout(
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Plot accuracy
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(y=val_accs, mode='lines', name='Val Accuracy'))
                            fig2.update_layout(
                                xaxis_title="Epoch",
                                yaxis_title="Accuracy (%)",
                                height=400
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
                        import traceback

                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
            elif 'dl_model' in st.session_state:
                st.info("✓ Model trained! Train again to update or use the Compare tab.")
                history = st.session_state.get('dl_history', {})
                if history:
                    final_acc = history['val_acc'][-1] if history.get('val_acc') else 0
                    st.metric("Final Validation Accuracy", f"{final_acc:.2f}%")
            else:
                st.info(
                    "👈 Configure model architecture and training parameters, then click 'Train Deep Learning Model'")

    except ImportError as e:
        st.warning("⚠️ Deep Learning requires PyTorch. Install with: `pip install torch torchvision`")
        st.code(str(e))

# TAB 5: Optimization
with tabs[4]:
    st.header("⚙️ Strategy Optimization")
    st.info("Optimize strategy parameters using grid search, random search, or walk-forward analysis.")
    st.warning("🚧 Feature coming soon!")

# TAB 6: Portfolio
with tabs[5]:
    st.header("💼 Portfolio Optimization")
    st.info("Multi-asset portfolio optimization with efficient frontier calculation.")
    st.warning("🚧 Feature coming soon!")

# TAB 7: Custom Strategy
with tabs[6]:
    st.header("🔧 Custom Strategy Builder")
    st.info("Build your own custom trading strategy with Python code.")
    st.warning("🚧 Feature coming soon!")

# Footer
st.markdown("---")
