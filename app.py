"""
Streamlit Quantitative Trading Platform - Main Application
Modular architecture with separate tab modules for better maintainability.
"""

# Add project root to Python path for cloud deployments
import sys
from pathlib import Path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Quant Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import strategy definitions
from quant_framework.models import (
    BuyHoldStrategy, MomentumStrategy, MeanReversionStrategy,
    RSIStrategy, MACDStrategy, BreakoutStrategy, TurtleStrategy,
    TripleMAStrategy, StochasticStrategy, FibonacciStrategy
)

# Import tab modules
from app_tabs import (
    render_strategy_backtest_tab,
    render_compare_strategies_tab,
    render_ml_models_tab,
    render_optimization_tab,
    render_portfolio_tab,
    render_custom_strategy_tab
)

# Title
st.title("📈 Quantitative Trading Platform")
st.markdown("*Complete platform for strategy backtesting, ML/DL training, and portfolio optimization*")

# Strategy definitions
STRATEGIES = {
    "📊 Buy & Hold (Baseline)": {
        "class": BuyHoldStrategy,
        "params": {}
    },
    "📈 Momentum (MA Crossover)": {
        "class": MomentumStrategy,
        "params": {
            "short_window": {"type": "number", "default": 20, "min": 5, "max": 100},
            "long_window": {"type": "number", "default": 50, "min": 20, "max": 200},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "📉 Mean Reversion": {
        "class": MeanReversionStrategy,
        "params": {
            "window": {"type": "number", "default": 20, "min": 10, "max": 100},
            "num_std": {"type": "number", "default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "🎯 RSI": {
        "class": RSIStrategy,
        "params": {
            "rsi_window": {"type": "number", "default": 14, "min": 5, "max": 30},
            "oversold_threshold": {"type": "number", "default": 30, "min": 20, "max": 40},
            "overbought_threshold": {"type": "number", "default": 70, "min": 60, "max": 80},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "💫 MACD": {
        "class": MACDStrategy,
        "params": {
            "fast_period": {"type": "number", "default": 12, "min": 5, "max": 20},
            "slow_period": {"type": "number", "default": 26, "min": 20, "max": 50},
            "signal_period": {"type": "number", "default": 9, "min": 5, "max": 15},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "🚀 Breakout": {
        "class": BreakoutStrategy,
        "params": {
            "lookback_period": {"type": "number", "default": 20, "min": 10, "max": 100},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "🐢 Turtle Trading": {
        "class": TurtleStrategy,
        "params": {
            "entry_period": {"type": "number", "default": 20, "min": 10, "max": 50},
            "exit_period": {"type": "number", "default": 10, "min": 5, "max": 30},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "🎲 Triple Moving Average": {
        "class": TripleMAStrategy,
        "params": {
            "fast_period": {"type": "number", "default": 10, "min": 5, "max": 30},
            "medium_period": {"type": "number", "default": 20, "min": 15, "max": 50},
            "slow_period": {"type": "number", "default": 50, "min": 30, "max": 100},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "📊 Stochastic Oscillator": {
        "class": StochasticStrategy,
        "params": {
            "window": {"type": "number", "default": 14, "min": 5, "max": 30},
            "oversold": {"type": "number", "default": 20, "min": 10, "max": 30},
            "overbought": {"type": "number", "default": 80, "min": 70, "max": 90},
            "allow_short": {"type": "checkbox", "default": False}
        }
    },
    "🌀 Fibonacci Retracement": {
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
        with col_a:
            fee_per_share = st.number_input("Fee/Share ($)", value=0.005, step=0.001, format="%.4f")
        with col_b:
            fee_minimum = st.number_input("Min Fee ($)", value=2.5, step=0.5)
        use_per_share_fee = True
        fee_percentage = 0.001
    else:
        fee_percentage = st.number_input("Fee (%)", value=0.1, step=0.01) / 100
        fee_per_share = 0.005
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

    max_position_pct = st.slider("Max Position (%)", 0, 100, 100, 1) / 100

    # Performance settings
    st.markdown("### ⚡ Performance")
    use_fast_backtester = st.checkbox(
        "Fast Mode (Recommended)",
        value=True,
        help="Use vectorized backtester for 10-50x faster execution. Disable for maximum accuracy."
    )

    # Store config
    config = {
        "start_date": start_date,
        "end_date": end_date,
        "symbol": symbol,
        "initial_capital": initial_capital,
        "fee_per_share": fee_per_share,
        "fee_percentage": fee_percentage,
        "fee_minimum": fee_minimum,
        "use_per_share_fee": use_per_share_fee,
        "slippage_perc": slippage_perc,
        "position_size": position_size,
        "use_fixed_trade_value": use_fixed_trade_value,
        "fixed_trade_value": fixed_trade_value,
        "max_position_pct": max_position_pct,
        "use_fast_backtester": use_fast_backtester
    }

# Main tabs
tabs = st.tabs([
    "🎯 Strategy Backtest",
    "🔧 Custom Strategy",
    "⚖️ Compare Strategies",
    "📊 Features & Labels",
    "🚀 Train Models",
    "⚙️ Optimization",
    "💼 Portfolio"
])

# Render each tab
with tabs[0]:
    render_strategy_backtest_tab(config, STRATEGIES)

with tabs[1]:
    render_custom_strategy_tab(config, STRATEGIES)

with tabs[2]:
    render_compare_strategies_tab(config, STRATEGIES)

with tabs[3]:
    st.header("📊 Features & Labels")
    st.markdown("Create features for ML/DL training")
    try:
        from app_tabs.ml_models import _render_feature_creation_tab
        _render_feature_creation_tab(config)
    except ImportError as e:
        st.warning("⚠️ Requires: `pip install scikit-learn xgboost lightgbm`")
        st.code(str(e))

with tabs[4]:
    st.header("🚀 Train Models")
    st.markdown("Train classical ML and deep learning models")
    try:
        from app_tabs.ml_models import _render_training_tab
        _render_training_tab(config)
    except ImportError as e:
        st.warning("⚠️ Requires: `pip install scikit-learn xgboost lightgbm torch`")
        st.code(str(e))

with tabs[5]:
    render_optimization_tab(config)

with tabs[6]:
    render_portfolio_tab(config)

