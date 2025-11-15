"""
Tab 2: Custom Strategy Builder
Build and test custom strategies with visual condition builder.
"""

import streamlit as st
import pandas as pd
import json
from quant_framework.models.custom_strategy import CustomStrategy
from .helpers import run_backtest, plot_equity_curve, plot_drawdown, plot_signals


# Available indicators and their configurations
INDICATOR_TYPES = {
    "SMA": {"name": "Simple Moving Average", "params": ["period"]},
    "EMA": {"name": "Exponential Moving Average", "params": ["period"]},
    "RSI": {"name": "RSI", "params": ["period"]},
    "MACD": {"name": "MACD", "params": ["fast", "slow", "signal"]},
    "BB": {"name": "Bollinger Bands", "params": ["period", "std"]},
    "ATR": {"name": "Average True Range", "params": ["period"]},
    "Stochastic": {"name": "Stochastic Oscillator", "params": ["period"]},
    "Volume_MA": {"name": "Volume Moving Average", "params": ["period"]}
}

OPERATORS = {
    ">": "Greater than",
    ">=": "Greater than or equal",
    "<": "Less than",
    "<=": "Less than or equal",
    "==": "Equal to",
    "crosses_above": "Crosses above",
    "crosses_below": "Crosses below"
}


def render_custom_strategy_tab(config, strategies_dict):
    """Render the Custom Strategy Builder tab."""
    
    # Initialize session state for custom strategies
    if 'custom_strategies' not in st.session_state:
        st.session_state['custom_strategies'] = {}
    
    st.header("🔧 Custom Strategy Builder")
    st.markdown("*Build your own trading strategy with custom indicators and conditions*")
    
    # Quick start templates
    with st.expander("🚀 Quick Start Templates", expanded=False):
        st.markdown("**Load a pre-configured template to get started quickly:**")
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            if st.button("📈 MA Crossover + RSI", use_container_width=True):
                _load_template_ma_rsi()
                st.rerun()
        
        with col_t2:
            if st.button("💫 MACD + BB Breakout", use_container_width=True):
                _load_template_macd_bb()
                st.rerun()
        
        with col_t3:
            if st.button("🎯 Multi-Indicator Combo", use_container_width=True):
                _load_template_multi()
                st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📋 Strategy Configuration")
        
        # Strategy name
        strategy_name = st.text_input("Strategy Name", value="My Custom Strategy", key="custom_strat_name")
        
        # Allow short
        allow_short = st.checkbox("Allow Short Positions", value=False, key="custom_allow_short")
        
        st.markdown("---")
        
        # INDICATORS SECTION
        st.subheader("📊 Indicators")
        st.caption("Add technical indicators to use in your strategy")
        
        if 'indicators' not in st.session_state:
            st.session_state['indicators'] = []
        if 'configuring_indicator' not in st.session_state:
            st.session_state['configuring_indicator'] = None
        
        # Display current indicators
        if st.session_state['indicators']:
            st.markdown("**Active Indicators:**")
            for i, ind in enumerate(st.session_state['indicators']):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    ind_label = f"{ind['type']}"
                    if 'period' in ind:
                        ind_label += f"({ind['period']})"
                    elif 'fast' in ind:
                        ind_label += f"({ind['fast']},{ind['slow']},{ind['signal']})"
                    st.text(f"• {ind_label}")
                with col_b:
                    if st.button("🗑️", key=f"del_ind_{i}"):
                        st.session_state['indicators'].pop(i)
                        st.rerun()
            st.markdown("---")
        
        # Show configuration UI if in configuration mode
        if st.session_state['configuring_indicator'] is not None:
            indicator_type = st.session_state['configuring_indicator']
            with st.container():
                st.markdown(f"**⚙️ Configure {INDICATOR_TYPES[indicator_type]['name']}**")
                params = {}
                
                for param in INDICATOR_TYPES[indicator_type]['params']:
                    if param == 'period':
                        params[param] = st.number_input("Period", value=14, min_value=1, max_value=200, key="config_period")
                    elif param == 'fast':
                        params[param] = st.number_input("Fast Period", value=12, min_value=1, max_value=200, key="config_fast")
                    elif param == 'slow':
                        params[param] = st.number_input("Slow Period", value=26, min_value=1, max_value=200, key="config_slow")
                    elif param == 'signal':
                        params[param] = st.number_input("Signal Period", value=9, min_value=1, max_value=200, key="config_signal")
                    elif param == 'std':
                        params[param] = st.number_input("Standard Deviations", value=2.0, min_value=0.5, max_value=5.0, step=0.1, key="config_std")
                
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ Add", key="confirm_ind", use_container_width=True):
                        ind_config = {"type": indicator_type, **params}
                        st.session_state['indicators'].append(ind_config)
                        st.session_state['configuring_indicator'] = None
                        st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel", key="cancel_ind", use_container_width=True):
                        st.session_state['configuring_indicator'] = None
                        st.rerun()
        else:
            # Add indicator selection
            indicator_type = st.selectbox(
                "Select Indicator Type",
                [""] + list(INDICATOR_TYPES.keys()),
                format_func=lambda x: "Choose an indicator..." if x == "" else f"{INDICATOR_TYPES[x]['name']}",
                key="ind_select"
            )
            
            if indicator_type and st.button("➕ Configure & Add", use_container_width=True, key="add_ind_btn"):
                st.session_state['configuring_indicator'] = indicator_type
                st.rerun()
        
        st.markdown("---")
        
        # ENTRY RULES SECTION
        st.subheader("🟢 Entry Rules")
        st.caption("Define when to enter a position")
        
        if 'entry_rules' not in st.session_state:
            st.session_state['entry_rules'] = []
        
        entry_logic = st.radio("Entry Logic", ["AND", "OR"], index=0, horizontal=True, key="entry_logic")
        
        if st.button("➕ Add Entry Condition", use_container_width=True):
            st.session_state['entry_rules'].append({
                "indicator": "Close",
                "operator": ">",
                "value_type": "number",
                "value": 100,
                "compare_indicator": "Close"
            })
            st.rerun()
        
        # Display entry rules
        for i, rule in enumerate(st.session_state['entry_rules']):
            with st.expander(f"Entry Condition {i+1}", expanded=True):
                _render_condition_editor(i, rule, "entry")
        
        st.markdown("---")
        
        # EXIT RULES SECTION
        st.subheader("🔴 Exit Rules")
        st.caption("Define when to exit a position")
        
        if 'exit_rules' not in st.session_state:
            st.session_state['exit_rules'] = []
        
        exit_logic = st.radio("Exit Logic", ["AND", "OR"], index=0, horizontal=True, key="exit_logic")
        
        if st.button("➕ Add Exit Condition", use_container_width=True):
            st.session_state['exit_rules'].append({
                "indicator": "Close",
                "operator": "<",
                "value_type": "number",
                "value": 100,
                "compare_indicator": "Close"
            })
            st.rerun()
        
        # Display exit rules
        for i, rule in enumerate(st.session_state['exit_rules']):
            with st.expander(f"Exit Condition {i+1}", expanded=True):
                _render_condition_editor(i, rule, "exit")
        
        st.markdown("---")
        
        # SAVE STRATEGY
        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 Save Strategy", use_container_width=True, type="primary"):
                if not st.session_state['indicators']:
                    st.warning("⚠️ Add at least one indicator")
                elif not st.session_state['entry_rules']:
                    st.warning("⚠️ Add at least one entry condition")
                elif not st.session_state['exit_rules']:
                    st.warning("⚠️ Add at least one exit condition")
                else:
                    # Save to cache
                    st.session_state['custom_strategies'][strategy_name] = {
                        "name": strategy_name,
                        "indicators": st.session_state['indicators'].copy(),
                        "entry_rules": st.session_state['entry_rules'].copy(),
                        "exit_rules": st.session_state['exit_rules'].copy(),
                        "entry_logic": entry_logic,
                        "exit_logic": exit_logic,
                        "allow_short": allow_short
                    }
                    st.success(f"✅ Strategy '{strategy_name}' saved!")
        
        with col_clear:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state['configuring_indicator'] = None
                st.session_state['indicators'] = []
                st.session_state['entry_rules'] = []
                st.session_state['exit_rules'] = []
                st.rerun()
    
    # RIGHT PANEL - Backtest and Results
    with col2:
        st.subheader("📊 Backtest & Results")
        
        # Show saved strategies
        if st.session_state['custom_strategies']:
            st.info(f"💾 **Saved Strategies:** {len(st.session_state['custom_strategies'])}")
            saved_strat_names = list(st.session_state['custom_strategies'].keys())
            
            selected_saved = st.selectbox(
                "Load Saved Strategy",
                [""] + saved_strat_names,
                format_func=lambda x: "Select a strategy..." if x == "" else x,
                key="load_saved"
            )
            
            col_load, col_del = st.columns([3, 1])
            with col_load:
                if selected_saved and st.button("📂 Load", use_container_width=True):
                    loaded = st.session_state['custom_strategies'][selected_saved]
                    st.session_state['configuring_indicator'] = None
                    st.session_state['indicators'] = loaded['indicators'].copy()
                    st.session_state['entry_rules'] = loaded['entry_rules'].copy()
                    st.session_state['exit_rules'] = loaded['exit_rules'].copy()
                    st.success(f"✅ Loaded '{selected_saved}'")
                    st.rerun()
            with col_del:
                if selected_saved and st.button("🗑️", use_container_width=True):
                    del st.session_state['custom_strategies'][selected_saved]
                    st.success(f"🗑️ Deleted '{selected_saved}'")
                    st.rerun()
            
            st.markdown("---")
        
        # Run backtest button
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True, key="run_custom_bt")
        
        if run_button:
            if not st.session_state.get('indicators'):
                st.error("❌ Please add at least one indicator")
            elif not st.session_state.get('entry_rules'):
                st.error("❌ Please add at least one entry condition")
            elif not st.session_state.get('exit_rules'):
                st.error("❌ Please add at least one exit condition")
            else:
                with st.spinner("Running backtest..."):
                    # Create custom strategy instance
                    strategy = CustomStrategy(
                        name=strategy_name,
                        indicators=st.session_state['indicators'],
                        entry_rules=st.session_state['entry_rules'],
                        exit_rules=st.session_state['exit_rules'],
                        entry_logic=entry_logic,
                        exit_logic=exit_logic,
                        allow_short=allow_short
                    )
                    
                    # Run backtest using helper
                    results, error = _run_custom_backtest(config, strategy)
                    
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
                            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                        with col4:
                            st.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
                        
                        col5, col6, col7, col8 = st.columns(4)
                        with col5:
                            st.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
                        with col6:
                            st.metric("Total Trades", f"{metrics['num_trades']}")
                        with col7:
                            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                        with col8:
                            st.metric("Volatility", f"{metrics['annual_volatility'] * 100:.2f}%")
                        
                        # Charts
                        st.markdown("---")
                        
                        tab1, tab2, tab3 = st.tabs(["💰 Equity Curve", "📉 Drawdown", "📊 Trading Signals"])
                        
                        with tab1:
                            fig = plot_equity_curve(equity_curve)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            fig = plot_drawdown(equity_curve)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab3:
                            if 'data' in results and 'signals' in results:
                                fig = plot_signals(results['data'], results['signals'], strategy_name)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Signal data not available")
                        
                        # Trade details
                        if 'trades' in results and len(results['trades']) > 0:
                            st.markdown("---")
                            st.markdown("### 📝 Trade History")
                            trades_df = pd.DataFrame(results['trades'])
                            if not trades_df.empty:
                                st.dataframe(trades_df.tail(20), use_container_width=True)


def _render_condition_editor(index: int, rule: dict, rule_type: str):
    """Render a condition editor."""
    # Get available indicators for dropdown
    base_indicators = ["Close", "Open", "High", "Low", "Volume"]
    custom_indicators = []
    
    # Get indicators from session state
    indicators_list = st.session_state.get('indicators', [])
    
    for ind in indicators_list:
        ind_type = ind['type']
        if ind_type == 'SMA':
            custom_indicators.append(f"SMA_{ind['period']}")
        elif ind_type == 'EMA':
            custom_indicators.append(f"EMA_{ind['period']}")
        elif ind_type == 'RSI':
            custom_indicators.append(f"RSI_{ind['period']}")
        elif ind_type == 'MACD':
            custom_indicators.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        elif ind_type == 'BB':
            custom_indicators.extend(['BB_Upper', 'BB_Middle', 'BB_Lower'])
        elif ind_type == 'ATR':
            custom_indicators.append(f"ATR_{ind['period']}")
        elif ind_type == 'Stochastic':
            custom_indicators.extend(['Stoch_%K', 'Stoch_%D'])
        elif ind_type == 'Volume_MA':
            custom_indicators.append(f"Volume_MA_{ind['period']}")
    
    all_indicators = base_indicators + custom_indicators
    
    # Show info if no custom indicators
    if not custom_indicators:
        st.info("ℹ️ Add indicators above to use them in conditions")
    
    # Indicator selection
    indicator = st.selectbox(
        "Indicator",
        all_indicators,
        index=all_indicators.index(rule['indicator']) if rule['indicator'] in all_indicators else 0,
        key=f"{rule_type}_ind_{index}"
    )
    rule['indicator'] = indicator
    
    # Operator selection
    operator = st.selectbox(
        "Operator",
        list(OPERATORS.keys()),
        format_func=lambda x: OPERATORS[x],
        index=list(OPERATORS.keys()).index(rule['operator']) if rule['operator'] in OPERATORS else 0,
        key=f"{rule_type}_op_{index}"
    )
    rule['operator'] = operator
    
    # Value type
    value_type = st.radio(
        "Compare with",
        ["number", "indicator"],
        format_func=lambda x: "Number" if x == "number" else "Another Indicator",
        index=0 if rule['value_type'] == "number" else 1,
        horizontal=True,
        key=f"{rule_type}_vtype_{index}"
    )
    rule['value_type'] = value_type
    
    if value_type == "number":
        value = st.number_input(
            "Value",
            value=float(rule['value']),
            step=0.1,
            format="%.2f",
            key=f"{rule_type}_val_{index}"
        )
        rule['value'] = value
    else:
        compare_indicator = st.selectbox(
            "Compare Indicator",
            all_indicators,
            index=all_indicators.index(rule['compare_indicator']) if rule['compare_indicator'] in all_indicators else 0,
            key=f"{rule_type}_comp_{index}"
        )
        rule['compare_indicator'] = compare_indicator
    
    # Delete button
    if st.button(f"🗑️ Delete Condition", key=f"{rule_type}_del_{index}", use_container_width=True):
        if rule_type == "entry":
            st.session_state['entry_rules'].pop(index)
        else:
            st.session_state['exit_rules'].pop(index)
        st.rerun()


def _run_custom_backtest(config, strategy):
    """Run backtest for custom strategy."""
    from quant_framework.data.loaders import YahooDataLoader
    from quant_framework.backtest.backtester import Backtester
    from quant_framework.backtest.fast_backtester import FastBacktester
    
    try:
        # Load data
        loader = YahooDataLoader()
        data = loader.load(
            config['symbol'],
            config['start_date'].strftime('%Y-%m-%d'),
            config['end_date'].strftime('%Y-%m-%d')
        )
        
        if data is None or data.empty:
            return None, "Failed to load data"
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Choose backtester
        if config.get('use_fast_backtester', True):
            backtester = FastBacktester(
                initial_capital=config['initial_capital'],
                fee_per_share=config['fee_per_share'],
                fee_minimum=config['fee_minimum'],
                use_per_share_fee=config['use_per_share_fee'],
                slippage_perc=config['slippage_perc'],
                position_size=config['position_size'],
                use_fixed_trade_value=config['use_fixed_trade_value'],
                fixed_trade_value=config['fixed_trade_value'],
                max_position_pct=config['max_position_pct']
            )
        else:
            backtester = Backtester(
                initial_capital=config['initial_capital'],
                fee_per_share=config['fee_per_share'],
                fee_minimum=config['fee_minimum'],
                use_per_share_fee=config['use_per_share_fee'],
                slippage_perc=config['slippage_perc'],
                position_size=config['position_size'],
                use_fixed_trade_value=config['use_fixed_trade_value'],
                fixed_trade_value=config['fixed_trade_value'],
                max_position_pct=config['max_position_pct']
            )
        
        results = backtester.run(data, signals)
        results['data'] = data
        results['signals'] = signals
        
        return results, None
    
    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"


# Template loading functions
def _load_template_ma_rsi():
    """Load MA Crossover + RSI template."""
    st.session_state['configuring_indicator'] = None
    st.session_state['indicators'] = [
        {"type": "SMA", "period": 20},
        {"type": "SMA", "period": 50},
        {"type": "RSI", "period": 14}
    ]
    st.session_state['entry_rules'] = [
        {
            "indicator": "SMA_20",
            "operator": "crosses_above",
            "value_type": "indicator",
            "value": 50,
            "compare_indicator": "SMA_50"
        },
        {
            "indicator": "RSI_14",
            "operator": ">",
            "value_type": "number",
            "value": 50,
            "compare_indicator": "Close"
        }
    ]
    st.session_state['exit_rules'] = [
        {
            "indicator": "SMA_20",
            "operator": "crosses_below",
            "value_type": "indicator",
            "value": 50,
            "compare_indicator": "SMA_50"
        }
    ]
    st.success("✅ Loaded: MA Crossover + RSI Filter")


def _load_template_macd_bb():
    """Load MACD + Bollinger Bands template."""
    st.session_state['configuring_indicator'] = None
    st.session_state['indicators'] = [
        {"type": "MACD", "fast": 12, "slow": 26, "signal": 9},
        {"type": "BB", "period": 20, "std": 2.0}
    ]
    st.session_state['entry_rules'] = [
        {
            "indicator": "MACD",
            "operator": "crosses_above",
            "value_type": "indicator",
            "value": 0,
            "compare_indicator": "MACD_Signal"
        },
        {
            "indicator": "Close",
            "operator": "<",
            "value_type": "indicator",
            "value": 100,
            "compare_indicator": "BB_Lower"
        }
    ]
    st.session_state['exit_rules'] = [
        {
            "indicator": "Close",
            "operator": ">",
            "value_type": "indicator",
            "value": 100,
            "compare_indicator": "BB_Upper"
        }
    ]
    st.success("✅ Loaded: MACD + Bollinger Bands Breakout")


def _load_template_multi():
    """Load Multi-Indicator template."""
    st.session_state['configuring_indicator'] = None
    st.session_state['indicators'] = [
        {"type": "EMA", "period": 9},
        {"type": "EMA", "period": 21},
        {"type": "RSI", "period": 14},
        {"type": "ATR", "period": 14},
        {"type": "Volume_MA", "period": 20}
    ]
    st.session_state['entry_rules'] = [
        {
            "indicator": "EMA_9",
            "operator": ">",
            "value_type": "indicator",
            "value": 50,
            "compare_indicator": "EMA_21"
        },
        {
            "indicator": "RSI_14",
            "operator": ">",
            "value_type": "number",
            "value": 50,
            "compare_indicator": "Close"
        },
        {
            "indicator": "Volume",
            "operator": ">",
            "value_type": "indicator",
            "value": 100,
            "compare_indicator": "Volume_MA_20"
        }
    ]
    st.session_state['exit_rules'] = [
        {
            "indicator": "RSI_14",
            "operator": ">",
            "value_type": "number",
            "value": 70,
            "compare_indicator": "Close"
        }
    ]
    st.success("✅ Loaded: Multi-Indicator Combo (EMA + RSI + Volume)")
