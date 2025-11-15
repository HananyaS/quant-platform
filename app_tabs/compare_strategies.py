"""
Tab 2: Compare Strategies
Side-by-side comparison of multiple strategies.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .helpers import run_backtest


def render_compare_strategies_tab(config, strategies_dict):
    """Render the Compare Strategies tab."""
    st.header("⚖️ Compare Multiple Strategies")

    st.markdown("### Select Strategies to Compare")
    
    # Get available built-in strategies
    available_strategies = [k for k in strategies_dict.keys() if k != "📊 Buy & Hold (Baseline)"]
    
    # Add custom strategies if any
    custom_strategies = []
    if 'custom_strategies' in st.session_state and st.session_state['custom_strategies']:
        custom_strategies = [f"🔧 {name}" for name in st.session_state['custom_strategies'].keys()]
        st.info(f"💾 {len(custom_strategies)} custom strategy(ies) available for comparison")
    
    all_strategies = available_strategies + custom_strategies
    
    selected_strategies = st.multiselect(
        "Strategies",
        options=all_strategies,
        default=all_strategies[:3] if len(all_strategies) >= 3 else all_strategies
    )

    col1, col2 = st.columns(2)
    with col1:
        include_baseline = st.checkbox("Include Buy & Hold Baseline", value=True)
    with col2:
        allow_short_comparison = st.checkbox("Allow Short Positions", value=False)

    if st.button("🔄 Run Comparison", type="primary"):
        if not selected_strategies and not include_baseline:
            st.warning("⚠️ Please select at least one strategy to compare")
        else:
            with st.spinner("Running comparison..."):
                strategies_to_compare = selected_strategies.copy()
                baseline_name = "📊 Buy & Hold (Baseline)"
                if include_baseline and baseline_name not in strategies_to_compare:
                    strategies_to_compare.insert(0, baseline_name)

                comparison_results = {}
                progress_bar = st.progress(0)

                for i, strat_name in enumerate(strategies_to_compare):
                    # Check if it's a custom strategy
                    if strat_name.startswith("🔧 "):
                        # It's a custom strategy
                        custom_name = strat_name[2:]  # Remove emoji prefix
                        custom_config = st.session_state['custom_strategies'][custom_name]
                        
                        # Create custom strategy instance
                        from quant_framework.models.custom_strategy import CustomStrategy
                        strategy = CustomStrategy(
                            name=custom_config['name'],
                            indicators=custom_config['indicators'],
                            entry_rules=custom_config['entry_rules'],
                            exit_rules=custom_config['exit_rules'],
                            entry_logic=custom_config['entry_logic'],
                            exit_logic=custom_config['exit_logic'],
                            allow_short=custom_config['allow_short']
                        )
                        
                        # Run custom backtest
                        from .custom_strategy import _run_custom_backtest
                        results, error = _run_custom_backtest(config, strategy)
                        if not error:
                            comparison_results[strat_name] = results
                    else:
                        # Built-in strategy
                        default_params = {k: v["default"] for k, v in strategies_dict[strat_name]["params"].items()}
                        if "allow_short" in default_params:
                            default_params["allow_short"] = allow_short_comparison

                        results, error = run_backtest(config, strat_name, default_params, strategies_dict)
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

