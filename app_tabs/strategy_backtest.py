"""
Tab 1: Strategy Backtest
Single strategy backtesting with full parameter control.
"""

import streamlit as st
from .helpers import render_strategy_params, run_backtest, plot_equity_curve, plot_drawdown, plot_signals


def render_strategy_backtest_tab(config, strategies_dict):
    """Render the Strategy Backtest tab."""
    st.header("🎯 Single Strategy Backtest")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Strategy Selection")
        strategy_name = st.selectbox("Choose Strategy", list(strategies_dict.keys()))

        st.markdown("---")
        strategy_params = render_strategy_params(strategy_name, strategies_dict)

        if strategy_name == "📊 Buy & Hold (Baseline)":
            st.info(
                "📊 This strategy invests 100% of capital at start and holds until end. Position sizing settings are automatically overridden.")

    with col2:
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

        if run_button:
            with st.spinner("Running backtest..."):
                results, error = run_backtest(config, strategy_name, strategy_params, strategies_dict)

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
                        st.markdown("## 📋 Trade Log")
                        st.dataframe(results['trades'], use_container_width=True)

