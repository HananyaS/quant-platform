"""
Tab 3: ML Models
Train classical ML models with full control over features, data splits, and labels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from quant_framework.data.loaders import YahooDataLoader


def render_ml_models_tab(config):
    """Render the ML Models tab."""
    st.header("🤖 Machine Learning Models")
    st.markdown("Train classical ML models with full control over features, data splits, and labels.")

    try:
        from quant_framework.ml.features import FeatureEngineering
        from quant_framework.ml.classifiers import (RandomForestClassifier, XGBoostClassifier,
                                                     LightGBMClassifier, SVMClassifier,
                                                     GradientBoostingClassifier)
        from quant_framework.ml.trainer import ModelTrainer
        from quant_framework.ml.preprocessing import DataPreprocessor

        col1, col2 = st.columns([1, 2])

        with col1:
            _render_feature_configuration(config)

        with col2:
            _render_model_training(config)

    except ImportError as e:
        st.warning(
            "⚠️ ML features require additional packages. Install with: `pip install scikit-learn xgboost lightgbm`")
        st.code(str(e))


def _render_feature_configuration(config):
    """Render feature configuration panel."""
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
                threshold = 0.0
            else:
                horizon = st.number_input(
                    "Future Horizon (days)",
                    min_value=1, max_value=30, value=5
                )
                threshold = 0.0
        else:
            forecast_horizon = st.number_input(
                "Forecast Horizon (days)",
                min_value=1, max_value=30, value=5
            )
            threshold = 0.0

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
            test_size = 0.2
        else:  # walk_forward
            train_window = st.number_input("Train Window (days)", 100, 1000, 252, 21)
            test_window = st.number_input("Test Window (days)", 20, 200, 63, 21)
            step_size = st.number_input("Step Size (days)", 5, 100, 21, 7)
            test_size = 0.2

    # Create features button
    create_features = st.button("🔧 Create Features & Labels", type="primary")

    # Store in session state for model training
    if 'ml_config_params' not in st.session_state:
        st.session_state['ml_config_params'] = {}

    st.session_state['ml_config_params'] = {
        'include_technical': include_technical,
        'include_statistical': include_statistical,
        'include_time': include_time,
        'include_lagged': include_lagged,
        'lookback_periods': lookback_periods,
        'target_type': target_type,
        'threshold': threshold,
        'split_method': split_method,
        'test_size': test_size if split_method == "holdout" else None,
        'n_splits': n_splits if split_method == "time_series_cv" else None,
        'train_window': train_window if split_method == "walk_forward" else None,
        'test_window': test_window if split_method == "walk_forward" else None,
        'step_size': step_size if split_method == "walk_forward" else None,
        'create_features': create_features
    }


def _render_model_training(config):
    """Render model training panel."""
    from quant_framework.ml.features import FeatureEngineering
    from quant_framework.ml.classifiers import (RandomForestClassifier, XGBoostClassifier,
                                                 LightGBMClassifier, SVMClassifier,
                                                 GradientBoostingClassifier)
    from quant_framework.ml.trainer import ModelTrainer
    from quant_framework.ml.preprocessing import DataPreprocessor

    ml_params = st.session_state.get('ml_config_params', {})
    create_features = ml_params.get('create_features', False)

    if create_features:
        with st.spinner("Creating features and labels..."):
            try:
                # Load data
                loader = YahooDataLoader(config["symbol"], config["start_date"], config["end_date"])
                data = loader.get_data()

                initial_rows = len(data)
                st.info(f"📥 Loaded {initial_rows} rows of data from {config['start_date']} to {config['end_date']}")

                # Feature engineering
                selected_periods = ml_params.get('lookback_periods', [5, 10, 20])
                max_lookback = max(selected_periods) if selected_periods else 20

                fe = FeatureEngineering(
                    include_technical=ml_params.get('include_technical', True),
                    include_statistical=ml_params.get('include_statistical', True),
                    include_time=ml_params.get('include_time', True),
                    include_lagged=ml_params.get('include_lagged', True),
                    lookback_periods=selected_periods if selected_periods else [5, 10, 20]
                )

                features_df = fe.create_features(data)

                # Create labels based on configuration
                threshold_val = ml_params.get('threshold', 0.0)
                target_type = ml_params.get('target_type', 'classification')
                X, y = fe.create_training_data(features_df, target_type, threshold_val)

                # Calculate sample loss
                samples_lost = initial_rows - len(X)
                loss_pct = (samples_lost / initial_rows) * 100

                if loss_pct > 50:
                    st.warning(
                        f"⚠️ {loss_pct:.1f}% of samples lost due to NaN values (first {max_lookback}+ rows). Consider using smaller lookback periods for more samples.")
                elif loss_pct > 30:
                    st.info(f"ℹ️ {loss_pct:.1f}% of samples lost to NaN values. This is normal with rolling features.")
                else:
                    st.success(f"✓ {loss_pct:.1f}% sample loss - good data retention!")

                st.session_state['X_ml'] = X
                st.session_state['y_ml'] = y
                st.session_state['ml_data'] = data
                st.session_state['ml_config'] = ml_params

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

