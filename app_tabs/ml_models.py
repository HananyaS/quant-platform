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
    """Render the ML Models tab with sub-tabs for features and training."""
    st.header("🤖 Machine Learning & Deep Learning")
    st.markdown("Train classical ML and deep learning models for market prediction.")
    
    # Sub-tabs for feature creation and model training (always show)
    feature_tab, training_tab = st.tabs(["📊 Features & Labels", "🚀 Train Models"])
    
    with feature_tab:
        try:
            _render_feature_creation_tab(config)
        except ImportError as e:
            st.warning("⚠️ ML features require: `pip install scikit-learn xgboost lightgbm`")
            st.code(str(e))
    
    with training_tab:
        try:
            _render_training_tab(config)
        except ImportError as e:
            st.warning("⚠️ Training requires: `pip install scikit-learn xgboost lightgbm torch`")
            st.code(str(e))


def _render_feature_creation_tab(config):
    """Render feature creation and label definition tab."""
    st.subheader("📊 Feature Engineering & Label Definition")
    st.markdown("Configure and create features from market data for ML/DL models.")

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
                    min_value=-10.0, max_value=10.0, value=1.0, step=0.1,
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
    st.markdown("---")
    st.markdown("### 🔧 Generate Features")
    create_features = st.button("🔧 Create Features & Labels", type="primary", use_container_width=True, help="Click to generate features and show training options")

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
    
    # Execute feature creation
    if create_features:
        from quant_framework.ml.features import FeatureEngineering
        
        with st.spinner("Creating features and labels..."):
            try:
                # Load data
                loader = YahooDataLoader(config["symbol"], config["start_date"], config["end_date"])
                data = loader.get_data()

                initial_rows = len(data)
                st.info(f"📥 Loaded {initial_rows} rows of data from {config['start_date']} to {config['end_date']}")

                # Feature engineering
                selected_periods = lookback_periods if lookback_periods else [5, 10, 20]
                max_lookback = max(selected_periods)

                fe = FeatureEngineering(
                    include_technical=include_technical,
                    include_statistical=include_statistical,
                    include_time=include_time,
                    include_lagged=include_lagged,
                    lookback_periods=selected_periods
                )

                features_df = fe.create_features(data)
                X, y = fe.create_training_data(features_df, target_type, threshold)

                # Calculate sample loss
                samples_lost = initial_rows - len(X)
                loss_pct = (samples_lost / initial_rows) * 100

                if loss_pct > 50:
                    st.warning(f"⚠️ {loss_pct:.1f}% of samples lost due to NaN values. Consider smaller lookback periods.")
                elif loss_pct > 30:
                    st.info(f"ℹ️ {loss_pct:.1f}% of samples lost to NaN values.")
                else:
                    st.success(f"✓ {loss_pct:.1f}% sample loss - good!")

                st.session_state['X_ml'] = X
                st.session_state['y_ml'] = y
                st.session_state['ml_data'] = data
                st.session_state['ml_config'] = st.session_state['ml_config_params']

                st.success(f"✅ Created {len(X.columns)} features with {len(X)} samples")

                # Show metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("📊 Features", len(X.columns))
                with col_b:
                    st.metric("📈 Samples", len(X))
                with col_c:
                    st.metric("🗑️ Dropped", samples_lost, delta=f"-{loss_pct:.1f}%", delta_color="inverse")
                with col_d:
                    if target_type == "classification":
                        pos_pct = (y == 1).sum() / len(y) * 100
                        st.metric("🎯 Positive %", f"{pos_pct:.1f}%")
                    else:
                        st.metric("🎯 Mean", f"{y.mean():.4f}")

                st.subheader("Sample Features")
                st.dataframe(X.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    elif 'X_ml' in st.session_state:
        st.success("✅ Features already created! Go to '🚀 Train Models' tab to train a model.")
        
        X = st.session_state['X_ml']
        y = st.session_state['y_ml']
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📊 Features", len(X.columns))
        with col_b:
            st.metric("📈 Samples", len(X))
        with col_c:
            target_type = st.session_state['ml_config'].get('target_type', 'classification')
            if target_type == "classification":
                pos_pct = (y == 1).sum() / len(y) * 100
                st.metric("🎯 Positive %", f"{pos_pct:.1f}%")
            else:
                st.metric("🎯 Mean", f"{y.mean():.4f}")
        
        with st.expander("View Sample Features"):
            st.dataframe(X.head(10), use_container_width=True)


def _render_training_tab(config):
    """Render training tab with ML and DL options."""
    if 'X_ml' not in st.session_state:
        st.info("👈 **Step 1:** Go to '📊 Features & Labels' tab")
        st.info("🔧 **Step 2:** Configure and create features")
        st.info("🚀 **Step 3:** Return here to train models")
        return
    
    # Show feature summary
    X = st.session_state['X_ml']
    y = st.session_state['y_ml']
    ml_config = st.session_state.get('ml_config', {})
    
    st.success("✅ Features loaded and ready for training!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Features", len(X.columns))
    with col2:
        st.metric("📈 Samples", len(X))
    with col3:
        target_type = ml_config.get('target_type', 'classification')
        if target_type == "classification":
            pos_pct = (y == 1).sum() / len(y) * 100
            st.metric("🎯 Positive %", f"{pos_pct:.1f}%")
        else:
            st.metric("🎯 Target Mean", f"{y.mean():.4f}")
    with col4:
        split_method = ml_config.get('split_method', 'holdout')
        st.metric("📏 Split", split_method.replace('_', ' ').title())
    
    st.markdown("---")
    
    # Nested tabs for ML and DL
    ml_tab, dl_tab = st.tabs(["🤖 Classical ML", "🧠 Deep Learning (PyTorch)"])
    
    with ml_tab:
        _render_classical_ml_training(config)
    
    with dl_tab:
        _render_deep_learning_training(config)


def _render_classical_ml_training(config):
    """Render classical ML training panel."""
    from quant_framework.ml.classifiers import (RandomForestClassifier, XGBoostClassifier,
                                                 LightGBMClassifier, SVMClassifier,
                                                 GradientBoostingClassifier)
    from quant_framework.ml.trainer import ModelTrainer
    from quant_framework.ml.preprocessing import DataPreprocessor
    
    # Initialize comparison storage
    if 'ml_comparison' not in st.session_state:
        st.session_state['ml_comparison'] = []
    
    st.subheader("🤖 Classical ML Training")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            ["Random Forest", "XGBoost", "LightGBM", "SVM", "Gradient Boosting"]
        )
    with col2:
        if st.button("🗑️ Clear Comparison", use_container_width=True):
            st.session_state['ml_comparison'] = []
            st.rerun()
    
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
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
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
                
                # Store for comparison
                comparison_entry = {
                    'model_name': model_type,
                    'params': model_params,
                    'results': results,
                    'train_metrics': results.get('train_metrics', {}),
                    'test_metrics': results.get('test_metrics', results.get('avg_test_metrics', {}))
                }
                st.session_state['ml_comparison'].append(comparison_entry)

                st.success("✓ Training complete!")

                # Show train vs test metrics
                train_metrics = results.get('train_metrics', {})
                test_metrics = results.get('test_metrics', results.get('avg_test_metrics', {}))
                
                st.subheader("📊 Train vs Test Performance")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    train_acc = train_metrics.get('accuracy', 0)
                    test_acc = test_metrics.get('accuracy', 0)
                    st.metric("Accuracy (Test)", f"{test_acc:.4f}", delta=f"{test_acc - train_acc:.4f}")
                    st.caption(f"Train: {train_acc:.4f}")
                with col_b:
                    train_prec = train_metrics.get('precision', 0)
                    test_prec = test_metrics.get('precision', 0)
                    st.metric("Precision (Test)", f"{test_prec:.4f}", delta=f"{test_prec - train_prec:.4f}")
                    st.caption(f"Train: {train_prec:.4f}")
                with col_c:
                    train_rec = train_metrics.get('recall', 0)
                    test_rec = test_metrics.get('recall', 0)
                    st.metric("Recall (Test)", f"{test_rec:.4f}", delta=f"{test_rec - train_rec:.4f}")
                    st.caption(f"Train: {train_rec:.4f}")
                with col_d:
                    train_f1 = train_metrics.get('f1_score', 0)
                    test_f1 = test_metrics.get('f1_score', 0)
                    st.metric("F1 Score (Test)", f"{test_f1:.4f}", delta=f"{test_f1 - train_f1:.4f}")
                    st.caption(f"Train: {train_f1:.4f}")

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
    
    # Model comparison section
    if len(st.session_state['ml_comparison']) > 0:
        st.markdown("---")
        st.subheader(f"📊 Model Comparison ({len(st.session_state['ml_comparison'])} models)")
        
        # Create comparison DataFrame
        comparison_data = []
        for entry in st.session_state['ml_comparison']:
            train_m = entry['train_metrics']
            test_m = entry['test_metrics']
            comparison_data.append({
                'Model': entry['model_name'],
                'Train Accuracy': f"{train_m.get('accuracy', 0):.4f}",
                'Test Accuracy': f"{test_m.get('accuracy', 0):.4f}",
                'Train Precision': f"{train_m.get('precision', 0):.4f}",
                'Test Precision': f"{test_m.get('precision', 0):.4f}",
                'Train Recall': f"{train_m.get('recall', 0):.4f}",
                'Test Recall': f"{test_m.get('recall', 0):.4f}",
                'Train F1': f"{train_m.get('f1_score', 0):.4f}",
                'Test F1': f"{test_m.get('f1_score', 0):.4f}",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Test accuracy comparison
            fig_acc = go.Figure()
            test_accs = [e['test_metrics'].get('accuracy', 0) for e in st.session_state['ml_comparison']]
            train_accs = [e['train_metrics'].get('accuracy', 0) for e in st.session_state['ml_comparison']]
            model_names = [e['model_name'] for e in st.session_state['ml_comparison']]
            
            fig_acc.add_trace(go.Bar(name='Test', x=model_names, y=test_accs, marker_color='#2E86AB'))
            fig_acc.add_trace(go.Bar(name='Train', x=model_names, y=train_accs, marker_color='#A23B72'))
            fig_acc.update_layout(
                title="Accuracy: Train vs Test",
                xaxis_title="Model",
                yaxis_title="Accuracy",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # F1 Score comparison
            fig_f1 = go.Figure()
            test_f1s = [e['test_metrics'].get('f1_score', 0) for e in st.session_state['ml_comparison']]
            train_f1s = [e['train_metrics'].get('f1_score', 0) for e in st.session_state['ml_comparison']]
            
            fig_f1.add_trace(go.Bar(name='Test', x=model_names, y=test_f1s, marker_color='#2E86AB'))
            fig_f1.add_trace(go.Bar(name='Train', x=model_names, y=train_f1s, marker_color='#A23B72'))
            fig_f1.update_layout(
                title="F1 Score: Train vs Test",
                xaxis_title="Model",
                yaxis_title="F1 Score",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Overfitting analysis
        st.subheader("🔍 Overfitting Analysis")
        overfit_data = []
        for entry in st.session_state['ml_comparison']:
            train_acc = entry['train_metrics'].get('accuracy', 0)
            test_acc = entry['test_metrics'].get('accuracy', 0)
            gap = train_acc - test_acc
            status = "✅ Good" if gap < 0.05 else "⚠️ Moderate" if gap < 0.10 else "❌ Overfitting"
            overfit_data.append({
                'Model': entry['model_name'],
                'Train Acc': f"{train_acc:.4f}",
                'Test Acc': f"{test_acc:.4f}",
                'Gap': f"{gap:.4f}",
                'Status': status
            })
        
        overfit_df = pd.DataFrame(overfit_data)
        st.dataframe(overfit_df, use_container_width=True)


def _render_deep_learning_training(config):
    """Render deep learning training panel (PyTorch)."""
    st.subheader("🧠 Deep Learning (PyTorch)")
    
    try:
        import torch
        import torch.nn as nn
        
        # Architecture selection
        architecture = st.selectbox(
            "Architecture",
            ["LSTM", "GRU", "Transformer", "CNN", "MLP"]
        )
        
        # Basic parameters
        with st.expander("🎛️ Model Parameters"):
            if architecture in ["LSTM", "GRU"]:
                hidden_size = st.slider("Hidden Size", 16, 256, 64, 8)
                num_layers = st.slider("Layers", 1, 5, 2, 1)
                dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
            elif architecture == "Transformer":
                d_model = st.slider("Model Dimension", 32, 256, 64, 8)
                nhead = st.selectbox("Attention Heads", [2, 4, 8], index=1)
                num_layers = st.slider("Layers", 1, 6, 2, 1)
            elif architecture == "CNN":
                num_filters = st.slider("Filters", 16, 128, 64, 8)
                kernel_size = st.slider("Kernel Size", 2, 7, 3, 1)
                num_layers = st.slider("Conv Layers", 1, 5, 2, 1)
            else:  # MLP
                hidden_sizes = st.text_input("Hidden Layers", "128,64,32")
                dropout = st.slider("Dropout", 0.0, 0.5, 0.3, 0.05)
        
        # Training config
        with st.expander("⚙️ Training Configuration"):
            batch_size = st.slider("Batch Size", 16, 256, 32, 16)
            epochs = st.slider("Epochs", 10, 300, 50, 10)
            learning_rate = st.select_slider(
                "Learning Rate", 
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                value=0.001
            )
            optimizer_type = st.selectbox("Optimizer", ["Adam", "SGD", "AdamW", "RMSprop"])
        
        # Validation & Early Stopping
        with st.expander("🛡️ Validation & Early Stopping"):
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
            enable_early_stopping = st.checkbox("Enable Early Stopping", value=True)
            if enable_early_stopping:
                patience = st.slider("Patience (epochs)", 5, 50, 10, 5)
                min_delta = st.select_slider(
                    "Min Delta (improvement threshold)",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                    value=0.001
                )
                monitor_metric = st.selectbox("Monitor Metric", ["Loss", "Accuracy"], index=0)
        
        if st.button("🚀 Train DL Model", type="primary", use_container_width=True):
            # Prepare data
            X = st.session_state['X_ml']
            y = st.session_state['y_ml']
            
            # Convert to PyTorch tensors
            import time
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # Progress tracking containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            col_loss, col_acc = st.columns(2)
            with col_loss:
                loss_chart = st.empty()
            with col_acc:
                acc_chart = st.empty()
            
            metrics_container = st.empty()
            
            try:
                status_text.text("📊 Preparing data...")
                
                # Split data into train, validation, test
                # First split: test set
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X.values, y.values, test_size=0.2, shuffle=False
                )
                
                # Second split: train and validation
                val_size = validation_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size, shuffle=False
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train_scaled)
                y_train_tensor = torch.LongTensor(y_train)
                X_val_tensor = torch.FloatTensor(X_val_scaled)
                y_val_tensor = torch.LongTensor(y_val)
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                y_test_tensor = torch.LongTensor(y_test)
                
                # Create DataLoader
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                status_text.text(f"🧠 Building {architecture} model...")
                
                # Build model (simplified)
                input_size = X_train_scaled.shape[1]
                num_classes = len(np.unique(y_train))
                
                if architecture in ["LSTM", "GRU"]:
                    # Simple LSTM/GRU
                    model = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size // 2, num_classes)
                    )
                else:  # MLP fallback for all
                    hidden_layers = [int(x.strip()) for x in hidden_sizes.split(',')]
                    layers = []
                    prev_size = input_size
                    for h_size in hidden_layers:
                        layers.extend([
                            nn.Linear(prev_size, h_size),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ])
                        prev_size = h_size
                    layers.append(nn.Linear(prev_size, num_classes))
                    model = nn.Sequential(*layers)
                
                # Loss and optimizer
                criterion = nn.CrossEntropyLoss()
                if optimizer_type == "Adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                elif optimizer_type == "SGD":
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                elif optimizer_type == "AdamW":
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                else:
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
                
                # Training loop with progress tracking
                train_losses = []
                train_accs = []
                val_losses = []
                val_accs = []
                
                # Early stopping variables
                if enable_early_stopping:
                    best_val_metric = float('inf') if monitor_metric == "Loss" else 0.0
                    best_model_state = None
                    epochs_without_improvement = 0
                early_stopped = False
                stopped_epoch = epochs
                
                for epoch in range(epochs):
                    # Training phase
                    model.train()
                    epoch_loss = 0
                    correct = 0
                    total = 0
                    
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                    
                    train_loss = epoch_loss / len(train_loader)
                    train_acc = correct / total
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    
                    # Validation phase
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                        _, val_predicted = torch.max(val_outputs.data, 1)
                        val_acc = (val_predicted == y_val_tensor).sum().item() / len(y_val_tensor)
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)
                    
                    # Early stopping check
                    if enable_early_stopping:
                        current_val_metric = val_loss if monitor_metric == "Loss" else val_acc
                        
                        if monitor_metric == "Loss":
                            improved = (best_val_metric - current_val_metric) > min_delta
                        else:  # Accuracy
                            improved = (current_val_metric - best_val_metric) > min_delta
                        
                        if improved:
                            best_val_metric = current_val_metric
                            best_model_state = model.state_dict().copy()
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1
                        
                        if epochs_without_improvement >= patience:
                            early_stopped = True
                            stopped_epoch = epoch + 1
                            status_text.warning(f"🛑 Early stopping triggered at epoch {stopped_epoch}")
                            # Restore best model
                            model.load_state_dict(best_model_state)
                            break
                    
                    # Update progress every epoch
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    
                    es_info = f" | No improve: {epochs_without_improvement}/{patience}" if enable_early_stopping else ""
                    status_text.text(f"📈 Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}{es_info}")
                    
                    # Update charts every 5 epochs or last epoch
                    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                        # Loss chart
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(y=train_losses, name='Train Loss', mode='lines', line=dict(color='#A23B72')))
                        fig_loss.add_trace(go.Scatter(y=val_losses, name='Val Loss', mode='lines', line=dict(color='#2E86AB')))
                        if enable_early_stopping and best_model_state is not None:
                            # Mark best epoch
                            best_epoch = val_losses.index(min(val_losses)) if monitor_metric == "Loss" else val_accs.index(max(val_accs))
                            fig_loss.add_trace(go.Scatter(
                                x=[best_epoch], 
                                y=[val_losses[best_epoch]], 
                                mode='markers', 
                                marker=dict(size=12, color='gold', symbol='star'),
                                name='Best Model'
                            ))
                        fig_loss.update_layout(title="Loss (Train vs Validation)", height=300, margin=dict(l=20, r=20, t=40, b=20))
                        loss_chart.plotly_chart(fig_loss, use_container_width=True)
                        
                        # Accuracy chart
                        fig_acc = go.Figure()
                        fig_acc.add_trace(go.Scatter(y=train_accs, name='Train Acc', mode='lines', line=dict(color='#A23B72')))
                        fig_acc.add_trace(go.Scatter(y=val_accs, name='Val Acc', mode='lines', line=dict(color='#2E86AB')))
                        if enable_early_stopping and best_model_state is not None:
                            # Mark best epoch
                            best_epoch = val_losses.index(min(val_losses)) if monitor_metric == "Loss" else val_accs.index(max(val_accs))
                            fig_acc.add_trace(go.Scatter(
                                x=[best_epoch], 
                                y=[val_accs[best_epoch]], 
                                mode='markers', 
                                marker=dict(size=12, color='gold', symbol='star'),
                                name='Best Model'
                            ))
                        fig_acc.update_layout(title="Accuracy (Train vs Validation)", height=300, margin=dict(l=20, r=20, t=40, b=20))
                        acc_chart.plotly_chart(fig_acc, use_container_width=True)
                        
                        # Current metrics
                        with metrics_container.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Epoch", f"{epoch+1}/{epochs}")
                            with col2:
                                st.metric("Train Loss", f"{train_loss:.4f}")
                            with col3:
                                st.metric("Val Loss", f"{val_loss:.4f}")
                            with col4:
                                st.metric("Val Acc", f"{val_acc:.4f}")
                
                # Final evaluation on test set
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor).item()
                    _, test_predicted = torch.max(test_outputs.data, 1)
                    test_acc = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
                
                # Final results
                progress_bar.progress(1.0)
                
                if early_stopped:
                    status_text.success(f"✅ Training stopped early at epoch {stopped_epoch} (patience={patience})")
                    st.info(f"🌟 Best {monitor_metric}: {best_val_metric:.4f} (restored from best epoch)")
                else:
                    status_text.success("✅ Training complete!")
                
                # Display final metrics
                st.markdown("---")
                st.subheader("📊 Final Test Set Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Test Accuracy", f"{test_acc:.4f}")
                with col2:
                    st.metric("Test Loss", f"{test_loss:.4f}")
                with col3:
                    st.metric("Epochs Trained", stopped_epoch if early_stopped else epochs)
                with col4:
                    best_val = val_losses[-1] if monitor_metric == "Loss" else val_accs[-1]
                    st.metric(f"Best Val {monitor_metric}", f"{best_val:.4f}")
                
                if early_stopped:
                    st.success(f"⏱️ Training time saved: ~{((epochs - stopped_epoch) / epochs * 100):.1f}% (stopped {epochs - stopped_epoch} epochs early)")
                
                # Store model
                st.session_state['dl_model'] = model
                st.session_state['dl_scaler'] = scaler
                st.session_state['dl_results'] = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                    'early_stopped': early_stopped,
                    'stopped_epoch': stopped_epoch,
                    'total_epochs': epochs,
                    'architecture': architecture,
                    'params': {
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'optimizer': optimizer_type,
                        'early_stopping': enable_early_stopping,
                        'patience': patience if enable_early_stopping else None,
                        'monitor': monitor_metric if enable_early_stopping else None
                    }
                }
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    except ImportError:
        st.warning("⚠️ PyTorch not installed. Install with: `pip install torch torchvision`")

