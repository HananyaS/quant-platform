"""
Feature analysis and importance tools.

Provides:
- Feature importance analysis
- SHAP values
- Correlation analysis
- Feature selection
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Analyze and visualize feature importance.
    
    Example:
        analyzer = FeatureAnalyzer()
        importance = analyzer.calculate_importance(model, X, y)
        analyzer.plot_importance(importance, top_n=20)
    """
    
    def __init__(self):
        """Initialize feature analyzer."""
        self.importance_scores = {}
    
    def calculate_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'native'
    ) -> pd.DataFrame:
        """
        Calculate feature importance.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            method: Method to use ('native', 'permutation', 'shap')
            
        Returns:
            DataFrame with feature importance
        """
        if method == 'native':
            return self._native_importance(model, X)
        elif method == 'permutation':
            return self._permutation_importance(model, X, y)
        elif method == 'shap':
            return self._shap_importance(model, X)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _native_importance(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """Get native feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError("Model does not have feature_importances_ or coef_")
        
        df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.importance_scores['native'] = df
        return df
    
    def _permutation_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """Calculate permutation importance."""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42
        )
        
        df = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        self.importance_scores['permutation'] = df
        return df
    
    def _shap_importance(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate SHAP values."""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For binary classification, use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        self.importance_scores['shap'] = df
        return df
    
    def plot_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance"
    ):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
            title: Plot title
            
        Returns:
            Plotly figure
        """
        import plotly.graph_objects as go
        
        df = importance_df.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['importance'],
            y=df['feature'],
            orientation='h',
            marker=dict(
                color=df['importance'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, top_n * 20),
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def correlation_heatmap(
        self,
        X: pd.DataFrame,
        method: str = 'pearson'
    ):
        """
        Create correlation heatmap.
        
        Args:
            X: Features
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            Plotly figure
        """
        import plotly.graph_objects as go
        
        corr = X.corr(method=method)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title=f'{method.capitalize()} Correlation Matrix',
            width=min(1200, len(corr.columns) * 40),
            height=min(1200, len(corr.columns) * 40)
        )
        
        return fig
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'mutual_info',
        k: int = 20
    ) -> List[str]:
        """
        Select top K features.
        
        Args:
            X: Features
            y: Target
            method: Selection method ('mutual_info', 'f_score', 'chi2')
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_score':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'chi2':
            # Chi2 requires non-negative features
            X_positive = X - X.min() + 1e-9
            selector = SelectKBest(chi2, k=k)
            selector.fit(X_positive, y)
            selected_features = X.columns[selector.get_support()].tolist()
            return selected_features
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        return selected_features
    
    def plot_shap_summary(self, model, X: pd.DataFrame, max_display: int = 20):
        """
        Plot SHAP summary plot.
        
        Args:
            model: Trained model
            X: Features
            max_display: Maximum features to display
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib required")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.tight_layout()
        
        return plt.gcf()
    
    def feature_interaction(
        self,
        X: pd.DataFrame,
        feature1: str,
        feature2: str,
        y: pd.Series
    ):
        """
        Analyze interaction between two features.
        
        Args:
            X: Features
            feature1: First feature name
            feature2: Second feature name
            y: Target
            
        Returns:
            Plotly figure
        """
        import plotly.graph_objects as go
        
        df = pd.DataFrame({
            'x': X[feature1],
            'y': X[feature2],
            'target': y
        })
        
        fig = go.Figure()
        
        for target_val in df['target'].unique():
            subset = df[df['target'] == target_val]
            fig.add_trace(go.Scatter(
                x=subset['x'],
                y=subset['y'],
                mode='markers',
                name=f'Class {target_val}',
                marker=dict(size=5, opacity=0.6)
            ))
        
        fig.update_layout(
            title=f'Feature Interaction: {feature1} vs {feature2}',
            xaxis_title=feature1,
            yaxis_title=feature2,
            hovermode='closest'
        )
        
        return fig


