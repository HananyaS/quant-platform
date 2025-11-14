"""
Classical ML classifiers for trading signal prediction.

Implements:
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine
- Gradient Boosting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from quant_framework.ml.base_model import BaseMLModel


class RandomForestClassifier(BaseMLModel):
    """
    Random Forest classifier for trading signals.
    
    Example:
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        importance = model.get_feature_importance()
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features for best split
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        super().__init__(name="RandomForestClassifier")
        
        from sklearn.ensemble import RandomForestClassifier as RFC
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
        self.model = RFC(**self.params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RandomForestClassifier':
        """Train Random Forest model."""
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        self.feature_importance_ = self.model.feature_importances_
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)


class XGBoostClassifier(BaseMLModel):
    """
    XGBoost classifier for trading signals.
    
    High-performance gradient boosting implementation.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            gamma: Minimum loss reduction
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        super().__init__(name="XGBoostClassifier")
        
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        self.model = XGBClassifier(**self.params)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = False,
        **kwargs
    ) -> 'XGBoostClassifier':
        """
        Train XGBoost model.
        
        Args:
            X: Training features
            y: Training target
            eval_set: Validation set for early stopping
            early_stopping_rounds: Stop if no improvement for N rounds
            verbose: Print training progress
            **kwargs: Additional training parameters
        """
        self.feature_names_ = list(X.columns)
        
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        if not verbose:
            fit_params['verbose'] = False
        
        self.model.fit(X, y, **fit_params)
        self.feature_importance_ = self.model.feature_importances_
        self.is_trained = True
        
        # Store training history
        if hasattr(self.model, 'evals_result_'):
            self.training_history_ = self.model.evals_result_()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)


class LightGBMClassifier(BaseMLModel):
    """
    LightGBM classifier for trading signals.
    
    Fast gradient boosting framework optimized for efficiency.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize LightGBM classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Learning rate
            num_leaves: Maximum tree leaves
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        super().__init__(name="LightGBMClassifier")
        
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'verbose': -1
        }
        
        self.model = LGBMClassifier(**self.params)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = 10,
        **kwargs
    ) -> 'LightGBMClassifier':
        """
        Train LightGBM model.
        
        Args:
            X: Training features
            y: Training target
            eval_set: Validation set for early stopping
            early_stopping_rounds: Stop if no improvement for N rounds
            **kwargs: Additional training parameters
        """
        self.feature_names_ = list(X.columns)
        
        fit_params = {'callbacks': []}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            
            # Add early stopping callback
            from lightgbm import early_stopping
            fit_params['callbacks'].append(early_stopping(early_stopping_rounds))
        
        self.model.fit(X, y, **fit_params)
        self.feature_importance_ = self.model.feature_importances_
        self.is_trained = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)


class SVMClassifier(BaseMLModel):
    """
    Support Vector Machine classifier for trading signals.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        probability: bool = True,
        random_state: int = 42
    ):
        """
        Initialize SVM classifier.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: Kernel coefficient
            probability: Enable probability estimates
            random_state: Random seed
        """
        super().__init__(name="SVMClassifier")
        
        from sklearn.svm import SVC
        
        self.params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'probability': probability,
            'random_state': random_state
        }
        
        self.model = SVC(**self.params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SVMClassifier':
        """Train SVM model."""
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)


class GradientBoostingClassifier(BaseMLModel):
    """
    Sklearn Gradient Boosting classifier.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42
    ):
        """Initialize Gradient Boosting classifier."""
        super().__init__(name="GradientBoostingClassifier")
        
        from sklearn.ensemble import GradientBoostingClassifier as GBC
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }
        
        self.model = GBC(**self.params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'GradientBoostingClassifier':
        """Train Gradient Boosting model."""
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        self.feature_importance_ = self.model.feature_importances_
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)


