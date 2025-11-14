"""
Base classes for ML trading models.

Provides abstract base for all ML models with common interface.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pickle
import json
from pathlib import Path


class BaseMLModel(ABC):
    """
    Abstract base class for ML trading models.
    
    All ML models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str = "BaseMLModel"):
        """
        Initialize base model.
        
        Args:
            name: Model name
        """
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_names_ = None
        self.feature_importance_ = None
        self.training_history_ = {}
        self.params = {}
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseMLModel':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training target
            **kwargs: Additional training parameters
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("Model does not support probability predictions")
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric values
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss
        )
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        predictions = self.predict(X)
        
        results = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y, predictions, average='weighted', zero_division=0)
        }
        
        # Add probability-based metrics if available
        try:
            probas = self.predict_proba(X)
            if len(probas.shape) == 2 and probas.shape[1] == 2:
                results['roc_auc'] = roc_auc_score(y, probas[:, 1])
            results['log_loss'] = log_loss(y, probas)
        except:
            pass
        
        return results
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Return top N features (None for all)
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance_ is None:
            raise ValueError("Feature importance not available")
        
        df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            df = df.head(top_n)
        
        return df
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'name': self.name,
            'model': self.model,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names_,
            'feature_importance': self.feature_importance_,
            'training_history': self.training_history_,
            'params': self.params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.name = model_data['name']
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        self.feature_names_ = model_data['feature_names']
        self.feature_importance_ = model_data['feature_importance']
        self.training_history_ = model_data['training_history']
        self.params = model_data['params']
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params
    
    def set_params(self, **params):
        """Set model parameters."""
        self.params.update(params)
        if self.model is not None and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)


