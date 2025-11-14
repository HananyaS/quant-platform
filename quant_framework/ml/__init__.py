"""
Machine Learning and Deep Learning components for quantitative trading.

This module provides:
- Feature engineering
- ML model training and evaluation
- Deep learning models (LSTM, Transformer)
- Model persistence
- Walk-forward analysis
"""

from quant_framework.ml.features import FeatureEngineering
from quant_framework.ml.preprocessing import DataPreprocessor
from quant_framework.ml.base_model import BaseMLModel
from quant_framework.ml.classifiers import (
    RandomForestClassifier,
    XGBoostClassifier,
    LightGBMClassifier,
    SVMClassifier,
    GradientBoostingClassifier
)
from quant_framework.ml.deep_models import (
    LSTMClassifier,
    GRUClassifier,
    CNNClassifier
)
from quant_framework.ml.trainer import ModelTrainer, ModelComparator

__all__ = [
    'FeatureEngineering',
    'DataPreprocessor',
    'BaseMLModel',
    'RandomForestClassifier',
    'XGBoostClassifier',
    'LightGBMClassifier',
    'SVMClassifier',
    'GradientBoostingClassifier',
    'LSTMClassifier',
    'GRUClassifier',
    'CNNClassifier',
    'ModelTrainer',
    'ModelComparator',
]

