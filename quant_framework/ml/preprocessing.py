"""
Data preprocessing for ML models.

Handles:
- Train/test/validation splits
- Scaling and normalization
- Time series cross-validation
- Walk-forward analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit


class DataPreprocessor:
    """
    Preprocess data for ML model training.
    
    Handles scaling, splitting, and validation for time series data.
    
    Example:
        preprocessor = DataPreprocessor(scaler_type='standard')
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
    """
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        test_size: float = 0.2,
        validation_size: float = 0.1
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust', 'none')
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
        """
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        self.is_fitted = False
    
    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        For time series, uses chronological split (no shuffle by default).
        
        Args:
            X: Features
            y: Target
            shuffle: Whether to shuffle (not recommended for time series)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        n = len(X)
        split_idx = int(n * (1 - self.test_size))
        
        if shuffle:
            indices = np.random.permutation(n)
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
        else:
            train_idx = range(split_idx)
            test_idx = range(split_idx, n)
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        return X_train, X_test, y_train, y_test
    
    def train_val_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val and test
        X_train_val, X_test, y_train_val, y_test = self.train_test_split(X, y)
        
        # Second split: train and val
        n_train_val = len(X_train_val)
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        split_idx = int(n_train_val * (1 - val_size_adjusted))
        
        X_train = X_train_val.iloc[:split_idx]
        X_val = X_train_val.iloc[split_idx:]
        y_train = y_train_val.iloc[:split_idx]
        y_val = y_train_val.iloc[split_idx:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform data.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Scaled features
        """
        if self.scaler is None:
            self.is_fitted = True
            return X
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            Scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        if self.scaler is None:
            return X
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            index=X.index,
            columns=X.columns
        )
        return X_scaled
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data.
        
        Args:
            X: Scaled features
            
        Returns:
            Original scale features
        """
        if self.scaler is None:
            return X
        
        X_original = pd.DataFrame(
            self.scaler.inverse_transform(X),
            index=X.index,
            columns=X.columns
        )
        return X_original
    
    def time_series_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> TimeSeriesSplit:
        """
        Create time series cross-validation splitter.
        
        Args:
            X: Features
            y: Target
            n_splits: Number of splits
            
        Returns:
            TimeSeriesSplit object
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv
    
    def walk_forward_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int = 252,  # 1 year
        test_size: int = 63,     # 1 quarter
        step_size: int = 21       # 1 month
    ) -> List[Tuple]:
        """
        Create walk-forward analysis splits.
        
        Args:
            X: Features
            y: Target
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step size for rolling window
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n = len(X)
        splits = []
        
        start = 0
        while start + train_size + test_size <= n:
            train_end = start + train_size
            test_end = train_end + test_size
            
            train_indices = list(range(start, train_end))
            test_indices = list(range(train_end, test_end))
            
            splits.append((train_indices, test_indices))
            
            start += step_size
        
        return splits
    
    def handle_missing_values(
        self,
        X: pd.DataFrame,
        strategy: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Args:
            X: Features with potential missing values
            strategy: Strategy for handling missing ('forward_fill', 'backward_fill', 
                     'interpolate', 'drop', 'mean')
            
        Returns:
            DataFrame with missing values handled
        """
        X = X.copy()
        
        if strategy == 'forward_fill':
            X = X.fillna(method='ffill')
        elif strategy == 'backward_fill':
            X = X.fillna(method='bfill')
        elif strategy == 'interpolate':
            X = X.interpolate(method='linear')
        elif strategy == 'drop':
            X = X.dropna()
        elif strategy == 'mean':
            X = X.fillna(X.mean())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Fill any remaining NaN with 0
        X = X.fillna(0)
        
        return X
    
    def remove_outliers(
        self,
        X: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from features.
        
        Args:
            X: Features
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        X = X.copy()
        
        if method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            # Clip values
            for col in X.columns:
                X[col] = X[col].clip(lower[col], upper[col])
                
        elif method == 'zscore':
            z_scores = np.abs((X - X.mean()) / X.std())
            X = X[(z_scores < threshold).all(axis=1)]
        
        return X

