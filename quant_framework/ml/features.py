"""
Feature engineering for ML-based trading strategies.

Provides comprehensive feature extraction from price data including:
- Technical indicators
- Statistical features
- Time-based features
- Lagged features
- Rolling window features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from quant_framework.data.indicators import TechnicalIndicators


class FeatureEngineering:
    """
    Feature engineering toolkit for ML trading models.
    
    Generates features from OHLCV data for training ML models.
    
    Example:
        fe = FeatureEngineering()
        features = fe.create_features(data)
        X, y = fe.create_training_data(features, target_column='returns')
    """
    
    def __init__(
        self,
        include_technical: bool = True,
        include_statistical: bool = True,
        include_time: bool = True,
        include_lagged: bool = True,
        lookback_periods: List[int] = [5, 10, 20, 50],
        forward_periods: int = 1
    ):
        """
        Initialize feature engineering.
        
        Args:
            include_technical: Include technical indicators
            include_statistical: Include statistical features
            include_time: Include time-based features
            include_lagged: Include lagged features
            lookback_periods: Periods for rolling calculations
            forward_periods: Periods forward for target calculation
        """
        self.include_technical = include_technical
        self.include_statistical = include_statistical
        self.include_time = include_time
        self.include_lagged = include_lagged
        self.lookback_periods = lookback_periods
        self.forward_periods = forward_periods
        
        self.feature_names_ = []
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all features
        """
        df = data.copy()
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Technical indicators
        if self.include_technical:
            df = self._add_technical_indicators(df)
        
        # Statistical features
        if self.include_statistical:
            df = self._add_statistical_features(df)
        
        # Time-based features
        if self.include_time:
            df = self._add_time_features(df)
        
        # Lagged features
        if self.include_lagged:
            df = self._add_lagged_features(df)
        
        # Store feature names
        self.feature_names_ = [col for col in df.columns if col not in 
                               ['open', 'high', 'low', 'close', 'volume']]
        
        # Ensure all feature columns are numeric (defensive check)
        for col in self.feature_names_:
            if df[col].dtype == 'object' or df[col].dtype == 'O':
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        column_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }
        return df.rename(columns=column_map)
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price changes
        df['price_change'] = df['close'] - df['close'].shift(1)
        df['high_low_range'] = df['high'] - df['low']
        df['close_open_change'] = df['close'] - df['open']
        
        # Normalized features
        df['norm_high'] = (df['high'] - df['low']) / df['close']
        df['norm_low'] = (df['low'] - df['close']) / df['close']
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma5'] = df['volume'].rolling(5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Use only user-selected lookback periods for moving averages
        # Always include at least [5, 10, 20] for basic indicators
        ma_periods = sorted(set(self.lookback_periods + [5, 10, 20]))
        
        # Moving averages
        for period in ma_periods:
            if period <= len(df):  # Only compute if we have enough data
                df[f'sma_{period}'] = TechnicalIndicators.sma(close, period)
                df[f'ema_{period}'] = TechnicalIndicators.ema(close, period)
                
                # Price relative to MA
                df[f'close_sma_{period}_ratio'] = close / df[f'sma_{period}']
        
        # MA crossovers (only if both periods exist)
        if 5 in ma_periods and 20 in ma_periods:
            df['sma_5_20_cross'] = df['sma_5'] - df['sma_20']
        if 20 in ma_periods and 50 in ma_periods:
            df['sma_20_50_cross'] = df['sma_20'] - df['sma_50']
        if 50 in ma_periods and 200 in ma_periods:
            df['sma_50_200_cross'] = df['sma_50'] - df['sma_200']
        
        # RSI
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = TechnicalIndicators.rsi(close, period)
        
        # Bollinger Bands
        for period in [10, 20]:
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close, period, 2)
            df[f'bb_upper_{period}'] = bb_upper
            df[f'bb_middle_{period}'] = bb_middle
            df[f'bb_lower_{period}'] = bb_lower
            df[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
            df[f'bb_position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram
        
        # ATR
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = TechnicalIndicators.atr(high, low, close, period)
            df[f'atr_{period}_pct'] = df[f'atr_{period}'] / close
        
        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic_oscillator(high, low, close)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        close = df['close']
        returns = df['returns']
        
        for period in self.lookback_periods:
            # Volatility
            df[f'volatility_{period}'] = returns.rolling(period).std()
            
            # Rolling statistics
            df[f'mean_{period}'] = close.rolling(period).mean()
            df[f'std_{period}'] = close.rolling(period).std()
            df[f'min_{period}'] = close.rolling(period).min()
            df[f'max_{period}'] = close.rolling(period).max()
            
            # Z-score
            df[f'zscore_{period}'] = (close - df[f'mean_{period}']) / df[f'std_{period}']
            
            # Percentile rank
            df[f'percentile_{period}'] = close.rolling(period).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
            )
            
            # Momentum
            df[f'momentum_{period}'] = close / close.shift(period) - 1
            
            # Rate of change
            df[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period)
            
            # Skewness and kurtosis
            if period >= 20:
                df[f'skew_{period}'] = returns.rolling(period).skew()
                df[f'kurt_{period}'] = returns.rolling(period).kurt()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Day of week (0=Monday, 4=Friday)
        df['day_of_week'] = df.index.dayofweek.astype(np.int32)
        df['is_monday'] = (df['day_of_week'] == 0).astype(np.int32)
        df['is_friday'] = (df['day_of_week'] == 4).astype(np.int32)
        
        # Month
        df['month'] = df.index.month.astype(np.int32)
        df['is_january'] = (df['month'] == 1).astype(np.int32)
        df['is_december'] = (df['month'] == 12).astype(np.int32)
        
        # Quarter
        df['quarter'] = df.index.quarter.astype(np.int32)
        
        # Day of month
        df['day_of_month'] = df.index.day.astype(np.int32)
        df['is_month_start'] = df.index.is_month_start.astype(np.int32)
        df['is_month_end'] = df.index.is_month_end.astype(np.int32)
        
        # Week of year - convert to numpy array first to avoid object dtype
        df['week_of_year'] = np.array(df.index.isocalendar().week, dtype=np.int32)
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        # Key features to lag
        lag_features = ['returns', 'volume_change', 'rsi_14', 'macd_hist']
        lag_features = [f for f in lag_features if f in df.columns]
        
        for feature in lag_features:
            for lag in [1, 2, 3, 5, 10]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def create_target(
        self,
        df: pd.DataFrame,
        target_type: str = 'classification',
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Create target variable for ML model.
        
        Args:
            df: DataFrame with features
            target_type: 'classification' or 'regression'
            threshold: Threshold for classification (default 0.0)
            
        Returns:
            Series with target values
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        # Calculate forward returns
        forward_returns = df['close'].shift(-self.forward_periods) / df['close'] - 1
        
        if target_type == 'classification':
            # Binary classification: 1 if return > threshold, 0 otherwise
            target = (forward_returns > threshold).astype(int)
        elif target_type == 'regression':
            # Regression: predict actual returns
            target = forward_returns
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        return target
    
    def create_training_data(
        self,
        df: pd.DataFrame,
        target_type: str = 'classification',
        threshold: float = 0.0,
        drop_na: bool = True
    ) -> tuple:
        """
        Create training data (X, y) from features.
        
        Args:
            df: DataFrame with features
            target_type: 'classification' or 'regression'
            threshold: Threshold for classification
            drop_na: Drop rows with NaN values
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Create target
        y = self.create_target(df, target_type, threshold)
        
        # Get feature columns (exclude original OHLCV and target)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Align X and y
        X = X.loc[y.index]
        
        if drop_na:
            # Drop rows with NaN in either X or y
            valid_idx = X.notna().all(axis=1) & y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names_
    
    def get_feature_importance_names(self, top_n: int = 20) -> List[str]:
        """Get top N most important feature names (after training)."""
        return self.feature_names_[:top_n]

