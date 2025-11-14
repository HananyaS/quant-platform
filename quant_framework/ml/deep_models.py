"""
Deep Learning models for time series prediction.

Implements:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- CNN (Convolutional Neural Network)
- Transformer
- Hybrid models
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from quant_framework.ml.base_model import BaseMLModel


class LSTMClassifier(BaseMLModel):
    """
    LSTM classifier for time series prediction.
    
    Uses Long Short-Term Memory networks to capture
    temporal dependencies in trading data.
    
    Example:
        model = LSTMClassifier(sequence_length=20, units=50, layers=2)
        model.fit(X_train, y_train, epochs=50, batch_size=32)
        predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        units: int = 50,
        layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """
        Initialize LSTM classifier.
        
        Args:
            sequence_length: Length of input sequences
            units: Number of LSTM units per layer
            layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            random_state: Random seed
        """
        super().__init__(name="LSTMClassifier")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.params = {
            'units': units,
            'layers': layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'random_state': random_state
        }
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
        self.scaler = None
    
    def _build_model(self, input_shape: Tuple, n_classes: int):
        """Build LSTM model architecture."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.params['units'],
            return_sequences=self.params['layers'] > 1,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(self.params['dropout']))
        
        # Additional LSTM layers
        for i in range(1, self.params['layers']):
            return_seq = i < self.params['layers'] - 1
            model.add(layers.LSTM(self.params['units'], return_sequences=return_seq))
            model.add(layers.Dropout(self.params['dropout']))
        
        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=keras.optimizers.Adam(self.params['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        else:
            model.add(layers.Dense(n_classes, activation='softmax'))
            model.compile(
                optimizer=keras.optimizers.Adam(self.params['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Create sequences for LSTM input."""
        n_samples = len(X) - self.sequence_length + 1
        n_features = X.shape[1]
        
        X_seq = np.zeros((n_samples, self.sequence_length, n_features))
        
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.sequence_length]
        
        if y is not None:
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq
        
        return X_seq
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
        **kwargs
    ) -> 'LSTMClassifier':
        """
        Train LSTM model.
        
        Args:
            X: Training features
            y: Training target
            validation_split: Fraction for validation
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity mode
            **kwargs: Additional training parameters
        """
        from tensorflow import keras
        
        self.feature_names_ = list(X.columns)
        
        # Convert to numpy and create sequences
        X_np = X.values
        y_np = y.values
        
        X_seq, y_seq = self._create_sequences(X_np, y_np)
        
        # Build model
        n_classes = len(np.unique(y_seq))
        input_shape = (self.sequence_length, X_np.shape[1])
        self.model = self._build_model(input_shape, n_classes)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_history_ = history.history
        self.is_trained = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_np = X.values
        X_seq = self._create_sequences(X_np)
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        if predictions.shape[1] == 1:
            # Binary classification
            predictions = (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class
            predictions = np.argmax(predictions, axis=1)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_np = X.values
        X_seq = self._create_sequences(X_np)
        
        probas = self.model.predict(X_seq, verbose=0)
        
        if probas.shape[1] == 1:
            # Binary classification - add complementary probability
            probas = np.hstack([1 - probas, probas])
        
        return probas


class GRUClassifier(BaseMLModel):
    """
    GRU classifier for time series prediction.
    
    Uses Gated Recurrent Units - faster than LSTM but similar performance.
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        units: int = 50,
        layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """Initialize GRU classifier."""
        super().__init__(name="GRUClassifier")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.params = {
            'units': units,
            'layers': layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'random_state': random_state
        }
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
    
    def _build_model(self, input_shape: Tuple, n_classes: int):
        """Build GRU model architecture."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential()
        
        # First GRU layer
        model.add(layers.GRU(
            self.params['units'],
            return_sequences=self.params['layers'] > 1,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(self.params['dropout']))
        
        # Additional GRU layers
        for i in range(1, self.params['layers']):
            return_seq = i < self.params['layers'] - 1
            model.add(layers.GRU(self.params['units'], return_sequences=return_seq))
            model.add(layers.Dropout(self.params['dropout']))
        
        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=keras.optimizers.Adam(self.params['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        else:
            model.add(layers.Dense(n_classes, activation='softmax'))
            model.compile(
                optimizer=keras.optimizers.Adam(self.params['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Create sequences for GRU input."""
        n_samples = len(X) - self.sequence_length + 1
        n_features = X.shape[1]
        
        X_seq = np.zeros((n_samples, self.sequence_length, n_features))
        
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.sequence_length]
        
        if y is not None:
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq
        
        return X_seq
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
        **kwargs
    ) -> 'GRUClassifier':
        """Train GRU model."""
        from tensorflow import keras
        
        self.feature_names_ = list(X.columns)
        
        X_np = X.values
        y_np = y.values
        
        X_seq, y_seq = self._create_sequences(X_np, y_np)
        
        n_classes = len(np.unique(y_seq))
        input_shape = (self.sequence_length, X_np.shape[1])
        self.model = self._build_model(input_shape, n_classes)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_history_ = history.history
        self.is_trained = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_np = X.values
        X_seq = self._create_sequences(X_np)
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        if predictions.shape[1] == 1:
            predictions = (predictions > 0.5).astype(int).flatten()
        else:
            predictions = np.argmax(predictions, axis=1)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_np = X.values
        X_seq = self._create_sequences(X_np)
        
        probas = self.model.predict(X_seq, verbose=0)
        
        if probas.shape[1] == 1:
            probas = np.hstack([1 - probas, probas])
        
        return probas


class CNNClassifier(BaseMLModel):
    """
    1D CNN classifier for time series.
    
    Uses convolutional layers to extract patterns from sequences.
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        filters: int = 64,
        kernel_size: int = 3,
        num_conv_layers: int = 2,
        dense_units: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """Initialize CNN classifier."""
        super().__init__(name="CNNClassifier")
        
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed")
        
        self.sequence_length = sequence_length
        self.params = {
            'filters': filters,
            'kernel_size': kernel_size,
            'num_conv_layers': num_conv_layers,
            'dense_units': dense_units,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'random_state': random_state
        }
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
    
    def _build_model(self, input_shape: Tuple, n_classes: int):
        """Build CNN model architecture."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential()
        
        # Conv layers
        for i in range(self.params['num_conv_layers']):
            if i == 0:
                model.add(layers.Conv1D(
                    self.params['filters'],
                    self.params['kernel_size'],
                    activation='relu',
                    input_shape=input_shape
                ))
            else:
                model.add(layers.Conv1D(
                    self.params['filters'],
                    self.params['kernel_size'],
                    activation='relu'
                ))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Dropout(self.params['dropout']))
        
        # Flatten and dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(self.params['dense_units'], activation='relu'))
        model.add(layers.Dropout(self.params['dropout']))
        
        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=keras.optimizers.Adam(self.params['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        else:
            model.add(layers.Dense(n_classes, activation='softmax'))
            model.compile(
                optimizer=keras.optimizers.Adam(self.params['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Create sequences."""
        n_samples = len(X) - self.sequence_length + 1
        n_features = X.shape[1]
        
        X_seq = np.zeros((n_samples, self.sequence_length, n_features))
        
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.sequence_length]
        
        if y is not None:
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq
        
        return X_seq
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
        **kwargs
    ) -> 'CNNClassifier':
        """Train CNN model."""
        from tensorflow import keras
        
        self.feature_names_ = list(X.columns)
        
        X_np = X.values
        y_np = y.values
        
        X_seq, y_seq = self._create_sequences(X_np, y_np)
        
        n_classes = len(np.unique(y_seq))
        input_shape = (self.sequence_length, X_np.shape[1])
        self.model = self._build_model(input_shape, n_classes)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_history_ = history.history
        self.is_trained = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_np = X.values
        X_seq = self._create_sequences(X_np)
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        if predictions.shape[1] == 1:
            predictions = (predictions > 0.5).astype(int).flatten()
        else:
            predictions = np.argmax(predictions, axis=1)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_np = X.values
        X_seq = self._create_sequences(X_np)
        
        probas = self.model.predict(X_seq, verbose=0)
        
        if probas.shape[1] == 1:
            probas = np.hstack([1 - probas, probas])
        
        return probas


