"""
Model training pipeline with validation and hyperparameter tuning.

Provides:
- Model training with cross-validation
- Hyperparameter optimization
- Walk-forward analysis
- Model evaluation and comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from quant_framework.ml.base_model import BaseMLModel
from quant_framework.ml.preprocessing import DataPreprocessor
import json
from pathlib import Path


class ModelTrainer:
    """
    Train and evaluate ML models with proper validation.
    
    Example:
        trainer = ModelTrainer(model, preprocessor)
        results = trainer.train(X, y, validation_method='time_series_cv')
        trainer.save_results('results.json')
    """
    
    def __init__(
        self,
        model: BaseMLModel,
        preprocessor: Optional[DataPreprocessor] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model: ML model to train
            preprocessor: Data preprocessor
        """
        self.model = model
        self.preprocessor = preprocessor or DataPreprocessor()
        self.results = {}
        self.best_params = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_method: str = 'holdout',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train model with validation.
        
        Args:
            X: Features
            y: Target
            validation_method: 'holdout', 'time_series_cv', or 'walk_forward'
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        if validation_method == 'holdout':
            return self._train_holdout(X, y, **kwargs)
        elif validation_method == 'time_series_cv':
            return self._train_time_series_cv(X, y, **kwargs)
        elif validation_method == 'walk_forward':
            return self._train_walk_forward(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown validation method: {validation_method}")
    
    def _train_holdout(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """Train with simple holdout validation."""
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split(
            X, y, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_test_scaled = self.preprocessor.transform(X_test)
        
        # Train model
        print(f"Training {self.model.name}...")
        self.model.fit(X_train_scaled, y_train, **kwargs)
        
        # Evaluate
        train_metrics = self.model.evaluate(X_train_scaled, y_train)
        test_metrics = self.model.evaluate(X_test_scaled, y_test)
        
        # Store results
        self.results = {
            'model_name': self.model.name,
            'validation_method': 'holdout',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_names': list(X.columns)
        }
        
        # Add feature importance if available
        if self.model.feature_importance_ is not None:
            self.results['feature_importance'] = self.model.get_feature_importance(top_n=20).to_dict()
        
        print(f"✓ Training complete!")
        print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
        
        return self.results
    
    def _train_time_series_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Train with time series cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        print(f"Training {self.model.name} with {n_splits}-fold time series CV...")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            print(f"  Fold {fold}/{n_splits}...")
            
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Scale
            preprocessor = DataPreprocessor(scaler_type=self.preprocessor.scaler_type)
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            
            # Train
            model_copy = self.model.__class__(**self.model.params)
            model_copy.fit(X_train_scaled, y_train, **kwargs)
            
            # Evaluate
            train_metrics = model_copy.evaluate(X_train_scaled, y_train)
            test_metrics = model_copy.evaluate(X_test_scaled, y_test)
            
            fold_results.append({
                'fold': fold,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            })
        
        # Aggregate results
        avg_train_metrics = self._average_metrics([f['train_metrics'] for f in fold_results])
        avg_test_metrics = self._average_metrics([f['test_metrics'] for f in fold_results])
        
        # Train final model on all data
        X_scaled = self.preprocessor.fit_transform(X)
        self.model.fit(X_scaled, y, **kwargs)
        
        self.results = {
            'model_name': self.model.name,
            'validation_method': 'time_series_cv',
            'n_splits': n_splits,
            'avg_train_metrics': avg_train_metrics,
            'avg_test_metrics': avg_test_metrics,
            'fold_results': fold_results,
            'feature_names': list(X.columns)
        }
        
        if self.model.feature_importance_ is not None:
            self.results['feature_importance'] = self.model.get_feature_importance(top_n=20).to_dict()
        
        print(f"✓ Training complete!")
        print(f"  Avg Train Accuracy: {avg_train_metrics['accuracy']:.4f}")
        print(f"  Avg Test Accuracy:  {avg_test_metrics['accuracy']:.4f}")
        
        return self.results
    
    def _train_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 21,
        **kwargs
    ) -> Dict[str, Any]:
        """Train with walk-forward analysis."""
        splits = self.preprocessor.walk_forward_splits(
            X, y, train_size, test_size, step_size
        )
        
        window_results = []
        
        print(f"Training {self.model.name} with walk-forward analysis...")
        print(f"  Windows: {len(splits)}")
        print(f"  Train size: {train_size}, Test size: {test_size}, Step: {step_size}")
        
        for window, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"  Window {window}/{len(splits)}...")
            
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Scale
            preprocessor = DataPreprocessor(scaler_type=self.preprocessor.scaler_type)
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            
            # Train
            model_copy = self.model.__class__(**self.model.params)
            model_copy.fit(X_train_scaled, y_train, **kwargs)
            
            # Evaluate
            test_metrics = model_copy.evaluate(X_test_scaled, y_test)
            
            window_results.append({
                'window': window,
                'test_metrics': test_metrics,
                'train_start': X_train.index[0],
                'train_end': X_train.index[-1],
                'test_start': X_test.index[0],
                'test_end': X_test.index[-1]
            })
        
        # Aggregate
        avg_test_metrics = self._average_metrics([w['test_metrics'] for w in window_results])
        
        # Train final model on all data
        X_scaled = self.preprocessor.fit_transform(X)
        self.model.fit(X_scaled, y, **kwargs)
        
        self.results = {
            'model_name': self.model.name,
            'validation_method': 'walk_forward',
            'n_windows': len(splits),
            'train_size': train_size,
            'test_size': test_size,
            'step_size': step_size,
            'avg_test_metrics': avg_test_metrics,
            'window_results': window_results,
            'feature_names': list(X.columns)
        }
        
        if self.model.feature_importance_ is not None:
            self.results['feature_importance'] = self.model.get_feature_importance(top_n=20).to_dict()
        
        print(f"✓ Training complete!")
        print(f"  Avg Test Accuracy: {avg_test_metrics['accuracy']:.4f}")
        
        return self.results
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Average metrics across folds/windows."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def hyperparameter_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List],
        method: str = 'grid',
        cv_splits: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            X: Features
            y: Target
            param_grid: Dictionary of parameters to search
            method: 'grid' or 'random'
            cv_splits: Number of CV splits
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with best parameters and results
        """
        if method == 'grid':
            return self._grid_search(X, y, param_grid, cv_splits, **kwargs)
        elif method == 'random':
            return self._random_search(X, y, param_grid, cv_splits, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List],
        cv_splits: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Grid search over parameter space."""
        from itertools import product
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        print(f"Grid search: {len(param_combinations)} combinations")
        
        results = []
        
        for i, params in enumerate(param_combinations, 1):
            param_dict = dict(zip(param_names, params))
            print(f"\n[{i}/{len(param_combinations)}] Testing: {param_dict}")
            
            # Create model with these parameters
            model = self.model.__class__(**param_dict)
            trainer = ModelTrainer(model, self.preprocessor)
            
            # Train with CV
            result = trainer.train(
                X, y,
                validation_method='time_series_cv',
                n_splits=cv_splits,
                **kwargs
            )
            
            results.append({
                'params': param_dict,
                'accuracy': result['avg_test_metrics']['accuracy'],
                'metrics': result['avg_test_metrics']
            })
        
        # Find best
        best_result = max(results, key=lambda x: x['accuracy'])
        self.best_params = best_result['params']
        
        print(f"\n✓ Grid search complete!")
        print(f"  Best params: {self.best_params}")
        print(f"  Best accuracy: {best_result['accuracy']:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_accuracy': best_result['accuracy'],
            'best_metrics': best_result['metrics'],
            'all_results': results
        }
    
    def save_results(self, filepath: str):
        """Save training results to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results_serializable = convert_types(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"✓ Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load training results from JSON."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        print(f"✓ Results loaded from {filepath}")


class ModelComparator:
    """
    Compare multiple ML models.
    
    Example:
        comparator = ModelComparator()
        comparator.add_model(rf_model, 'RandomForest')
        comparator.add_model(xgb_model, 'XGBoost')
        results = comparator.compare(X, y)
    """
    
    def __init__(self):
        """Initialize model comparator."""
        self.models = {}
        self.results = {}
    
    def add_model(self, model: BaseMLModel, name: str):
        """Add a model to compare."""
        self.models[name] = model
    
    def compare(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_method: str = 'time_series_cv',
        **kwargs
    ) -> pd.DataFrame:
        """
        Compare all models.
        
        Args:
            X: Features
            y: Target
            validation_method: Validation method
            **kwargs: Additional training parameters
            
        Returns:
            DataFrame with comparison results
        """
        print(f"Comparing {len(self.models)} models...\n")
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Model: {name}")
            print('='*60)
            
            trainer = ModelTrainer(model)
            result = trainer.train(X, y, validation_method=validation_method, **kwargs)
            
            self.results[name] = result
        
        # Create comparison DataFrame
        comparison_data = []
        
        for name, result in self.results.items():
            if validation_method == 'holdout':
                metrics = result['test_metrics']
            else:
                metrics = result['avg_test_metrics']
            
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics.get('roc_auc', np.nan)
            })
        
        df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
        
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print('='*60)
        print(df.to_string(index=False))
        
        return df


