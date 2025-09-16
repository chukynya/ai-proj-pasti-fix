"""
Hyperparameter tuning utilities for the AI Disease Prediction Project.

This module provides automated hyperparameter tuning with focus on optimizing
recall score for the best performing classification models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, recall_score, f1_score
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    A class for hyperparameter tuning with focus on recall optimization.
    """
    
    def __init__(self, random_state=42, n_jobs=-1):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            random_state (int): Random seed for reproducibility
            n_jobs (int): Number of parallel jobs
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_models = {}
        self.tuning_results = {}
        
    def get_param_grids(self):
        """
        Get parameter grids for different models.
        
        Returns:
            dict: Parameter grids for each model type
        """
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            },
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 3000]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'degree': [2, 3, 4]  # Only relevant for poly kernel
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        return param_grids
    
    def tune_with_grid_search(self, model_type, X_train, y_train, cv_folds=5, 
                             scoring='recall', n_jobs=None):
        """
        Tune hyperparameters using Grid Search CV.
        
        Args:
            model_type (str): Type of model to tune
            X_train (pd.DataFrame): Training features
            y_train (array-like): Training target
            cv_folds (int): Number of CV folds
            scoring (str): Scoring metric to optimize
            n_jobs (int): Number of parallel jobs
            
        Returns:
            sklearn estimator: Best model after tuning
        """
        n_jobs = n_jobs or self.n_jobs
        param_grids = self.get_param_grids()
        
        if model_type not in param_grids:
            raise ValueError(f"Model type {model_type} not supported")
        
        # Initialize model
        if model_type == 'RandomForest':
            model = RandomForestClassifier(random_state=self.random_state)
        elif model_type == 'XGBoost':
            model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(random_state=self.random_state)
        elif model_type == 'SVM':
            model = SVC(random_state=self.random_state, probability=True)
        elif model_type == 'GradientBoosting':
            model = GradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Model type {model_type} not implemented")
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Grid search
        print(f"Starting Grid Search for {model_type}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_type],
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best {scoring} score: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Store results
        self.best_models[model_type] = grid_search.best_estimator_
        self.tuning_results[model_type] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }
        
        return grid_search.best_estimator_
    
    def tune_with_random_search(self, model_type, X_train, y_train, cv_folds=5,
                               scoring='recall', n_iter=50, n_jobs=None):
        """
        Tune hyperparameters using Random Search CV.
        
        Args:
            model_type (str): Type of model to tune
            X_train (pd.DataFrame): Training features
            y_train (array-like): Training target
            cv_folds (int): Number of CV folds
            scoring (str): Scoring metric to optimize
            n_iter (int): Number of parameter settings sampled
            n_jobs (int): Number of parallel jobs
            
        Returns:
            sklearn estimator: Best model after tuning
        """
        n_jobs = n_jobs or self.n_jobs
        param_grids = self.get_param_grids()
        
        if model_type not in param_grids:
            raise ValueError(f"Model type {model_type} not supported")
        
        # Initialize model
        if model_type == 'RandomForest':
            model = RandomForestClassifier(random_state=self.random_state)
        elif model_type == 'XGBoost':
            model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(random_state=self.random_state)
        elif model_type == 'SVM':
            model = SVC(random_state=self.random_state, probability=True)
        elif model_type == 'GradientBoosting':
            model = GradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Model type {model_type} not implemented")
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Random search
        print(f"Starting Random Search for {model_type}...")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[model_type],
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best {scoring} score: {random_search.best_score_:.4f}")
        print(f"Best parameters: {random_search.best_params_}")
        
        # Store results
        self.best_models[model_type] = random_search.best_estimator_
        self.tuning_results[model_type] = {
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'cv_results': random_search.cv_results_
        }
        
        return random_search.best_estimator_
    
    def tune_with_optuna(self, model_type, X_train, y_train, cv_folds=5,
                        scoring='recall', n_trials=100):
        """
        Tune hyperparameters using Optuna optimization.
        
        Args:
            model_type (str): Type of model to tune
            X_train (pd.DataFrame): Training features
            y_train (array-like): Training target
            cv_folds (int): Number of CV folds
            scoring (str): Scoring metric to optimize
            n_trials (int): Number of optimization trials
            
        Returns:
            sklearn estimator: Best model after tuning
        """
        def objective(trial):
            if model_type == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': self.random_state
                }
                model = RandomForestClassifier(**params)
                
            elif model_type == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                    'random_state': self.random_state,
                    'eval_metric': 'logloss'
                }
                model = XGBClassifier(**params)
                
            else:
                raise ValueError(f"Optuna optimization not implemented for {model_type}")
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            if scoring == 'recall':
                scorer = make_scorer(recall_score)
            elif scoring == 'f1':
                scorer = make_scorer(f1_score)
            else:
                scorer = scoring
                
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                if scoring == 'recall':
                    score = recall_score(y_val_fold, y_pred)
                elif scoring == 'f1':
                    score = f1_score(y_val_fold, y_pred)
                else:
                    score = scorer(model, X_val_fold, y_val_fold)
                    
                scores.append(score)
            
            return np.mean(scores)
        
        print(f"Starting Optuna optimization for {model_type}...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best {scoring} score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        # Create best model
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        
        if model_type == 'RandomForest':
            best_model = RandomForestClassifier(**best_params)
        elif model_type == 'XGBoost':
            best_params['eval_metric'] = 'logloss'
            best_model = XGBClassifier(**best_params)
        
        # Train on full dataset
        best_model.fit(X_train, y_train)
        
        # Store results
        self.best_models[model_type] = best_model
        self.tuning_results[model_type] = {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'study': study
        }
        
        return best_model
    
    def get_tuning_summary(self):
        """
        Get summary of all tuning results.
        
        Returns:
            pd.DataFrame: Summary of tuning results
        """
        if not self.tuning_results:
            print("No tuning results available.")
            return None
        
        summary_data = {}
        for model_type, results in self.tuning_results.items():
            summary_data[model_type] = {
                'Best Score': results['best_score'],
                'Best Parameters': str(results['best_params'])
            }
        
        return pd.DataFrame(summary_data).T.round(4)


def tune_best_model(X_train, y_train, model_type='RandomForest', method='random_search',
                   cv_folds=5, scoring='recall', random_state=42, **kwargs):
    """
    Convenience function to tune hyperparameters for a specific model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (array-like): Training target
        model_type (str): Type of model to tune
        method (str): Tuning method ('grid_search', 'random_search', 'optuna')
        cv_folds (int): Number of CV folds
        scoring (str): Scoring metric to optimize
        random_state (int): Random seed
        **kwargs: Additional arguments for tuning methods
        
    Returns:
        sklearn estimator: Best tuned model
    """
    tuner = HyperparameterTuner(random_state=random_state)
    
    if method == 'grid_search':
        return tuner.tune_with_grid_search(model_type, X_train, y_train, cv_folds, scoring, **kwargs)
    elif method == 'random_search':
        return tuner.tune_with_random_search(model_type, X_train, y_train, cv_folds, scoring, **kwargs)
    elif method == 'optuna':
        return tuner.tune_with_optuna(model_type, X_train, y_train, cv_folds, scoring, **kwargs)
    else:
        raise ValueError(f"Method {method} not supported. Use 'grid_search', 'random_search', or 'optuna'")


if __name__ == "__main__":
    # Example usage
    print("Hyperparameter Tuning Module")
    print("===========================")
    print("This module provides utilities for hyperparameter tuning with recall optimization.")
    print("Main functions:")
    print("- HyperparameterTuner class for comprehensive tuning")
    print("- tune_best_model() for quick model tuning")
    print("- Support for Grid Search, Random Search, and Optuna optimization")