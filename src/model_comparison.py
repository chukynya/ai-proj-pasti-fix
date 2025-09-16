"""
Model comparison utilities for the AI Disease Prediction Project.

This module implements various classification algorithms and provides
comprehensive comparison with focus on recall score optimization.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc, recall_score,
    precision_score, f1_score, accuracy_score
)
import warnings

warnings.filterwarnings('ignore')


class ModelComparison:
    """
    A class for comparing multiple classification models with focus on recall optimization.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model comparison class.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = self._initialize_models()
        self.results = {}
        self.trained_models = {}
        
    def _initialize_models(self):
        """
        Initialize all classification models to be compared.
        
        Returns:
            dict: Dictionary of model name to model object
        """
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state, eval_metric='logloss'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'SVM': SVC(
                random_state=self.random_state, probability=True
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'AdaBoost': AdaBoostClassifier(
                random_state=self.random_state
            )
        }
        return models
    
    def cross_validate_models(self, X_train, y_train, cv_folds=5):
        """
        Perform cross-validation for all models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (array-like): Training target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            pd.DataFrame: Cross-validation results
        """
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        print("Performing cross-validation...")
        print("-" * 50)
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Cross-validation for different metrics
            accuracy_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
            precision_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='precision')
            recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall')
            f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
            
            cv_results[name] = {
                'Accuracy': {
                    'mean': accuracy_scores.mean(),
                    'std': accuracy_scores.std()
                },
                'Precision': {
                    'mean': precision_scores.mean(),
                    'std': precision_scores.std()
                },
                'Recall': {
                    'mean': recall_scores.mean(),
                    'std': recall_scores.std()
                },
                'F1-Score': {
                    'mean': f1_scores.mean(),
                    'std': f1_scores.std()
                }
            }
            
            print(f"  Recall: {recall_scores.mean():.4f} (+/- {recall_scores.std() * 2:.4f})")
        
        # Convert to DataFrame for easier analysis
        cv_df = pd.DataFrame(cv_results).T
        cv_df = cv_df.round(4)
        
        return cv_df
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate on test set.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            y_train (array-like): Training target
            y_test (array-like): Test target
            
        Returns:
            pd.DataFrame: Evaluation results
        """
        results = {}
        
        print("Training and evaluating models...")
        print("-" * 50)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # ROC AUC (if probability predictions available)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            }
            
            print(f"  Recall: {recall:.4f}")
            
        # Store results
        self.results = results
        results_df = pd.DataFrame(results).T.round(4)
        
        return results_df
    
    def get_best_model_by_recall(self):
        """
        Get the best performing model based on recall score.
        
        Returns:
            tuple: (model_name, model_object, recall_score)
        """
        if not self.results:
            print("No results available. Please run train_and_evaluate_models first.")
            return None, None, None
        
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['Recall'])
        best_model = self.trained_models[best_model_name]
        best_recall = self.results[best_model_name]['Recall']
        
        return best_model_name, best_model, best_recall
    
    def get_detailed_classification_report(self, X_test, y_test, model_name=None):
        """
        Get detailed classification report for a specific model or the best model.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (array-like): Test target
            model_name (str): Name of the model to analyze
            
        Returns:
            dict: Detailed classification metrics
        """
        if model_name is None:
            model_name, _, _ = self.get_best_model_by_recall()
        
        if model_name not in self.trained_models:
            print(f"Model {model_name} not found in trained models.")
            return None
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        detailed_report = {
            'model_name': model_name,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'specificity': specificity,
            'sensitivity': sensitivity,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        return detailed_report
    
    def compare_models(self, X_train, X_test, y_train, y_test, cv_folds=5):
        """
        Complete model comparison pipeline.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            y_train (array-like): Training target
            y_test (array-like): Test target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Complete comparison results
        """
        print("Starting comprehensive model comparison...")
        print("=" * 60)
        
        # Cross-validation results
        cv_results = self.cross_validate_models(X_train, y_train, cv_folds)
        
        print("\nCross-Validation Results (Mean ± Std):")
        print("-" * 50)
        for model in cv_results.index:
            recall_mean = cv_results.loc[model, ('Recall', 'mean')]
            recall_std = cv_results.loc[model, ('Recall', 'std')]
            print(f"{model:20}: Recall = {recall_mean:.4f} ± {recall_std:.4f}")
        
        # Test set evaluation
        test_results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        print("\nTest Set Results:")
        print("-" * 30)
        for model in test_results.index:
            recall_score = test_results.loc[model, 'Recall']
            print(f"{model:20}: Recall = {recall_score:.4f}")
        
        # Best model information
        best_model_name, best_model, best_recall = self.get_best_model_by_recall()
        
        print(f"\nBest Model by Recall Score:")
        print(f"Model: {best_model_name}")
        print(f"Recall Score: {best_recall:.4f}")
        
        # Detailed report for best model
        detailed_report = self.get_detailed_classification_report(X_test, y_test, best_model_name)
        
        return {
            'cv_results': cv_results,
            'test_results': test_results,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'best_recall': best_recall,
            'detailed_report': detailed_report,
            'trained_models': self.trained_models
        }


def compare_classification_models(X_train, X_test, y_train, y_test, cv_folds=5, random_state=42):
    """
    Convenience function to compare classification models.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (array-like): Training target
        y_test (array-like): Test target
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random seed
        
    Returns:
        dict: Comparison results
    """
    comparator = ModelComparison(random_state=random_state)
    return comparator.compare_models(X_train, X_test, y_train, y_test, cv_folds)


if __name__ == "__main__":
    # Example usage
    print("Model Comparison Module")
    print("======================")
    print("This module provides utilities for comparing multiple classification models.")
    print("Main functions:")
    print("- ModelComparison class for comprehensive model evaluation")
    print("- compare_classification_models() for quick model comparison")
    print("- Focus on recall score optimization for medical diagnosis")