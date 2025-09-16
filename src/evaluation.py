"""
Evaluation utilities for the AI Disease Prediction Project.

This module provides comprehensive evaluation metrics and analysis tools
with focus on recall score and medical diagnosis requirements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc, recall_score, precision_score, f1_score,
    accuracy_score, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    A comprehensive model evaluation class focused on medical diagnosis metrics.
    """
    
    def __init__(self, model, model_name="Model"):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained scikit-learn model
            model_name (str): Name of the model for reporting
        """
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}
        
    def evaluate_classification(self, X_test, y_test, class_names=None):
        """
        Comprehensive classification evaluation.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (array-like): Test target
            class_names (list): Names of classes
            
        Returns:
            dict: Comprehensive evaluation results
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            y_pred_proba = self.model.decision_function(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # ROC AUC and PR AUC
        roc_auc = None
        pr_auc = None
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Additional medical metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        ppv = precision  # Positive Predictive Value (same as precision)
        
        # Medical interpretation metrics
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=class_names or ['Healthy', 'Disease'],
            output_dict=True
        )
        
        results = {
            'model_name': self.model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'npv': npv,
            'ppv': ppv,
            'false_negative_rate': false_negative_rate,
            'false_positive_rate': false_positive_rate,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        self.evaluation_results = results
        return results
    
    def print_evaluation_summary(self):
        """
        Print a comprehensive evaluation summary.
        """
        if not self.evaluation_results:
            print("No evaluation results available. Please run evaluate_classification first.")
            return
        
        results = self.evaluation_results
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY: {results['model_name']}")
        print(f"{'='*60}")
        
        print(f"\nOVERALL PERFORMANCE METRICS:")
        print(f"-" * 30)
        print(f"Accuracy:      {results['accuracy']:.4f}")
        print(f"Precision:     {results['precision']:.4f}")
        print(f"Recall:        {results['recall']:.4f}")
        print(f"F1-Score:      {results['f1_score']:.4f}")
        
        if results['roc_auc']:
            print(f"ROC AUC:       {results['roc_auc']:.4f}")
        if results['pr_auc']:
            print(f"PR AUC:        {results['pr_auc']:.4f}")
        
        print(f"\nMEDICAL DIAGNOSIS METRICS:")
        print(f"-" * 30)
        print(f"Sensitivity:   {results['sensitivity']:.4f} (True Positive Rate)")
        print(f"Specificity:   {results['specificity']:.4f} (True Negative Rate)")
        print(f"PPV:           {results['ppv']:.4f} (Positive Predictive Value)")
        print(f"NPV:           {results['npv']:.4f} (Negative Predictive Value)")
        
        print(f"\nERROR RATES:")
        print(f"-" * 15)
        print(f"False Negative Rate: {results['false_negative_rate']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        
        print(f"\nCONFUSION MATRIX:")
        print(f"-" * 17)
        cm = results['confusion_matrix']
        print(f"                 Predicted")
        print(f"                 Healthy  Disease")
        print(f"Actual Healthy   {cm[0,0]:7d}  {cm[0,1]:7d}")
        print(f"       Disease   {cm[1,0]:7d}  {cm[1,1]:7d}")
        
        print(f"\nDETAILED COUNTS:")
        print(f"-" * 16)
        print(f"True Negatives:   {results['true_negatives']:4d}")
        print(f"False Positives:  {results['false_positives']:4d}")
        print(f"False Negatives:  {results['false_negatives']:4d}")
        print(f"True Positives:   {results['true_positives']:4d}")
        
        # Medical interpretation
        print(f"\nMEDICAL INTERPRETATION:")
        print(f"-" * 23)
        if results['recall'] >= 0.9:
            print("✓ HIGH RECALL: Model catches most disease cases (good for screening)")
        elif results['recall'] >= 0.8:
            print("○ MODERATE RECALL: Model catches most disease cases")
        else:
            print("⚠ LOW RECALL: Model may miss disease cases (concerning for diagnosis)")
            
        if results['precision'] >= 0.8:
            print("✓ HIGH PRECISION: Few false alarms")
        elif results['precision'] >= 0.6:
            print("○ MODERATE PRECISION: Some false alarms")
        else:
            print("⚠ LOW PRECISION: Many false alarms")
    
    def get_feature_importance(self, feature_names=None, top_n=10):
        """
        Get feature importance from the model.
        
        Args:
            feature_names (list): Names of features
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance rankings
        """
        importance = None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            print("Model doesn't support feature importance extraction.")
            return None
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def calculate_cost_benefit_analysis(self, cost_fn=100, cost_fp=10, benefit_tp=500, benefit_tn=1):
        """
        Calculate cost-benefit analysis for medical diagnosis.
        
        Args:
            cost_fn (float): Cost of false negative (missing disease)
            cost_fp (float): Cost of false positive (false alarm)
            benefit_tp (float): Benefit of true positive (correct diagnosis)
            benefit_tn (float): Benefit of true negative (correct healthy)
            
        Returns:
            dict: Cost-benefit analysis results
        """
        if not self.evaluation_results:
            print("No evaluation results available.")
            return None
        
        results = self.evaluation_results
        tn, fp, fn, tp = (results['true_negatives'], results['false_positives'],
                         results['false_negatives'], results['true_positives'])
        
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        total_benefit = (tp * benefit_tp) + (tn * benefit_tn)
        net_benefit = total_benefit - total_cost
        
        cost_benefit_results = {
            'total_cost': total_cost,
            'total_benefit': total_benefit,
            'net_benefit': net_benefit,
            'cost_per_case': total_cost / (tp + tn + fp + fn),
            'benefit_per_case': total_benefit / (tp + tn + fp + fn),
            'net_benefit_per_case': net_benefit / (tp + tn + fp + fn)
        }
        
        return cost_benefit_results
    
    def generate_learning_curve(self, X_train, y_train, cv=5, train_sizes=None):
        """
        Generate learning curve data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (array-like): Training target
            cv (int): Number of CV folds
            train_sizes (array-like): Training set sizes to use
            
        Returns:
            dict: Learning curve data
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X_train, y_train, 
            cv=cv, train_sizes=train_sizes,
            scoring='recall', n_jobs=-1
        )
        
        return {
            'train_sizes': train_sizes,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }


def evaluate_multiple_models(models_dict, X_test, y_test, class_names=None):
    """
    Evaluate multiple models and compare their performance.
    
    Args:
        models_dict (dict): Dictionary of model_name -> model_object
        X_test (pd.DataFrame): Test features
        y_test (array-like): Test target
        class_names (list): Names of classes
        
    Returns:
        pd.DataFrame: Comparison of model performances
    """
    results = []
    
    print("Evaluating multiple models...")
    print("-" * 40)
    
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}...")
        
        evaluator = ModelEvaluator(model, model_name)
        evaluation = evaluator.evaluate_classification(X_test, y_test, class_names)
        
        # Extract key metrics
        result_row = {
            'Model': model_name,
            'Accuracy': evaluation['accuracy'],
            'Precision': evaluation['precision'],
            'Recall': evaluation['recall'],
            'F1-Score': evaluation['f1_score'],
            'Specificity': evaluation['specificity'],
            'ROC-AUC': evaluation['roc_auc'],
            'PR-AUC': evaluation['pr_auc'],
            'False Negative Rate': evaluation['false_negative_rate'],
            'False Positive Rate': evaluation['false_positive_rate']
        }
        
        results.append(result_row)
        
        # Print brief summary
        print(f"  Recall: {evaluation['recall']:.4f}")
        print(f"  Precision: {evaluation['precision']:.4f}")
        print(f"  F1-Score: {evaluation['f1_score']:.4f}")
        print()
    
    results_df = pd.DataFrame(results).round(4)
    
    # Sort by recall (primary metric for medical diagnosis)
    results_df = results_df.sort_values('Recall', ascending=False)
    
    print("MODELS RANKED BY RECALL SCORE:")
    print("=" * 35)
    for idx, row in results_df.iterrows():
        print(f"{row['Model']:20}: Recall = {row['Recall']:.4f}")
    
    return results_df


def create_evaluation_report(model, X_test, y_test, model_name="Model", 
                           feature_names=None, class_names=None):
    """
    Create a comprehensive evaluation report for a single model.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (array-like): Test target
        model_name (str): Name of the model
        feature_names (list): Names of features
        class_names (list): Names of classes
        
    Returns:
        dict: Comprehensive evaluation report
    """
    evaluator = ModelEvaluator(model, model_name)
    
    # Main evaluation
    evaluation = evaluator.evaluate_classification(X_test, y_test, class_names)
    
    # Feature importance
    feature_importance = evaluator.get_feature_importance(feature_names)
    
    # Cost-benefit analysis
    cost_benefit = evaluator.calculate_cost_benefit_analysis()
    
    # Print summary
    evaluator.print_evaluation_summary()
    
    return {
        'evaluation_metrics': evaluation,
        'feature_importance': feature_importance,
        'cost_benefit_analysis': cost_benefit,
        'evaluator': evaluator
    }


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("======================")
    print("This module provides comprehensive model evaluation utilities.")
    print("Main functions:")
    print("- ModelEvaluator class for detailed single model evaluation")
    print("- evaluate_multiple_models() for comparing multiple models")
    print("- create_evaluation_report() for comprehensive model analysis")
    print("- Focus on medical diagnosis metrics including recall optimization")