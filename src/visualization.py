"""
Visualization utilities for the AI Disease Prediction Project.

This module provides comprehensive visualization tools for model performance
analysis, data exploration, and results presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, auc
)
import warnings

warnings.filterwarnings('ignore')

# Set default style
plt.style.use('default')
sns.set_palette("husl")


class ModelVisualizer:
    """
    A comprehensive visualization class for machine learning model analysis.
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (tuple): Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                            title="Confusion Matrix", save_path=None):
        """
        Plot confusion matrix with annotations.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            class_names (list): Names of classes
            title (str): Plot title
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The confusion matrix plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if class_names is None:
            class_names = ['Healthy', 'Disease']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('Actual Label', fontsize=12)
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1%})',
                       ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, models_dict, X_test, y_test, title="ROC Curves Comparison", 
                      save_path=None):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_dict (dict): Dictionary of model_name -> model_object
            X_test (pd.DataFrame): Test features
            y_test (array-like): Test target
            title (str): Plot title
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The ROC curves plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (model_name, model) in enumerate(models_dict.items()):
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
            else:
                continue
                
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            ax.plot(fpr, tpr, color=self.colors[i], lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, models_dict, X_test, y_test, 
                                   title="Precision-Recall Curves", save_path=None):
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            models_dict (dict): Dictionary of model_name -> model_object
            X_test (pd.DataFrame): Test features
            y_test (array-like): Test target
            title (str): Plot title
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The PR curves plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate baseline (random classifier)
        baseline = y_test.sum() / len(y_test)
        
        for i, (model_name, model) in enumerate(models_dict.items()):
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
            else:
                continue
                
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Plot
            ax.plot(recall, precision, color=self.colors[i], lw=2,
                   label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        # Plot baseline
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=2,
                  label=f'Random Classifier (AUC = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, results_df, metrics=None, 
                            title="Model Performance Comparison", save_path=None):
        """
        Plot comparison of multiple models across different metrics.
        
        Args:
            results_df (pd.DataFrame): Results dataframe from evaluate_multiple_models
            metrics (list): List of metrics to include in comparison
            title (str): Plot title
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The comparison plot
        """
        if metrics is None:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        fig, axes = plt.subplots(1, len(available_metrics), 
                                figsize=(4 * len(available_metrics), 6))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Create bar plot
            bars = ax.bar(range(len(results_df)), results_df[metric], 
                         color=self.colors[:len(results_df)], alpha=0.8)
            
            # Customize
            ax.set_xlabel('Models', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(results_df)))
            ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value annotations
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, importance_df, title="Feature Importance", 
                              save_path=None):
        """
        Plot feature importance.
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            title (str): Plot title
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The feature importance plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by importance
        importance_df_sorted = importance_df.sort_values('importance', ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(importance_df_sorted)), 
                      importance_df_sorted['importance'],
                      color=self.colors[0], alpha=0.8)
        
        # Customize
        ax.set_yticks(range(len(importance_df_sorted)))
        ax.set_yticklabels(importance_df_sorted['feature'])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_learning_curve(self, learning_curve_data, title="Learning Curve", 
                          save_path=None):
        """
        Plot learning curve showing training and validation scores.
        
        Args:
            learning_curve_data (dict): Learning curve data from evaluation module
            title (str): Plot title
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The learning curve plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        train_sizes = learning_curve_data['train_sizes']
        train_mean = learning_curve_data['train_scores_mean']
        train_std = learning_curve_data['train_scores_std']
        val_mean = learning_curve_data['val_scores_mean']
        val_std = learning_curve_data['val_scores_std']
        
        # Plot training scores
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.2, color=self.colors[0])
        ax.plot(train_sizes, train_mean, 'o-', color=self.colors[0], 
               label='Training Score')
        
        # Plot validation scores
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.2, color=self.colors[1])
        ax.plot(train_sizes, val_mean, 'o-', color=self.colors[1], 
               label='Validation Score')
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Recall Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, results_df, models_dict, X_test, y_test):
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            results_df (pd.DataFrame): Model comparison results
            models_dict (dict): Dictionary of trained models
            X_test (pd.DataFrame): Test features
            y_test (array-like): Test target
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'ROC Curves',
                          'Precision-Recall Curves', 'Confusion Matrix'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # 1. Model Performance Comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                fig.add_trace(
                    go.Bar(name=metric, x=results_df['Model'], y=results_df[metric],
                          marker_color=px.colors.qualitative.Set3[i]),
                    row=1, col=1
                )
        
        # 2. ROC Curves
        for i, (model_name, model) in enumerate(models_dict.items()):
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'{model_name} (AUC={roc_auc:.3f})',
                             line=dict(color=px.colors.qualitative.Set3[i])),
                    row=1, col=2
                )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                     name='Random', line=dict(dash='dash', color='gray')),
            row=1, col=2
        )
        
        # 3. Precision-Recall Curves
        for i, (model_name, model) in enumerate(models_dict.items()):
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                fig.add_trace(
                    go.Scatter(x=recall, y=precision, mode='lines',
                             name=f'{model_name} (AUC={pr_auc:.3f})',
                             line=dict(color=px.colors.qualitative.Set3[i])),
                    row=2, col=1
                )
        
        # 4. Summary Table
        summary_data = results_df.round(3)
        fig.add_trace(
            go.Table(
                header=dict(values=list(summary_data.columns),
                           fill_color='lightblue'),
                cells=dict(values=[summary_data[col] for col in summary_data.columns],
                          fill_color='white')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="AI Disease Prediction Model Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig


def create_comprehensive_visualization_report(models_dict, results_df, X_test, y_test,
                                           feature_importance_dict=None, 
                                           save_directory=None):
    """
    Create a comprehensive visualization report for all models.
    
    Args:
        models_dict (dict): Dictionary of trained models
        results_df (pd.DataFrame): Model comparison results
        X_test (pd.DataFrame): Test features
        y_test (array-like): Test target
        feature_importance_dict (dict): Feature importance for each model
        save_directory (str): Directory to save plots
        
    Returns:
        dict: Dictionary of created figures
    """
    visualizer = ModelVisualizer()
    figures = {}
    
    print("Creating comprehensive visualization report...")
    print("-" * 50)
    
    # 1. Model Performance Comparison
    print("Creating model performance comparison...")
    fig_comparison = visualizer.plot_model_comparison(results_df)
    figures['model_comparison'] = fig_comparison
    if save_directory:
        fig_comparison.savefig(f"{save_directory}/model_comparison.png")
    
    # 2. ROC Curves
    print("Creating ROC curves...")
    fig_roc = visualizer.plot_roc_curve(models_dict, X_test, y_test)
    figures['roc_curves'] = fig_roc
    if save_directory:
        fig_roc.savefig(f"{save_directory}/roc_curves.png")
    
    # 3. Precision-Recall Curves
    print("Creating Precision-Recall curves...")
    fig_pr = visualizer.plot_precision_recall_curve(models_dict, X_test, y_test)
    figures['pr_curves'] = fig_pr
    if save_directory:
        fig_pr.savefig(f"{save_directory}/pr_curves.png")
    
    # 4. Confusion Matrices for top models
    print("Creating confusion matrices...")
    top_models = results_df.head(3)['Model'].tolist()
    for model_name in top_models:
        if model_name in models_dict:
            model = models_dict[model_name]
            y_pred = model.predict(X_test)
            fig_cm = visualizer.plot_confusion_matrix(
                y_test, y_pred, title=f"Confusion Matrix - {model_name}"
            )
            figures[f'confusion_matrix_{model_name}'] = fig_cm
            if save_directory:
                fig_cm.savefig(f"{save_directory}/confusion_matrix_{model_name}.png")
    
    # 5. Feature Importance (if available)
    if feature_importance_dict:
        print("Creating feature importance plots...")
        for model_name, importance_df in feature_importance_dict.items():
            if importance_df is not None:
                fig_fi = visualizer.plot_feature_importance(
                    importance_df, title=f"Feature Importance - {model_name}"
                )
                figures[f'feature_importance_{model_name}'] = fig_fi
                if save_directory:
                    fig_fi.savefig(f"{save_directory}/feature_importance_{model_name}.png")
    
    # 6. Interactive Dashboard
    print("Creating interactive dashboard...")
    fig_dashboard = visualizer.create_interactive_dashboard(
        results_df, models_dict, X_test, y_test
    )
    figures['interactive_dashboard'] = fig_dashboard
    if save_directory:
        fig_dashboard.write_html(f"{save_directory}/interactive_dashboard.html")
    
    print(f"Visualization report completed! Created {len(figures)} visualizations.")
    
    return figures


if __name__ == "__main__":
    # Example usage
    print("Visualization Module")
    print("===================")
    print("This module provides comprehensive visualization utilities.")
    print("Main functions:")
    print("- ModelVisualizer class for creating various plots")
    print("- create_comprehensive_visualization_report() for full report")
    print("- Interactive dashboard with Plotly")
    print("- Focus on medical diagnosis visualization needs")