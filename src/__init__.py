"""
AI Disease Prediction Project

A comprehensive machine learning project for disease prediction using blood test data.
This package provides utilities for data preprocessing, model comparison, hyperparameter tuning,
evaluation, and visualization with focus on recall score optimization.
"""

__version__ = "1.0.0"
__author__ = "AI Disease Prediction Team"
__email__ = "team@ai-disease-prediction.edu"

# Import main classes and functions for easy access
from .data_preprocessing import load_and_preprocess_data, load_data, preprocess_data
from .model_comparison import compare_classification_models, ModelComparison
from .hyperparameter_tuning import tune_best_model, HyperparameterTuner
from .evaluation import create_evaluation_report, evaluate_multiple_models, ModelEvaluator
from .visualization import create_comprehensive_visualization_report, ModelVisualizer
from .main import DiseasePredictor

__all__ = [
    'load_and_preprocess_data',
    'load_data', 
    'preprocess_data',
    'compare_classification_models',
    'ModelComparison',
    'tune_best_model',
    'HyperparameterTuner',
    'create_evaluation_report',
    'evaluate_multiple_models',
    'ModelEvaluator',
    'create_comprehensive_visualization_report',
    'ModelVisualizer',
    'DiseasePredictor'
]