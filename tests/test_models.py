"""
Unit tests for the AI Disease Prediction Project.

This module contains basic tests to ensure the main components work correctly.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os
import tempfile
import shutil

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import load_data, preprocess_data
from model_comparison import ModelComparison
from evaluation import ModelEvaluator
from generate_sample_data import create_synthetic_blood_dataset


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = create_synthetic_blood_dataset(n_samples=100, random_state=42)
        
    def test_preprocess_data(self):
        """Test data preprocessing function."""
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
            self.test_data, target_column='Disease', test_size=0.2, random_state=42
        )
        
        # Check that preprocessing worked
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertIsNotNone(feature_names)
        self.assertIsNotNone(scaler)
        
        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), 100)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check feature names
        self.assertIsInstance(feature_names, list)
        self.assertTrue(len(feature_names) > 0)
        
    def test_data_loading(self):
        """Test data loading with CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Test loading
            data = load_data(temp_path)
            self.assertIsNotNone(data)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 100)
        finally:
            # Clean up
            os.unlink(temp_path)


class TestModelComparison(unittest.TestCase):
    """Test model comparison functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create simple synthetic data
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5,
            n_redundant=0, random_state=42
        )
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Convert to DataFrame for consistency
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.X_train = pd.DataFrame(self.X_train, columns=feature_names)
        self.X_test = pd.DataFrame(self.X_test, columns=feature_names)
        
    def test_model_comparison_initialization(self):
        """Test ModelComparison class initialization."""
        comparator = ModelComparison(random_state=42)
        self.assertIsNotNone(comparator.models)
        self.assertTrue(len(comparator.models) > 0)
        self.assertEqual(comparator.random_state, 42)
        
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        comparator = ModelComparison(random_state=42)
        cv_results = comparator.cross_validate_models(
            self.X_train, self.y_train, cv_folds=3
        )
        
        self.assertIsNotNone(cv_results)
        self.assertIsInstance(cv_results, pd.DataFrame)
        self.assertTrue(len(cv_results) > 0)
        
        # Check that recall column exists
        self.assertIn(('Recall', 'mean'), cv_results.columns)
        
    def test_model_training_and_evaluation(self):
        """Test model training and evaluation."""
        comparator = ModelComparison(random_state=42)
        results = comparator.train_and_evaluate_models(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        self.assertIsNotNone(results)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertTrue(len(results) > 0)
        
        # Check that required metrics exist
        required_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for metric in required_metrics:
            self.assertIn(metric, results.columns)
            
        # Check that best model can be identified
        best_name, best_model, best_recall = comparator.get_best_model_by_recall()
        self.assertIsNotNone(best_name)
        self.assertIsNotNone(best_model)
        self.assertIsInstance(best_recall, (int, float))


class TestModelEvaluator(unittest.TestCase):
    """Test model evaluation functionality."""
    
    def setUp(self):
        """Set up test data and trained model."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create synthetic data
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5,
            n_redundant=0, random_state=42
        )
        
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.X_test = pd.DataFrame(self.X_test, columns=feature_names)
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X_train, y_train)
        
    def test_model_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(self.model, "Test Model")
        self.assertEqual(evaluator.model_name, "Test Model")
        self.assertEqual(evaluator.model, self.model)
        
    def test_classification_evaluation(self):
        """Test classification evaluation."""
        evaluator = ModelEvaluator(self.model, "Test Model")
        results = evaluator.evaluate_classification(self.X_test, self.y_test)
        
        self.assertIsNotNone(results)
        self.assertIsInstance(results, dict)
        
        # Check required metrics
        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'specificity', 'sensitivity', 'confusion_matrix'
        ]
        for metric in required_metrics:
            self.assertIn(metric, results)
            
        # Check that confusion matrix has correct shape
        cm = results['confusion_matrix']
        self.assertEqual(cm.shape, (2, 2))
        
    def test_feature_importance(self):
        """Test feature importance extraction."""
        evaluator = ModelEvaluator(self.model, "Test Model")
        feature_names = [f'feature_{i}' for i in range(10)]
        importance_df = evaluator.get_feature_importance(feature_names)
        
        self.assertIsNotNone(importance_df)
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertTrue(len(importance_df) > 0)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)


class TestSampleDataGeneration(unittest.TestCase):
    """Test sample data generation functionality."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic blood test data generation."""
        df = create_synthetic_blood_dataset(n_samples=50, random_state=42)
        
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 50)
        
        # Check required columns
        required_cols = ['Patient_ID', 'Age', 'Gender', 'Disease']
        for col in required_cols:
            self.assertIn(col, df.columns)
            
        # Check data types and ranges
        self.assertTrue(df['Age'].between(18, 85).all())
        self.assertTrue(df['Gender'].isin(['Male', 'Female']).all())
        self.assertTrue(df['Disease'].isin(['Healthy', 'Disease']).all())
        
        # Check that we have blood test features
        blood_features = [col for col in df.columns 
                         if col not in ['Patient_ID', 'Age', 'Gender', 'Disease']]
        self.assertTrue(len(blood_features) > 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test dataset
        self.test_data = create_synthetic_blood_dataset(n_samples=100, random_state=42)
        self.data_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.test_data.to_csv(self.data_path, index=False)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_complete_pipeline_components(self):
        """Test that main pipeline components work together."""
        from data_preprocessing import load_and_preprocess_data
        from model_comparison import compare_classification_models
        
        # Test data preprocessing
        X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data(
            self.data_path, target_column='Disease', test_size=0.3, random_state=42
        )
        
        self.assertIsNotNone(X_train)
        self.assertTrue(len(X_train) > 0)
        
        # Test model comparison (with reduced models for speed)
        from model_comparison import ModelComparison
        comparator = ModelComparison(random_state=42)
        
        # Use only a few models for testing
        test_models = {
            'Random Forest': comparator.models['Random Forest'],
            'Logistic Regression': comparator.models['Logistic Regression']
        }
        comparator.models = test_models
        
        results = comparator.train_and_evaluate_models(
            X_train, X_test, y_train, y_test
        )
        
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
        
        # Test that we can get best model
        best_name, best_model, best_recall = comparator.get_best_model_by_recall()
        self.assertIsNotNone(best_name)
        self.assertIsNotNone(best_model)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestDataPreprocessing,
        TestModelComparison,
        TestModelEvaluator,
        TestSampleDataGeneration,
        TestIntegration
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)