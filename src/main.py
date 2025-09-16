"""
Main pipeline for the AI Disease Prediction Project.

This module provides the main entry point for running the complete
machine learning pipeline for disease prediction using blood data.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings

# Import project modules
from data_preprocessing import load_and_preprocess_data
from model_comparison import compare_classification_models
from hyperparameter_tuning import tune_best_model
from evaluation import create_evaluation_report, evaluate_multiple_models
from visualization import create_comprehensive_visualization_report

warnings.filterwarnings('ignore')


class DiseasePredictor:
    """
    Main class for the disease prediction pipeline.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the disease predictor.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.data_info = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.scaler = None
        
    def run_complete_pipeline(self, data_path, target_column='Disease',
                            test_size=0.2, apply_balancing=True,
                            tune_hyperparameters=True, save_models=True,
                            save_results=True, results_dir='results'):
        """
        Run the complete machine learning pipeline.
        
        Args:
            data_path (str): Path to the dataset
            target_column (str): Name of the target column
            test_size (float): Proportion of data for testing
            apply_balancing (bool): Whether to apply SMOTE balancing
            tune_hyperparameters (bool): Whether to tune hyperparameters
            save_models (bool): Whether to save trained models
            save_results (bool): Whether to save results and plots
            results_dir (str): Directory to save results
            
        Returns:
            dict: Complete pipeline results
        """
        print("="*80)
        print("AI DISEASE PREDICTION PROJECT - COMPLETE PIPELINE")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Create results directory
        if save_results and not os.path.exists(results_dir):
            os.makedirs(results_dir)
            os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
        
        # Step 1: Data Loading and Preprocessing
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("-" * 50)
        
        X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data(
            data_path, target_column, test_size, apply_balancing, self.random_state
        )
        
        if X_train is None:
            print("Error: Data preprocessing failed!")
            return None
        
        self.feature_names = feature_names
        self.scaler = scaler
        self.data_info = {
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'feature_count': len(feature_names),
            'class_distribution_train': pd.Series(y_train).value_counts().to_dict()
        }
        
        print(f"✓ Data preprocessing completed successfully!")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Features: {len(feature_names)}")
        print()
        
        # Step 2: Model Comparison
        print("STEP 2: CLASSIFICATION MODELS COMPARISON")
        print("-" * 50)
        
        comparison_results = compare_classification_models(
            X_train, X_test, y_train, y_test, 
            cv_folds=5, random_state=self.random_state
        )
        
        self.model_results = comparison_results
        self.best_model_name = comparison_results['best_model_name']
        self.best_model = comparison_results['best_model']
        
        print(f"✓ Model comparison completed!")
        print(f"  Best model: {self.best_model_name}")
        print(f"  Best recall score: {comparison_results['best_recall']:.4f}")
        print()
        
        # Step 3: Hyperparameter Tuning (if requested)
        tuned_model = None
        if tune_hyperparameters:
            print("STEP 3: HYPERPARAMETER TUNING")
            print("-" * 35)
            
            print(f"Tuning hyperparameters for {self.best_model_name}...")
            
            # Map model names to tuning-compatible names
            model_type_mapping = {
                'Random Forest': 'RandomForest',
                'XGBoost': 'XGBoost',
                'Logistic Regression': 'LogisticRegression',
                'SVM': 'SVM',
                'Gradient Boosting': 'GradientBoosting'
            }
            
            model_type = model_type_mapping.get(self.best_model_name)
            
            if model_type:
                try:
                    tuned_model = tune_best_model(
                        X_train, y_train, model_type=model_type,
                        method='random_search', cv_folds=5,
                        scoring='recall', random_state=self.random_state,
                        n_iter=20  # Reduced for faster execution
                    )
                    
                    # Evaluate tuned model
                    tuned_pred = tuned_model.predict(X_test)
                    from sklearn.metrics import recall_score
                    tuned_recall = recall_score(y_test, tuned_pred)
                    
                    print(f"✓ Hyperparameter tuning completed!")
                    print(f"  Original recall: {comparison_results['best_recall']:.4f}")
                    print(f"  Tuned recall: {tuned_recall:.4f}")
                    
                    # Use tuned model if it's better
                    if tuned_recall > comparison_results['best_recall']:
                        self.best_model = tuned_model
                        print(f"  ✓ Using tuned model (better performance)")
                    else:
                        print(f"  → Using original model (no improvement)")
                    
                except Exception as e:
                    print(f"  ⚠ Hyperparameter tuning failed: {str(e)}")
                    print(f"  → Using original best model")
            else:
                print(f"  ⚠ Hyperparameter tuning not available for {self.best_model_name}")
                print(f"  → Using original best model")
        print()
        
        # Step 4: Detailed Evaluation
        print("STEP 4: DETAILED MODEL EVALUATION")
        print("-" * 40)
        
        evaluation_report = create_evaluation_report(
            self.best_model, X_test, y_test, 
            model_name=self.best_model_name,
            feature_names=feature_names
        )
        
        print("✓ Detailed evaluation completed!")
        print()
        
        # Step 5: Visualization
        if save_results:
            print("STEP 5: CREATING VISUALIZATIONS")
            print("-" * 35)
            
            # Prepare models dict for visualization
            models_for_viz = comparison_results['trained_models']
            results_df = comparison_results['test_results']
            
            # Get feature importance for visualizable models
            feature_importance_dict = {}
            for model_name, model in models_for_viz.items():
                try:
                    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                        from evaluation import ModelEvaluator
                        evaluator = ModelEvaluator(model, model_name)
                        importance = evaluator.get_feature_importance(feature_names)
                        feature_importance_dict[model_name] = importance
                except:
                    pass
            
            # Create visualization report
            plot_dir = os.path.join(results_dir, 'plots')
            figures = create_comprehensive_visualization_report(
                models_for_viz, results_df, X_test, y_test,
                feature_importance_dict, plot_dir
            )
            
            print(f"✓ Visualizations saved to {plot_dir}")
            print()
        
        # Step 6: Save Models and Results
        if save_models or save_results:
            print("STEP 6: SAVING MODELS AND RESULTS")
            print("-" * 40)
            
            if save_models:
                # Save best model
                model_path = os.path.join(results_dir, 'models', f'best_model_{self.best_model_name.replace(" ", "_").lower()}.joblib')
                joblib.dump(self.best_model, model_path)
                
                # Save scaler
                scaler_path = os.path.join(results_dir, 'models', 'scaler.joblib')
                joblib.dump(self.scaler, scaler_path)
                
                print(f"✓ Best model saved: {model_path}")
                print(f"✓ Scaler saved: {scaler_path}")
            
            if save_results:
                # Save results summary
                results_summary = {
                    'pipeline_info': {
                        'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'data_path': data_path,
                        'target_column': target_column,
                        'test_size': test_size,
                        'apply_balancing': apply_balancing,
                        'tune_hyperparameters': tune_hyperparameters
                    },
                    'data_info': self.data_info,
                    'best_model_name': self.best_model_name,
                    'best_recall_score': comparison_results['best_recall'],
                    'model_comparison_results': comparison_results['test_results'].to_dict(),
                    'detailed_evaluation': evaluation_report['evaluation_metrics']
                }
                
                # Save as JSON-serializable format
                import json
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    return obj
                
                results_path = os.path.join(results_dir, 'pipeline_results.json')
                with open(results_path, 'w') as f:
                    json.dump(results_summary, f, indent=2, default=convert_numpy)
                
                print(f"✓ Results summary saved: {results_path}")
            
            print()
        
        # Final Summary
        print("PIPELINE COMPLETION SUMMARY")
        print("="*30)
        print(f"✓ Best Model: {self.best_model_name}")
        print(f"✓ Best Recall Score: {comparison_results['best_recall']:.4f}")
        print(f"✓ Total Features: {len(feature_names)}")
        print(f"✓ Test Samples: {X_test.shape[0]}")
        
        if save_results:
            print(f"✓ Results saved to: {results_dir}")
        
        print(f"\nPipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        return {
            'data_info': self.data_info,
            'model_comparison_results': comparison_results,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'evaluation_report': evaluation_report,
            'feature_names': feature_names,
            'scaler': self.scaler
        }
    
    def predict_new_samples(self, new_data):
        """
        Make predictions on new data samples.
        
        Args:
            new_data (pd.DataFrame): New samples to predict
            
        Returns:
            dict: Predictions and probabilities
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please run the pipeline first.")
        
        if self.scaler is None:
            raise ValueError("No scaler available. Please run the pipeline first.")
        
        # Preprocess new data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make predictions
        predictions = self.best_model.predict(new_data_scaled)
        
        prediction_probabilities = None
        if hasattr(self.best_model, 'predict_proba'):
            prediction_probabilities = self.best_model.predict_proba(new_data_scaled)
        
        return {
            'predictions': predictions,
            'prediction_probabilities': prediction_probabilities,
            'model_used': self.best_model_name
        }


def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="AI Disease Prediction Project - Complete Pipeline"
    )
    
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the dataset CSV file')
    parser.add_argument('--target-column', type=str, default='Disease',
                       help='Name of the target column (default: Disease)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--no-balancing', action='store_true',
                       help='Disable SMOTE balancing')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--no-save-models', action='store_true',
                       help='Do not save trained models')
    parser.add_argument('--no-save-results', action='store_true',
                       help='Do not save results and plots')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DiseasePredictor(random_state=args.random_seed)
    
    # Run pipeline
    try:
        results = predictor.run_complete_pipeline(
            data_path=args.data_path,
            target_column=args.target_column,
            test_size=args.test_size,
            apply_balancing=not args.no_balancing,
            tune_hyperparameters=not args.no_tuning,
            save_models=not args.no_save_models,
            save_results=not args.no_save_results,
            results_dir=args.results_dir
        )
        
        if results:
            print("\n🎉 Pipeline executed successfully!")
        else:
            print("\n❌ Pipeline execution failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Pipeline execution failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()