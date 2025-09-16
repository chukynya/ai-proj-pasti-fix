"""
Data preprocessing utilities for the AI Disease Prediction Project.

This module provides functions for loading, cleaning, and preprocessing blood test data
for machine learning model training and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load blood test data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing blood test data
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def preprocess_data(data, target_column='Disease', test_size=0.2, random_state=42):
    """
    Preprocess the blood test data for machine learning.
    
    Args:
        data (pd.DataFrame): Raw dataset
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    if data is None:
        return None, None, None, None, None, None
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Handle categorical features if any
    categorical_columns = X.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        le = LabelEncoder()
        for col in categorical_columns:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Encode target variable if categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"Data preprocessing completed.")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Target distribution in training set:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns), scaler


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (array-like): Training target
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train_balanced, y_train_balanced)
    """
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"SMOTE applied. Original shape: {X_train.shape}")
    print(f"Balanced shape: {X_balanced.shape}")
    print(f"Balanced target distribution:")
    print(pd.Series(y_balanced).value_counts().sort_index())
    
    return X_balanced, y_balanced


def load_and_preprocess_data(file_path, target_column='Disease', 
                           test_size=0.2, apply_balancing=True, random_state=42):
    """
    Complete pipeline for loading and preprocessing data.
    
    Args:
        file_path (str): Path to the CSV file
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing
        apply_balancing (bool): Whether to apply SMOTE for class balancing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    # Load data
    data = load_data(file_path)
    if data is None:
        return None, None, None, None, None, None
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
        data, target_column, test_size, random_state
    )
    
    # Apply SMOTE if requested
    if apply_balancing and X_train is not None:
        X_train, y_train = apply_smote(X_train, y_train, random_state)
        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train, columns=feature_names)
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def get_data_summary(data):
    """
    Generate a comprehensive summary of the dataset.
    
    Args:
        data (pd.DataFrame): Dataset to summarize
        
    Returns:
        dict: Summary statistics and information
    """
    if data is None:
        return None
    
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numerical_summary': data.describe().to_dict(),
        'categorical_summary': {}
    }
    
    # Add categorical column summaries
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = data[col].value_counts().to_dict()
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("========================")
    print("This module provides utilities for loading and preprocessing blood test data.")
    print("Main functions:")
    print("- load_data(file_path)")
    print("- preprocess_data(data, target_column)")
    print("- apply_smote(X_train, y_train)")
    print("- load_and_preprocess_data(file_path, target_column)")