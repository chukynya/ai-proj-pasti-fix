"""
Generate sample blood test data for testing the AI Disease Prediction Project.

This script creates a realistic synthetic dataset with blood test parameters
that could be used for disease prediction.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from datetime import datetime
import os


def generate_blood_test_features():
    """
    Generate realistic blood test feature names and their normal ranges.
    
    Returns:
        dict: Feature names with their normal ranges
    """
    features = {
        # Complete Blood Count (CBC)
        'WBC_count': (4.0, 11.0),  # White Blood Cell count (10^3/μL)
        'RBC_count': (4.5, 5.5),  # Red Blood Cell count (10^6/μL)
        'Hemoglobin': (12.0, 16.0),  # Hemoglobin (g/dL)
        'Hematocrit': (35.0, 45.0),  # Hematocrit (%)
        'Platelets': (150.0, 450.0),  # Platelet count (10^3/μL)
        
        # Basic Metabolic Panel (BMP)
        'Glucose': (70.0, 100.0),  # Glucose (mg/dL)
        'BUN': (7.0, 20.0),  # Blood Urea Nitrogen (mg/dL)
        'Creatinine': (0.6, 1.2),  # Creatinine (mg/dL)
        'Sodium': (135.0, 145.0),  # Sodium (mmol/L)
        'Potassium': (3.5, 5.0),  # Potassium (mmol/L)
        'Chloride': (96.0, 106.0),  # Chloride (mmol/L)
        'CO2': (22.0, 28.0),  # Carbon Dioxide (mmol/L)
        
        # Liver Function Tests
        'ALT': (7.0, 56.0),  # Alanine Aminotransferase (U/L)
        'AST': (10.0, 40.0),  # Aspartate Aminotransferase (U/L)
        'Bilirubin_total': (0.3, 1.2),  # Total Bilirubin (mg/dL)
        'Albumin': (3.5, 5.0),  # Albumin (g/dL)
        
        # Lipid Panel
        'Total_cholesterol': (125.0, 200.0),  # Total Cholesterol (mg/dL)
        'HDL_cholesterol': (40.0, 60.0),  # HDL Cholesterol (mg/dL)
        'LDL_cholesterol': (0.0, 100.0),  # LDL Cholesterol (mg/dL)
        'Triglycerides': (0.0, 150.0),  # Triglycerides (mg/dL)
        
        # Additional Biomarkers
        'CRP': (0.0, 3.0),  # C-Reactive Protein (mg/L)
        'ESR': (0.0, 30.0),  # Erythrocyte Sedimentation Rate (mm/hr)
        'Iron': (60.0, 170.0),  # Iron (μg/dL)
        'TIBC': (250.0, 400.0),  # Total Iron Binding Capacity (μg/dL)
        'Ferritin': (15.0, 300.0),  # Ferritin (ng/mL)
        'B12': (200.0, 900.0),  # Vitamin B12 (pg/mL)
        'Folate': (2.0, 20.0),  # Folate (ng/mL)
        
        # Thyroid Function
        'TSH': (0.4, 4.0),  # Thyroid Stimulating Hormone (mIU/L)
        'T3': (80.0, 200.0),  # Triiodothyronine (ng/dL)
        'T4': (5.0, 12.0),  # Thyroxine (μg/dL)
        
        # Cardiac Biomarkers
        'Troponin': (0.0, 0.04),  # Troponin (ng/mL)
        'CK_MB': (0.0, 6.3),  # Creatine Kinase-MB (ng/mL)
    }
    
    return features


def create_synthetic_blood_dataset(n_samples=1000, n_informative=20, 
                                  disease_prevalence=0.3, random_state=42):
    """
    Create a synthetic blood test dataset for disease prediction.
    
    Args:
        n_samples (int): Number of samples to generate
        n_informative (int): Number of informative features
        disease_prevalence (float): Proportion of positive cases
        random_state (int): Random seed
        
    Returns:
        pd.DataFrame: Synthetic dataset
    """
    np.random.seed(random_state)
    
    # Get feature information
    feature_info = generate_blood_test_features()
    feature_names = list(feature_info.keys())
    n_features = len(feature_names)
    
    # Generate base classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_informative, n_features),
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[1-disease_prevalence, disease_prevalence],
        class_sep=1.2,
        random_state=random_state
    )
    
    # Transform features to realistic blood test ranges
    X_realistic = np.zeros_like(X)
    
    for i, feature_name in enumerate(feature_names):
        min_val, max_val = feature_info[feature_name]
        
        # Normalize to [0, 1] then scale to realistic range
        X_normalized = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
        X_realistic[:, i] = X_normalized * (max_val - min_val) + min_val
        
        # Add some noise and outliers for realism
        noise = np.random.normal(0, (max_val - min_val) * 0.05, n_samples)
        X_realistic[:, i] += noise
        
        # Create some outliers for disease cases
        disease_mask = (y == 1)
        n_outliers = int(0.1 * disease_mask.sum())  # 10% outliers in disease cases
        
        if n_outliers > 0:
            outlier_indices = np.random.choice(
                np.where(disease_mask)[0], n_outliers, replace=False
            )
            
            # Make outliers more extreme
            outlier_multiplier = np.random.choice([0.5, 1.5, 2.0], n_outliers)
            X_realistic[outlier_indices, i] *= outlier_multiplier
    
    # Create DataFrame
    df = pd.DataFrame(X_realistic, columns=feature_names)
    df['Disease'] = y
    df['Disease'] = df['Disease'].map({0: 'Healthy', 1: 'Disease'})
    
    # Add some demographic information
    df['Age'] = np.random.normal(45, 15, n_samples).astype(int)
    df['Age'] = np.clip(df['Age'], 18, 85)
    
    df['Gender'] = np.random.choice(['Male', 'Female'], n_samples)
    
    # Add patient ID
    df['Patient_ID'] = [f'P{i:06d}' for i in range(1, n_samples + 1)]
    
    # Reorder columns
    id_cols = ['Patient_ID', 'Age', 'Gender']
    feature_cols = feature_names
    target_col = ['Disease']
    
    df = df[id_cols + feature_cols + target_col]
    
    return df


def add_missing_values(df, missing_rate=0.05, random_state=42):
    """
    Add realistic missing values to the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to add missing values to
        missing_rate (float): Proportion of values to make missing
        random_state (int): Random seed
        
    Returns:
        pd.DataFrame: Dataset with missing values
    """
    np.random.seed(random_state)
    df_missing = df.copy()
    
    # Get only the blood test feature columns (exclude ID, Age, Gender, Disease)
    blood_features = [col for col in df.columns 
                     if col not in ['Patient_ID', 'Age', 'Gender', 'Disease']]
    
    # Add missing values randomly
    for col in blood_features:
        n_missing = int(len(df) * missing_rate)
        if n_missing > 0:
            missing_indices = np.random.choice(len(df), n_missing, replace=False)
            df_missing.loc[missing_indices, col] = np.nan
    
    return df_missing


def save_sample_datasets(output_dir='data', n_samples=1000):
    """
    Generate and save sample datasets.
    
    Args:
        output_dir (str): Directory to save datasets
        n_samples (int): Number of samples to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating sample blood test datasets...")
    print(f"Number of samples: {n_samples}")
    
    # Generate main dataset
    df = create_synthetic_blood_dataset(n_samples=n_samples)
    
    # Save complete dataset
    complete_path = os.path.join(output_dir, 'sample_blood_data.csv')
    df.to_csv(complete_path, index=False)
    print(f"✓ Complete dataset saved: {complete_path}")
    
    # Generate dataset with missing values
    df_missing = add_missing_values(df, missing_rate=0.08)
    missing_path = os.path.join(output_dir, 'sample_blood_data_missing.csv')
    df_missing.to_csv(missing_path, index=False)
    print(f"✓ Dataset with missing values saved: {missing_path}")
    
    # Generate smaller dataset for quick testing
    df_small = df.sample(n=min(200, n_samples), random_state=42)
    small_path = os.path.join(output_dir, 'sample_blood_data_small.csv')
    df_small.to_csv(small_path, index=False)
    print(f"✓ Small test dataset saved: {small_path}")
    
    # Print dataset summary
    print(f"\nDataset Summary:")
    print(f"- Total samples: {len(df)}")
    print(f"- Total features: {len(df.columns) - 4}")  # Exclude ID, Age, Gender, Disease
    print(f"- Blood test features: {len([col for col in df.columns if col not in ['Patient_ID', 'Age', 'Gender', 'Disease']])}")
    print(f"- Disease prevalence: {(df['Disease'] == 'Disease').mean():.1%}")
    print(f"- Age range: {df['Age'].min()}-{df['Age'].max()} years")
    print(f"- Gender distribution: {df['Gender'].value_counts().to_dict()}")
    
    return df


def create_data_readme(output_dir='data'):
    """
    Create a README file explaining the dataset.
    
    Args:
        output_dir (str): Directory to save README
    """
    readme_content = """# Blood Test Dataset for Disease Prediction

This directory contains synthetic blood test datasets generated for the AI Disease Prediction Project.

## Dataset Files

1. **sample_blood_data.csv** - Complete dataset with all features
2. **sample_blood_data_missing.csv** - Dataset with realistic missing values (~8%)
3. **sample_blood_data_small.csv** - Smaller dataset for quick testing (200 samples)

## Dataset Description

The dataset contains synthetic blood test results for disease prediction with the following characteristics:

### Features

#### Demographic Information
- `Patient_ID`: Unique patient identifier
- `Age`: Patient age (18-85 years)
- `Gender`: Patient gender (Male/Female)

#### Blood Test Parameters (30 features)

**Complete Blood Count (CBC)**
- `WBC_count`: White Blood Cell count (10³/μL)
- `RBC_count`: Red Blood Cell count (10⁶/μL)
- `Hemoglobin`: Hemoglobin level (g/dL)
- `Hematocrit`: Hematocrit percentage (%)
- `Platelets`: Platelet count (10³/μL)

**Basic Metabolic Panel (BMP)**
- `Glucose`: Blood glucose (mg/dL)
- `BUN`: Blood Urea Nitrogen (mg/dL)
- `Creatinine`: Creatinine level (mg/dL)
- `Sodium`: Sodium level (mmol/L)
- `Potassium`: Potassium level (mmol/L)
- `Chloride`: Chloride level (mmol/L)
- `CO2`: Carbon Dioxide level (mmol/L)

**Liver Function Tests**
- `ALT`: Alanine Aminotransferase (U/L)
- `AST`: Aspartate Aminotransferase (U/L)
- `Bilirubin_total`: Total Bilirubin (mg/dL)
- `Albumin`: Albumin level (g/dL)

**Lipid Panel**
- `Total_cholesterol`: Total Cholesterol (mg/dL)
- `HDL_cholesterol`: HDL Cholesterol (mg/dL)
- `LDL_cholesterol`: LDL Cholesterol (mg/dL)
- `Triglycerides`: Triglycerides level (mg/dL)

**Additional Biomarkers**
- `CRP`: C-Reactive Protein (mg/L)
- `ESR`: Erythrocyte Sedimentation Rate (mm/hr)
- `Iron`: Iron level (μg/dL)
- `TIBC`: Total Iron Binding Capacity (μg/dL)
- `Ferritin`: Ferritin level (ng/mL)
- `B12`: Vitamin B12 (pg/mL)
- `Folate`: Folate level (ng/mL)

**Thyroid Function**
- `TSH`: Thyroid Stimulating Hormone (mIU/L)
- `T3`: Triiodothyronine (ng/dL)
- `T4`: Thyroxine (μg/dL)

**Cardiac Biomarkers**
- `Troponin`: Troponin level (ng/mL)
- `CK_MB`: Creatine Kinase-MB (ng/mL)

#### Target Variable
- `Disease`: Disease status (Healthy/Disease)

### Dataset Characteristics

- **Total Samples**: 1000 (default)
- **Disease Prevalence**: ~30%
- **Missing Values**: Some datasets include realistic missing values
- **Feature Ranges**: Based on standard clinical reference ranges
- **Data Quality**: Includes realistic noise and outliers

### Usage

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/sample_blood_data.csv')

# Basic information
print(df.info())
print(df['Disease'].value_counts())

# Use with the pipeline
from src.main import DiseasePredictor

predictor = DiseasePredictor()
results = predictor.run_complete_pipeline('data/sample_blood_data.csv')
```

### Notes

- This is **synthetic data** generated for educational purposes
- The relationships between features and disease status are artificially created
- Real medical datasets would require proper clinical validation
- Always consult medical professionals for actual health decisions

### Data Generation

The dataset was generated using:
```python
from src.generate_sample_data import save_sample_datasets
save_sample_datasets(n_samples=1000)
```

Generated on: {generation_time}
"""
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content.format(
            generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    print(f"✓ Data README created: {readme_path}")


if __name__ == "__main__":
    print("Generating Sample Blood Test Dataset")
    print("=" * 40)
    
    # Generate sample datasets
    df = save_sample_datasets(n_samples=1000)
    
    # Create README
    create_data_readme()
    
    print("\n✅ Sample dataset generation completed!")
    print("\nTo use the dataset:")
    print("python -m src.main --data-path data/sample_blood_data.csv")