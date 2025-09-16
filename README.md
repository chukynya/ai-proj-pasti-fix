# AI Disease Prediction Project

A comprehensive Machine Learning project for predicting diseases based on blood test data. This project compares multiple classification algorithms and optimizes for recall score to minimize false negatives in medical diagnosis.

## Project Overview

This is a Semester 3 AI project that implements and compares various machine learning classification models for disease prediction using blood test parameters. The project focuses on maximizing recall score to ensure minimal false negatives in medical diagnosis.

## Features

- **Multiple Classification Models**: Random Forest, SVM, Logistic Regression, XGBoost, and more
- **Comprehensive Evaluation**: Focus on recall score optimization with detailed metrics
- **Hyperparameter Tuning**: Automated optimization for best performing models
- **Data Visualization**: Interactive plots and model performance comparisons
- **Cross-Validation**: Robust model validation techniques
- **Modular Design**: Clean, maintainable code structure for team collaboration

## Project Structure

```
ai-proj-pasti-fix/
├── src/
│   ├── data_preprocessing.py    # Data loading and preprocessing utilities
│   ├── model_comparison.py      # Classification models comparison
│   ├── hyperparameter_tuning.py # Model optimization
│   ├── evaluation.py            # Model evaluation metrics
│   └── visualization.py         # Plotting and visualization
├── data/
│   ├── sample_blood_data.csv    # Sample dataset
│   └── README.md                # Data description
├── notebooks/
│   ├── data_exploration.ipynb   # Exploratory data analysis
│   ├── model_comparison.ipynb   # Model comparison analysis
│   └── results_analysis.ipynb   # Final results and insights
├── tests/
│   └── test_models.py           # Unit tests
├── models/                      # Saved trained models
├── results/                     # Output results and reports
└── requirements.txt             # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chukynya/ai-proj-pasti-fix.git
cd ai-proj-pasti-fix
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Data Preparation**:
```python
from src.data_preprocessing import load_and_preprocess_data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/blood_data.csv')
```

2. **Model Comparison**:
```python
from src.model_comparison import compare_classification_models
results = compare_classification_models(X_train, X_test, y_train, y_test)
```

3. **Hyperparameter Tuning**:
```python
from src.hyperparameter_tuning import tune_best_model
best_model = tune_best_model(X_train, y_train, model_type='RandomForest')
```

### Running the Complete Pipeline

```bash
python -m src.main --data-path data/blood_data.csv --optimize-recall
```

## Model Performance Metrics

The project evaluates models using multiple metrics with primary focus on:
- **Recall Score**: Minimizing false negatives (primary metric)
- **Precision**: Minimizing false positives
- **F1-Score**: Balanced performance measure
- **AUC-ROC**: Overall classification performance
- **Confusion Matrix**: Detailed error analysis

## Team Collaboration

This project is designed for a 4-person team with clear module separation:
- **Person 1**: Data preprocessing and feature engineering
- **Person 2**: Model implementation and comparison
- **Person 3**: Hyperparameter tuning and optimization
- **Person 4**: Evaluation, visualization, and documentation

## Contributing

1. Create a feature branch for your module
2. Follow PEP 8 coding standards
3. Add unit tests for new functionality
4. Update documentation as needed
5. Submit pull request for review

## Results

The project aims to:
- Compare 5+ classification algorithms
- Achieve recall score > 0.90 for disease detection
- Provide comprehensive model performance analysis
- Generate publication-ready visualizations

## License

This project is for educational purposes as part of Semester 3 AI coursework.

## Authors

- Team Members: [Add team member names]
- Course: AI/Machine Learning - Semester 3
- Institution: [Add institution name]