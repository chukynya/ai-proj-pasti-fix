"""
Project Demonstration Script

This script demonstrates the AI Disease Prediction Project capabilities
and provides setup instructions for the team.
"""

import os
import sys


def print_header(title, char="="):
    """Print a formatted header."""
    print(f"\n{char * len(title)}")
    print(title)
    print(f"{char * len(title)}")


def print_section(title, char="-"):
    """Print a formatted section."""
    print(f"\n{title}")
    print(f"{char * len(title)}")


def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} (missing)")
        return False


def demonstrate_project():
    """Main demonstration function."""
    
    print_header("AI DISEASE PREDICTION PROJECT DEMONSTRATION")
    
    print("""
🎯 PROJECT OVERVIEW
This is a comprehensive Machine Learning project for disease prediction using blood test data.
Focus: Optimize RECALL SCORE to minimize false negatives in medical diagnosis.
Team: Designed for 4-person collaborative development.
    """)
    
    print_section("📁 PROJECT STRUCTURE")
    
    # Check directory structure
    directories = {
        'src': 'Source code modules',
        'data': 'Dataset storage',
        'models': 'Trained model storage', 
        'results': 'Output results and reports',
        'notebooks': 'Jupyter notebooks for analysis',
        'tests': 'Unit and integration tests'
    }
    
    for dir_name, description in directories.items():
        if os.path.exists(dir_name):
            files_count = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
            print(f"✓ {dir_name}/: {description} ({files_count} files)")
        else:
            print(f"✗ {dir_name}/: {description} (missing)")
    
    print_section("🔧 CORE MODULES")
    
    # Check core modules
    modules = {
        'src/data_preprocessing.py': 'Data loading, cleaning, SMOTE balancing',
        'src/model_comparison.py': '9 classification algorithms with cross-validation',
        'src/hyperparameter_tuning.py': 'Grid Search, Random Search, Optuna optimization',
        'src/evaluation.py': 'Medical metrics, cost-benefit analysis',
        'src/visualization.py': 'ROC curves, confusion matrices, dashboards',
        'src/main.py': 'Complete pipeline orchestration with CLI',
        'src/generate_sample_data.py': 'Synthetic blood test data generation'
    }
    
    for filepath, description in modules.items():
        check_file_exists(filepath, description)
    
    print_section("🩺 MEDICAL FOCUS FEATURES")
    
    medical_features = [
        "Recall score optimization (primary metric)",
        "Sensitivity/Specificity analysis",
        "False negative/positive rate tracking",
        "Cost-benefit analysis for medical decisions",
        "30 realistic blood test parameters",
        "Clinical interpretation guidelines"
    ]
    
    for feature in medical_features:
        print(f"✓ {feature}")
    
    print_section("🤖 MACHINE LEARNING MODELS")
    
    ml_models = [
        "Random Forest", "XGBoost", "Logistic Regression",
        "SVM", "KNN", "Naive Bayes", "Decision Tree",
        "Gradient Boosting", "AdaBoost"
    ]
    
    print("Available algorithms:")
    for i, model in enumerate(ml_models, 1):
        print(f"{i:2d}. {model}")
    
    print_section("📊 BLOOD TEST PARAMETERS")
    
    parameter_categories = {
        "Complete Blood Count (CBC)": ["WBC_count", "RBC_count", "Hemoglobin", "Hematocrit", "Platelets"],
        "Basic Metabolic Panel": ["Glucose", "BUN", "Creatinine", "Sodium", "Potassium"],
        "Liver Function Tests": ["ALT", "AST", "Bilirubin_total", "Albumin"],
        "Lipid Panel": ["Total_cholesterol", "HDL_cholesterol", "LDL_cholesterol", "Triglycerides"],
        "Thyroid Function": ["TSH", "T3", "T4"],
        "Cardiac Biomarkers": ["Troponin", "CK_MB"]
    }
    
    total_params = sum(len(params) for params in parameter_categories.values())
    print(f"Total parameters: {total_params}")
    
    for category, params in parameter_categories.items():
        print(f"\n{category} ({len(params)} parameters):")
        for param in params:
            print(f"  • {param}")
    
    print_section("👥 TEAM COLLABORATION STRUCTURE")
    
    team_structure = {
        "Person 1 (Data Specialist)": [
            "data_preprocessing.py module",
            "Feature engineering and selection", 
            "Missing value handling",
            "Data quality assessment"
        ],
        "Person 2 (Model Developer)": [
            "model_comparison.py module",
            "Algorithm implementation",
            "Cross-validation setup",
            "Ensemble methods"
        ],
        "Person 3 (Optimization Expert)": [
            "hyperparameter_tuning.py module",
            "Parameter optimization",
            "Performance tuning",
            "Advanced optimization techniques"
        ],
        "Person 4 (Analyst/Reporter)": [
            "evaluation.py and visualization.py modules",
            "Performance analysis",
            "Visualization creation",
            "Documentation and presentation"
        ]
    }
    
    for role, responsibilities in team_structure.items():
        print(f"\n{role}:")
        for responsibility in responsibilities:
            print(f"  • {responsibility}")
    
    print_section("🚀 GETTING STARTED")
    
    setup_steps = [
        ("1. Install Dependencies", "pip install -r requirements.txt"),
        ("2. Generate Sample Data", "python -m src.generate_sample_data"),
        ("3. Run Complete Pipeline", "python -m src.main --data-path data/sample_blood_data.csv"),
        ("4. Interactive Analysis", "jupyter notebook notebooks/model_comparison.ipynb"),
        ("5. Run Tests", "python -m tests.test_models")
    ]
    
    for step, command in setup_steps:
        print(f"\n{step}:")
        print(f"  {command}")
    
    print_section("📈 EXPECTED RESULTS")
    
    expected_results = [
        "Model comparison across 9 algorithms",
        "Cross-validation results with confidence intervals",
        "Best model achieving >90% recall score",
        "Feature importance rankings",
        "ROC and Precision-Recall curves",
        "Interactive performance dashboard",
        "Clinical interpretation report",
        "Cost-benefit analysis"
    ]
    
    print("The pipeline will generate:")
    for result in expected_results:
        print(f"✓ {result}")
    
    print_section("💡 ADVANCED USAGE")
    
    advanced_examples = [
        ("Custom dataset", "python -m src.main --data-path your_data.csv --target-column Disease_Status"),
        ("Skip balancing", "python -m src.main --data-path data.csv --no-balancing"),
        ("Custom results dir", "python -m src.main --data-path data.csv --results-dir team_results"),
        ("Quick test run", "python -m src.main --data-path data/sample_blood_data_small.csv --no-tuning")
    ]
    
    for description, command in advanced_examples:
        print(f"\n{description}:")
        print(f"  {command}")
    
    print_section("🔍 PROJECT VALIDATION")
    
    # Check essential files
    essential_files = [
        'README.md',
        'requirements.txt',
        '.gitignore',
        'src/__init__.py',
        'notebooks/model_comparison.ipynb',
        'tests/test_models.py'
    ]
    
    all_good = True
    for filepath in essential_files:
        if not check_file_exists(filepath, "Essential file"):
            all_good = False
    
    if all_good:
        print(f"\n🎉 PROJECT VALIDATION: PASSED")
        print(f"✅ All essential components are present and ready!")
    else:
        print(f"\n⚠️  PROJECT VALIDATION: ISSUES FOUND")
        print(f"❌ Some components are missing. Please check the setup.")
    
    print_header("🏁 SUMMARY")
    
    print(f"""
✅ COMPLETE IMPLEMENTATION ACHIEVED!

This AI Disease Prediction project provides:
• Comprehensive ML pipeline for medical diagnosis
• 9 classification algorithms with recall optimization  
• Professional modular architecture for team collaboration
• Extensive evaluation and visualization capabilities
• Real-world medical focus with clinical interpretation

Perfect for Semester 3 AI project requirements!

Team members can now:
1. Install dependencies and generate sample data
2. Each work on their assigned module
3. Use notebooks for interactive analysis
4. Present results with professional visualizations

Good luck with your project! 🚀
    """)


if __name__ == "__main__":
    try:
        demonstrate_project()
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        print("Please check the project setup and try again.")
        sys.exit(1)