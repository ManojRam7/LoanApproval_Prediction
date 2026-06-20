"""
Configuration settings for the Loan Approval Prediction system.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "LoanApprovalPrediction.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"

# Logging configuration
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "app.log"
LOG_LEVEL = "INFO"

# Model configuration
MODEL_CONFIG = {
    "n_estimators": 7,
    "criterion": "entropy",
    "random_state": 7,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}

# Train-test split configuration
TRAIN_TEST_SPLIT_RATIO = 0.4
RANDOM_STATE_SPLIT = 1

# Feature names in correct order
FEATURE_NAMES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

# Encoding mappings
ENCODING_MAPPINGS = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Urban": 0, "Rural": 1, "Semiurban": 2},
}

# Streamlit configuration
STREAMLIT_THEME = "light"
STREAMLIT_PAGE_LAYOUT = "wide"
