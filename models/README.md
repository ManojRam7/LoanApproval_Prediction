# Models Directory

This directory stores the trained machine learning models in pickle format.

## Contents

- **random_forest_model.pkl** - Trained Random Forest Classifier model
  - Accuracy: 82% on test data
  - Trained on 609 samples
  - Predicts loan approval status based on 11 features

## Model Training

The model was trained using the following configuration:
- Algorithm: Random Forest Classifier
- Parameters:
  - n_estimators: 7
  - criterion: entropy
  - random_state: 7
- Train/Test Split: 60/40

## Usage

Models are loaded via the `ModelManager` class from `src/model_manager.py`
