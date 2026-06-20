"""
Model Training Script for Loan Approval Prediction

This script trains a Random Forest classifier on loan approval data
and saves the trained model for production use.
"""

import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import (
    RAW_DATA_PATH,
    MODEL_PATH,
    MODEL_CONFIG,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_STATE_SPLIT,
)
from src.logger import logger


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load raw data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess data for model training."""
    try:
        # Drop Loan_ID as it's not predictive
        if "Loan_ID" in df.columns:
            df = df.drop("Loan_ID", axis=1)

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        # Remove target variable if present
        if "Loan_Status" in categorical_cols:
            categorical_cols.remove("Loan_Status")

        # Encode categorical variables
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col])

        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        logger.info("Data preprocessing completed")

        # Separate features and target
        X = df.drop("Loan_Status", axis=1)
        y = df["Loan_Status"]

        # Ensure target is numeric binary for sklearn metrics and consistency.
        if y.dtype == "object":
            if set(y.unique()).issubset({"Y", "N"}):
                y = y.map({"N": 0, "Y": 1})
            else:
                y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    try:
        model = RandomForestClassifier(**MODEL_CONFIG)
        model.fit(X_train, y_train)

        # Training accuracy
        train_predictions = model.predict(X_train)
        train_accuracy = metrics.accuracy_score(y_train, train_predictions)
        logger.info(f"Training accuracy: {train_accuracy*100:.2f}%")

        return model

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate model performance."""
    try:
        predictions = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        unique_labels = np.unique(y_test)
        average_mode = "binary" if len(unique_labels) == 2 else "weighted"

        precision = metrics.precision_score(
            y_test,
            predictions,
            average=average_mode,
            zero_division=0,
        )
        recall = metrics.recall_score(
            y_test,
            predictions,
            average=average_mode,
            zero_division=0,
        )
        f1 = metrics.f1_score(
            y_test,
            predictions,
            average=average_mode,
            zero_division=0,
        )

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        logger.info(
            f"Model Evaluation - Accuracy: {accuracy*100:.2f}%, "
            f"Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1:.2f}"
        )

        return metrics_dict

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise


def save_model(model: RandomForestClassifier, save_path: str | Path) -> None:
    """Save trained model to pickle file."""
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved successfully to {save_path}")

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def main():
    """Main training pipeline."""
    logger.info("Starting model training pipeline")

    try:
        # Load data
        df = load_data(RAW_DATA_PATH)

        # Preprocess
        X, y = preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TRAIN_TEST_SPLIT_RATIO,
            random_state=RANDOM_STATE_SPLIT,
        )

        logger.info(
            f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}"
        )

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate
        eval_metrics = evaluate_model(model, X_test, y_test)

        # Save model
        save_model(model, MODEL_PATH)

        logger.info("Model training pipeline completed successfully")

        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED")
        print("="*50)
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        print(f"Model accuracy: {eval_metrics['accuracy']*100:.2f}%")
        print(f"Model saved to: {MODEL_PATH}")
        print("="*50)

        return model

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
