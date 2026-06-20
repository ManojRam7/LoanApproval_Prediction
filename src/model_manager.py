"""
Model management module for loading and using the trained Random Forest model.
"""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from src.logger import logger
from src.config import MODEL_PATH


class ModelManager:
    """Manages model loading, prediction, and basic operations."""

    def __init__(self, model_path: str | Path = MODEL_PATH):
        """
        Initialize the ModelManager.

        Parameters
        ----------
        model_path : str or Path
            Path to the pickled model file
        """
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from pickle file."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.

        Parameters
        ----------
        X : pd.DataFrame
            Input features for prediction

        Returns
        -------
        np.ndarray
            Prediction results (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")

        try:
            predictions = self.model.predict(X)
            logger.debug(f"Predictions made for {len(X)} samples")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using the loaded model.

        Parameters
        ----------
        X : pd.DataFrame
            Input features for prediction

        Returns
        -------
        np.ndarray
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")

        try:
            if not hasattr(self.model, "predict_proba"):
                logger.warning("Model does not support probability predictions")
                return None

            proba = self.model.predict_proba(X)
            logger.debug(f"Probabilities calculated for {len(X)} samples")
            return proba
        except Exception as e:
            logger.error(f"Error during probability prediction: {str(e)}")
            raise

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns
        -------
        dict
            Model information including type and parameters
        """
        if self.model is None:
            return {"status": "No model loaded"}

        return {
            "model_type": type(self.model).__name__,
            "model_path": str(self.model_path),
            "parameters": self.model.get_params(),
        }


# Singleton instance for easy access
_model_manager = None


def get_model_manager(model_path: str | Path = MODEL_PATH) -> ModelManager:
    """
    Get or create a singleton ModelManager instance.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the model file

    Returns
    -------
    ModelManager
        Singleton instance of ModelManager
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(model_path)
    return _model_manager
