"""
Data processing and transformation module for loan approval predictions.
"""

import pandas as pd
import numpy as np
from src.logger import logger
from src.config import FEATURE_NAMES, ENCODING_MAPPINGS


class DataProcessor:
    """Handles data preprocessing and feature engineering."""

    def __init__(self):
        """Initialize the DataProcessor."""
        self.feature_names = FEATURE_NAMES
        self.encoding_mappings = ENCODING_MAPPINGS

    def process_input(
        self,
        gender: str,
        married: str,
        dependents: str,
        education: str,
        self_employed: str,
        applicant_income: float,
        coapplicant_income: float,
        loan_amount: float,
        loan_amount_term: float,
        credit_history: int,
        property_area: str,
    ) -> pd.DataFrame:
        """
        Process and encode user input for model prediction.

        Parameters
        ----------
        gender : str
            Applicant's gender
        married : str
            Marital status
        dependents : str
            Number of dependents
        education : str
            Education level
        self_employed : str
            Self-employment status
        applicant_income : float
            Applicant's income
        coapplicant_income : float
            Co-applicant's income
        loan_amount : float
            Requested loan amount
        loan_amount_term : float
            Loan term in days
        credit_history : int
            Credit history status
        property_area : str
            Property area type

        Returns
        -------
        pd.DataFrame
            Processed input ready for model prediction
        """
        try:
            # Create input dataframe with encoded values
            input_data = pd.DataFrame(
                {
                    "Gender": [self.encoding_mappings["Gender"].get(gender, 0)],
                    "Married": [self.encoding_mappings["Married"].get(married, 0)],
                    "Dependents": [
                        0 if dependents == "0"
                        else (3 if dependents == "3+" else int(dependents))
                    ],
                    "Education": [self.encoding_mappings["Education"].get(education, 0)],
                    "Self_Employed": [
                        self.encoding_mappings["Self_Employed"].get(self_employed, 0)
                    ],
                    "ApplicantIncome": [applicant_income],
                    "CoapplicantIncome": [coapplicant_income],
                    "LoanAmount": [loan_amount],
                    "Loan_Amount_Term": [loan_amount_term],
                    "Credit_History": [credit_history],
                    "Property_Area": [
                        self.encoding_mappings["Property_Area"].get(property_area, 0)
                    ],
                }
            )

            logger.info("Input data processed successfully")
            return input_data
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            raise

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate user inputs for correctness.

        Parameters
        ----------
        **kwargs
            Input parameters to validate

        Returns
        -------
        bool
            True if all inputs are valid
        """
        try:
            # Validate income fields are non-negative
            if kwargs.get("applicant_income", 0) < 0:
                raise ValueError("Applicant income cannot be negative")
            if kwargs.get("coapplicant_income", 0) < 0:
                raise ValueError("Co-applicant income cannot be negative")
            if kwargs.get("loan_amount", 0) <= 0:
                raise ValueError("Loan amount must be positive")
            if kwargs.get("loan_amount_term", 0) <= 0:
                raise ValueError("Loan term must be positive")

            logger.info("Input validation passed")
            return True
        except ValueError as e:
            logger.warning(f"Input validation failed: {str(e)}")
            raise

    @staticmethod
    def load_and_prepare_data(file_path: str) -> pd.DataFrame:
        """
        Load and prepare raw data for analysis.

        Parameters
        ----------
        file_path : str
            Path to the CSV file

        Returns
        -------
        pd.DataFrame
            Prepared dataframe
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded from {file_path}. Shape: {df.shape}")

            # Drop Loan_ID as it's not predictive
            if "Loan_ID" in df.columns:
                df = df.drop("Loan_ID", axis=1)

            # Handle missing values
            for col in df.columns:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

            logger.info("Data preparation completed")
            return df
        except Exception as e:
            logger.error(f"Error loading/preparing data: {str(e)}")
            raise
