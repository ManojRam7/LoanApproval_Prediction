"""
Utility functions for the Loan Approval Prediction system.
"""

from src.logger import logger


def format_currency(amount: float) -> str:
    """
    Format amount as currency string.

    Parameters
    ----------
    amount : float
        Amount to format

    Returns
    -------
    str
        Formatted currency string
    """
    return f"${amount:,.2f}"


def get_prediction_message(prediction: int, probability: float = None) -> dict:
    """
    Generate a user-friendly prediction message.

    Parameters
    ----------
    prediction : int
        Model prediction (0 or 1)
    probability : float, optional
        Prediction probability

    Returns
    -------
    dict
        Dictionary with status, message, and confidence
    """
    if prediction == 1:
        status = "✅ APPROVED"
        message = "Your loan application has been approved!"
    else:
        status = "❌ NOT APPROVED"
        message = "Unfortunately, your loan application was not approved."

    result = {"status": status, "message": message}

    if probability is not None:
        confidence = max(probability) * 100
        result["confidence"] = f"{confidence:.1f}%"

    logger.info(f"Prediction message generated: {status}")
    return result


def validate_income_ratio(
    applicant_income: float, coapplicant_income: float, loan_amount: float
) -> dict:
    """
    Validate loan amount against total income.

    Parameters
    ----------
    applicant_income : float
        Applicant's income
    coapplicant_income : float
        Co-applicant's income
    loan_amount : float
        Requested loan amount

    Returns
    -------
    dict
        Validation results including debt-to-income ratio
    """
    total_income = applicant_income + coapplicant_income

    if total_income == 0:
        return {"valid": False, "ratio": None, "message": "Total income cannot be zero"}

    debt_to_income_ratio = loan_amount / total_income

    return {
        "valid": True,
        "ratio": round(debt_to_income_ratio, 2),
        "message": f"Debt-to-Income Ratio: {debt_to_income_ratio:.2%}",
    }
