"""
Streamlit application for Loan Approval Prediction.
An interactive web interface for predicting loan approval status.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model_manager import get_model_manager
from src.data_processor import DataProcessor
from src.utils import format_currency, get_prediction_message, validate_income_ratio
from src.logger import logger
from src.config import STREAMLIT_PAGE_LAYOUT, STREAMLIT_THEME

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout=STREAMLIT_PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .header-style {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subheader-style {
        font-size: 1.2em;
        color: #555;
        margin-bottom: 20px;
    }
    .prediction-approved {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: #155724;
        font-weight: bold;
    }
    .prediction-rejected {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #721c24;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediction_result = None


def main():
    """Main Streamlit application."""
    # Title
    st.markdown(
        '<div class="header-style">💰 Loan Approval Predictor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subheader-style">Instant ML-Based Loan Approval Assessment</div>',
        unsafe_allow_html=True,
    )

    # Sidebar information
    with st.sidebar:
        st.markdown("### 📋 Application Information")
        st.info(
            """
            **How it works:**
            - Enter your loan details
            - Our ML model analyzes your profile
            - Get instant approval prediction
            
            **Model Details:**
            - Algorithm: Random Forest Classifier
            - Accuracy: 82% on test data
            - Features: 11 applicant attributes
            """
        )

        st.markdown("### 🎯 Feature Importance Tips")
        st.warning(
            """
            These factors impact approval:
            - Credit History (High Impact)
            - Income & Co-income
            - Loan Amount
            - Education Level
            - Employment Status
            """
        )

    # Create columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Personal Information")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            help="Applicant's gender",
        )

        married = st.selectbox(
            "Marital Status",
            ["Yes", "No"],
            help="Are you married?",
        )

        dependents = st.selectbox(
            "Number of Dependents",
            ["0", "1", "2", "3+"],
            help="Number of people depending on you",
        )

        education = st.selectbox(
            "Education Level",
            ["Graduate", "Not Graduate"],
            help="Highest education level completed",
        )

        self_employed = st.selectbox(
            "Employment Type",
            ["No", "Yes"],
            help="Are you self-employed?",
        )

    with col2:
        st.subheader("💼 Financial Information")

        applicant_income = st.number_input(
            "Annual Income ($)",
            min_value=0,
            value=5000,
            step=100,
            help="Your annual income",
        )

        coapplicant_income = st.number_input(
            "Co-Applicant Income ($)",
            min_value=0,
            value=0,
            step=100,
            help="Co-applicant's annual income (if any)",
        )

        loan_amount = st.number_input(
            "Loan Amount Requested ($)",
            min_value=1,
            value=100000,
            step=1000,
            help="Amount of loan you want to borrow",
        )

        loan_amount_term = st.number_input(
            "Loan Term (Days)",
            min_value=1,
            value=360,
            step=30,
            help="Duration of the loan in days",
        )

    # Second row of inputs
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📍 Additional Details")

        credit_history = st.selectbox(
            "Credit History",
            [1, 0],
            format_func=lambda x: "Good (1)" if x == 1 else "Poor/No History (0)",
            help="Your credit history status",
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Rural", "Semiurban"],
            help="Location of the property",
        )

    # Validation and income ratio
    with col4:
        st.subheader("💡 Quick Metrics")

        try:
            income_validation = validate_income_ratio(
                applicant_income, coapplicant_income, loan_amount
            )

            if income_validation["valid"]:
                total_income = applicant_income + coapplicant_income
                st.metric("Total Income", format_currency(total_income))
                st.metric(
                    "Debt-to-Income Ratio",
                    f"{income_validation['ratio']:.2%}",
                    help="Lower is better",
                )
            else:
                st.error(income_validation["message"])
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            logger.error(f"Metrics calculation error: {e}")

    # Prediction button
    st.divider()

    col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])

    with col_btn_2:
        if st.button(
            "🔍 Predict Loan Approval",
            use_container_width=True,
            type="primary",
        ):
            try:
                # Initialize components
                model_manager = get_model_manager()
                data_processor = DataProcessor()

                # Validate inputs
                data_processor.validate_inputs(
                    applicant_income=applicant_income,
                    coapplicant_income=coapplicant_income,
                    loan_amount=loan_amount,
                    loan_amount_term=loan_amount_term,
                )

                # Process input
                processed_input = data_processor.process_input(
                    gender=gender,
                    married=married,
                    dependents=dependents,
                    education=education,
                    self_employed=self_employed,
                    applicant_income=applicant_income,
                    coapplicant_income=coapplicant_income,
                    loan_amount=loan_amount,
                    loan_amount_term=loan_amount_term,
                    credit_history=credit_history,
                    property_area=property_area,
                )

                # Make prediction
                prediction = model_manager.predict(processed_input)[0]

                # Get probabilities
                proba = model_manager.predict_proba(processed_input)[0]

                # Store results in session state
                st.session_state.prediction_made = True
                st.session_state.prediction_result = {
                    "prediction": prediction,
                    "probability": proba,
                }

                logger.info(f"Prediction made: {prediction}")

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                logger.error(f"Prediction error: {e}")

    # Display results
    if st.session_state.prediction_made and st.session_state.prediction_result:
        st.divider()
        st.subheader("🎯 Prediction Results")

        prediction = st.session_state.prediction_result["prediction"]
        probability = st.session_state.prediction_result["probability"]

        result = get_prediction_message(prediction, probability)

        if prediction == 1:
            st.markdown(
                f"""<div class="prediction-approved">
                {result['status']}<br/>
                {result['message']}<br/>
                Confidence: {result.get('confidence', 'N/A')}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="prediction-rejected">
                {result['status']}<br/>
                {result['message']}<br/>
                Confidence: {result.get('confidence', 'N/A')}
                </div>""",
                unsafe_allow_html=True,
            )

        # Detailed breakdown
        st.subheader("📊 Prediction Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Approval Probability",
                f"{probability[1]*100:.1f}%",
                help="Confidence in approval",
            )

        with col2:
            st.metric(
                "Rejection Probability",
                f"{probability[0]*100:.1f}%",
                help="Confidence in rejection",
            )

        with col3:
            confidence_level = "High" if max(probability) > 0.8 else "Medium" if max(probability) > 0.6 else "Low"
            st.metric("Confidence Level", confidence_level)

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 0.9em;'>
        <p>🤖 Powered by Random Forest Machine Learning Model</p>
        <p>This prediction is based on historical data patterns. Always verify with financial advisors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

