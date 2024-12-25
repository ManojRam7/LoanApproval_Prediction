import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Title of the app
st.title("Loan Approval Prediction")

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open("random_forest_model_new.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Input features
st.header("Enter Loan Details")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Convert inputs into a dataframe
input_data = pd.DataFrame({
    "Gender": [1 if gender == "Male" else 0],
    "Married": [1 if married == "Yes" else 0],
    "Dependents": [0 if dependents == "0" else (3 if dependents == "3+" else int(dependents))],
    "Education": [0 if education == "Graduate" else 1],
    "Self_Employed": [1 if self_employed == "Yes" else 0],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [0 if property_area == "Urban" else (1 if property_area == "Rural" else 2)]
})

# Prediction button
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    st.write(f"Loan Application Status: {result}")

