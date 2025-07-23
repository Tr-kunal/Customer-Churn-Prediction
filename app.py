import streamlit as st
import pickle
import numpy as np

# Load model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Dummy encoding for simplicity (you should align with training data)
input_data = np.array([[1 if gender == "Male" else 0,
                        senior,
                        1 if partner == "Yes" else 0,
                        1 if dependents == "Yes" else 0,
                        tenure,
                        monthly,
                        total,
                        1 if contract == "One year" else 0,
                        1 if contract == "Two year" else 0]])

if st.button("Predict Churn"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.warning("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
