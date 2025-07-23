import streamlit as st
import pandas as pd
import pickle

# --- Load the Trained Model [cite: 62] ---
try:
    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'logistic_model.pkl' is in the same directory.")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📈 Customer Churn Prediction Model")
st.write("This app predicts customer churn for a telecom company using a logistic regression model. [cite: 6] Enter the customer's details on the left to see the churn prediction. [cite: 7]")

# --- Sidebar for User Inputs [cite: 57] ---
st.sidebar.header("Customer Details")

# Input fields for user
gender = st.sidebar.selectbox('Gender', ('Male', 'Female')) # [cite: 16]
senior_citizen = st.sidebar.selectbox('Senior Citizen', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No') # [cite: 17]
partner = st.sidebar.selectbox('Has a Partner?', ('Yes', 'No')) # [cite: 18]
dependents = st.sidebar.selectbox('Has Dependents?', ('Yes', 'No')) # [cite: 19]
phone_service = st.sidebar.selectbox('Has Phone Service?', ('Yes', 'No')) # [cite: 21]
multiple_lines = st.sidebar.selectbox('Has Multiple Lines?', ('Yes', 'No', 'No phone service')) # [cite: 22]
internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No')) # [cite: 23]
online_security = st.sidebar.selectbox('Has Online Security?', ('Yes', 'No', 'No internet service')) # [cite: 24]
online_backup = st.sidebar.selectbox('Has Online Backup?', ('Yes', 'No', 'No internet service')) # [cite: 25]
device_protection = st.sidebar.selectbox('Has Device Protection?', ('Yes', 'No', 'No internet service')) # [cite: 27]
tech_support = st.sidebar.selectbox('Has Tech Support?', ('Yes', 'No', 'No internet service')) # [cite: 28]
streaming_tv = st.sidebar.selectbox('Streams TV?', ('Yes', 'No', 'No internet service')) # [cite: 29]
streaming_movies = st.sidebar.selectbox('Streams Movies?', ('Yes', 'No', 'No internet service')) # [cite: 30]
contract = st.sidebar.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year')) # [cite: 31]
paperless_billing = st.sidebar.selectbox('Uses Paperless Billing?', ('Yes', 'No')) # [cite: 32]
payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')) # [cite: 33]

st.sidebar.header("Usage and Charges")
tenure = st.sidebar.slider('Tenure (months)', 0, 72, 1) # [cite: 20]
monthly_charges = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=70.5, step=0.1) # [cite: 34]
total_charges = st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=150.0, step=1.0) # [cite: 35]

# --- Prediction Logic ---
# Create a DataFrame from the inputs
input_data = pd.DataFrame({
    'gender': [gender], 'SeniorCitizen': [senior_citizen], 'Partner': [partner], 'Dependents': [dependents],
    'tenure': [tenure], 'PhoneService': [phone_service], 'MultipleLines': [multiple_lines],
    'InternetService': [internet_service], 'OnlineSecurity': [online_security], 'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection], 'TechSupport': [tech_support], 'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies], 'Contract': [contract], 'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method], 'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]
})

# Make prediction and get probability
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# --- Display Prediction [cite: 58] ---
st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"🔴 This customer is likely to **churn**.", icon="🚨")
    churn_probability = prediction_proba[1]
else:
    st.success(f"🟢 This customer is likely to **stay**.", icon="✅")
    churn_probability = prediction_proba[1]

st.metric(label="Probability of Churn", value=f"{churn_probability:.2%}")
st.progress(churn_probability)