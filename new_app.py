import streamlit as st
import pandas as pd
import pickle

# --- Load the Trained Model ---
# The model is a pipeline that includes preprocessing and the logistic regression classifier.
try:
    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'logistic_model.pkl' is in the same directory.")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📱 Telecom Customer Churn Prediction")
st.write("This app uses a logistic regression model to predict whether a customer is likely to churn based on their account details.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Enter Customer Details")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ('Male', 'Female'))
    partner = st.selectbox('Has a Partner?', ('Yes', 'No'))
    dependents = st.selectbox('Has Dependents?', ('Yes', 'No'))
    phone_service = st.selectbox('Has Phone Service?', ('Yes', 'No'))
    paperless_billing = st.selectbox('Uses Paperless Billing?', ('Yes', 'No'))
    senior_citizen = st.selectbox('Is a Senior Citizen?', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    tenure = st.slider('Tenure (months)', 0, 72, 24)

with col2:
    contract = st.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    payment_method = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    multiple_lines = st.selectbox('Has Multiple Lines?', ('No phone service', 'No', 'Yes'))
    internet_service = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_security = st.selectbox('Has Online Security?', ('No internet service', 'No', 'Yes'))
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=70.0, step=0.1)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=1500.0, step=1.0)

# --- Prediction Logic ---
# Create a DataFrame from the inputs with the correct column names
# These names must match the columns used to train the model.
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': ['No' for _ in [internet_service]],  # Placeholder, will be updated below
    'OnlineBackup': ['No' for _ in [internet_service]],    # Placeholder
    'DeviceProtection': ['No' for _ in [internet_service]],# Placeholder
    'TechSupport': ['No' for _ in [internet_service]],     # Placeholder
    'StreamingTV': ['No' for _ in [internet_service]],     # Placeholder
    'StreamingMovies': ['No' for _ in [internet_service]], # Placeholder
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
}, index=[0])


# --- Display Prediction ---
st.header("Prediction Result")

# When the 'Predict' button is clicked
if st.button('Predict Churn'):
    # Make prediction using the loaded pipeline
    # The pipeline handles all preprocessing (scaling, encoding) automatically
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"🔴 This customer is likely to **churn**.", icon="🚨")
        churn_probability = prediction_proba[1]
    else:
        st.success(f"🟢 This customer is likely to **stay**.", icon="✅")
        churn_probability = prediction_proba[0]

    st.metric(label="Confidence", value=f"{churn_probability:.2%}")
    st.progress(churn_probability)
