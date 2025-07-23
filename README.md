# 📱 Customer Churn Prediction Web App

This is a Streamlit-based web app that predicts whether a telecom customer is likely to churn. It uses a logistic regression model trained on customer data.

## 🚀 Features
- **Real-time Predictions**: Instantly predict churn based on user-input features.
- **Simple UI**: Clean and user-friendly interface built with Streamlit.
- **Logistic Regression Model**: Utilizes a classic and interpretable model for classification.
- **Easy Deployment**: Fully containerized and deployable on Streamlit Cloud.

## 🧠 ML Model
The prediction model is a **Logistic Regression** classifier trained on 19 customer features. The preprocessing pipeline includes:
- **Handling Missing Values**: Fills missing `TotalCharges` with the median value.
- **One-Hot Encoding**: Converts all categorical features into a numerical format.
- **Standardization**: Scales numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) to have a mean of 0 and a standard deviation of 1.

## 📦 Requirements
All necessary Python libraries are listed in the `requirements.txt` file.

## ▶️ Run Locally
1.  **Clone the repository and navigate to the project folder.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
