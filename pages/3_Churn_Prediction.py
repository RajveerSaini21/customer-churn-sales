import streamlit as st
import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CUSTOMERS_PATH = os.path.join(BASE_DIR, "data", "processed", "customers_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

customers = pd.read_csv(CUSTOMERS_PATH)

if customers["Churn"].dtype == object:
    customers["Churn"] = customers["Churn"].map({"No": 0, "Yes": 1})

st.title("Churn Prediction")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

    features = customers.drop("Churn", axis=1)
    customers["Churn_Probability"] = model.predict_proba(features)[:, 1]

    st.subheader("High Risk Customers")
    high_risk = customers[customers["Churn_Probability"] > 0.7]

    st.dataframe(high_risk.head(10))
else:
    st.warning("Model not found. Train the model first.")
