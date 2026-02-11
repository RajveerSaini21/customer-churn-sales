import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CUSTOMERS_PATH = os.path.join(BASE_DIR, "data", "processed", "customers_cleaned.csv")
TRANSACTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "transactions_cleaned.csv")

customers = pd.read_csv(CUSTOMERS_PATH)
transactions = pd.read_csv(TRANSACTIONS_PATH)

if customers["Churn"].dtype == object:
    customers["Churn"] = customers["Churn"].map({"No": 0, "Yes": 1})

st.title("Executive Overview")

total_customers = len(customers)
churn_rate = round(customers["Churn"].mean() * 100, 2)
total_revenue = int(transactions["Revenue"].sum())

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate (%)", churn_rate)
col3.metric("Total Revenue", f"{total_revenue:,}")

st.markdown("---")

st.subheader("Churn Distribution")
st.bar_chart(customers["Churn"].value_counts())
