import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSACTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "transactions_cleaned.csv")

transactions = pd.read_csv(TRANSACTIONS_PATH)

st.title("Sales Analytics")

transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
transactions["Month"] = transactions["InvoiceDate"].dt.to_period("M").astype(str)

monthly_sales = transactions.groupby("Month")["Revenue"].sum()

st.subheader("Monthly Revenue Trend")
st.line_chart(monthly_sales)

st.subheader("Top 10 Customers by Revenue")
top_customers = (
    transactions.groupby("CustomerID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(top_customers)
