# import streamlit as st
# import pandas as pd
# import os

# # ---------------- PATH SETUP ---------------- #
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CUSTOMERS_PATH = os.path.join(
#     BASE_DIR, "data", "processed", "customers_cleaned.csv"
# )
# TRANSACTIONS_PATH = os.path.join(
#     BASE_DIR, "data", "processed", "transactions_cleaned.csv"
# )

# # ---------------- LOAD DATA ---------------- #
# customers = pd.read_csv(CUSTOMERS_PATH)
# transactions = pd.read_csv(TRANSACTIONS_PATH)

# # ---------------- DASHBOARD ---------------- #
# st.title("Customer Churn & Sales Dashboard")

# st.subheader("Churn Distribution")

# churn_counts = customers["Churn"].round().value_counts()
# churn_counts.index = ["Not Churned", "Churned"]

# st.bar_chart(churn_counts)


# st.subheader("Monthly Sales Trend")
# transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
# transactions["month"] = transactions["InvoiceDate"].dt.to_period("M").astype(str)

# monthly_sales = transactions.groupby("month")["Revenue"].sum()
# st.line_chart(monthly_sales)


import streamlit as st
import pandas as pd
import os
import joblib

# ================= PATH SETUP ================= #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CUSTOMERS_PATH = os.path.join(BASE_DIR, "data", "processed", "customers_cleaned.csv")
TRANSACTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "transactions_cleaned.csv")
SEGMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_segments.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

# ================= LOAD DATA ================= #
customers = pd.read_csv(CUSTOMERS_PATH)
transactions = pd.read_csv(TRANSACTIONS_PATH)
segments = pd.read_csv(SEGMENTS_PATH)

# ---------------- FIX CHURN COLUMN ---------------- #
if customers["Churn"].dtype == object:
    customers["Churn"] = customers["Churn"].map({"No": 0, "Yes": 1})

if segments["Churn"].dtype == object:
    segments["Churn"] = segments["Churn"].map({"No": 0, "Yes": 1})

# ---------------- LOAD MODEL SAFELY ---------------- #
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.warning("Churn model not found. Please train the model first.")

# ================= PAGE CONFIG ================= #
st.set_page_config(
    page_title="Customer Churn & Sales Dashboard",
    layout="wide"
)

st.title("Customer Churn & Sales Dashboard")

# ================= KPI CARDS ================= #
total_customers = len(customers)
churn_rate = round(customers["Churn"].mean() * 100, 2)
total_revenue = int(transactions["Revenue"].sum())
avg_revenue = int(
    transactions["Revenue"].sum() / transactions["CustomerID"].nunique()
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate (%)", churn_rate)
col3.metric("Total Revenue", f"{total_revenue:,}")
col4.metric("Avg Revenue / Customer", f"{avg_revenue:,}")

st.markdown("---")

# ================= SIDEBAR FILTER ================= #
st.sidebar.header("Filters")

min_revenue = st.sidebar.slider(
    "Minimum Transaction Revenue",
    int(transactions["Revenue"].min()),
    int(transactions["Revenue"].max()),
    int(transactions["Revenue"].min())
)

filtered_txn = transactions.loc[transactions["Revenue"] >= min_revenue].copy()

# ================= CHURN DISTRIBUTION ================= #
st.subheader("Churn Distribution")

churn_counts = customers["Churn"].value_counts()
churn_counts.index = ["Not Churned", "Churned"]

st.bar_chart(churn_counts)

# ================= SALES TREND ================= #
st.subheader("Monthly Sales Trend")

filtered_txn["InvoiceDate"] = pd.to_datetime(filtered_txn["InvoiceDate"])
filtered_txn["Month"] = filtered_txn["InvoiceDate"].dt.to_period("M").astype(str)

monthly_sales = filtered_txn.groupby("Month")["Revenue"].sum()
st.line_chart(monthly_sales)

st.markdown("---")

# ================= CHURN PREDICTION ================= #
if model is not None:
    customers_features = customers.drop("Churn", axis=1)

    customers["Churn_Probability"] = model.predict_proba(customers_features)[:, 1]

    st.subheader("High Risk Customers (Churn Probability > 70%)")

    high_risk = customers[customers["Churn_Probability"] > 0.7]
    st.dataframe(high_risk.head(10))
else:
    st.info("Train the churn model to see churn probability predictions.")

st.markdown("---")

# ================= CUSTOMER SEGMENTATION ================= #
st.subheader("Customer Segmentation Analysis")

segment_churn = segments.groupby("Segment")["Churn"].mean()
st.bar_chart(segment_churn)

st.subheader("Top Customers by Revenue")

top_customers = (
    filtered_txn.groupby("CustomerID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(top_customers)

# ================= FOOTER ================= #
st.markdown(
    """
    ---
    **Customer Churn Prediction & Sales Dashboard**  
    Built using Python, Machine Learning, and Streamlit
    """
)
