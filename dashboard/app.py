# import streamlit as st
# import pandas as pd
# import os
# import joblib

# # ================= PATH SETUP ================= #
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CUSTOMERS_PATH = os.path.join(BASE_DIR, "data", "processed", "customers_cleaned.csv")
# TRANSACTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "transactions_cleaned.csv")
# SEGMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_segments.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

# # ================= LOAD DATA ================= #
# customers = pd.read_csv(CUSTOMERS_PATH)
# transactions = pd.read_csv(TRANSACTIONS_PATH)
# segments = pd.read_csv(SEGMENTS_PATH)

# # ---------------- FIX CHURN COLUMN ---------------- #
# if customers["Churn"].dtype == object:
#     customers["Churn"] = customers["Churn"].map({"No": 0, "Yes": 1})

# if segments["Churn"].dtype == object:
#     segments["Churn"] = segments["Churn"].map({"No": 0, "Yes": 1})

# # ---------------- LOAD MODEL SAFELY ---------------- #
# model = None
# if os.path.exists(MODEL_PATH):
#     model = joblib.load(MODEL_PATH)
# else:
#     st.warning("Churn model not found. Please train the model first.")

# # ================= PAGE CONFIG ================= #
# st.set_page_config(
#     page_title="Customer Churn & Sales Dashboard",
#     layout="wide"
# )

# st.title("Customer Churn & Sales Dashboard")

# # ================= KPI CARDS ================= #
# total_customers = len(customers)
# churn_rate = round(customers["Churn"].mean() * 100, 2)
# total_revenue = int(transactions["Revenue"].sum())
# avg_revenue = int(
#     transactions["Revenue"].sum() / transactions["CustomerID"].nunique()
# )

# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Total Customers", total_customers)
# col2.metric("Churn Rate (%)", churn_rate)
# col3.metric("Total Revenue", f"{total_revenue:,}")
# col4.metric("Avg Revenue / Customer", f"{avg_revenue:,}")

# st.markdown("---")

# # ================= SIDEBAR FILTER ================= #
# st.sidebar.header("Filters")

# min_revenue = st.sidebar.slider(
#     "Minimum Transaction Revenue",
#     int(transactions["Revenue"].min()),
#     int(transactions["Revenue"].max()),
#     int(transactions["Revenue"].min())
# )

# filtered_txn = transactions.loc[transactions["Revenue"] >= min_revenue].copy()

# # ================= CHURN DISTRIBUTION ================= #
# st.subheader("Churn Distribution")

# churn_counts = customers["Churn"].value_counts()
# churn_counts.index = ["Not Churned", "Churned"]

# st.bar_chart(churn_counts)

# # ================= SALES TREND ================= #
# st.subheader("Monthly Sales Trend")

# filtered_txn["InvoiceDate"] = pd.to_datetime(filtered_txn["InvoiceDate"])
# filtered_txn["Month"] = filtered_txn["InvoiceDate"].dt.to_period("M").astype(str)

# monthly_sales = filtered_txn.groupby("Month")["Revenue"].sum()
# st.line_chart(monthly_sales)

# st.markdown("---")

# # ================= CHURN PREDICTION ================= #
# if model is not None:
#     customers_features = customers.drop("Churn", axis=1)

#     customers["Churn_Probability"] = model.predict_proba(customers_features)[:, 1]

#     st.subheader("High Risk Customers (Churn Probability > 70%)")

#     high_risk = customers[customers["Churn_Probability"] > 0.7]
#     st.dataframe(high_risk.head(10))
# else:
#     st.info("Train the churn model to see churn probability predictions.")

# st.markdown("---")

# # ================= CUSTOMER SEGMENTATION ================= #
# st.subheader("Customer Segmentation Analysis")

# segment_churn = segments.groupby("Segment")["Churn"].mean()
# st.bar_chart(segment_churn)

# st.subheader("Top Customers by Revenue")

# top_customers = (
#     filtered_txn.groupby("CustomerID")["Revenue"]
#     .sum()
#     .sort_values(ascending=False)
#     .head(10)
# )

# st.dataframe(top_customers)

# # ================= FOOTER ================= #
# st.markdown(
#     """
#     ---
#     **Customer Churn Prediction & Sales Dashboard**  
#     Built using Python, Machine Learning, and Streamlit
#     """
# )


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

st.set_page_config(
    page_title="Customer Churn & Sales Dashboard",
    layout="wide"
)

customers = pd.read_csv(CUSTOMERS_PATH)
transactions = pd.read_csv(TRANSACTIONS_PATH)
segments = pd.read_csv(SEGMENTS_PATH)

if customers["Churn"].dtype == object:
    customers["Churn"] = customers["Churn"].map({"No": 0, "Yes": 1})

if segments["Churn"].dtype == object:
    segments["Churn"] = segments["Churn"].map({"No": 0, "Yes": 1})

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "Home",
        "Executive Overview",
        "Customer Insights",
        "Churn Prediction",
        "Sales Analytics",
        "Customer Segmentation"
    ]
)

if page == "Home":

    st.markdown(
        """
        <h1 style='text-align: center; font-size: 42px;'>
        Customer Churn Prediction & Sales Analytics Platform
        </h1>
        <p style='text-align: center; font-size:18px; color:gray;'>
        AI-powered customer intelligence dashboard for retention and revenue growth
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ================= KPI PREVIEW ================= #
    total_customers = len(customers)
    churn_rate = round(customers["Churn"].mean() * 100, 2)
    total_revenue = int(transactions["Revenue"].sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churn Rate", f"{churn_rate}%")
    col3.metric("Total Revenue", f"{total_revenue:,}")

    st.markdown("---")

    # ================= FEATURE CARDS ================= #
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Intelligence")
        st.markdown("""
        • Behavioral analysis  
        • Churn risk detection  
        • Customer segmentation  
        • Retention insights  
        """)

    with col2:
        st.subheader("Sales Analytics")
        st.markdown("""
        • Revenue trends  
        • Top customers  
        • Monthly growth analysis  
        • Revenue impact of churn  
        """)

    st.markdown("---")

    st.subheader("What This System Does")

    st.info("""
    This platform combines machine learning and business analytics 
    to help organizations identify at-risk customers, 
    understand churn drivers, and improve revenue performance.
    """)

    st.markdown("---")

    st.success("Use the sidebar to explore insights and predictions.")


elif page == "Executive Overview":

    st.title("Executive Overview")

    total_customers = len(customers)
    churn_rate = round(customers["Churn"].mean() * 100, 2)
    total_revenue = int(transactions["Revenue"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total_customers)
    col2.metric("Churn Rate (%)", churn_rate)
    col3.metric("Total Revenue", f"{total_revenue:,}")

    st.subheader("Churn Distribution")
    st.bar_chart(customers["Churn"].value_counts())


elif page == "Customer Insights":

    st.title("Customer Insights")

    st.subheader("Tenure Distribution")
    st.bar_chart(customers["tenure"].value_counts().sort_index())

    st.subheader("Average Monthly Charges by Churn")
    charges = customers.groupby("Churn")["MonthlyCharges"].mean()
    charges.index = ["Not Churned", "Churned"]
    st.bar_chart(charges)

elif page == "Churn Prediction":

    st.title("Churn Prediction")

    if model is not None:
        features = customers.drop("Churn", axis=1)
        customers["Churn_Probability"] = model.predict_proba(features)[:, 1]

        st.subheader("High Risk Customers (> 70%)")
        high_risk = customers[customers["Churn_Probability"] > 0.7]
        st.dataframe(high_risk.head(10))
    else:
        st.warning("Model not found. Train the model first.")


elif page == "Sales Analytics":

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

elif page == "Customer Segmentation":

    st.title("Customer Segmentation")

    st.subheader("Segment Distribution")
    st.bar_chart(segments["Segment"].value_counts())

    st.subheader("Churn Rate by Segment")
    segment_churn = segments.groupby("Segment")["Churn"].mean()
    st.bar_chart(segment_churn)
