import streamlit as st
import pandas as pd
import os

# ================= PATH SETUP ================= #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUSTOMERS_PATH = os.path.join(BASE_DIR, "data", "processed", "customers_cleaned.csv")

# ================= LOAD DATA ================= #
customers = pd.read_csv(CUSTOMERS_PATH)

# Convert Churn if needed
if customers["Churn"].dtype == object:
    customers["Churn"] = customers["Churn"].map({"No": 0, "Yes": 1})

st.set_page_config(layout="wide")
st.title("Customer Insights & Behavior Analysis")

st.markdown("""
This page analyzes customer behavior patterns to understand
what factors influence churn.
""")

# ================= SIDEBAR FILTERS ================= #
st.sidebar.header("Filters")

min_tenure, max_tenure = st.sidebar.slider(
    "Select Tenure Range",
    int(customers["tenure"].min()),
    int(customers["tenure"].max()),
    (
        int(customers["tenure"].min()),
        int(customers["tenure"].max())
    )
)

filtered = customers[
    (customers["tenure"] >= min_tenure) &
    (customers["tenure"] <= max_tenure)
]

# ================= TENURE DISTRIBUTION ================= #
st.subheader("Tenure Distribution")

st.bar_chart(filtered["tenure"].value_counts().sort_index())

st.markdown("---")

# ================= MONTHLY CHARGES ANALYSIS ================= #
st.subheader("Monthly Charges vs Churn")

charges_by_churn = filtered.groupby("Churn")["MonthlyCharges"].mean()
charges_by_churn.index = ["Not Churned", "Churned"]

st.bar_chart(charges_by_churn)

st.markdown("---")

# ================= CONTRACT TYPE ANALYSIS ================= #
if "Contract" in filtered.columns:
    st.subheader("Contract Type vs Churn")

    contract_churn = (
        filtered.groupby("Contract")["Churn"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(contract_churn)

st.markdown("---")

# ================= PAYMENT METHOD ANALYSIS ================= #
if "PaymentMethod" in filtered.columns:
    st.subheader("Payment Method vs Churn")

    payment_churn = (
        filtered.groupby("PaymentMethod")["Churn"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(payment_churn)

st.markdown("---")

# ================= SUMMARY INSIGHTS ================= #
st.subheader("Key Insights")

high_risk_contract = contract_churn.idxmax() if "Contract" in filtered.columns else "N/A"

st.markdown(f"""
• Customers with **short tenure** show higher churn rates.  
• Higher **monthly charges** are associated with increased churn.  
• The contract type with highest churn rate: **{high_risk_contract}**.  
• Certain payment methods correlate with churn behavior.  
""")
