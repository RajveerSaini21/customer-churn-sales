import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEGMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_segments.csv")

segments = pd.read_csv(SEGMENTS_PATH)

if segments["Churn"].dtype == object:
    segments["Churn"] = segments["Churn"].map({"No": 0, "Yes": 1})

st.title("Customer Segmentation")

st.subheader("Segment Distribution")
st.bar_chart(segments["Segment"].value_counts())

st.subheader("Churn Rate by Segment")
segment_churn = segments.groupby("Segment")["Churn"].mean()
st.bar_chart(segment_churn)
