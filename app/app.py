import sys
import os

# Ensure project root is in Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st

# Import dashboard tabs
from dashboard.segmentation_tab import segmentation_tab
from dashboard.fraud_tab import fraud_tab


# Streamlit App Layout
st.set_page_config(
    page_title="Health Insurance ML Dashboard",
    layout="wide"
)

st.title("🏥 Health Insurance ML Dashboard")

tab1, tab2 = st.tabs(["🔍 Fraud Detection", "🧩 Customer Segmentation"])

with tab1:
    fraud_tab()

with tab2:
    segmentation_tab()