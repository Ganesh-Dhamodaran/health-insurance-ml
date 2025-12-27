import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from fraud_detection.inference import predict_fraud

def fraud_tab():
    st.header("🔍 Fraud Detection")

    # -----------------------------
    # ✅ CSV UPLOAD MODE
    # -----------------------------
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### ✅ Uploaded Data")
        st.dataframe(df)

        if st.button("Run Fraud Prediction"):
            results = predict_fraud(df)

            st.write("### 🔎 Prediction Results")
            st.dataframe(results)

            # -----------------------------
            # ✅ PLOTS (Compact + Clean)
            # -----------------------------
            col1, col2 = st.columns(2)

            # ✅ Plot 1: Fraud Probability Distribution
            with col1:
                st.write("### 📊 Fraud Probability Distribution")
                fig, ax = plt.subplots(figsize=(4, 3))
                results["fraud_probability"].hist(
                    bins=10, ax=ax, color="teal", edgecolor="black"
                )
                ax.set_xlabel("Probability")
                ax.set_ylabel("Count")
                plt.tight_layout()
                st.pyplot(fig)

            # ✅ Plot 2: Fraud vs Non-Fraud Count
            with col2:
                st.write("### 📊 Fraud Prediction Counts")

                fraud_counts = (
                    results["fraud_prediction"]
                    .value_counts()
                    .sort_index()
                )
                fraud_counts.index = fraud_counts.index.map({0: "Not Fraud", 1: "Fraud"})

                fig2, ax2 = plt.subplots(figsize=(4, 3))
                fraud_counts.plot(
                    kind="bar",
                    ax=ax2,
                    color=["green", "red"][: len(fraud_counts)]
                )
                ax2.set_ylabel("Count")
                ax2.set_xticklabels(fraud_counts.index, rotation=0)
                plt.tight_layout()
                st.pyplot(fig2)