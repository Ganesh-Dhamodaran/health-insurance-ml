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

    # -----------------------------
    # ✅ SINGLE RECORD MODE
    # -----------------------------
    st.write("---")
    st.write("### Or enter a single customer record")

    age = st.number_input("Age", min_value=18, max_value=100)
    income = st.number_input("Income Level", min_value=0)
    premium = st.number_input("Premium Amount", min_value=0)
    deductible = st.number_input("Deductible", min_value=0)

    start_date = st.date_input("Policy Start Date")
    renewal_date = st.date_input("Policy Renewal Date")

    if st.button("Predict Single Record"):
        record = {
            "Age": age,
            "Income Level": income,
            "Premium Amount": premium,
            "Deductible": deductible,
            "Policy Start Date": str(start_date),
            "Policy Renewal Date": str(renewal_date)
        }

        df_single = pd.DataFrame([record])
        results = predict_fraud(df_single)

        st.write("### ✅ Prediction")
        st.dataframe(results)

        # ✅ Compact single-record probability bar
        st.write("### 📊 Fraud Probability")
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.bar(["Probability"], [results["fraud_probability"].iloc[0]], color="orange")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)