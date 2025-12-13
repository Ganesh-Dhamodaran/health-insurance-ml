import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fraud_detection.inference import predict_fraud


def fraud_tab():
    st.header("🔍 Fraud Detection")

    st.write("""
    Upload customer or claim data to detect potential fraud using the trained ML model.
    """)

    uploaded_file = st.file_uploader("Upload claim/customer CSV", type=["csv"])

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Run prediction
    try:
        fraud_df = predict_fraud(df)
    except FileNotFoundError:
        st.error("❌ Fraud detection model not found.")
        st.info("Ensure fraud_detection_model.pkl and preprocessing artifacts exist in /models.")
        return
    except ValueError as e:
        st.error(f"❌ Data validation error:\n{e}")
        return
    except Exception as e:
        st.error(f"❌ Unexpected error:\n{e}")
        return

    st.subheader("🔎 Fraud Predictions")
    st.dataframe(fraud_df.head())

    # Probability distribution
    if "fraud_probability" in fraud_df.columns:
        st.subheader("📊 Fraud Probability Distribution")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(fraud_df["fraud_probability"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    # Fraud vs Non-Fraud count
    if "fraud_prediction" in fraud_df.columns:
        st.subheader("📈 Fraud vs Non‑Fraud Count")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=fraud_df, x="fraud_prediction", palette="Set2", ax=ax)
        st.pyplot(fig)

    # Download results
    st.subheader("⬇️ Download Fraud Predictions")
    st.download_button(
        label="Download fraud_predictions.csv",
        data=fraud_df.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )