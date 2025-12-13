import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from segmentation.inference import segment_customers
from segmentation.profiling import cluster_profile_summary


def segmentation_tab():
    st.header("🧩 Customer Segmentation")

    st.write("""
    Upload customer data to view their segment, PCA position, and cluster characteristics.
    This uses the trained segmentation pipeline (preprocessor + KMeans + PCA).
    """)

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is None:
        return

    # Load CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Run segmentation with error handling
    try:
        segmented_df = segment_customers(df)
    except FileNotFoundError as e:
        st.error("❌ Segmentation models not found.")
        st.info("Run the training script first:\n\n`python train_segmentation.py`")
        return
    except ValueError as e:
        st.error(f"❌ Data validation error:\n{e}")
        return
    except Exception as e:
        st.error(f"❌ Unexpected error during segmentation:\n{e}")
        return

    # Show segmented output
    st.subheader("🧩 Segmented Customers")
    st.dataframe(segmented_df.head())

    # PCA scatter plot
    st.subheader("📊 PCA Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=segmented_df,
        x="pca_x",
        y="pca_y",
        hue="cluster",
        palette="tab10",
        alpha=0.7,
        ax=ax
    )
    ax.set_title("Customer Segments (PCA + KMeans)")
    st.pyplot(fig)

    # Cluster profiling
    st.subheader("📈 Cluster Profile Summary")
    profile = cluster_profile_summary(segmented_df)
    st.dataframe(profile)

    # Download segmented output
    st.subheader("⬇️ Download Segmented Data")
    st.download_button(
        label="Download segmented_customers.csv",
        data=segmented_df.to_csv(index=False),
        file_name="segmented_customers.csv",
        mime="text/csv"
    )