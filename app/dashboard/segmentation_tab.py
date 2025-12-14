import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

    # Run segmentation
    try:
        segmented_df = segment_customers(df)
    except Exception as e:
        st.error(f"❌ Segmentation error:\n{e}")
        return

    st.subheader("🧩 Segmented Customers")
    st.dataframe(segmented_df.head())

    # ---------------------------------------------------------
    # ✅ PCA Scatter Plot (Compact)
    # ---------------------------------------------------------
    st.subheader("📊 PCA Cluster Visualization")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.scatterplot(
        data=segmented_df,
        x="pca_x",
        y="pca_y",
        hue="cluster",
        palette="tab10",
        alpha=0.7,
        ax=ax
    )
    ax.set_title("Customer Segments (PCA + KMeans)", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, bbox_inches="tight")

    # ---------------------------------------------------------
    # ✅ Cluster Profile Summary
    # ---------------------------------------------------------
    st.subheader("📈 Cluster Profile Summary")
    profile = cluster_profile_summary(segmented_df)
    st.dataframe(profile)

    # ---------------------------------------------------------
    # ✅ Cluster Distribution Pie Chart
    # ---------------------------------------------------------
    st.subheader("🥧 Cluster Distribution")

    cluster_counts = segmented_df["cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.pie(
        cluster_counts,
        labels=[f"Cluster {i}" for i in cluster_counts.index],
        autopct="%1.1f%%",
        colors=sns.color_palette("tab10", len(cluster_counts))
    )
    ax.set_title("Cluster Distribution", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, bbox_inches="tight")

    # ---------------------------------------------------------
    # ✅ Cluster-wise Feature Comparison (Grouped + 2-Column Grid)
    # ---------------------------------------------------------
    st.subheader("📊 Cluster-wise Feature Comparison")

    numeric_cols = segmented_df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["pca_x", "pca_y"]]

    group_size = 4
    feature_groups = [
        numeric_cols[i:i + group_size] for i in range(0, len(numeric_cols), group_size)
    ]

    for idx, group in enumerate(feature_groups):
        with st.expander(f"Feature Group {idx + 1}", expanded=False):
            col1, col2 = st.columns(2)

            for i, feature in enumerate(group):
                target_col = col1 if i % 2 == 0 else col2

                with target_col:
                    st.write(f"### 🔹 {feature}")

                    fig, ax = plt.subplots(figsize=(4.5, 3))
                    sns.barplot(
                        data=segmented_df,
                        x="cluster",
                        y=feature,
                        palette="tab10",
                        estimator=np.mean,
                        ax=ax
                    )
                    ax.set_title(f"Avg {feature} per Cluster", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig, bbox_inches="tight")

                    means = segmented_df.groupby("cluster")[feature].mean()
                    best_cluster = means.idxmax()
                    st.info(f"📌 **Cluster {best_cluster} has the highest {feature} ({means.max():.2f}).**")

    # ---------------------------------------------------------
    # ✅ Premium Responsive Cluster Tiles Using st.button
    # ---------------------------------------------------------
    st.subheader("🕸️ Radar Chart: Cluster Characteristics")

    clusters = sorted(segmented_df["cluster"].unique())
    icons = ["🔥", "🌟", "⚡", "🌙", "🌈", "💎", "🚀", "🎯", "💠", "⭐"]

    # ✅ CSS for tile styling
    tile_css = """
    <style>
    div.stButton > button {
        background-color: #f0f2f6;
        color: #333;
        padding: 12px 18px;
        border-radius: 10px;
        border: 2px solid transparent;
        font-weight: 600;
        transition: 0.2s ease-in-out;
        width: 100%;
    }
    div.stButton > button:hover {
        border: 2px solid #4a90e2;
        background-color: #e8f1ff;
        color: #000;
    }
    div.stButton.selected > button {
        border: 2px solid #1f77b4;
        background-color: #dcecff;
        color: #000;
    }
    </style>
    """
    st.markdown(tile_css, unsafe_allow_html=True)

    if "selected_cluster" not in st.session_state:
        st.session_state["selected_cluster"] = clusters[0]

    cols = st.columns(len(clusters))

    for i, c in enumerate(clusters):
        icon = icons[c % len(icons)]
        label = f"{icon} Cluster {c}"

        with cols[i]:
            container_class = "selected" if st.session_state["selected_cluster"] == c else ""
            st.markdown(f'<div class="stButton {container_class}">', unsafe_allow_html=True)
            if st.button(label, key=f"tile_{i}_{c}"):
                st.session_state["selected_cluster"] = c
            st.markdown("</div>", unsafe_allow_html=True)

    selected_cluster = st.session_state["selected_cluster"]

    # ---------------------------------------------------------
    # ✅ Radar Chart for Selected Cluster
    # ---------------------------------------------------------
    cluster_data = profile.loc[selected_cluster]

    labels = cluster_data.index
    values = cluster_data.values
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(3.8, 3.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=8)
    ax.set_title(f"Cluster {selected_cluster} Profile", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, bbox_inches="tight")

    # ---------------------------------------------------------
    # ✅ Download segmented output
    # ---------------------------------------------------------
    st.subheader("⬇️ Download Segmented Data")
    st.download_button(
        label="Download segmented_customers.csv",
        data=segmented_df.to_csv(index=False),
        file_name="segmented_customers.csv",
        mime="text/csv"
    )