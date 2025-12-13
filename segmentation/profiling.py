import pandas as pd

def cluster_profile_summary(df):
    """Generate summary statistics for each cluster."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Exclude PCA columns
    numeric_cols = [col for col in numeric_cols if col not in ["pca_x", "pca_y"]]

    profile = df.groupby("cluster")[numeric_cols].mean().round(2)
    return profile