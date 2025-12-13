import os
import json
import joblib
import pandas as pd


# Compute absolute paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models")


def load_segmentation_artifacts():
    """
    Load segmentation artifacts:
    - Preprocessor
    - KMeans model
    - PCA model
    - Feature metadata
    """

    pre_path = os.path.join(MODEL_DIR, "segmentation_preprocessor.pkl")
    kmeans_path = os.path.join(MODEL_DIR, "segmentation_kmeans.pkl")
    pca_path = os.path.join(MODEL_DIR, "segmentation_pca.pkl")
    meta_path = os.path.join(MODEL_DIR, "segmentation_features.json")

    # Check existence of all required files
    missing = []
    for p in [pre_path, kmeans_path, pca_path, meta_path]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            "❌ Missing segmentation model artifacts:\n" +
            "\n".join(missing) +
            "\n\nRun: python train_segmentation.py"
        )

    # Load artifacts
    preprocessor = joblib.load(pre_path)
    kmeans = joblib.load(kmeans_path)
    pca = joblib.load(pca_path)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    return preprocessor, kmeans, pca, numeric_features, categorical_features



def segment_customers(df):
    """
    Apply preprocessing, clustering, and PCA to new customer data.
    Returns a dataframe with:
    - cluster
    - pca_x
    - pca_y
    """

    # Load artifacts safely
    preprocessor, kmeans, pca, numeric_features, categorical_features = load_segmentation_artifacts()

    # Ensure required columns exist
    required_cols = numeric_features + categorical_features
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "❌ Missing required columns in uploaded data:\n" +
            "\n".join(missing_cols)
        )

    # Select only required columns
    df_input = df[required_cols].copy()

    # Preprocess
    X_processed = preprocessor.transform(df_input)

    # Predict clusters
    clusters = kmeans.predict(X_processed)

    # PCA for visualization
    pca_components = pca.transform(X_processed)

    # Build output dataframe
    df_output = df.copy()
    df_output["cluster"] = clusters
    df_output["pca_x"] = pca_components[:, 0]
    df_output["pca_y"] = pca_components[:, 1]

    return df_output