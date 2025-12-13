import pandas as pd
from preprocessing import fit_preprocessor
from model import train_kmeans, train_pca, save_segmentation_models
import json


def train_segmentation_pipeline(data_path="../data/insurance_synthetic.csv"):
    """Train full segmentation pipeline: preprocessing + KMeans + PCA."""

    # Load dataset
    df = pd.read_csv(data_path)

    # Fit preprocessing and save artifacts
    preprocessor = fit_preprocessor(df)

    # Transform data
    metadata = json.load(open("../models/segmentation_features.json"))
    numeric = metadata["numeric_features"]
    categorical = metadata["categorical_features"]

    X_processed = preprocessor.transform(df[numeric + categorical])

    # Train models
    kmeans = train_kmeans(X_processed, n_clusters=5)
    pca = train_pca(X_processed)

    # Save models
    save_segmentation_models(kmeans, pca)

    print("Full segmentation pipeline trained and saved successfully!")