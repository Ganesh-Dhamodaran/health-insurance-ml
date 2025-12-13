import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path


def train_kmeans(X_processed, n_clusters=5):
    """Train KMeans clustering model."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_processed)
    return kmeans


def train_pca(X_processed, n_components=2):
    """Train PCA for 2D visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_processed)
    return pca


def save_segmentation_models(kmeans, pca, save_path="../models"):
    """Save clustering models."""
    Path(save_path).mkdir(exist_ok=True)

    with open(f"{save_path}/segmentation_kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(f"{save_path}/segmentation_pca.pkl", "wb") as f:
        pickle.dump(pca, f)

    print("KMeans and PCA models saved successfully!")