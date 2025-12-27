import os
import joblib
from sklearn.decomposition import PCA

from utils.preprocess_fraud import preprocess_input

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models")

def load_fraud_artifacts():
    """Load fraud detection model + preprocessing artifacts."""

    required_files = {
        "model": "fraud_detection_model.pkl",
        "scaler": "scaler.pkl",
        "num_cols": "numerical_cols.pkl",
        "cat_cols": "categorical_cols.pkl",
        "dummy_cols": "dummy_columns.pkl"
    }

    loaded = {}

    for key, filename in required_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fraud artifact: {path}")
        loaded[key] = joblib.load(path)

    return (
        loaded["model"],
        loaded["scaler"],
        loaded["num_cols"],
        loaded["cat_cols"],
        loaded["dummy_cols"]
    )


def predict_fraud(df):
    """
    Run fraud detection on new data.
    df can be:
    ✅ Single-row DataFrame
    ✅ Multi-row DataFrame
    """

    model, scaler, numerical_cols, categorical_cols, dummy_columns = load_fraud_artifacts()

    # ✅ Preprocess using shared function
    X = preprocess_input(df)

    # ✅ Apply PCA (reduce to 10 components)
    pca = PCA(n_components=10, random_state=42)
    X = pca.fit_transform(X)

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Build output
    df_out = df.copy()
    df_out["fraud_prediction"] = predictions
    df_out["fraud_probability"] = probabilities.round(4)

    return df_out