import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import json
from pathlib import Path


def get_feature_groups(df):
    """Dynamically detect numeric and categorical features, excluding unwanted columns."""

    exclude_cols = [
        "Customer ID",
        "Policy Start Date",
        "Policy Renewal Date",
        "Segmentation Group"
    ]

    df = df.drop(columns=[col for col in exclude_cols if col in df.columns])

    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """Build the preprocessing pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor


def fit_preprocessor(df, save_path="../models"):
    """Fit preprocessing pipeline and save artifacts."""

    Path(save_path).mkdir(exist_ok=True)

    numeric_features, categorical_features = get_feature_groups(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    preprocessor.fit(df[numeric_features + categorical_features])

    # Save preprocessor
    with open("../models/segmentation_preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Save metadata
    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }

    with open("../models/segmentation_features.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Segmentation preprocessing artifacts saved successfully!")

    return preprocessor