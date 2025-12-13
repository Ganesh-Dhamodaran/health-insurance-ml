import os
import joblib
import pandas as pd


# Absolute paths
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


def preprocess_fraud(df, numerical_cols, categorical_cols, scaler, dummy_columns):
    """
    Apply the SAME preprocessing steps used during training:
    ✅ Date feature engineering
    ✅ Missing value handling
    ✅ One-hot encoding
    ✅ Dummy alignment
    ✅ Scaling
    """

    df = df.copy()

    # -----------------------------
    # DATE FEATURE ENGINEERING
    # -----------------------------
    if "Policy Start Date" in df.columns and "Policy Renewal Date" in df.columns:
        df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], errors="coerce")
        df["Policy Renewal Date"] = pd.to_datetime(df["Policy Renewal Date"], errors="coerce")

        df["Policy_Duration_Days"] = (df["Policy Renewal Date"] - df["Policy Start Date"]).dt.days
        df["Policy_Start_Year"] = df["Policy Start Date"].dt.year
        df["Policy_Start_Month"] = df["Policy Start Date"].dt.month

        df.drop(columns=["Policy Start Date", "Policy Renewal Date"], inplace=True)

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # -----------------------------
    # ONE-HOT ENCODING
    # -----------------------------
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align with training dummy columns
    df = df.reindex(columns=dummy_columns, fill_value=0)

    # -----------------------------
    # SCALING
    # -----------------------------
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df


def predict_fraud(df):
    """Run fraud detection on new data."""

    model, scaler, numerical_cols, categorical_cols, dummy_columns = load_fraud_artifacts()

    # Preprocess
    X = preprocess_fraud(df, numerical_cols, categorical_cols, scaler, dummy_columns)

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Build output
    df_out = df.copy()
    df_out["fraud_prediction"] = predictions
    df_out["fraud_probability"] = probabilities.round(4)

    return df_out