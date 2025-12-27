import pandas as pd
import pickle
import os

# Root paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models")

def preprocess_input(data):
    """
    Preprocess input for fraud detection.
    Supports:
    ✅ Single record (dict)
    ✅ Multiple records (DataFrame)
    """

    # Convert dict → DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # -----------------------------
    # Load artifacts
    # -----------------------------
    with open(os.path.join(MODEL_DIR, "numerical_cols.pkl"), "rb") as f:
        numerical_cols = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "categorical_cols.pkl"), "rb") as f:
        categorical_cols = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "dummy_columns.pkl"), "rb") as f:
        dummy_cols = pickle.load(f)

    # -----------------------------
    # DATE FEATURE ENGINEERING
    # -----------------------------
    if "Policy Start Date" in df.columns:
        df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], errors="coerce")

    if "Policy Renewal Date" in df.columns:
        df["Policy Renewal Date"] = pd.to_datetime(df["Policy Renewal Date"], errors="coerce")

    if "Policy Start Date" in df.columns and "Policy Renewal Date" in df.columns:
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
    # SCALE NUMERICAL FEATURES
    # -----------------------------
    if all(col in df.columns for col in numerical_cols):
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    # -----------------------------
    # ONE-HOT ENCODING
    # -----------------------------
    existing_cats = [col for col in categorical_cols if col in df.columns]
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    # -----------------------------
    # ALIGN WITH TRAINING DUMMIES
    # -----------------------------
    df = df.reindex(columns=dummy_cols, fill_value=0)

    return df