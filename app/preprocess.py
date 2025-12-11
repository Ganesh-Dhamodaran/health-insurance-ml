import pandas as pd
import pickle

def preprocess_input(data):
    """
    Preprocess a single customer record using the same steps
    used in 02_preprocessing.ipynb.
    """

    # Convert input dict → DataFrame
    df = pd.DataFrame([data])

    # -----------------------------
    # Load preprocessing artifacts
    # -----------------------------
    with open("../models/numerical_cols.pkl", "rb") as f:
        numerical_cols = pickle.load(f)

    with open("../models/categorical_cols.pkl", "rb") as f:
        categorical_cols = pickle.load(f)

    with open("../models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("../models/dummy_columns.pkl", "rb") as f:
        dummy_cols = pickle.load(f)

    # -----------------------------
    # DATE FEATURE ENGINEERING
    # -----------------------------
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
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # -----------------------------
    # ONE-HOT ENCODING
    # -----------------------------
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align with training dummy columns
    df = df.reindex(columns=dummy_cols, fill_value=0)

    # -----------------------------
    # SCALING
    # -----------------------------
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df