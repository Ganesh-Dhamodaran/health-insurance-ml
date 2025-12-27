from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.preprocess_fraud import preprocess_input

app = Flask(__name__)

# -----------------------------
# Load trained model
# -----------------------------
with open(os.path.join(ROOT_DIR, "models", "fraud_detection_model.pkl"), "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Read JSON input
        data = request.get_json()

        # 2. Apply preprocessing
        X = preprocess_input(data)

        # 3. Align schema if needed
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            # Add missing columns with 0
            for col in expected:
                if col not in X.columns:
                    X[col] = 0
            # Drop extras
            X = X[expected]

        # 4. Predict
        if hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(X)[0][1]
        else:
            # fallback if model has no predict_proba
            pred_proba = float(model.predict(X)[0])

        pred_class = int(pred_proba >= 0.5)

        # 5. Return response
        return jsonify({
            "fraud_probability": float(pred_proba),
            "fraud_flag": pred_class
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------
# Run the API
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)