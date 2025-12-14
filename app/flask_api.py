from flask import Flask, request, jsonify
import pickle


import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.preprocess_fraud import preprocess_input

app = Flask(__name__)

# -----------------------------
# Load trained model
# -----------------------------
with open("../models/fraud_detection_model.pkl", "rb") as f:
    model = pickle.load(f)


# -----------------------------
# Landing Page
# -----------------------------
@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>Health Insurance Fraud Detection API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f7f9;
                    padding: 40px;
                    text-align: center;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    display: inline-block;
                }
                h1 {
                    color: #008080;
                }
                p {
                    font-size: 18px;
                    color: #333;
                }
                code {
                    background: #eee;
                    padding: 5px 10px;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Health Insurance Fraud Detection API</h1>
                <p>Your API is running successfully.</p>
                <p>Use the <code>/predict</code> endpoint to submit JSON data and get fraud predictions.</p>
            </div>
        </body>
    </html>
    """


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

        # 3. Predict
        pred_proba = model.predict_proba(X)[0][1]
        pred_class = int(pred_proba >= 0.5)

        # 4. Return response
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