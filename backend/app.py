import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import cv2

app = Flask(__name__)
CORS(app)

# Load models (ensure paths are correct)
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return "Soil Vision AI API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            data["nitrogen"], data["phosphorus"], data["potassium"],
            data["temperature"], data["humidity"], data["ph"], data["rainfall"]
        ]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        crop = label_encoder.inverse_transform(prediction)[0]
        return jsonify({"recommended_crop": crop})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/soil-scan", methods=["POST"])
def soil_scan():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Dummy detection - replace later
        soil_type = "Clay Soil"
        crop = "Rice" if soil_type == "Clay Soil" else "Groundnut" if soil_type == "Sandy Soil" else "Wheat"
        return jsonify({"soil_type": soil_type, "recommended_crop": crop})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)