import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import cv2

app = Flask(__name__)
CORS(app)

# Load models
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

FEATURE_ORDER = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


@app.route("/")
def home():
    return "Soil Vision AI API is running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Ensure correct feature order
        features = []

        for feature in FEATURE_ORDER:
            if feature not in data:
                return jsonify({"error": f"Missing field: {feature}"}), 400

            features.append(float(data[feature]))

        features = np.array([features])

        print("Raw Input:", features)

        # Scale input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)

        crop = label_encoder.inverse_transform(prediction)[0]

        print("Prediction:", crop)

        return jsonify({
            "recommended_crop": crop
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/soil-scan", methods=["POST"])
def soil_scan():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        file_bytes = np.frombuffer(file.read(), np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        # Dummy soil detection (placeholder)
        soil_type = "Clay Soil"

        if soil_type == "Clay Soil":
            crop = "rice"
        elif soil_type == "Sandy Soil":
            crop = "groundnut"
        else:
            crop = "wheat"

        return jsonify({
            "soil_type": soil_type,
            "recommended_crop": crop
        })

    except Exception as e:
        print("Soil Scan Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
