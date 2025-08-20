from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Initialize Flask app
diabetes_app = Flask(__name__)

@diabetes_app.route("/")
def home():
    return "Diabetes Prediction API is Running!"

@diabetes_app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get input data from request
        features = np.array(data["features"]).reshape(1, -1)  # Reshape input
        prediction = model.predict(features)  # Predict using model
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    diabetes_app.run(debug=True)

