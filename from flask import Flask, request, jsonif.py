from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('diabetes_model.pkl')

@app.route("/")
def index():
    return render_template("form.html")  # Halaman input akan berada di file form.html

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        data = request.json
        input_features = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age']),
        ]
        # Convert to numpy array and predict
        input_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_array)

        # Interpret the prediction
        diagnosis = "Positive for diabetes" if prediction[0] == 1 else "Negative for diabetes"
        return jsonify({"diagnosis": diagnosis})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
