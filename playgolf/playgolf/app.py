from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('playgolf.pkl')
scaler = joblib.load('scaler.pkl')

le_outlook = joblib.load('le_outlook.pkl')
le_temperature = joblib.load('le_temperature.pkl')
le_humidity = joblib.load('le_humidity.pkl')
le_windy = joblib.load('le_windy.pkl')
le_target = joblib.load('le_target.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    outlook = request.form['outlook']
    temperature = request.form['temperature']
    humidity = request.form['humidity']
    windy = request.form['windy']

    try:
        outlook_encoded = le_outlook.transform([outlook])[0]
        temperature_encoded = le_temperature.transform([temperature])[0]
        humidity_encoded = le_humidity.transform([humidity])[0]
        windy_encoded = le_windy.transform([windy])[0]
    except ValueError:
        return render_template('index.html', prediction_text="⚠️ Invalid input. Please try again.")

    input_array = np.array([[outlook_encoded, temperature_encoded, humidity_encoded, windy_encoded]])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)

    if isinstance(prediction[0], str):
        prediction_label = prediction[0]
    else:
        prediction_label = le_target.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f"Prediction: {prediction_label}")

if __name__ == '__main__':
    app.run(debug=True)
