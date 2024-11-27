from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# Function to load the ARIMA model
def load_arima_model():
    try:
        print("Attempting to load ARIMA model...")
        model = joblib.load('arima_model.joblib')


        # Ensure the model file is correct
        if model:
            print("Model loaded successfully")
        else:
            print("Model loading failed: Model is None")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Home route to render the dashboard
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    model = load_arima_model()
    if model is None:
        return jsonify({'error': 'Model not loaded properly'})

    # Get input data (you can pass the date range or other inputs as per your needs)
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    # Example: Forecast next 30 days (adjust according to your need)
    try:
        print(f"Generating forecast for dates {start_date} to {end_date}")
        forecast = model.forecast()  # Replace with the correct number of steps
        return jsonify({'forecast': forecast.tolist()})
    except Exception as e:
        return jsonify({'error': f"Error generating forecast: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
