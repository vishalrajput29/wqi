from flask import Flask, request, render_template, jsonify
import joblib
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

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
    return render_template('index1.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    model = load_arima_model()
    if model is None:
        return jsonify({'error': 'Model not loaded properly'})

    # Get input data (start and end dates)
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    # Example: Forecast next 30 days (adjust according to your need)
    try:
        print(f"Generating forecast for dates {start_date} to {end_date}")
        # Assuming your model's forecast method takes a number of periods (steps) as input
        forecast_steps = 30  # Example for 30 days forecast (adjust as needed)
        forecast = model.forecast(steps=forecast_steps)
        
        # Check if WQI > 70 (assuming forecast values represent WQI)
        wqi_impure = any(value > 70 for value in forecast)  # Check if any value in the forecast is greater than 70

        # Plotting the forecast
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(0, forecast_steps), forecast, label="Forecasted WQI", color='blue')
        plt.title(f'WQI Prediction for {start_date} to {end_date}', fontsize=16)
        plt.xlabel('Days', fontsize=14)
        plt.ylabel('Water Quality Index (WQI)', fontsize=14)
        plt.legend()

        # Save plot to a BytesIO object and convert to base64 for embedding in HTML
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf-8')

        # Return the result with the plot URL and WQI impurity message
        return render_template('result1.html', plot_url=plot_url, wqi_impure=wqi_impure)
    except Exception as e:
        return jsonify({'error': f"Error generating forecast: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
