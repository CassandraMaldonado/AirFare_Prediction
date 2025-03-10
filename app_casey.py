from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
from models import FlightPricePredictor
from utils import preprocess_flight_data, predict_future_prices, load_flight_data

app = Flask(__name__)

# Set up device
device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

# Load the pre-trained model
model_path = 'model/flight_price_model.pth'
MODEL = None

def load_model():
    global MODEL
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please run train.py first.")
        return False
    
    MODEL = FlightPricePredictor().to(device)
    MODEL.load_state_dict(torch.load(model_path, map_location=device))
    MODEL.eval()
    print("Model loaded successfully!")
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None and not load_model():
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get form data
        data = request.json
        origin = data.get('origin')
        destination = data.get('destination')
        date_str = data.get('date')
        time_str = data.get('time')
        current_price = float(data.get('price', 500))  # Default if not provided
        
        # Create flight data dictionary
        flight_data = {
            'origin': origin,
            'destination': destination,
            'date': date_str,
            'time': time_str,
            'price': current_price
        }
        
        # Create a DataFrame with the flight data
        df = pd.DataFrame([flight_data])
        
        # Add mock price history for demonstration
        # In a real app, you'd retrieve historical data from a database
        price_history = []
        price = current_price
        for i in range(5):
            price = price * np.random.uniform(0.98, 1.02)  # Small random variation
            price_history.append(round(price, 2))
        
        for i, price in enumerate(reversed(price_history)):
            df[f'price_t-{i+1}'] = price
        
        # Preprocess the data
        processed_data = preprocess_flight_data(df)
        processed_flight = processed_data.iloc[0]
        
        # Predict future prices
        future_prices = predict_future_prices(MODEL, processed_flight, days=7)
        
        # Denormalize predictions (assuming mean=500, std=100 for demo)
        price_mean = 500
        price_std = 100
        normalized_predictions = [round(price * price_std + price_mean, 2) for price in future_prices]
        
        # Find the lowest price day
        lowest_price = min(normalized_predictions)
        best_day = normalized_predictions.index(lowest_price)
        
        # Create dates for results
        dates = [(datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                 for i in range(len(normalized_predictions))]
        
        # Calculate savings
        savings = current_price - lowest_price
        savings_percent = (savings / current_price) * 100
        
        # Prepare response
        result = {
            'flight': {
                'origin': origin,
                'destination': destination,
                'date': date_str,
                'time': time_str
            },
            'current_price': current_price,
            'predictions': [
                {'date': date, 'price': price} 
                for date, price in zip(dates, normalized_predictions)
            ],
            'best_buy': {
                'date': dates[best_day],
                'day': best_day + 1,
                'price': lowest_price,
                'savings': round(savings, 2),
                'savings_percent': round(savings_percent, 2)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/flights', methods=['GET'])
def get_flights():
    try:
        # Load sample flight data
        df = load_flight_data('flight_data.csv')
        
        # Return only necessary columns
        flights = df[['origin', 'destination', 'date', 'time', 'price']].to_dict('records')
        return jsonify(flights)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to load the model at startup
    load_model()
    app.run(debug=True)
    st.session_state.page = "input"
    st.rerun()
        
    display_results(model)
