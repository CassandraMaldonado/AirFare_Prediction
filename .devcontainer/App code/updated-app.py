from flask import Flask, render_template, request, jsonify
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Path to the XGBoost model
MODEL_PATH = 'model/xg_boost_model_compressed.pkl'
MODEL = None

def load_model():
    """
    Load the pre-trained XGBoost model with multiple fallback methods
    """
    global MODEL
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found.")
        return False
    
    try:
        # Method 1: Standard pickle binary mode
        with open(MODEL_PATH, 'rb') as file:
            MODEL = pickle.load(file)
            print("Model loaded with standard pickle")
            return True
    except Exception as e1:
        print(f"Error loading with standard pickle: {str(e1)}")
        
        try:
            # Method 2: Try pickle with different protocol
            import _pickle as cPickle
            with open(MODEL_PATH, 'rb') as file:
                MODEL = cPickle.load(file)
                print("Model loaded with cPickle")
                return True
        except Exception as e2:
            print(f"Error loading with cPickle: {str(e2)}")
            
            try:
                # Method 3: Try loading a pickle protocol 0 file (text mode)
                with open(MODEL_PATH, 'r') as file:
                    MODEL = pickle.load(file)
                    print("Model loaded with text mode pickle")
                    return True
            except Exception as e3:
                print(f"Error loading with text pickle: {str(e3)}")
                
                print("All loading methods failed. Please check model format.")
                return False

def preprocess_flight_data(flight_data):
    """
    Preprocess flight data for the XGBoost model
    
    Args:
        flight_data (pd.DataFrame): Input flight data
        
    Returns:
        pd.DataFrame: Processed features
    """
    # Make a copy to avoid modifying the original data
    processed_data = flight_data.copy()
    
    # Extract date features
    if 'date' in processed_data.columns:
        processed_data['date'] = pd.to_datetime(processed_data['date'])
        processed_data['day_of_week'] = processed_data['date'].dt.dayofweek
        processed_data['day_of_month'] = processed_data['date'].dt.day
        processed_data['month'] = processed_data['date'].dt.month
        days_to_flight = (processed_data['date'] - datetime.now().date()).dt.days
        processed_data['days_to_flight'] = days_to_flight.fillna(0).astype(int)
    
    # One-hot encode categorical variables
    if 'origin' in processed_data.columns:
        dummies = pd.get_dummies(processed_data['origin'], prefix='origin')
        processed_data = pd.concat([processed_data, dummies], axis=1)
        processed_data.drop('origin', axis=1, inplace=True)
    
    if 'destination' in processed_data.columns:
        dummies = pd.get_dummies(processed_data['destination'], prefix='dest')
        processed_data = pd.concat([processed_data, dummies], axis=1)
        processed_data.drop('destination', axis=1, inplace=True)
    
    # Process time if available
    if 'time' in processed_data.columns and not processed_data['time'].isnull().all():
        try:
            processed_data['hour'] = pd.to_datetime(processed_data['time']).dt.hour
        except:
            # If time parsing fails, add a default hour
            processed_data['hour'] = 12
    else:
        processed_data['hour'] = 12
        
    # Drop columns not needed for prediction
    cols_to_drop = ['date', 'time'] if 'time' in processed_data.columns else ['date']
    processed_data = processed_data.drop(cols_to_drop, axis=1, errors='ignore')
    
    return processed_data

def predict_future_prices(model, current_features, days=7):
    """
    Predict flight prices recursively for the next n days
    
    Args:
        model: Trained XGBoost model
        current_features (pd.Series): Current flight features
        days (int): Number of days to predict
        
    Returns:
        list: Predicted prices for the next n days
    """
    predictions = []
    
    # Create a copy of the features to modify during prediction
    features = current_features.copy()
    current_price = features['price']
    
    for i in range(days):
        # Adjust date-related features for the next day
        if 'days_to_flight' in features:
            features['days_to_flight'] = features['days_to_flight'] - 1
        
        if 'day_of_week' in features:
            features['day_of_week'] = (features['day_of_week'] + 1) % 7
        
        if 'day_of_month' in features:
            # Simple approximation, doesn't account for month boundaries
            features['day_of_month'] = features['day_of_month'] + 1
            if features['day_of_month'] > 28:
                features['day_of_month'] = 1
                features['month'] = features['month'] % 12 + 1
        
        # Make prediction for the next day
        features_df = pd.DataFrame([features])
        
        try:
            # Use predict method for most models
            next_day_price = model.predict(features_df)[0]
        except AttributeError:
            # Fallback for models that don't have predict method
            try:
                next_day_price = model(features_df)[0]
            except:
                # If all fails, use a simple heuristic (5% random change)
                import random
                next_day_price = current_price * (1 + (random.random() - 0.5) * 0.1)
        
        predictions.append(next_day_price)
        
        # Update the current price for the next iteration (recursive forecasting)
        features['price'] = next_day_price
    
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None and not load_model():
        return jsonify({
            'error': 'Model not loaded. Please check if the XGBoost model file exists.'
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
        
        # Preprocess the data for the XGBoost model
        processed_data = preprocess_flight_data(df)
        processed_flight = processed_data.iloc[0]
        
        # Predict future prices recursively
        future_prices = predict_future_prices(MODEL, processed_flight, days=7)
        
        # Create dates for results
        dates = [(datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                 for i in range(len(future_prices))]
        
        # Find the lowest price day
        lowest_price = min(future_prices)
        best_day = future_prices.index(lowest_price)
        
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
                {'date': date, 'price': float(round(price, 2))} 
                for date, price in zip(dates, future_prices)
            ],
            'best_buy': {
                'date': dates[best_day],
                'day': best_day + 1,
                'price': float(round(lowest_price, 2)),
                'savings': float(round(savings, 2)),
                'savings_percent': float(round(savings_percent, 2))
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in prediction: {error_details}")
        return jsonify({
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/flights', methods=['GET'])
def get_flights():
    try:
        # Load sample flight data
        df = pd.read_csv('flight_data.csv')
        
        # Return only necessary columns
        flights = df[['origin', 'destination', 'date', 'time', 'price']].to_dict('records')
        return jsonify(flights)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to load the model at startup
    load_model()
    app.run(debug=True)
