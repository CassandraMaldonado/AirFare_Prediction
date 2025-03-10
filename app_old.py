import streamlit as st
import os
import pickle
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Flight Price Predictor")

# Load the XGBoost model
def load_model():
    model_path = 'model/xg_boost_model_compressed.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found.")
        return None
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Collect input data
st.title("Flight Price Predictor")
st.write("Predict flight prices for the next 7 days")

# Origin and destination
origin = st.selectbox("Origin", ["New York", "Los Angeles", "Chicago", "Dallas"])
destination = st.selectbox("Destination", ["Los Angeles", "New York", "Chicago", "Dallas"])

# Date and price inputs
departure_date = st.date_input("Departure Date")
current_price = st.number_input("Current Price ($)", min_value=50, value=500)

# Load model when button is clicked
if st.button("Predict Prices"):
    model = load_model()
    
    if model is None:
        st.error("Could not load the model. Please check that xg_boost_model_compressed.pkl exists in the model directory.")
    else:
        # Manually create simple feature data (avoiding pandas dependency)
        # Use a Python dictionary for our features
        features = {
            'price': current_price,
            'day_of_week': departure_date.weekday(),
            'days_to_flight': (departure_date - datetime.now().date()).days
        }
        
        # Add origin one-hot encoding
        if origin == "New York":
            features['origin_New York'] = 1
        else:
            features['origin_New York'] = 0
            
        if origin == "Los Angeles":
            features['origin_Los Angeles'] = 1
        else:
            features['origin_Los Angeles'] = 0
            
        if origin == "Chicago":
            features['origin_Chicago'] = 1
        else:
            features['origin_Chicago'] = 0
            
        if origin == "Dallas":
            features['origin_Dallas'] = 1
        else:
            features['origin_Dallas'] = 0
            
        # Add destination one-hot encoding
        if destination == "New York":
            features['dest_New York'] = 1
        else:
            features['dest_New York'] = 0
            
        if destination == "Los Angeles":
            features['dest_Los Angeles'] = 1
        else:
            features['dest_Los Angeles'] = 0
            
        if destination == "Chicago":
            features['dest_Chicago'] = 1
        else:
            features['dest_Chicago'] = 0
            
        if destination == "Dallas":
            features['dest_Dallas'] = 1
        else:
            features['dest_Dallas'] = 0
        
        # Try to convert features to format needed by model
        try:
            import numpy as np
            import pandas as pd
            
            # Try pandas conversion
            features_df = pd.DataFrame([features])
            
        except ImportError:
            st.error("Could not import pandas or numpy. Using fallback method.")
            
            # Fallback prediction method (if pandas/numpy aren't available)
            # Basic pattern: prices tend to be lower midweek
            day_multipliers = {
                0: 0.99,  # Monday
                1: 0.98,  # Tuesday - typically lowest
                2: 0.97,  # Wednesday - typically lowest
                3: 0.98,  # Thursday
                4: 1.01,  # Friday - rising for weekend
                5: 1.03,  # Saturday - peak
                6: 1.01   # Sunday
            }
            
            # Make predictions for 7 days
            predictions = []
            current = current_price
            current_day = departure_date.weekday()
            
            for i in range(7):
                next_day = (current_day + i + 1) % 7
                next_price = current * day_multipliers.get(next_day, 1.0)
                predictions.append(next_price)
                current = next_price
            
            # Display predictions
            st.subheader("Predicted Prices for Next 7 Days")
            for i, price in enumerate(predictions):
                day = departure_date + timedelta(days=i+1)
                st.write(f"{day.strftime('%A, %b %d')}: ${price:.2f}")
            
            # Find best day
            min_price = min(predictions)
            best_day_index = predictions.index(min_price)
            best_day = departure_date + timedelta(days=best_day_index+1)
            
            st.subheader("Recommendation")
            if min_price < current_price:
                savings = current_price - min_price
                st.success(f"Wait until {best_day.strftime('%A, %b %d')} to buy. Price: ${min_price:.2f}")
                st.write(f"Potential savings: ${savings:.2f}")
            else:
                st.warning("Buy now! Prices are expected to rise.")
            
            # Exit early since we used fallback method
            st.info("Note: Using simplified prediction model since pandas/numpy aren't available.")
            return
        
        # Make predictions for 7 days
        predictions = []
        next_price = current_price
        
        st.subheader("Predicted Prices for Next 7 Days")
        
        for i in range(7):
            # Update date features
            features['days_to_flight'] = features['days_to_flight'] - 1
            features['day_of_week'] = (features['day_of_week'] + 1) % 7
            
            # Update price from previous prediction
            features['price'] = next_price
            
            # Convert to DataFrame for prediction
            df = pd.DataFrame([features])
            
            # Predict price
            try:
                next_price = float(model.predict(df)[0])
                predictions.append(next_price)
                
                # Display prediction
                day = departure_date + timedelta(days=i+1)
                st.write(f"{day.strftime('%A, %b %d')}: ${next_price:.2f}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                break
        
        # If we have predictions, display recommendation
        if predictions:
            # Find best day
            min_price = min(predictions)
            best_day_index = predictions.index(min_price)
            best_day = departure_date + timedelta(days=best_day_index+1)
            
            st.subheader("Recommendation")
            if min_price < current_price:
                savings = current_price - min_price
                st.success(f"Wait until {best_day.strftime('%A, %b %d')} to buy. Price: ${min_price:.2f}")
                st.write(f"Potential savings: ${savings:.2f}")
            else:
                st.warning("Buy now! Prices are expected to rise.")
