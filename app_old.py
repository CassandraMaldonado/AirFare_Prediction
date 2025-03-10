import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Flight Price Predictor")

# Load the XGBoost model
def load_model():
    model_path = 'xg_boost_model_compressed.pkl'
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

# Origin and destination
origin = st.selectbox("Origin", ["New York", "Los Angeles", "Chicago", "Dallas"])
destination = st.selectbox("Destination", ["Los Angeles", "New York", "Chicago", "Dallas"])

# Date and price inputs
departure_date = st.date_input("Departure Date")
current_price = st.number_input("Current Price ($)", min_value=50, value=500)
passengers = st.number_input("Passengers", min_value=1, value=1)

# Load model when button is clicked
if st.button("Predict Prices"):
    model = load_model()
    
    if model is None:
        st.error("Could not load the model. Please check that xg_boost_model_compressed.pkl exists in the model directory.")
    else:
        # Create input data frame
        input_data = pd.DataFrame({
            'origin': [origin],
            'destination': [destination],
            'date': [departure_date],
            'price': [current_price]
        })
        
        # Process features
        input_data['day_of_week'] = pd.to_datetime(input_data['date']).dt.dayofweek
        input_data['days_to_flight'] = (pd.to_datetime(input_data['date']) - datetime.now().date()).dt.days
        
        # One-hot encode origin/destination
        origin_cols = pd.get_dummies(input_data['origin'], prefix='origin')
        dest_cols = pd.get_dummies(input_data['destination'], prefix='dest')
        
        # Combine dataframes
        processed_data = pd.concat([input_data, origin_cols, dest_cols], axis=1)
        
        # Remove original columns
        processed_data = processed_data.drop(['origin', 'destination', 'date'], axis=1)
        
        # Make predictions for 7 days
        predictions = []
        df = processed_data.copy()
        
        st.subheader("Predicted Prices for Next 7 Days")
        
        for i in range(7):
            # Update date features
            if 'days_to_flight' in df:
                df['days_to_flight'] = df['days_to_flight'] - 1
                
            if 'day_of_week' in df:
                df['day_of_week'] = (df['day_of_week'] + 1) % 7
            
            # Predict price
            try:
                next_price = float(model.predict(df)[0])
                predictions.append(next_price)
                
                # Update price for next prediction
                df['price'] = next_price
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                break
        
        # If we have predictions, display them
        if predictions:
            # Create dates
            dates = [(departure_date + timedelta(days=i+1)) for i in range(len(predictions))]
            date_strs = [d.strftime("%a, %b %d") for d in dates]
            
            # Show results
            results = pd.DataFrame({
                'Date': date_strs,
                'Price': [f"${p:.2f}" for p in predictions],
                'Total': [f"${p*passengers:.2f}" for p in predictions]
            })
            
            st.table(results)
            
            # Find best day
            best_day_index = np.argmin(predictions)
            best_day = dates[best_day_index]
            best_price = predictions[best_day_index]
            
            savings = current_price - best_price
            
            if savings > 0:
                st.success(f"Best day to buy: {best_day.strftime('%A, %B %d')} for ${best_price:.2f}")
                st.write(f"Potential savings: ${savings:.2f}")
            else:
                st.warning("Buy now! Prices are expected to increase.")
