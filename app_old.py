import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# Load the XGBoost model
@st.cache_resource
def load_model():
    """Load the XGBoost model"""
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

def collect_inputs():
    """
    Collect user inputs for flight price prediction.
    
    Returns:
        dict: Dictionary containing user input values
    """
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Route Information")
        
        # Origin and destination inputs
        origin = st.selectbox(
            "Origin City",
            options=["New York", "Los Angeles", "Chicago", "Dallas", "Atlanta", "San Francisco", "Denver", "Miami"],
            index=0
        )
        
        destination = st.selectbox(
            "Destination City",
            options=["Los Angeles", "New York", "Chicago", "Dallas", "Atlanta", "San Francisco", "Denver", "Miami"],
            index=0
        )
        
        # Prevent same origin and destination
        if origin == destination:
            st.warning("⚠️ Origin and destination cannot be the same")
        
        # Flight class
        flight_class = st.radio(
            "Cabin Class",
            options=["Economy", "Premium Economy", "Business", "First"],
            horizontal=True
        )
        
    with col2:
        st.subheader("🗓️ Travel Information")
        
        # Date inputs
        departure_date = st.date_input("Departure Date")
        
        # Number of passengers
        passengers = st.number_input(
            "Number of Passengers",
            min_value=1,
            max_value=10,
            value=1,
            step=1
        )
        
        # Direct flight only
        direct_flight = st.checkbox("Direct Flights Only")
        
        # Current price
        current_price = st.number_input(
            "Current Price ($)",
            min_value=50,
            max_value=2000,
            value=500,
            step=10
        )
    
    # Return all inputs as a dictionary
    return {
        "origin": origin,
        "destination": destination,
        "flight_class": flight_class,
        "departure_date": departure_date,
        "passengers": passengers,
        "direct_flight": direct_flight,
        "current_price": current_price
    }

def preprocess_inputs(user_inputs):
    """
    Preprocess user inputs for XGBoost model prediction.
    
    Args:
        user_inputs (dict): Raw user inputs
        
    Returns:
        pd.DataFrame: Processed inputs ready for model prediction
    """
    # Create a DataFrame from the inputs
    input_df = pd.DataFrame([{
        'origin': user_inputs['origin'],
        'destination': user_inputs['destination'],
        'date': user_inputs['departure_date'],
        'price': user_inputs['current_price'],
        'cabin_class': user_inputs['flight_class'],
        'direct_flight': 1 if user_inputs['direct_flight'] else 0
    }])
    
    # Extract date features
    input_df['date'] = pd.to_datetime(input_df['date'])
    input_df['day_of_week'] = input_df['date'].dt.dayofweek
    input_df['day_of_month'] = input_df['date'].dt.day
    input_df['month'] = input_df['date'].dt.month
    input_df['days_to_flight'] = (input_df['date'] - datetime.now().date()).dt.days
    
    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['origin', 'destination', 'cabin_class'], drop_first=True)
    
    # Drop date column as it's not needed for prediction
    input_df = input_df.drop(['date'], axis=1)
    
    return input_df

def predict_prices_recursive(model, features_df, days=7):
    """
    Predict flight prices recursively for the next n days
    
    Args:
        model: Trained XGBoost model
        features_df (pd.DataFrame): Current flight features
        days (int): Number of days to predict
        
    Returns:
        list: Predicted prices for the next n days
    """
    predictions = []
    
    # Make a copy to avoid modifying the original
    df = features_df.copy()
    
    for i in range(days):
        # Update date-related features for the next day
        df['days_to_flight'] = df['days_to_flight'] - 1
        df['day_of_week'] = (df['day_of_week'] + 1) % 7
        df['day_of_month'] = df['day_of_month'] + 1
        
        # Handle month boundaries
        if df['day_of_month'].values[0] > 28:
            df['day_of_month'] = 1
            df['month'] = df['month'] % 12 + 1
        
        # Predict the next day's price
        next_price = model.predict(df)[0]
        predictions.append(next_price)
        
        # Update the price for the next iteration
        df['price'] = next_price
    
    return predictions

def display_results(model, user_inputs):
    """
    Display flight price prediction results using the XGBoost model.
    
    Args:
        model: The trained XGBoost model
        user_inputs: User input dictionary
    """
    if model is None:
        st.error("❌ Model not loaded. Please check if the XGBoost model file exists.")
        return
    
    st.title("✈️ Flight Price Predictions")
    
    # Display search criteria
    st.subheader("🔍 Search Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**From:** {user_inputs['origin']}")
        st.write(f"**To:** {user_inputs['destination']}")
        st.write(f"**Class:** {user_inputs['flight_class']}")
    
    with col2:
        st.write(f"**Date:** {user_inputs['departure_date'].strftime('%B %d, %Y')}")
        st.write(f"**Passengers:** {user_inputs['passengers']}")
        st.write(f"**Current Price:** ${user_inputs['current_price']:.2f}")
    
    # Process inputs for model prediction
    processed_inputs = preprocess_inputs(user_inputs)
    
    # Make recursive predictions
    try:
        predictions = predict_prices_recursive(model, processed_inputs, days=7)
        
        # Create dates for each prediction
        dates = [(user_inputs['departure_date'] + timedelta(days=i+1)) for i in range(len(predictions))]
        
        # Convert prices to account for passengers
        price_per_passenger = [price for price in predictions]
        total_prices = [price * user_inputs['passengers'] for price in predictions]
        
        # Find best day to buy
        best_day_index = np.argmin(price_per_passenger)
        best_day = dates[best_day_index]
        best_price = price_per_passenger[best_day_index]
        
        # Calculate potential savings
        current_price = user_inputs['current_price']
        savings = current_price - best_price
        savings_percent = (savings / current_price) * 100 if current_price > 0 else 0
        
        # Display prediction results
        st.subheader("📊 Price Predictions for Next 7 Days")
        
        # Create data for the chart
        chart_data = pd.DataFrame({
            'Date': [d.strftime('%b %d') for d in dates],
            'Price': price_per_passenger
        })
        
        # Plot the prices
        st.line_chart(chart_data.set_index('Date'))
        
        # Display price table
        price_data = pd.DataFrame({
            'Date': [d.strftime('%A, %b %d') for d in dates],
            'Price Per Passenger': [f"${p:.2f}" for p in price_per_passenger],
            'Total Price': [f"${p:.2f}" for p in total_prices]
        })
        
        st.table(price_data)
        
        # Show recommendation
        st.subheader("💰 Recommendation")
        
        if savings > 0:
            st.success(f"✅ Wait until **{best_day.strftime('%A, %B %d')}** to purchase for the best price!")
            st.write(f"**Predicted best price:** ${best_price:.2f} per passenger (${best_price * user_inputs['passengers']:.2f} total)")
            st.write(f"**Potential savings:** ${savings:.2f} per passenger (${savings * user_inputs['passengers']:.2f} total)")
            st.write(f"**Savings percentage:** {savings_percent:.1f}%")
        else:
            st.warning("⚠️ Buy now! Prices are expected to increase in the coming days.")
            
    except Exception as e:
        st.error(f"❌ Error making predictions: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

def main():
    st.title("✈️ Flight Price Predictor")
    st.write("Get price predictions for flights over the next 7 days and find the best time to buy!")
    
    # Load the model
    model = load_model()
    
    # Tab for input and results
    tabs = st.tabs(["Search Flights", "About"])
    
    with tabs[0]:
        # Collect user inputs
        user_inputs = collect_inputs()
        
        # Search button
        if st.button("🔍 Search Flights"):
            display_results(model, user_inputs)
    
    with tabs[1]:
        st.header("About this App")
        st.write("""
        This app uses an XGBoost machine learning model to predict flight prices over the next 7 days based on historical data.
        
        **How it works:**
        1. Enter your flight details including origin, destination, date, and current price
        2. Our model predicts how prices might change over the next week
        3. We recommend the best day to purchase your ticket
        
        **Note:** Predictions are based on historical patterns and should be used as guidance only.
        Actual prices may vary due to many factors outside the model's scope.
        """)

if __name__ == "__main__":
    main()
