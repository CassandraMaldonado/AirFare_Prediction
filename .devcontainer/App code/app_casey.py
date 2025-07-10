import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️"
)

# Initialize session state variables if they don't exist
if 'page' not in st.session_state:
    st.session_state.page = "input"
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'selected_flight' not in st.session_state:
    st.session_state.selected_flight = None

# Function to predict prices (simplified)
def predict_prices(origin, destination, date, time):
    """Generate simple price predictions for the next 7 days"""
    # Generate a base price based on origin and destination
    base_price = 500 + (ord(origin[0]) + ord(destination[0])) % 200
    
    # Set seed for reproducible results
    np.random.seed(int(ord(origin[0]) + ord(destination[0])))
    
    # Generate prices for 7 days
    prices = []
    current_price = base_price
    for i in range(7):
        # Random fluctuation between -5% to +5%
        change = np.random.uniform(-0.05, 0.05)
        current_price = current_price * (1 + change)
        prices.append(round(current_price, 2))
    
    # Generate dates
    start_date = datetime.now()
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    
    # Create DataFrame with predictions
    predictions = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    return predictions, base_price

# Define navigation functions
def show_prediction_page(origin, destination, date, time):
    st.session_state.page = "results"
    predictions, base_price = predict_prices(origin, destination, date, time)
    st.session_state.predictions = predictions
    st.session_state.selected_flight = {
        'origin': origin,
        'destination': destination,
        'date': date,
        'time': time,
        'price': base_price
    }

def back_to_input():
    st.session_state.page = "input"

# Airport codes
airport_codes = ['NYC', 'LAX', 'CHI', 'MIA', 'SFO', 'LON', 'PAR', 'TOK', 'SYD', 'BER', 
                'DFW', 'ATL', 'DEN', 'SEA', 'JFK', 'ORD', 'LHR', 'CDG', 'FRA', 'DXB']

# Main app logic
if st.session_state.page == "input":
    # Input Page
    st.title("✈️ Flight Price Predictor")
    
    # Flight selection form
    st.header("Enter Flight Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        origin = st.selectbox("Origin", options=airport_codes, index=0)
    
    with col2:
        destination = st.selectbox("Destination", options=airport_codes, index=5, 
                                 key="dest_select")
    
    col3, col4 = st.columns(2)
    
    with col3:
        date = st.date_input("Date", value=datetime.now() + timedelta(days=30))
    
    with col4:
        time = st.time_input("Time", value=datetime.strptime("12:00", "%H:%M").time())
    
    # Submit button
    if st.button("Predict Prices"):
        show_prediction_page(origin, destination, date.strftime('%Y-%m-%d'), 
                           time.strftime('%H:%M'))

else:
    # Results Page
    flight = st.session_state.selected_flight
    predictions = st.session_state.predictions
    
    st.title("✈️ Flight Price Prediction Results")
    
    # Back button
    if st.button("← Back to Flight Selection"):
        back_to_input()
    
    # Flight details
    st.header("Flight Details")
    st.write(f"**From:** {flight['origin']} **To:** {flight['destination']}")
    st.write(f"**Date:** {flight['date']} **Time:** {flight['time']}")
    st.write(f"**Current Price:** ${flight['price']:.2f}")
    
    # Find best price
    min_price_idx = predictions['price'].idxmin()
    min_price = predictions.iloc[min_price_idx]
    
    savings = flight['price'] - min_price['price']
    savings_percent = (savings / flight['price']) * 100
    
    # Display chart
    st.header("Price Predictions")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(predictions['date'], predictions['price'], marker='o', linestyle='-', color='blue')
    ax.axhline(y=flight['price'], color='gray', linestyle='--', label='Current Price')
    
    # Highlight best day
    ax.scatter(min_price['date'], min_price['price'], color='green', s=100, zorder=5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title('Predicted Prices for Next 7 Days')
    ax.grid(True, alpha=0.3)
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    st.pyplot(fig)
    
    # Best time to buy
    st.header("Best Time to Buy")
    st.success(f"The best time to buy is on **{min_price['date']}** at **${min_price['price']:.2f}**")
    
    if savings > 0:
        st.write(f"You could save **${savings:.2f}** ({savings_percent:.1f}%) compared to the current price.")
    else:
        st.write("The current price is the best price. Consider buying now!")
    
    # Price prediction table
    st.header("Daily Price Predictions")
    
    # Format the prediction data for display
    display_predictions = predictions.copy()
    display_predictions['formatted_price'] = display_predictions['price'].apply(lambda x: f"${x:.2f}")
    
    st.table(display_predictions[['date', 'formatted_price']])
