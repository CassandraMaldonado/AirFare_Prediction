import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# Define a simple price predictor model
class SimpleFlightPricePredictor:
    def __init__(self):
        # This is a placeholder for the actual model
        pass
        
    def predict(self, current_price, days=7):
        """Simple price prediction for demo purposes"""
        predictions = []
        price = current_price
        
        # Generate some random price fluctuations for demo
        # Use a seed for reproducible results
        np.random.seed(42)
        for _ in range(days):
            # Random price change between -5% and +5%
            change = np.random.uniform(-0.05, 0.05)
            price = price * (1 + change)
            predictions.append(price)
            
        return predictions

# Function to create sample flight data
def create_sample_flights():
    # Set seed for reproducible results
    np.random.seed(42)
    origins = ['NYC', 'LAX', 'CHI', 'MIA', 'SFO']
    destinations = ['LON', 'PAR', 'TOK', 'SYD', 'BER']
    
    flights = []
    for i in range(10):
        origin = np.random.choice(origins)
        destination = np.random.choice(destinations)
        if origin != destination:
            date = (datetime.now() + timedelta(days=np.random.randint(7, 60))).strftime('%Y-%m-%d')
            time = f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}"
            price = round(np.random.uniform(300, 1200), 2)
            
            flights.append({
                'origin': origin,
                'destination': destination,
                'date': date,
                'time': time,
                'price': price
            })
    
    return pd.DataFrame(flights)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "input"
    
if 'flights' not in st.session_state:
    st.session_state.flights = create_sample_flights()
    
if 'selected_flight' not in st.session_state:
    st.session_state.selected_flight = None
    
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Function to predict prices
def predict_prices(flight):
    """Generate price predictions for the next 7 days"""
    model = SimpleFlightPricePredictor()
    current_price = flight['price']
    
    # Predict prices for next 7 days
    prices = model.predict(current_price, days=7)
    
    # Create dates for predictions
    start_date = datetime.strptime(flight['date'], '%Y-%m-%d')
    dates = [(start_date - timedelta(days=7-i)).strftime('%Y-%m-%d') for i in range(7)]
    
    # Create prediction dataframe
    predictions = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    return predictions

# Function to select a flight
def select_flight(index):
    st.session_state.selected_flight = st.session_state.flights.iloc[index]
    st.session_state.predictions = predict_prices(st.session_state.selected_flight)
    st.session_state.page = "results"

# Function to return to input page
def back_to_input():
    st.session_state.page = "input"
    st.session_state.selected_flight = None
    st.session_state.predictions = None

# Input page
def show_input_page():
    st.title("✈️ Flight Price Predictor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Select a Flight")
        
        # Display available flights
        for i, flight in st.session_state.flights.iterrows():
            with st.container():
                col_info, col_price, col_action = st.columns([3, 1, 1])
                
                with col_info:
                    st.write(f"**{flight['origin']} → {flight['destination']}**")
                    st.write(f"Date: {flight['date']} at {flight['time']}")
                
                with col_price:
                    st.write(f"**${flight['price']:.2f}**")
                
                with col_action:
                    if st.button("Select", key=f"select_{i}"):
                        select_flight(i)
    
    with col2:
        st.header("Add Custom Flight")
        
        with st.form("flight_form"):
            origin = st.text_input("Origin (airport code)", max_chars=3)
            destination = st.text_input("Destination (airport code)", max_chars=3)
            date = st.date_input("Date", min_value=datetime.now())
            time = st.time_input("Time")
            price = st.number_input("Current Price ($)", min_value=50.0, max_value=5000.0, value=500.0)
            
            submit = st.form_submit_button("Predict Prices")
            
            if submit:
                # Create flight data
                flight = {
                    'origin': origin.upper(),
                    'destination': destination.upper(),
                    'date': date.strftime('%Y-%m-%d'),
                    'time': time.strftime('%H:%M'),
                    'price': price
                }
                
                # Add to flights dataframe
                st.session_state.flights = pd.concat([
                    st.session_state.flights, 
                    pd.DataFrame([flight])
                ], ignore_index=True)
                
                # Select this flight
                st.session_state.selected_flight = flight
                st.session_state.predictions = predict_prices(flight)
                st.session_state.page = "results"

# Results page
def show_results_page():
    flight = st.session_state.selected_flight
    predictions = st.session_state.predictions
    
    st.title("✈️ Flight Price Prediction Results")
    
    # Back button
    if st.button("← Back to Flights"):
        back_to_input()
        st.experimental_rerun()
    
    # Flight details
    st.header("Flight Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**From:** {flight['origin']} **To:** {flight['destination']}")
        st.write(f"**Date:** {flight['date']} **Time:** {flight['time']}")
    
    with col2:
        st.write(f"**Current Price:** ${flight['price']:.2f}")
    
    # Price predictions
    st.header("Price Predictions")
    
    # Find best price
    min_price_idx = predictions['price'].idxmin()
    min_price = predictions.iloc[min_price_idx]
    
    savings = flight['price'] - min_price['price']
    savings_percent = (savings / flight['price']) * 100
    
    # Display chart
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
    st.subheader("Best Time to Buy")
    st.success(f"The best time to buy is on **{min_price['date']}** at **${min_price['price']:.2f}**")
    
    if savings > 0:
        st.write(f"You could save **${savings:.2f}** ({savings_percent:.1f}%) compared to the current price.")
    else:
        st.write("The current price is the best price. Consider buying now!")
    
    # Price prediction table
    st.subheader("Daily Price Predictions")
    
    # Add current price to table
    full_predictions = pd.DataFrame([{
        'date': 'Current',
        'price': flight['price'],
        'change': 0.0
    }])
    
    # Add predicted prices with change
    for i, row in predictions.iterrows():
        prev_price = flight['price'] if i == 0 else predictions.iloc[i-1]['price']
        change = (row['price'] - prev_price) / prev_price * 100
        
        full_predictions = pd.concat([
            full_predictions,
            pd.DataFrame([{
                'date': row['date'],
                'price': row['price'],
                'change': change
            }])
        ], ignore_index=True)
    
    # Format the table
    formatted_predictions = full_predictions.copy()
    formatted_predictions['price'] = formatted_predictions['price'].apply(lambda x: f"${x:.2f}")
    formatted_predictions['change'] = formatted_predictions['change'].apply(
        lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
    )
    
    st.dataframe(
        formatted_predictions,
        column_config={
            "date": "Date",
            "price": "Price",
            "change": "Change from Previous"
        },
        hide_index=True
    )

# Main app logic - show the appropriate page
if st.session_state.page == "input":
    show_input_page()
elif st.session_state.page == "results":
    show_results_page()
