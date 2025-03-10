import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set page title
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️")

# Initialize state
if 'input_completed' not in st.session_state:
    st.session_state.input_completed = False

if 'flight_data' not in st.session_state:
    st.session_state.flight_data = None

if 'price_predictions' not in st.session_state:
    st.session_state.price_predictions = None

# Airport codes
AIRPORTS = {
    'NYC': 'New York',
    'LAX': 'Los Angeles',
    'CHI': 'Chicago',
    'MIA': 'Miami',
    'SFO': 'San Francisco',
    'LON': 'London',
    'PAR': 'Paris',
    'TOK': 'Tokyo',
    'SYD': 'Sydney',
    'BER': 'Berlin'
}

# Simple price predictor function
def predict_prices(origin, destination, base_price):
    """Generate simple price predictions"""
    # Set seed based on inputs for consistent results
    np.random.seed(sum(ord(c) for c in origin + destination))
    
    # Generate prices for 7 days
    prices = []
    current_price = base_price
    for _ in range(7):
        # Random fluctuation
        change = np.random.uniform(-0.05, 0.05)
        current_price = current_price * (1 + change)
        prices.append(round(current_price, 2))
    
    # Generate dates
    start_date = datetime.now()
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    return df

# Function to reset app
def reset_app():
    st.session_state.input_completed = False
    st.session_state.flight_data = None
    st.session_state.price_predictions = None

# Main app logic
if not st.session_state.input_completed:
    # Input page
    st.title("Flight Price Predictor")
    
    # Flight details form
    st.header("Enter Flight Details")
    
    with st.form("flight_form"):
        # Origin and destination selection
        col1, col2 = st.columns(2)
        with col1:
            origin = st.selectbox("Origin", options=list(AIRPORTS.keys()), 
                                format_func=lambda x: f"{x} - {AIRPORTS[x]}")
        with col2:
            destination = st.selectbox("Destination", options=list(AIRPORTS.keys()), 
                                     format_func=lambda x: f"{x} - {AIRPORTS[x]}",
                                     index=1)
        
        # Date and time
        date = st.date_input("Travel Date", value=datetime.now() + timedelta(days=30), 
                           min_value=datetime.now())
        time = st.time_input("Departure Time", value=datetime.strptime("10:00", "%H:%M").time())
        
        # Submit button
        submitted = st.form_submit_button("Find Best Time to Buy")
        
        if submitted:
            if origin == destination:
                st.error("Origin and destination cannot be the same!")
            else:
                # Generate base price
                base_price = 300 + abs(ord(origin[0]) - ord(destination[0])) * 50
                
                # Store flight data
                st.session_state.flight_data = {
                    "origin": origin,
                    "origin_name": AIRPORTS[origin],
                    "destination": destination,
                    "destination_name": AIRPORTS[destination],
                    "date": date.strftime('%Y-%m-%d'),
                    "time": time.strftime('%H:%M'),
                    "base_price": base_price
                }
                
                # Generate predictions
                st.session_state.price_predictions = predict_prices(
                    origin, destination, base_price
                )
                
                # Mark input as completed
                st.session_state.input_completed = True
                
                # Force refresh
                st.rerun()

else:
    # Results page
    flight = st.session_state.flight_data
    predictions = st.session_state.price_predictions
    
    # Back button
    if st.button("← New Search"):
        reset_app()
        st.rerun()
    
    st.title("Flight Price Prediction")
    
    # Flight details card
    st.markdown("### Flight Details")
    st.markdown(f"""
    **{flight['origin']} ({flight['origin_name']})** → **{flight['destination']} ({flight['destination_name']})**  
    **Date:** {flight['date']}  
    **Time:** {flight['time']}  
    **Current Price:** ${flight['base_price']:.2f}
    """)
    
    # Find best day to buy
    min_price_idx = predictions['price'].idxmin()
    min_price = predictions.iloc[min_price_idx]
    
    # Calculate savings
    savings = flight['base_price'] - min_price['price']
    savings_percent = (savings / flight['base_price']) * 100
    
    # Display prediction results
    st.markdown("### Price Prediction Results")
    
    if savings > 0:
        st.success(f"""
        **Best day to buy:** {min_price['date']}  
        **Best price:** ${min_price['price']:.2f}  
        **Potential savings:** ${savings:.2f} ({savings_percent:.1f}%)
        """)
    else:
        st.warning("The current price is the best price. Consider buying now!")
    
    # Simple chart using Streamlit
    st.markdown("### Price Trend")
    
    # Add current price to chart data
    chart_data = pd.DataFrame({
        'Date': ['Current'] + predictions['date'].tolist(),
        'Price': [flight['base_price']] + predictions['price'].tolist()
    })
    
    st.line_chart(chart_data.set_index('Date')['Price'])
    
    # Price table
    st.markdown("### Daily Prices")
    
    # Format table
    display_data = predictions.copy()
    display_data['price'] = display_data['price'].apply(lambda x: f"${x:.2f}")
    
    # Display as table
    st.table(display_data.rename(columns={'date': 'Date', 'price': 'Price'}))
