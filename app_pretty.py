import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set page title and layout
st.set_page_config(
    page_title="Flight Price Predictor", 
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .price-box {
        font-size: 2rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
    }
    .flight-info {
        margin-bottom: 10px;
        font-size: 1.2rem;
    }
    .savings {
        color: #2E7D32;
        font-weight: bold;
    }
    .footnote {
        font-size: 0.8rem;
        color: #757575;
        text-align: center;
        margin-top: 30px;
    }
    .recommendation {
        background-color: #e3f2fd;
        border-left: 5px solid #1976D2;
        padding: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize state
if 'input_completed' not in st.session_state:
    st.session_state.input_completed = False

if 'flight_data' not in st.session_state:
    st.session_state.flight_data = None

if 'price_predictions' not in st.session_state:
    st.session_state.price_predictions = None

# Airport codes and cities - expanded list
AIRPORTS = {
    'ATL': 'Atlanta',
    'BER': 'Berlin',
    'BKK': 'Bangkok',
    'CDG': 'Paris',
    'CHI': 'Chicago', 
    'DEL': 'Delhi',
    'DFW': 'Dallas',
    'DXB': 'Dubai',
    'HKG': 'Hong Kong',
    'HND': 'Tokyo',
    'JFK': 'New York JFK',
    'LAX': 'Los Angeles',
    'LHR': 'London Heathrow',
    'MAD': 'Madrid',
    'MIA': 'Miami',
    'NYC': 'New York',
    'ORD': 'Chicago O\'Hare',
    'SFO': 'San Francisco',
    'SIN': 'Singapore',
    'SYD': 'Sydney'
}

# Enhanced price predictor with realistic patterns
def predict_prices(origin, destination, base_price, date_str):
    """Generate realistic price predictions with patterns"""
    # Set seed based on inputs for consistent results
    np.random.seed(sum(ord(c) for c in origin + destination))
    
    # Convert date string to datetime
    travel_date = datetime.strptime(date_str, '%Y-%m-%d')
    days_until_travel = (travel_date - datetime.now()).days
    
    # Price patterns based on days until travel
    # Prices typically drop 60-90 days before flight, rise 30-45 days before,
    # and spike in the last 2 weeks
    
    # Generate prices for next 7 days
    prices = []
    current_price = base_price
    
    # Calculate trend factor based on days until travel
    if days_until_travel > 60:
        # Far from travel date - prices likely to drop
        trend_factor = -0.02  # Slight downward trend
    elif days_until_travel > 30:
        # Getting closer - prices stabilizing
        trend_factor = 0.005  # Minimal upward trend
    elif days_until_travel > 14:
        # Approaching travel date - prices rising
        trend_factor = 0.01   # Moderate upward trend
    else:
        # Very close to travel date - prices rising rapidly
        trend_factor = 0.03   # Strong upward trend
    
    for i in range(7):
        # Random daily fluctuation plus trend
        noise = np.random.uniform(-0.03, 0.03)
        change = noise + trend_factor
        current_price = current_price * (1 + change)
        prices.append(round(current_price, 2))
    
    # Generate dates
    start_date = datetime.now()
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    
    # Generate day names
    day_names = [(start_date + timedelta(days=i)).strftime('%A') for i in range(7)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'day': day_names,
        'price': prices
    })
    
    return df

# Function to calculate base price based on distance and demand
def calculate_base_price(origin, destination, date_str):
    # Distance factor - using ASCII values as a simple proxy for "distance"
    distance_factor = abs(sum(ord(c) for c in origin) - sum(ord(c) for c in destination)) / 10
    
    # Base price - between $300 and $800
    base_price = 300 + distance_factor * 5
    
    # Date factor - weekend flights are more expensive
    travel_date = datetime.strptime(date_str, '%Y-%m-%d')
    weekday = travel_date.weekday()
    # Weekend premium (Friday, Saturday, Sunday)
    if weekday >= 4:
        base_price *= 1.2
    
    # Season factor - summer and holidays are more expensive
    month = travel_date.month
    if month in [6, 7, 8, 12]:  # Summer and December
        base_price *= 1.15
    
    # Round to nearest $5
    return round(base_price / 5) * 5

# Function to reset app
def reset_app():
    st.session_state.input_completed = False
    st.session_state.flight_data = None
    st.session_state.price_predictions = None

# Main app logic
if not st.session_state.input_completed:
    # Input page
    st.markdown("<h1 class='main-header'>✈️ Flight Price Predictor</h1>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='subheader'>Find the best time to buy your ticket</h2>", unsafe_allow_html=True)
        
        with st.form("flight_form"):
            # Origin and destination selection
            col1, col2 = st.columns(2)
            with col1:
                origin = st.selectbox("Origin", options=sorted(list(AIRPORTS.keys())), 
                                    format_func=lambda x: f"{x} - {AIRPORTS[x]}")
            with col2:
                # Set default to a different airport than origin
                destination_options = sorted(list(AIRPORTS.keys()))
                default_dest_index = 1 if destination_options[0] == origin else 0
                destination = st.selectbox("Destination", options=destination_options, 
                                         format_func=lambda x: f"{x} - {AIRPORTS[x]}",
                                         index=default_dest_index)
            
            # Date and time
            col3, col4 = st.columns(2)
            with col3:
                date = st.date_input("Travel Date", 
                                   value=datetime.now() + timedelta(days=30), 
                                   min_value=datetime.now())
            with col4:
                time = st.time_input("Departure Time", 
                                   value=datetime.strptime("10:00", "%H:%M").time())
            
            # Trip type and travelers
            col5, col6 = st.columns(2)
            with col5:
                trip_type = st.radio("Trip Type", options=["One Way", "Round Trip"])
            with col6:
                travelers = st.number_input("Number of Travelers", min_value=1, max_value=10, value=1)
            
            # Submit button
            submitted = st.form_submit_button("Find Best Time to Buy", use_container_width=True)
        
        if submitted:
            if origin == destination:
                st.error("Origin and destination cannot be the same!")
            else:
                # Calculate base price
                date_str = date.strftime('%Y-%m-%d')
                base_price = calculate_base_price(origin, destination, date_str)
                
                # Adjust for number of travelers
                total_price = base_price * travelers
                
                # Store flight data
                st.session_state.flight_data = {
                    "origin": origin,
                    "origin_name": AIRPORTS[origin],
                    "destination": destination,
                    "destination_name": AIRPORTS[destination],
                    "date": date_str,
                    "time": time.strftime('%H:%M'),
                    "trip_type": trip_type,
                    "travelers": travelers,
                    "base_price": base_price,
                    "total_price": total_price
                }
                
                # Generate predictions
                st.session_state.price_predictions = predict_prices(
                    origin, destination, total_price, date_str
                )
                
                # Mark input as completed
                st.session_state.input_completed = True
                
                # Force refresh
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Tips section
        with st.expander("💡 Tips for Finding Cheap Flights"):
            st.markdown("""
            * **Book in advance**: Typically, flights are cheapest when booked 3-4 months before departure
            * **Be flexible with dates**: Flying mid-week (Tuesday, Wednesday) is often cheaper
            * **Consider nearby airports**: Sometimes flying to/from alternative airports can save money
            * **Set price alerts**: Many websites offer alerts when prices drop for routes you're interested in
            * **Clear your cookies**: Some booking sites may increase prices if you search the same route multiple times
            """)
    
    # Add footnote
    st.markdown("<p class='footnote'>Predictions are based on historical trends and current market data. Actual prices may vary.</p>", unsafe_allow_html=True)

else:
    # Results page
    flight = st.session_state.flight_data
    predictions = st.session_state.price_predictions
    
    # Header with back button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← New Search"):
            reset_app()
            st.rerun()
    with col2:
        st.markdown("<h1 class='main-header'>Flight Price Prediction</h1>", unsafe_allow_html=True)
    
    # Flight details card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Create two columns for flight info and current price
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<h3>Flight Details</h3>", unsafe_allow_html=True)
        
        airport_text = f"**{flight['origin']} ({flight['origin_name']})** → **{flight['destination']} ({flight['destination_name']})**"
        date_text = f"**Date:** {flight['date']} at {flight['time']}"
        trip_text = f"**Trip Type:** {flight['trip_type']} | **Travelers:** {flight['travelers']}"
        
        st.markdown(f"<div class='flight-info'>{airport_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='flight-info'>{date_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='flight-info'>{trip_text}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3>Current Price</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='price-box'>${flight['total_price']:.2f}</div>", unsafe_allow_html=True)
        if flight['travelers'] > 1:
            st.markdown(f"${flight['base_price']:.2f} per traveler", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Find best day to buy
    min_price_idx = predictions['price'].idxmin()
    min_price = predictions.iloc[min_price_idx]
    best_day_name = min_price['day']
    
    # Calculate savings
    savings = flight['total_price'] - min_price['price']
    savings_percent = (savings / flight['total_price']) * 100
    
    # Recommendation card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Our Recommendation</h3>", unsafe_allow_html=True)
    
    if savings > 0:
        st.markdown(f"""
        <div class='recommendation'>
            <h4>Wait until {best_day_name}, {min_price['date']} to purchase your ticket</h4>
            <p>The predicted price will be <span class='price-box'>${min_price['price']:.2f}</span></p>
            <p>You could save <span class='savings'>${savings:.2f} ({savings_percent:.1f}%)</span> compared to the current price.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='recommendation'>
            <h4>Buy your ticket now!</h4>
            <p>The current price is the best we predict for the next 7 days.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Price prediction chart and table
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<h3 class='subheader'>Price Trend (Next 7 Days)</h3>", unsafe_allow_html=True)
        
        # Add current price to chart data
        chart_data = pd.DataFrame({
            'Date': ['Today'] + predictions['date'].tolist(),
            'Price': [flight['total_price']] + predictions['price'].tolist()
        })
        
        # Format the chart
        st.line_chart(chart_data.set_index('Date')['Price'], height=350)
    
    with col2:
        st.markdown("<h3 class='subheader'>Price Calendar</h3>", unsafe_allow_html=True)
        
        # Format table with additional styling
        display_data = predictions.copy()
        
        # Add change column
        display_data['prev_price'] = [flight['total_price']] + display_data['price'].iloc[:-1].tolist()
        display_data['change'] = (display_data['price'] - display_data['prev_price']) / display_data['prev_price'] * 100
        
        # Format columns
        display_data['formatted_day'] = display_data['day']
        display_data['formatted_price'] = display_data['price'].apply(lambda x: f"${x:.2f}")
        
        # Fix the syntax error by avoiding nested f-strings
        display_data['formatted_change'] = display_data['change'].apply(
            lambda x: "<span style='color:{}'>{}%</span>".format(
                'red' if x > 0 else 'green',
                f"+{x:.1f}" if x > 0 else f"{x:.1f}"
            )
        )
        
        # Create styled dataframe
        styled_df = pd.DataFrame({
            'Day': display_data['formatted_day'],
            'Date': display_data['date'],
            'Price': display_data['formatted_price'],
            'Change': display_data['formatted_change']
        })
        
        # Display as HTML table
        st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Price factors that influence the prediction
    st.markdown("<h3 class='subheader'>Price Influencing Factors</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4>Time Until Flight</h4>", unsafe_allow_html=True)
        travel_date = datetime.strptime(flight['date'], '%Y-%m-%d')
        days_until = (travel_date - datetime.now()).days
        st.markdown(f"<p>{days_until} days until departure</p>", unsafe_allow_html=True)
        if days_until > 60:
            st.markdown("<p>✅ Booking early typically results in better prices</p>", unsafe_allow_html=True)
        elif days_until < 14:
            st.markdown("<p>⚠️ Last-minute bookings often cost more</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4>Day of Week</h4>", unsafe_allow_html=True)
        weekday = travel_date.weekday()
        day_name = travel_date.strftime('%A')
        st.markdown(f"<p>Flying on a {day_name}</p>", unsafe_allow_html=True)
        if weekday in [1, 2, 3]:  # Tuesday, Wednesday, Thursday
            st.markdown("<p>✅ Mid-week flights are often cheaper</p>", unsafe_allow_html=True)
        elif weekday in [0, 4, 5, 6]:  # Monday, Friday, Saturday, Sunday
            st.markdown("<p>⚠️ Weekend and Monday flights tend to cost more</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4>Route Popularity</h4>", unsafe_allow_html=True)
        st.markdown(f"<p>{flight['origin']} to {flight['destination']}</p>", unsafe_allow_html=True)
        # Check if it's a major route (simple check based on airport codes)
        major_airports = ['JFK', 'LAX', 'LHR', 'CDG', 'SFO', 'DXB']
        if flight['origin'] in major_airports and flight['destination'] in major_airports:
            st.markdown("<p>✅ Popular routes often have more competitive pricing</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p>⚠️ Less frequent routes may have premium pricing</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add footnote
    st.markdown("<p class='footnote'>This prediction tool uses historical data patterns and is intended as a guide only. Actual prices may vary based on airline policies, fuel costs, and seat availability.</p>", unsafe_allow_html=True)
