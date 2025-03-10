import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os

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
    .stats-container {
        display: flex;
        justify-content: space-between;
        margin: 10px 0;
    }
    .stat-box {
        text-align: center;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
        flex: 1;
        margin: 0 5px;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1976D2;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #616161;
    }
    .volatile-high {
        color: #d32f2f;
    }
    .volatile-medium {
        color: #ff9800;
    }
    .volatile-low {
        color: #2e7d32;
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

if 'flight_combinations' not in st.session_state:
    # Load flight combinations from dataset
    # In production, you would read the actual CSV file
    # For now, we'll create it based on the provided images
    
    # Define the combinations data from the screenshots
    flight_combos_data = {
        'combination_id': ['1_4_83_0', '4_5_110_0', '2_11_83_0', '2_1_34_0', '1_4_34_0',
                           '10_1_29_1', '8_11_61_1', '6_12_125_0', '10_9_124_0', '14_2_34_0'],
        'route': ['1 to 4', '4 to 5', '2 to 11', '2 to 1', '1 to 4',
                  '10 to 1', '8 to 11', '6 to 12', '10 to 9', '14 to 2'],
        'startingAirport': [1, 4, 2, 2, 1, 10, 8, 6, 10, 14],
        'destinationAirport': [4, 5, 11, 1, 4, 1, 11, 12, 9, 2],
        'airlineCode': [83, 110, 83, 34, 34, 29, 61, 125, 124, 34],
        'isNonStop': [False, False, False, False, False, True, True, False, False, False],
        'search_count': [977, 482, 919, 839, 1291, 961, 1041, 53, 167, 1499],
        'min_price': [107.60, 47.58, 77.10, 101.10, 102.60, 68.60, 48.60, 412.09, 370.60, 107.60],
        'max_price': [1213.60, 436.58, 667.10, 574.10, 1364.60, 361.16, 398.60, 875.10, 1550.61, 691.61],
        'avg_price': [280.77, 224.10, 284.06, 260.84, 321.57, 115.16, 158.17, 626.72, 622.98, 248.68],
        'price_volatility': [0.41, 0.36, 0.36, 0.39, 0.47, 0.28, 0.33, 0.14, 0.26, 0.27],
        'travelDistance': [1624.35, 1609.90, 1189.23, 914.87, 1702.74, 193.92, 1106.64, 3344.71, 2515.71, 886.90],
        'travelDuration': [475.44, 579.09, 381.50, 342.84, 456.58, 80.07, 188.02, 946.35, 611.41, 352.61]
    }
    
    st.session_state.flight_combinations = pd.DataFrame(flight_combos_data)

# Map airport codes to user-friendly names
# We'll create a mapping based on the data we have
def generate_airport_names():
    """Generate airport names from the flight combination data"""
    combos = st.session_state.flight_combinations
    
    # Extract unique airports
    starting_airports = combos['startingAirport'].unique()
    destination_airports = combos['destinationAirport'].unique()
    all_airports = np.unique(np.concatenate([starting_airports, destination_airports]))
    
    # Create a mapping
    airport_map = {}
    for airport in all_airports:
        # Use a placeholder name based on the ID
        airport_map[str(airport)] = f"Airport {airport}"
    
    return airport_map

# Generate airline names based on the data
def generate_airline_names():
    """Generate airline names from the flight combination data"""
    combos = st.session_state.flight_combinations
    
    # Extract unique airlines
    airlines = combos['airlineCode'].unique()
    
    # Create a mapping
    airline_map = {}
    for airline in airlines:
        # Use a placeholder name based on the ID
        airline_map[str(airline)] = f"Airline {airline}"
    
    return airline_map

# Load price prediction model (in a real app, you would load your XGBoost model here)
def load_prediction_model():
    """Load or simulate the price prediction model"""
    # In a real scenario, you would load your XGBoost model:
    # if os.path.exists('xgboost_model.pkl'):
    #     with open('xgboost_model.pkl', 'rb') as f:
    #         model = pickle.load(f)
    #     return model
    
    # For the demo, we'll just use a simple function
    class SimplePredictionModel:
        def predict(self, features):
            """Simulate price predictions using the features"""
            # Base the prediction on the average price for the combination
            combo_id = features.get('flight_combo_id')
            combo_data = st.session_state.flight_combinations[
                st.session_state.flight_combinations['combination_id'] == combo_id
            ]
            
            if len(combo_data) == 0:
                # Fallback if combination not found
                return np.array([300.0] * 7)
            
            # Get the base statistics
            base_price = combo_data['avg_price'].values[0]
            volatility = combo_data['price_volatility'].values[0]
            min_price = combo_data['min_price'].values[0]
            max_price = combo_data['max_price'].values[0]
            
            # Generate a 7-day price forecast using the volatility and price range
            np.random.seed(int(hash(combo_id) % 2**32))
            
            # Calculate days to departure
            days_to_departure = features.get('days_to_departure', 30)
            
            # Determine price trend based on days to departure
            if days_to_departure > 60:
                # Far from travel date - prices likely to drop slightly
                trend_factor = -0.02
            elif days_to_departure > 30:
                # Getting closer - prices stabilizing
                trend_factor = 0.005
            elif days_to_departure > 14:
                # Approaching travel date - prices rising
                trend_factor = 0.01
            else:
                # Very close to travel date - prices rising rapidly
                trend_factor = 0.03
            
            # Generate price predictions
            prices = []
            current_price = base_price
            
            for i in range(7):
                # Apply volatility and trend
                random_factor = np.random.normal(0, volatility * 0.1)
                day_effect = trend_factor + random_factor
                current_price = current_price * (1 + day_effect)
                
                # Ensure the price is within reasonable bounds
                current_price = max(min_price * 0.9, min(current_price, max_price * 1.1))
                prices.append(current_price)
            
            return np.array(prices)
    
    return SimplePredictionModel()

# Enhanced price predictor using the model
def predict_prices(flight_combo_id, travel_date_str):
    """Generate price predictions using the model or simulation"""
    # Convert date string to datetime
    travel_date = datetime.strptime(travel_date_str, '%Y-%m-%d')
    current_date = datetime.now()
    days_until_travel = (travel_date - current_date).days
    
    # Get the flight combination data
    combo_data = st.session_state.flight_combinations[
        st.session_state.flight_combinations['combination_id'] == flight_combo_id
    ]
    
    if len(combo_data) == 0:
        st.error(f"Flight combination {flight_combo_id} not found!")
        return None
    
    # Load or get the prediction model
    model = load_prediction_model()
    
    # Prepare features for the model
    features = {
        'flight_combo_id': flight_combo_id,
        'days_to_departure': days_until_travel,
        'is_weekend': 1 if travel_date.weekday() >= 5 else 0,
        'travel_month': travel_date.month,
        'travel_day': travel_date.day,
        'search_date': current_date.strftime('%Y-%m-%d')
    }
    
    # Get price predictions
    predicted_prices = model.predict(features)
    
    # Generate dates for the predictions
    dates = [(current_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    days = [(current_date + timedelta(days=i)).strftime('%A') for i in range(7)]
    
    # Create DataFrame with the predictions
    df = pd.DataFrame({
        'date': dates,
        'day': days,
        'price': [round(price, 2) for price in predicted_prices]
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
    st.markdown("<h1 class='main-header'>✈️ Flight Price Predictor</h1>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='subheader'>Find the best time to buy your ticket</h2>", unsafe_allow_html=True)
        
        # Generate airport names
        AIRPORTS = generate_airport_names()
        # Generate airline names
        AIRLINES = generate_airline_names()
        
        # Display flight combinations tab
        flight_combos = st.session_state.flight_combinations
        
        with st.form("flight_form"):
            # Two options: Choose from existing combinations or custom selection
            selection_method = st.radio(
                "Select flight",
                ["Choose from existing routes", "Custom selection"]
            )
            
            if selection_method == "Choose from existing routes":
                # Show existing flight combinations
                combo_options = flight_combos['combination_id'].tolist()
                combo_display = [
                    f"{row['route']} (Airline: {row['airlineCode']}, {'Nonstop' if row['isNonStop'] else 'Connecting'})"
                    for _, row in flight_combos.iterrows()
                ]
                
                selected_combo_idx = st.selectbox(
                    "Select flight route",
                    options=range(len(combo_options)),
                    format_func=lambda i: combo_display[i]
                )
                
                selected_combo_id = combo_options[selected_combo_idx]
                selected_combo = flight_combos[flight_combos['combination_id'] == selected_combo_id].iloc[0]
                
                # Show statistics about the selected route
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg. Price", f"${selected_combo['avg_price']:.2f}")
                with col2:
                    st.metric("Min Price", f"${selected_combo['min_price']:.2f}")
                with col3:
                    st.metric("Max Price", f"${selected_combo['max_price']:.2f}")
                
                # Show volatility
                volatility = selected_combo['price_volatility']
                volatility_class = "volatile-high" if volatility > 0.4 else "volatile-medium" if volatility > 0.2 else "volatile-low"
                
                st.markdown(f"""
                <div class='stats-container'>
                    <div class='stat-box'>
                        <div class='stat-value {volatility_class}'>{volatility:.2f}</div>
                        <div class='stat-label'>Price Volatility</div>
                    </div>
                    <div class='stat-box'>
                        <div class='stat-value'>{selected_combo['search_count']}</div>
                        <div class='stat-label'>Search Count</div>
                    </div>
                    <div class='stat-box'>
                        <div class='stat-value'>{selected_combo['travelDuration']:.0f} min</div>
                        <div class='stat-label'>Avg. Duration</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a clear visual indicator of whether this is a good route to book
                if volatility < 0.2:
                    st.success("💰 This route has stable pricing - good for budget planning!")
                elif volatility > 0.4:
                    st.warning("⚠️ This route has highly variable pricing - timing matters!")
                
            else:  # Custom selection
                # Origin and destination selection
                col1, col2 = st.columns(2)
                with col1:
                    origin = st.selectbox(
                        "Origin", 
                        options=sorted(AIRPORTS.keys()), 
                        format_func=lambda x: f"{x} - {AIRPORTS[x]}"
                    )
                with col2:
                    # Set default to a different airport than origin
                    destination_options = sorted([k for k in AIRPORTS.keys() if k != origin])
                    destination = st.selectbox(
                        "Destination", 
                        options=destination_options, 
                        format_func=lambda x: f"{x} - {AIRPORTS[x]}"
                    )
                
                # Airline and flight type
                col3, col4 = st.columns(2)
                with col3:
                    airline = st.selectbox(
                        "Airline",
                        options=sorted(AIRLINES.keys()),
                        format_func=lambda x: AIRLINES[x]
                    )
                with col4:
                    is_nonstop = st.checkbox("Nonstop flight", value=False)
                
                # Create a combination ID
                selected_combo_id = f"{origin}_{destination}_{airline}_{1 if is_nonstop else 0}"
                
                # Check if this combination exists in our data
                matching_combos = flight_combos[flight_combos['combination_id'] == selected_combo_id]
                if not matching_combos.empty:
                    selected_combo = matching_combos.iloc[0]
                    st.success("✅ This flight combination exists in our database!")
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg. Price", f"${selected_combo['avg_price']:.2f}")
                    with col2:
                        st.metric("Min Price", f"${selected_combo['min_price']:.2f}")
                    with col3:
                        st.metric("Max Price", f"${selected_combo['max_price']:.2f}")
                else:
                    st.warning("⚠️ This specific combination doesn't exist in our database. Price predictions may be less accurate.")
                    selected_combo = None
            
            # Date selection
            travel_date = st.date_input(
                "Travel Date", 
                value=datetime.now() + timedelta(days=30), 
                min_value=datetime.now()
            )
            
            # Number of travelers
            travelers = st.number_input("Number of Travelers", min_value=1, max_value=10, value=1)
            
            # Submit button
            submitted = st.form_submit_button("Find Best Time to Buy", use_container_width=True)
        
        if submitted:
            if selection_method == "Custom selection" and origin == destination:
                st.error("Origin and destination cannot be the same!")
            else:
                # Generate price predictions
                predictions = predict_prices(selected_combo_id, travel_date.strftime('%Y-%m-%d'))
                
                if predictions is not None:
                    # Store flight data
                    if selection_method == "Choose from existing routes":
                        # Get data from the selected combination
                        st.session_state.flight_data = {
                            "combo_id": selected_combo_id,
                            "origin": str(selected_combo['startingAirport']),
                            "origin_name": AIRPORTS.get(str(selected_combo['startingAirport']), f"Airport {selected_combo['startingAirport']}"),
                            "destination": str(selected_combo['destinationAirport']),
                            "destination_name": AIRPORTS.get(str(selected_combo['destinationAirport']), f"Airport {selected_combo['destinationAirport']}"),
                            "airline": str(selected_combo['airlineCode']),
                            "airline_name": AIRLINES.get(str(selected_combo['airlineCode']), f"Airline {selected_combo['airlineCode']}"),
                            "isNonStop": selected_combo['isNonStop'],
                            "date": travel_date.strftime('%Y-%m-%d'),
                            "travelers": travelers,
                            "base_price": selected_combo['avg_price'],
                            "total_price": selected_combo['avg_price'] * travelers,
                            "min_price": selected_combo['min_price'],
                            "max_price": selected_combo['max_price'],
                            "travelDuration": selected_combo['travelDuration'] if 'travelDuration' in selected_combo else None,
                            "price_volatility": selected_combo['price_volatility'] if 'price_volatility' in selected_combo else None
                        }
                    else:  # Custom selection
                        # Use data from the form
                        avg_price = selected_combo['avg_price'] if selected_combo is not None else 300.0
                        min_price = selected_combo['min_price'] if selected_combo is not None else 200.0
                        max_price = selected_combo['max_price'] if selected_combo is not None else 500.0
                        
                        st.session_state.flight_data = {
                            "combo_id": selected_combo_id,
                            "origin": origin,
                            "origin_name": AIRPORTS.get(origin, f"Airport {origin}"),
                            "destination": destination,
                            "destination_name": AIRPORTS.get(destination, f"Airport {destination}"),
                            "airline": airline,
                            "airline_name": AIRLINES.get(airline, f"Airline {airline}"),
                            "isNonStop": is_nonstop,
                            "date": travel_date.strftime('%Y-%m-%d'),
                            "travelers": travelers,
                            "base_price": avg_price,
                            "total_price": avg_price * travelers,
                            "min_price": min_price,
                            "max_price": max_price,
                            "travelDuration": selected_combo['travelDuration'] if selected_combo is not None and 'travelDuration' in selected_combo else None,
                            "price_volatility": selected_combo['price_volatility'] if selected_combo is not None and 'price_volatility' in selected_combo else 0.3
                        }
                    
                    # Store predictions
                    st.session_state.price_predictions = predictions
                    
                    # Mark input as completed
                    st.session_state.input_completed = True
                    
                    # Force refresh
                    st.rerun()
                else:
                    st.error("Could not generate price predictions. Please try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Tips section
        with st.expander("💡 Tips for Finding Cheap Flights"):
            st.markdown("""
            * **Book in advance**: Typically, flights are cheapest when booked 3-4 months before departure
            * **Be flexible with dates**: Flying mid-week (Tuesday, Wednesday) is often cheaper
            * **Consider nearby airports**: Sometimes flying to/from alternative airports can save money
            * **Check flight combinations with lower price volatility**: Routes with low volatility (< 0.2) tend to have more predictable prices
            * **Monitor routes with high search counts**: Popular routes often have more competitive pricing
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
        airline_text = f"**Airline:** {flight['airline_name']} | **{'Nonstop' if flight['isNonStop'] else 'Connecting'}**"
        date_text = f"**Date:** {flight['date']}"
        travelers_text = f"**Travelers:** {flight['travelers']}"
        
        st.markdown(f"<div class='flight-info'>{airport_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='flight-info'>{airline_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='flight-info'>{date_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='flight-info'>{travelers_text}</div>", unsafe_allow_html=True)
        
        if flight['travelDuration'] is not None:
            duration_hours = int(flight['travelDuration'] // 60)
            duration_minutes = int(flight['travelDuration'] % 60)
            duration_text = f"**Flight Duration:** {duration_hours}h {duration_minutes}m"
            st.markdown(f"<div class='flight-info'>{duration_text}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3>Current Price</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='price-box'>${flight['total_price']:.2f}</div>", unsafe_allow_html=True)
        if flight['travelers'] > 1:
            st.markdown(f"${flight['base_price']:.2f} per traveler", unsafe_allow_html=True)
        
        # Add price volatility indicator
        if 'price_volatility' in flight and flight['price_volatility'] is not None:
            volatility = flight['price_volatility']
            volatility_class = "volatile-high" if volatility > 0.4 else "volatile-medium" if volatility > 0.2 else "volatile-low"
            volatility_label = "High" if volatility > 0.4 else "Medium" if volatility > 0.2 else "Low"
            
            st.markdown(f"""
            <div style='margin-top: 10px; text-align: center;'>
                <span class='{volatility_class}' style='font-weight: bold;'>
                    {volatility_label} Volatility ({volatility:.2f})
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Find best day to buy
    min_price_idx = predictions['price'].idxmin()
    min_price = predictions.iloc[min_price_idx]
    best_day_name = min_price['day']
    
    # Calculate savings
    savings = flight['total_price'] - min_price['price']
    savings_percent = (savings / flight['total_price']) * 100 if flight['total_price'] > 0 else 0
    
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
    
    # Add a confidence indicator based on price volatility
    if 'price_volatility' in flight and flight['price_volatility'] is not None:
        volatility = flight['price_volatility']
        confidence = "high" if volatility < 0.2 else "medium" if volatility < 0.4 else "low"
        confidence_text = {
            "high": "Our prediction has high confidence due to the stable pricing of this route.",
            "medium": "Our prediction has moderate confidence as this route shows some price variability.",
            "low": "Our prediction has lower confidence as this route has highly variable pricing."
        }
        confidence_icon = {
            "high": "✅", 
            "medium": "⚠️",
            "low": "⚠️"
        }
        
        st.markdown(f"""
        <div style='margin-top: 10px; font-style: italic;'>
            {confidence_icon[confidence]} <strong>Prediction Confidence:</strong> {confidence_text[confidence]}
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
        display_data['formatted_price
