import streamlit as st
import pandas as pd
import numpy as np
import os

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
    
    # Return all inputs as a dictionary
    return {
        "origin": origin,
        "destination": destination,
        "flight_class": flight_class,
        "departure_date": departure_date,
        "passengers": passengers,
        "direct_flight": direct_flight
    }

def display_results(model):
    """
    Display flight price prediction results.
    
    Args:
        model: The trained model for price prediction
    """
    st.title("✈️ Flight Price Results")
    
    # Get user inputs from session state
    user_inputs = st.session_state.user_inputs
    
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
        st.write(f"**Direct Flight:** {'Yes' if user_inputs['direct_flight'] else 'No'}")
    
    # Process inputs for model prediction
    processed_inputs = preprocess_inputs(user_inputs)
    
    # Make predictions
    try:
        predictions = predict_prices(model, processed_inputs)
        
        # Display flight options
        st.subheader("✈️ Available Flights")
        
        for i, flight in enumerate(predictions):
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.subheader(f"{user_inputs['origin']} → {user_inputs['destination']}")
                    st.write(f"**Departure:** {flight['departure_time']}")
                    if not user_inputs['direct_flight']:
                        st.write(f"**Stops:** {flight['stops']}")
                    st.write(f"**Duration:** {flight['duration']}")
                
                with col2:
                    st.write("**Airline:** " + flight['airline'])
                    st.write(f"**Aircraft:** {flight['aircraft']}")
                
                with col3:
                    st.subheader(f"${flight['price']:,.2f}")
                    st.write(f"per passenger")
                    
                    if st.button("Select", key=f"select_{i}"):
                        st.session_state.selected_flight = flight
                        st.success("✅ Flight selected! Continue to booking.")
                
                st.divider()
        
    except Exception as e:
        st.error(f"❌ Error making predictions: {e}")
        import traceback
        st.text(traceback.format_exc())

def preprocess_inputs(user_inputs):
    """
    Preprocess user inputs for model prediction.
    
    Args:
        user_inputs (dict): Raw user inputs
        
    Returns:
        dict: Processed inputs ready for model prediction
    """
    # This would normally convert raw inputs to the format expected by the model
    # For demonstration, we'll return a simplified version
    
    # Encode categorical variables
    origin_encoded = encode_city(user_inputs['origin'])
    destination_encoded = encode_city(user_inputs['destination'])
    
    # Calculate days until departure
    import datetime
    days_until_departure = (user_inputs['departure_date'] - datetime.date.today()).days
    
    # Encode flight class
    class_mapping = {"Economy": 0, "Premium Economy": 1, "Business": 2, "First": 3}
    class_encoded = class_mapping[user_inputs['flight_class']]
    
    # Return processed features
    return {
        "origin_encoded": origin_encoded,
        "destination_encoded": destination_encoded,
        "days_until_departure": days_until_departure,
        "class_encoded": class_encoded,
        "direct_flight": int(user_inputs['direct_flight']),
        "passengers": user_inputs['passengers']
    }

def encode_city(city):
    """
    Encode city names to numerical values.
    
    Args:
        city (str): City name
        
    Returns:
        int: Encoded value
    """
    city_mapping = {
        "New York": 0,
        "Los Angeles": 1,
        "Chicago": 2,
        "Dallas": 3,
        "Atlanta": 4,
        "San Francisco": 5,
        "Denver": 6,
        "Miami": 7
    }
    return city_mapping.get(city, 0)

def predict_prices(model, processed_inputs):
    """
    Generate flight predictions using the trained model.
    
    Args:
        model: Trained prediction model
        processed_inputs (dict): Processed user inputs
        
    Returns:
        list: List of flight options with predictions
    """
    # Create a feature vector that the model expects
    features = [
        processed_inputs['origin_encoded'],
        processed_inputs['destination_encoded'],
        processed_inputs['days_until_departure'],
        processed_inputs['class_encoded'],
        processed_inputs['direct_flight'],
    ]
    
    # Get base price from the actual model
    base_price = model.predict([features])[0]
    
    # Create flight options with real price prediction
    import random
    airlines = ["American Airlines", "Delta", "United", "Southwest", "JetBlue", "Alaska"]
    aircraft = ["Boeing 737", "Airbus A320", "Boeing 787", "Airbus A380", "Embraer E190"]
    
    # Create 3 flight options with slight price variations based on the model's prediction
    flights = []
    
    for i in range(3):
        # Apply a small variation to reflect different flight options
        # Each subsequent option is slightly more expensive
        price_factor = 1.0 + (i * 0.05)  # 0%, 5%, 10% increase
        price = base_price * price_factor * processed_inputs['passengers']
        
        # Create realistic departure times
        morning_departure = "08:30" if i == 0 else "10:15"
        afternoon_departure = "13:45" if i == 0 else "15:20"
        evening_departure = "18:30" if i == 0 else "20:45"
        departure_options = [morning_departure, afternoon_departure, evening_departure]
        departure_time = departure_options[i % 3]
        
        # Set duration based on distance
        city_distance = abs(processed_inputs['origin_encoded'] - processed_inputs['destination_encoded'])
        if city_distance <= 1:
            duration = "1h 45m"
        elif city_distance <= 3:
            duration = "3h 15m"
        else:
            duration = "5h 30m"
        
        # Set stops based on direct flight preference
        stops = "Direct"
        if not processed_inputs['direct_flight'] and i > 0:
            stops = "1 stop" if i == 1 else "2 stops"
        
        flights.append({
            "airline": airlines[i % len(airlines)],
            "aircraft": aircraft[i % len(aircraft)],
            "departure_time": departure_time,
            "duration": duration,
            "stops": stops,
            "price": price
        })
    
    # Sort by price
    flights.sort(key=lambda x: x['price'])
    
    return flights
