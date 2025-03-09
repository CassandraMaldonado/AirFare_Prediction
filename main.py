import streamlit as st
import pandas as pd
import numpy as np

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
    # Convert inputs to the format expected by the model
    # For a real app, this would use the actual model prediction
    
    # Create a feature vector that the model expects
    features = [
        processed_inputs['origin_encoded'],
        processed_inputs['destination_encoded'],
        processed_inputs['days_until_departure'],
        processed_inputs['class_encoded'],
        processed_inputs['direct_flight'],
    ]
    
    # For demonstration, we'll create some sample flights
    # In a real app, you would use: predicted_price = model.predict([features])[0]
    base_price = model.predict([features])[0]
    
    # Create 3-5 flight options with slightly different prices
    import random
    num_flights = random.randint(3, 5)
    
    airlines = ["American Airlines", "Delta", "United", "Southwest", "JetBlue", "Alaska"]
    aircraft = ["Boeing 737", "Airbus A320", "Boeing 787", "Airbus A380", "Embraer E190"]
    
    flights = []
    
    for _ in range(num_flights):
        # Vary price by up to 20%
        price_variation = random.uniform(0.9, 1.1)
        price = base_price * price_variation * processed_inputs['passengers']
        
        # Generate departure time
        hour = random.randint(6, 21)
        minute = random.choice([0, 15, 30, 45])
        departure_time = f"{hour:02d}:{minute:02d}"
        
        # Generate duration based on distance between cities
        if abs(processed_inputs['origin_encoded'] - processed_inputs['destination_encoded']) <= 2:
            # Short flight
            hours = random.randint(1, 3)
            minutes = random.randint(0, 59)
        else:
            # Long flight
            hours = random.randint(3, 6)
            minutes = random.randint(0, 59)
        
        duration = f"{hours}h {minutes}m"
        
        # Generate stops if not direct
        stops = "Direct"
        if not processed_inputs['direct_flight']:
            stops = random.choice(["Direct", "1 stop", "2 stops"])
        
        flights.append({
            "airline": random.choice(airlines),
            "aircraft": random.choice(aircraft),
            "departure_time": departure_time,
            "duration": duration,
            "stops": stops,
            "price": price
        })
    
    # Sort by price
    flights.sort(key=lambda x: x['price'])
    
    return flights