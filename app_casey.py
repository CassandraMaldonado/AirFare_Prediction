import os
import joblib
import streamlit as st
import traceback
from main import collect_inputs, display_results

st.set_page_config(layout="wide")

# Define model file path
MODEL_PATH = "xg_boost_model_compressed.pkl"

@st.cache_resource
def load_model():
    """Load the trained XGBoost model from the local file."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()
        
    return joblib.load(MODEL_PATH)

# Load model with error handling
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.text(traceback.format_exc())  # Show full traceback
    st.stop()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "input"

# **Page 1: Collect Inputs**
if st.session_state.page == "input":
    st.title("✈️ Flight Price Predictor")

    # Get user inputs with error handling
    try:
        user_inputs = collect_inputs()
    except Exception as e:
        st.error("❌ Error collecting inputs")
        st.text(traceback.format_exc())  # Show full traceback
        st.stop()

    if st.button("Search Flights"):
        st.session_state.page = "results"
        st.session_state.user_inputs = user_inputs
        st.rerun()

# **Page 2: Show Predictions**
elif st.session_state.page == "results":
    st.title("✈️ Flight Price Results")
    
    # Add a "Back" button
    if st.button("⬅️ Back to Search"):
        st.session_state.page = "input"
        st.rerun()
        
    display_results(model)
