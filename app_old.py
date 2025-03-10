import os
import joblib
import streamlit as st
import traceback
from main import collect_inputs, display_results

st.set_page_config(layout="wide")

# Define model file path
MODEL_PATH = "xc_boost_model_compressed.pkl"

# Ensure XGBoost is imported if needed
try:
    import xgboost as xgb
except ImportError:
    st.error("❌ XGBoost package is required but not installed")
    st.info("Try installing it with: pip install xgboost")
    st.stop()

@st.cache_resource
def load_model():
    """Load the trained XGBoost model from the local file."""
    try:
        # First try standard joblib loading
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Standard loading failed: {e}. Trying alternative method...")
        
        # If that fails, try XGBoost specific loading (if XGBoost is installed)
        try:
            import xgboost as xgb
            # Try loading as an XGBoost model
            return xgb.Booster(model_file=MODEL_PATH)
        except Exception as e2:
            raise Exception(f"Failed to load model with both methods. Original error: {e}, XGBoost error: {e2}")

# Load model with error handling
try:
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.info("Please ensure the model file is in the correct location")
        st.stop()
    
    # Try to load the model
    model = load_model()
    st.success(f"✅ Successfully loaded {MODEL_PATH}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.text(traceback.format_exc())  # Show full traceback
    st.info("This could be due to version incompatibility or model format issues")
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
    display_results(model)
