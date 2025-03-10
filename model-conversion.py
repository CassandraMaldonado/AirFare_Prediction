"""
Script to convert the XGBoost model to different formats.
Run this if you have issues loading the model.
"""
import os
import pickle
import sys

def convert_model():
    """Convert model between different serialization formats."""
    try:
        import joblib
        have_joblib = True
    except ImportError:
        print("Warning: joblib not installed. Limited conversion options available.")
        print("Install joblib with: pip install joblib")
        have_joblib = False
    
    # Define file paths
    input_model_path = 'model/xg_boost_model_compressed.pkl'
    output_pickle_path = 'model/xg_boost_model_pickle.pkl'
    output_joblib_path = 'model/xg_boost_model_joblib.pkl' if have_joblib else None
    
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    print(f"Attempting to load model from: {input_model_path}")
    
    # Try different loading methods
    model = None
    error_messages = []
    
    # Method 1: Standard pickle with binary mode
    try:
        with open(input_model_path, 'rb') as f:
            model = pickle.load(f)
        print("Successfully loaded model with pickle (binary mode)")
    except Exception as e:
        error_message = f"Failed to load with pickle (binary mode): {str(e)}"
        error_messages.append(error_message)
        print(error_message)
    
    # Method 2: If joblib is available, try that
    if have_joblib and model is None:
        try:
            model = joblib.load(input_model_path)
            print("Successfully loaded model with joblib")
        except Exception as e:
            error_message = f"Failed to load with joblib: {str(e)}"
            error_messages.append(error_message)
            print(error_message)
    
    # If we couldn't load the model
    if model is None:
        print("\nFailed to load the model using any method.")
        print("Error details:")
        for i, msg in enumerate(error_messages, 1):
            print(f"{i}. {msg}")
        print("\nPlease ensure the model file exists and is in the correct format.")
        return False
    
    # Save the model in different formats
    print("\nSaving model in different formats:")
    
    # Save with pickle
    try:
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(model, f, protocol=4)  # Protocol 4 is compatible with Python 3.4+
        print(f"Saved model with pickle to: {output_pickle_path}")
    except Exception as e:
        print(f"Failed to save with pickle: {str(e)}")
    
    # Save with joblib if available
    if have_joblib:
        try:
            joblib.dump(model, output_joblib_path)
            print(f"Saved model with joblib to: {output_joblib_path}")
        except Exception as e:
            print(f"Failed to save with joblib: {str(e)}")
    
    print("\nConversion complete!")
    print("Update your code to use one of the newly created model files.")
    return True

if __name__ == "__main__":
    print("Model Conversion Utility")
    print("========================")
    
    success = convert_model()
    
    if success:
        print("\nRecommended code for loading the model:")
        print("\nOption 1 (pickle):")
        print("```python")
        print("import pickle")
        print("with open('model/xg_boost_model_pickle.pkl', 'rb') as f:")
        print("    model = pickle.load(f)")
        print("```")
        
        print("\nOption 2 (joblib):")
        print("```python")
        print("import joblib")
        print("model = joblib.load('model/xg_boost_model_joblib.pkl')")
        print("```")
    else:
        sys.exit(1)
