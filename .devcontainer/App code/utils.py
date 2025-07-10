import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch

# Set global device
device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

def load_flight_data(file_path):
    """
    Load flight data from a CSV file or create mock data if file doesn't exist
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with flight data
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        # Create mock data for demonstration
        print(f"File {file_path} not found. Creating mock data.")
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
        times = ['08:00', '12:30', '16:45', '19:20']
        origins = ['NYC', 'LAX', 'CHI', 'MIA', 'SFO']
        destinations = ['LON', 'PAR', 'TOK', 'SYD', 'BER']
        
        mock_data = []
        for i in range(100):
            date = np.random.choice(dates)
            time = np.random.choice(times)
            origin = np.random.choice(origins)
            dest = np.random.choice(destinations)
            
            # Base price with some randomness
            price = np.random.normal(500, 100)
            
            # Add price history (last 5 days)
            price_history = []
            current_price = price
            for _ in range(5):
                current_price = current_price * np.random.uniform(0.95, 1.05)
                price_history.append(round(current_price, 2))
            
            # Most recent price is the current price
            price = round(price_history[-1], 2)
            
            # Create row
            row = {
                'date': date,
                'time': time,
                'origin': origin,
                'destination': dest,
                'price': price
            }
            
            # Add price history columns
            for j, hist_price in enumerate(price_history):
                row[f'price_t-{5-j}'] = hist_price
                
            mock_data.append(row)
        
        df = pd.DataFrame(mock_data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        # Save to file
        df.to_csv(file_path, index=False)
        return df

def preprocess_flight_data(df):
    """
    Preprocess flight data for the model
    
    Args:
        df: pandas DataFrame with flight data
        
    Returns:
        Preprocessed pandas DataFrame
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert date strings to datetime objects
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    
    # Convert time to minutes since midnight
    df_processed['time_minutes'] = df_processed['time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    
    # One-hot encode categorical variables
    df_processed = pd.get_dummies(df_processed, columns=['origin', 'destination'])
    
    # Extract day of week, day of month, month
    df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
    df_processed['day_of_month'] = df_processed['date'].dt.day
    df_processed['month'] = df_processed['date'].dt.month
    
    # Normalize numerical features
    for col in ['day_of_week', 'day_of_month', 'month', 'time_minutes', 'price']:
        if col in df_processed.columns:
            df_processed[col] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
    
    # Normalize price history columns
    price_cols = [col for col in df_processed.columns if col.startswith('price_t-')]
    for col in price_cols:
        if col in df_processed.columns:
            # Use same normalization as price to keep consistent scale
            df_processed[col] = (df_processed[col] - df_processed['price'].mean()) / df_processed['price'].std()
    
    return df_processed

def prepare_training_data(df):
    """
    Prepare data for training the model
    
    Args:
        df: Preprocessed pandas DataFrame
        
    Returns:
        X, y as torch tensors
    """
    # Create feature matrix X and target vector y
    X = []
    y = []
    
    # Find price history columns
    price_cols = [col for col in df.columns if col.startswith('price_t-')]
    price_cols.sort()  # Ensure correct order
    
    # For each flight, create a sequence of price history
    for _, row in df.iterrows():
        price_history = [row[col] for col in price_cols]
        
        # Add features
        features = []
        for price in price_history:
            # Day of week, day of month, month, time_minutes, price
            feature = [
                row['day_of_week'], 
                row['day_of_month'], 
                row['month'], 
                row['time_minutes'], 
                price
            ]
            features.append(feature)
        
        X.append(features)
        y.append(row['price'])
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    
    return X, y

def predict_future_prices(model, current_data, days=7):
    """
    Predict flight prices for the next 'days' days
    
    Args:
        model: Trained model
        current_data: Current flight data (processed)
        days: Number of days to predict
        
    Returns:
        List of predicted prices for the next 'days' days
    """
    model.eval()
    with torch.no_grad():
        # Extract current features
        features = current_data.copy()
        
        # Predict prices for next 'days' days
        predicted_prices = []
        
        for _ in range(days):
            # Create input sequence (last 5 prices)
            input_seq = []
            for i in range(5):
                # Day of week, day of month, month, time_minutes, price
                feature = [
                    features['day_of_week'],
                    features['day_of_month'],
                    features['month'],
                    features['time_minutes'],
                    features[f'price_t-{i+1}']
                ]
                input_seq.append(feature)
            
            # Convert to tensor
            input_tensor = torch.tensor([input_seq], dtype=torch.float32).to(device)
            
            # Predict next price
            next_price = model(input_tensor).item()
            predicted_prices.append(next_price)
            
            # Shift price history
            for i in range(1, 5):
                features[f'price_t-{i}'] = features[f'price_t-{i+1}']
            features[f'price_t-5'] = next_price
            
            # Update date features (simplistic approach)
            features['day_of_week'] = (features['day_of_week'] + 1) % 7
            
            # Update day of month (simplified)
            features['day_of_month'] += 1
            if features['day_of_month'] > 28:  # Simplified
                features['day_of_month'] = 1
                features['month'] += 1
                if features['month'] > 12:
                    features['month'] = 1
        
        return predicted_prices
