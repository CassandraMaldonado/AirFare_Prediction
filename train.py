import os
import torch
import torch.nn as nn
from models import FlightPricePredictor
from utils import load_flight_data, preprocess_flight_data, prepare_training_data

def train_model(data_path="flight_data.csv", model_dir="model", epochs=50, batch_size=32):
    """
    Train the flight price prediction model and save it to disk
    
    Args:
        data_path: Path to the flight data CSV
        model_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model
    """
    # Set up device
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading flight data...")
    flight_data = load_flight_data(data_path)
    
    print("Preprocessing data...")
    processed_data = preprocess_flight_data(flight_data)
    
    # Prepare training data
    X, y = prepare_training_data(processed_data)
    
    # Create model
    model = FlightPricePredictor().to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, "flight_price_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train_model()
