import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from baseline_tcn import BaselineTCN

def train_model():
    # Check if Apple Silicon GPU (MPS) acceleration is available on your Mac Mini M4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")
    
    # Load dataset generated from your simulation
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    
    # Convert to PyTorch tensors and create data loaders
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = BaselineTCN().to(device)
    criterion = nn.MSELoss() # Mean Squared Error for phase tracking accuracy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("🧠 Training neural network on simulated ionospheric data...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.6f}")
        
    # Save the trained baseline model weights
    torch.save(model.state_dict(), "models/baseline_tcn.pth")
    print("✅ Model training complete and saved to `models/baseline_tcn.pth`!")

if __name__ == "__main__":
    train_model()