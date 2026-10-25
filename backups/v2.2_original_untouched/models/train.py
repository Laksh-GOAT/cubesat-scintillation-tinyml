import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

try:
    from models.baseline_tcn import BaselineTCN
except ImportError:
    from baseline_tcn import BaselineTCN

def load_batch_dataset(data_dir="data/processed/v2.2_space_weather_telemetry"):
    batch_files = sorted(glob.glob(os.path.join(data_dir, "batch_*.npz")))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {data_dir}. Run generate_dataset.py first!")

    X_list, y_list = [], []
    for file in batch_files:
        data = np.load(file)
        X_list.append(data['X'])
        y_list.append(data['y'])

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    print(f"Loaded {X_all.shape[0]} total signals from {len(batch_files)} batches.")
    return torch.tensor(X_all, dtype=torch.float32), torch.tensor(y_all, dtype=torch.float32)

def train_model():
    X, y = load_batch_dataset()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = BaselineTCN(input_channel=3, output_channel=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    print("Starting Multi-Task training (Signal Recovery + Cycle Slip + S4 Telemetry)...")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        total_loss = epoch_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Multi-Task Loss (MSE): {total_loss:.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_tcn.pth")
    print("Trained model weights saved to models/baseline_tcn.pth!")

if __name__ == "__main__":
    train_model()