import glob
import numpy as np
import torch
import torch.nn as nn
from models.baseline_tcn import BaselineTCN

def main():
    print("=== V2.2 READ-ONLY DIAGNOSTIC REPORT ===")
    print("NOTE: All batches (batch_0 to batch_9) were used in train.py.")
    print("      NO independent unseen test dataset exists in V2.2.\n")

    batch_files = sorted(glob.glob("data/processed/v2.2_space_weather_telemetry/batch_*.npz"))
    print(f"Found {len(batch_files)} batch files.")

    X_all, y_all = [], []
    for f in batch_files:
        d = np.load(f)
        X_all.append(d["X"])
        y_all.append(d["y"])
    X = torch.tensor(np.concatenate(X_all, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y_all, axis=0), dtype=torch.float32)

    print(f"Full Dataset Input Shape (X):  {tuple(X.shape)}")
    print(f"Full Dataset Target Shape (y): {tuple(y.shape)}\n")

    names = [
        "Task 1 (Ch 0: Clean Carrier Envelope)",
        "Task 2 (Ch 1: Cycle Slip Binary Mask)",
        "Task 3 (Ch 2: S4 Scintillation Index)"
    ]
    print("--- TARGET RANGES ACROSS ALL 10,000 TRAINING SAMPLES ---")
    for c in range(3):
        t_min = y[:, c, :].min().item()
        t_max = y[:, c, :].max().item()
        t_mean = y[:, c, :].mean().item()
        t_std = y[:, c, :].std().item()
        print(f"{names[c]}: min={t_min:.4f}, max={t_max:.4f}, mean={t_mean:.4f}, std={t_std:.4f}")

    model = BaselineTCN(input_channel=3, output_channel=3)
    model.load_state_dict(torch.load("models/baseline_tcn.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds_all = model(X)
        preds_b0 = preds_all[:100]

    print("\n--- FP32 PREDICTION ERRORS (ON ALL 10,000 TRAINING SAMPLES) ---")
    for c in range(3):
        mse = nn.MSELoss()(preds_all[:, c, :], y[:, c, :]).item()
        mae = torch.mean(torch.abs(preds_all[:, c, :] - y[:, c, :])).item()
        print(f"{names[c]}: MSE = {mse:.6f} | MAE = {mae:.6f}")

    print(f"Combined 3-Task MSE (All 10,000): {nn.MSELoss()(preds_all, y).item():.6f}")
    print(f"Combined 3-Task MSE (First 100 of batch_0): {nn.MSELoss()(preds_b0, y[:100]).item():.6f}")

    print("\n--- SAMPLE #0 CHECK (FP32 MODEL) ---")
    print(f"Sample #0 True S4:      {y[0, 2, :].mean().item():.4f}")
    print(f"Sample #0 FP32 Pred S4: {preds_all[0, 2, :].mean().item():.4f}")

if __name__ == "__main__":
    main()
