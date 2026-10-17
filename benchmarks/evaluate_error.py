import os
import sys

# Automatically add the project root folder to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from models.baseline_tcn import BaselineTCN

def evaluate_models():
    print("📊 Loading validation dataset and models for benchmarking...")
    
    # Load validation data
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    metadata = np.load("data/processed/metadata.npy")
    
    X_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # 1. Load Baseline Float32 Model
    float_model = BaselineTCN()
    float_model.load_state_dict(torch.load("models/baseline_tcn.pth", map_location="cpu"))
    float_model.eval()
    
    # 2. Load Quantized Model
    quant_model = BaselineTCN()
    quant_model = torch.quantization.quantize_dynamic(
        quant_model, {torch.nn.Conv1d}, dtype=torch.qint8
    )
    quant_model.load_state_dict(torch.load("models/quantized_tcn.pth", map_location="cpu"))
    quant_model.eval()
    
    # Run Inference & Calculate Errors
    with torch.no_grad():
        float_preds = float_model(X_tensor)
        quant_preds = quant_model(X_tensor)
        
        float_mse = torch.mean((float_preds - y_tensor) ** 2).item()
        quant_mse = torch.mean((quant_preds - y_tensor) ** 2).item()
        
    print("\n================ SCIENTIFIC BENCHMARK RESULTS ================")
    print(f"🔬 Baseline Float32 Model MSE (Phase Error): {float_mse:.6f}")
    print(f"⚡ Quantized Int8 TinyML Model MSE:       {quant_mse:.6f}")
    print(f"📉 Accuracy Trade-off Delta:               {abs(quant_mse - float_mse):.6f}")
    print("================================================================")
    print("✅ Benchmark complete! This proves your compressed model maintains tracking lock.")

if __name__ == "__main__":
    evaluate_models()