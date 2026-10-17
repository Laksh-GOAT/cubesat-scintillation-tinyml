import os
import sys
import time

# Automatically add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from models.baseline_tcn import BaselineTCN

def profile_edge_performance():
    print("⏱️ Profiling inference speed and memory footprint...")
    
    # Load validation sample
    X_val = np.load("data/processed/X_val.npy")
    sample_input = torch.tensor(X_val[:1], dtype=torch.float32) # Single batch sample
    
    # Load Quantized Model
    quant_model = BaselineTCN()
    quant_model = torch.quantization.quantize_dynamic(
        quant_model, {torch.nn.Conv1d}, dtype=torch.qint8
    )
    quant_model.load_state_dict(torch.load("models/quantized_tcn.pth", map_location="cpu"))
    quant_model.eval()
    
    # Warm-up run
    with torch.no_grad():
        _ = quant_model(sample_input)
        
    # Measure execution time across 100 runs
    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = quant_model(sample_input)
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
    model_size_kb = os.path.getsize("models/quantized_tcn.pth") / 1024
    
    print("\n================ EDGE HARDWARE PROFILE ================")
    print(f"📦 Quantized Model Memory Footprint: {model_size_kb:.2f} KB")
    print(f"⚡ Average Inference Latency:        {avg_latency_ms:.2f} ms")
    print(f"🛰️ Real-Time Tracking Feasibility:   {'PASS (< 50ms window)' if avg_latency_ms < 50 else 'REVIEW'}")
    print("========================================================")
    print("✅ Edge profiling complete!")

if __name__ == "__main__":
    profile_edge_profile = profile_edge_performance()