import os
import glob
import torch
import torch.nn as nn
import numpy as np

try:
    from models.baseline_tcn import BaselineTCN
except ImportError:
    from baseline_tcn import BaselineTCN

def evaluate_quantization():
    print("Loading trained FP32 Multi-Task model...")
    model_fp32 = BaselineTCN(input_channel=3, output_channel=3)
    model_path = "models/baseline_tcn.pth"
    model_fp32.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_fp32.eval()

    fp32_size = os.path.getsize(model_path) / 1024
    print(f"-> Original FP32 Model Footprint: {fp32_size:.2f} KB")

    print("\nApplying Post-Training Quantization (PTQ) to INT8...")
    quantized_state_dict = {}
    scales = {}

    for name, param in model_fp32.named_parameters():
        max_val = torch.max(torch.abs(param.data)).item()
        scale = max_val / 127.0 if max_val > 0 else 1.0
        quantized_tensor = torch.clamp(torch.round(param.data / scale), -128, 127).to(torch.int8)
        quantized_state_dict[name] = quantized_tensor
        scales[name] = scale

    quantized_path = "models/quantized_tcn.pth"
    torch.save({"quantized_weights": quantized_state_dict, "scales": scales}, quantized_path)
    int8_size = os.path.getsize(quantized_path) / 1024
    print(f"-> Quantized INT8 Model Footprint: {int8_size:.2f} KB")
    print(f"-> Compression Ratio: {fp32_size / int8_size:.2f}x reduction")

    print("\nMeasuring Quantization Delta Error & Space Weather Telemetry on Test Data...")
    batch_files = sorted(glob.glob("data/processed/v2.2_space_weather_telemetry/batch_*.npz"))
    if not batch_files:
        print("No batch files found for evaluation.")
        return

    data = np.load(batch_files[0])
    X_test = torch.tensor(data['X'][:100], dtype=torch.float32)
    y_test = torch.tensor(data['y'][:100], dtype=torch.float32)

    criterion = nn.MSELoss()

    with torch.no_grad():
        out_fp32 = model_fp32(X_test)
        loss_fp32 = criterion(out_fp32, y_test).item()

    model_int8_sim = BaselineTCN(input_channel=3, output_channel=3)
    int8_state_dict = model_fp32.state_dict().copy()
    for name, param in model_fp32.named_parameters():
        if name in quantized_state_dict:
            int8_state_dict[name] = quantized_state_dict[name].float() * scales[name]
    model_int8_sim.load_state_dict(int8_state_dict)
    model_int8_sim.eval()

    with torch.no_grad():
        out_int8 = model_int8_sim(X_test)
        loss_int8 = criterion(out_int8, y_test).item()

    delta_error = loss_int8 - loss_fp32
    print(f"-> FP32 Test MSE: {loss_fp32:.6f}")
    print(f"-> INT8 Quantized Test MSE: {loss_int8:.6f}")
    print(f"-> Quantization Delta Error: {delta_error:+.6f}")

    # Sample Telemetry Check on Signal #0
    pred_s4 = out_int8[0, 2, :].mean().item()
    true_s4 = y_test[0, 2, :].mean().item()
    slip_detected = (out_int8[0, 1, :].max().item() > 0.4)
    storm_alert = "CRITICAL SOLAR STORM (S4 >= 0.6) - SAVING HIGH-RATE LOGS" if pred_s4 >= 0.6 else "NOMINAL IONOSPHERE"

    print("\n--- ONBOARD CUBESAT TELEMETRY SAMPLE #0 ---")
    print(f"-> True S4 Severity Index:      {true_s4:.3f}")
    print(f"-> AI Predicted S4 Index:       {pred_s4:.3f}")
    print(f"-> Cycle Slip Flagged:          {slip_detected}")
    print(f"-> Autonomous Telemetry Alert:  [{storm_alert}]")

if __name__ == "__main__":
    evaluate_quantization()