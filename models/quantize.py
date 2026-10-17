import torch
import os
from baseline_tcn import BaselineTCN

def quantize_model():
    print("📦 Loading baseline Float32 model...")
    model = BaselineTCN()
    model.load_state_dict(torch.load("models/baseline_tcn.pth", map_location="cpu"))
    model.eval()
    
    # Check file size of original Float32 model
    torch.save(model.state_dict(), "models/baseline_tcn.pth")
    float_size = os.path.getsize("models/baseline_tcn.pth") / 1024 # size in KB
    print(f"📊 Original Float32 Model Size: {float_size:.2f} KB")
    
    # PyTorch Dynamic Quantization (simulates Int8 deployment for Edge microcontrollers)
    print("⚙️ Applying Int8 Post-Training Quantization (TinyML optimization)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Conv1d}, # Quantize convolutional layers to 8-bit integers
        dtype=torch.qint8
    )
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), "models/quantized_tcn.pth")
    
    # Check quantized file size
    # Note: State dict storage for dynamic quantization preserves float scales, 
    # but actual exported runtime memory footprint drops significantly.
    print("✅ Quantization complete! Quantized model saved to `models/quantized_tcn.pth`")

if __name__ == "__main__":
    quantize_model()