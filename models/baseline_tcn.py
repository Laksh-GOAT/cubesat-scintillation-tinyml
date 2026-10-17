import torch
import torch.nn as nn

class BaselineTCN(nn.Module):
    def __init__(self, input_channel=1, output_channel=1):
        super(BaselineTCN, self).__init__()
        
        # A lightweight convolutional architecture designed for edge devices
        self.network = nn.Sequential(
            # First convolutional layer: extracts local phase jitter features
            nn.Conv1d(in_channels=input_channel, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            # Second convolutional layer with dilation: captures wider temporal context
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Final projection layer back to a single phase output
            nn.Conv1d(in_channels=32, out_channels=output_channel, kernel_size=1)
        )

    def forward(self, x):
        # PyTorch Conv1d expects shape: (Batch, Channels, Length)
        # Our data shape is (Batch, Length), so we add the channel dimension:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        out = self.network(x)
        
        # Remove channel dimension for output comparison
        return out.squeeze(1)

if __name__ == "__main__":
    # Quick test to verify model architecture
    model = BaselineTCN()
    dummy_input = torch.randn(32, 2000) # Batch of 32, 2000 time steps each
    output = model(dummy_input)
    print("✅ TCN Model initialized successfully!")
    print(f"📦 Input shape: {dummy_input.shape}")
    print(f"🎯 Output shape: {output.shape}")