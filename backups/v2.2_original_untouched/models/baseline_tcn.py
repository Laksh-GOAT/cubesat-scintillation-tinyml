import torch
import torch.nn as nn

class BaselineTCN(nn.Module):
    def __init__(self, input_channel=3, output_channel=3):
        super(BaselineTCN, self).__init__()

        # Lightweight 3-channel TCN for:
        # Ch 0: Signal Recovery | Ch 1: Cycle Slip Detection | Ch 2: S4 Severity Regression
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=output_channel, kernel_size=1)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    model = BaselineTCN()
    dummy_input = torch.randn(32, 3, 1000)
    output = model(dummy_input)
    print("TCN Multi-Task Model initialized successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")