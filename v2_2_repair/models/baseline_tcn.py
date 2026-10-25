import torch
import torch.nn as nn

class BaselineTCN(nn.Module):
    def __init__(self, input_channel=3, output_channel=3):
        super(BaselineTCN, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv1d(input_channel, 24, kernel_size=5, dilation=1, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 32, kernel_size=5, dilation=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, dilation=4, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=8, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.head_envelope = nn.Conv1d(32, 1, kernel_size=1)

        self.head_slip_logits = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

        self.head_s4 = nn.Sequential(
            nn.Conv1d(96, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        out_env = self.head_envelope(feats)
        out_slip_logits = self.head_slip_logits(feats)

        feat_mean = feats.mean(dim=-1, keepdim=True).expand_as(feats)
        feat_std = feats.std(dim=-1, keepdim=True, unbiased=False).expand_as(feats)
        s4_in = torch.cat([feats, feat_mean, feat_std], dim=1)
        out_s4 = self.head_s4(s4_in)

        return torch.cat([out_env, out_slip_logits, out_s4], dim=1)

if __name__ == "__main__":
    model = BaselineTCN()
    params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(4, 3, 1000)
    out = model(dummy)
    assert out.shape == (4, 3, 1000)
    print(f"[PASS] Repaired BaselineTCN initialized: {params} parameters, output shape={tuple(out.shape)}")