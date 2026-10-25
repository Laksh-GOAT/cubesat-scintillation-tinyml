import os
import glob
import json
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from v2_2_repair.models.baseline_tcn import BaselineTCN
except ImportError:
    from baseline_tcn import BaselineTCN


# ==============================================================================
# 1. BATCHNORM FOLDING & STRUCTURAL VERIFICATION
# ==============================================================================
def fold_conv_bn_pair(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    """
    Mathematically folds a trained BatchNorm1d module into its preceding Conv1d module.
    For input x, Conv1d output is z = W * x + b_conv.
    BatchNorm1d output is y = gamma * (z - mean) / sqrt(var + eps) + beta
                            = (gamma / sqrt(var + eps)) * W * x + [gamma * (b_conv - mean) / sqrt(var + eps) + beta].
    """
    if not isinstance(conv, nn.Conv1d):
        raise TypeError(f"Expected nn.Conv1d, got {type(conv)}")
    if not isinstance(bn, nn.BatchNorm1d):
        raise TypeError(f"Expected nn.BatchNorm1d, got {type(bn)}")
    if conv.out_channels != bn.num_features:
        raise ValueError(f"Channel mismatch: conv.out_channels={conv.out_channels} vs bn.num_features={bn.num_features}")

    device = conv.weight.device
    dtype = torch.float32

    w_conv = conv.weight.detach().to(device=device, dtype=dtype)
    if conv.bias is not None:
        b_conv = conv.bias.detach().to(device=device, dtype=dtype)
    else:
        b_conv = torch.zeros(conv.out_channels, device=device, dtype=dtype)

    rm = bn.running_mean.detach().to(device=device, dtype=dtype)
    rv = bn.running_var.detach().to(device=device, dtype=dtype)
    eps = float(bn.eps)

    if bn.affine and bn.weight is not None and bn.bias is not None:
        gamma = bn.weight.detach().to(device=device, dtype=dtype)
        beta = bn.bias.detach().to(device=device, dtype=dtype)
    else:
        gamma = torch.ones(conv.out_channels, device=device, dtype=dtype)
        beta = torch.zeros(conv.out_channels, device=device, dtype=dtype)

    inv_std = torch.rsqrt(rv + eps)
    factor = gamma * inv_std  # shape: (C_out,)

    w_folded = w_conv * factor.view(-1, 1, 1)
    b_folded = (b_conv - rm) * factor + beta
    return w_folded, b_folded


def extract_folded_layers(model: BaselineTCN):
    """
    Inspects the live BaselineTCN instance, verifies layer ordering and adjacencies,
    and returns an ordered dictionary of folded FP32 (weight, bias, config) tuples.
    """
    model.eval()
    backbone_blocks = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
    layers = {}

    for block_idx, (c_i, bn_i, act_i) in enumerate(backbone_blocks):
        conv_mod = model.backbone[c_i]
        bn_mod = model.backbone[bn_i]
        act_mod = model.backbone[act_i]
        if not isinstance(act_mod, nn.ReLU):
            raise TypeError(f"Expected ReLU at backbone[{act_i}], found {type(act_mod)}")

        w_f, b_f = fold_conv_bn_pair(conv_mod, bn_mod)
        layers[f"bb_{block_idx}"] = {
            "weight_fp32": w_f,
            "bias_fp32": b_f,
            "in_channels": int(conv_mod.in_channels),
            "out_channels": int(conv_mod.out_channels),
            "kernel_size": int(conv_mod.kernel_size[0]),
            "dilation": int(conv_mod.dilation[0]),
            "padding": int(conv_mod.padding[0]),
            "relu": True,
        }

    # Direct Conv1d heads (no BatchNorm present)
    head_specs = [
        ("env_0", model.head_envelope, False),
        ("slip_0", model.head_slip_logits[0], True),
        ("slip_2", model.head_slip_logits[2], False),
        ("s4_0", model.head_s4[0], True),
        ("s4_2", model.head_s4[2], False),
    ]
    for name, conv_mod, use_relu in head_specs:
        if not isinstance(conv_mod, nn.Conv1d):
            raise TypeError(f"Expected nn.Conv1d for {name}, got {type(conv_mod)}")
        w = conv_mod.weight.detach().to(torch.float32).clone()
        b = (
            conv_mod.bias.detach().to(torch.float32).clone()
            if conv_mod.bias is not None
            else torch.zeros(conv_mod.out_channels, dtype=torch.float32)
        )
        layers[name] = {
            "weight_fp32": w,
            "bias_fp32": b,
            "in_channels": int(conv_mod.in_channels),
            "out_channels": int(conv_mod.out_channels),
            "kernel_size": int(conv_mod.kernel_size[0]),
            "dilation": int(conv_mod.dilation[0]),
            "padding": int(conv_mod.padding[0]),
            "relu": bool(use_relu),
        }

    return layers


# ==============================================================================
# 2. SYMMETRIC PER-OUTPUT-CHANNEL INT8 WEIGHT QUANTIZATION
# ==============================================================================
def quantize_weight_per_channel(weight_fp32: torch.Tensor):
    """
    Symmetric per-output-channel INT8 quantization for Conv1d weight tensor (C_out, C_in, K).
      scale[c] = max_{i,k} |W[c, i, k]| / 127.0  (or 1.0 if max == 0)
      q_weight[c, i, k] = clip(round(W[c, i, k] / scale[c]), -127, 127)
    Returns:
      q_weight: torch.int8 tensor of shape (C_out, C_in, K)
      scales:   torch.float32 tensor of shape (C_out,)
    """
    if weight_fp32.ndim != 3:
        raise ValueError(f"Expected 3D Conv1d weight tensor, got shape {tuple(weight_fp32.shape)}")

    max_abs = torch.amax(torch.abs(weight_fp32), dim=(1, 2))  # shape: (C_out,)
    scales = torch.where(max_abs > 0.0, max_abs / 127.0, torch.ones_like(max_abs)).to(torch.float32)
    scaled = weight_fp32 / scales.view(-1, 1, 1)
    q_weight = torch.clamp(torch.round(scaled), -127, 127).to(torch.int8)
    return q_weight, scales


# ==============================================================================
# 3. RUNNABLE INFERENCE MODULES (FOLDED FP32, W8A32 INT8-WEIGHT, W8A8 FULL-INT8)
# ==============================================================================
class QuantizedTCNInference(nn.Module):
    """
    Executes the BatchNorm-folded BaselineTCN in one of three explicit modes:
      - mode="folded_fp32": Uses exact folded FP32 weights and biases.
      - mode="w8a32":       Uses INT8 per-channel weights (dequantized during MAC) and FP32 activations.
                            (Matches the exported C++ inference engine).
      - mode="w8a8":        Quantizes both activations (using validation-calibrated scales) and
                            weights to INT8, accumulates in INT32, and rescales to FP32.
    """
    def __init__(self, layer_dict, mode="w8a32", act_scales=None):
        super().__init__()
        if mode not in ("folded_fp32", "w8a32", "w8a8"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.layers = layer_dict
        self.mode = mode
        self.act_scales = act_scales or {}

    def _conv1d_op(self, x: torch.Tensor, name: str) -> torch.Tensor:
        spec = self.layers[name]
        b = spec["bias_fp32"]
        d = spec["dilation"]
        p = spec["padding"]

        if self.mode == "folded_fp32":
            w = spec["weight_fp32"]
            out = F.conv1d(x, w, b, stride=1, padding=p, dilation=d)
        elif self.mode == "w8a32":
            q_w = spec["weight_int8"].to(torch.float32)
            s_w = spec["scale_fp32"].view(-1, 1, 1)
            w_deq = q_w * s_w
            out = F.conv1d(x, w_deq, b, stride=1, padding=p, dilation=d)
        elif self.mode == "w8a8":
            # Symmetric INT8 activation quantization using validation-calibrated scale
            s_x = float(self.act_scales[name])
            q_x = torch.clamp(torch.round(x / s_x), -127, 127)
            q_w = spec["weight_int8"].to(torch.float32)
            # Integer MAC accumulated in exact float32/int32 (max accum < 2^24, exact in float32)
            int32_acc = F.conv1d(q_x, q_w, bias=None, stride=1, padding=p, dilation=d)
            s_w = spec["scale_fp32"].view(1, -1, 1)
            out = int32_acc * (s_x * s_w) + b.view(1, -1, 1)

        if spec["relu"]:
            out = F.relu(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._conv1d_op(x, "bb_0")
        h = self._conv1d_op(h, "bb_1")
        h = self._conv1d_op(h, "bb_2")
        feats = self._conv1d_op(h, "bb_3")

        out_env = self._conv1d_op(feats, "env_0")

        s = self._conv1d_op(feats, "slip_0")
        out_slip = self._conv1d_op(s, "slip_2")

        feat_mean = feats.mean(dim=-1, keepdim=True).expand_as(feats)
        feat_std = feats.std(dim=-1, keepdim=True, unbiased=False).expand_as(feats)
        s4_in = torch.cat([feats, feat_mean, feat_std], dim=1)

        u = self._conv1d_op(s4_in, "s4_0")
        out_s4 = self._conv1d_op(u, "s4_2")

        return torch.cat([out_env, out_slip, out_s4], dim=1)


def calibrate_w8a8_activation_scales(layer_dict, X_val_calib: torch.Tensor):
    """
    Calibrates per-layer input activation scales strictly on a subset of the
    VALIDATION split (never the test split) for W8A8 comparison.
    """
    runner = QuantizedTCNInference(layer_dict, mode="w8a32")
    runner.eval()
    act_scales = {}

    def _scale_of(tensor: torch.Tensor) -> float:
        m = float(torch.max(torch.abs(tensor)).item())
        return m / 127.0 if m > 0.0 else 1.0

    with torch.no_grad():
        act_scales["bb_0"] = _scale_of(X_val_calib)
        h0 = runner._conv1d_op(X_val_calib, "bb_0")
        act_scales["bb_1"] = _scale_of(h0)
        h1 = runner._conv1d_op(h0, "bb_1")
        act_scales["bb_2"] = _scale_of(h1)
        h2 = runner._conv1d_op(h1, "bb_2")
        act_scales["bb_3"] = _scale_of(h2)
        feats = runner._conv1d_op(h2, "bb_3")

        act_scales["env_0"] = _scale_of(feats)
        act_scales["slip_0"] = _scale_of(feats)
        s0 = runner._conv1d_op(feats, "slip_0")
        act_scales["slip_2"] = _scale_of(s0)

        feat_mean = feats.mean(dim=-1, keepdim=True).expand_as(feats)
        feat_std = feats.std(dim=-1, keepdim=True, unbiased=False).expand_as(feats)
        s4_in = torch.cat([feats, feat_mean, feat_std], dim=1)
        act_scales["s4_0"] = _scale_of(s4_in)
        u0 = runner._conv1d_op(s4_in, "s4_0")
        act_scales["s4_2"] = _scale_of(u0)

    return act_scales


# ==============================================================================
# 4. DATASET LOADING & TASK-LEVEL METRIC EVALUATION
# ==============================================================================
def load_split(base_dir: str, split_name: str):
    pattern = os.path.join(base_dir, split_name, "batch_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Missing dataset split files at: {pattern}")
    X_list, y_list = [], []
    for f in files:
        d = np.load(f)
        X_list.append(d["X"])
        y_list.append(d["y"])
    X = torch.tensor(np.concatenate(X_list, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.float32)
    return X, y


def compute_classification_metrics(y_true_flat: np.ndarray, y_pred_flat: np.ndarray):
    tp = int(np.sum((y_true_flat == 1) & (y_pred_flat == 1)))
    fp = int(np.sum((y_true_flat == 0) & (y_pred_flat == 1)))
    tn = int(np.sum((y_true_flat == 0) & (y_pred_flat == 0)))
    fn = int(np.sum((y_true_flat == 1) & (y_pred_flat == 0)))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_predictions(preds: np.ndarray, targets: np.ndarray, slip_threshold: float = 0.82):
    # Task 1: Envelope Reconstruction
    t1_diff = preds[:, 0, :] - targets[:, 0, :]
    t1_mse = float(np.mean(t1_diff ** 2))
    t1_mae = float(np.mean(np.abs(t1_diff)))

    # Task 2: Cycle-Slip Classification at validation-selected probability threshold (0.82)
    probs = 1.0 / (1.0 + np.exp(-np.clip(preds[:, 1, :], -20.0, 20.0)))
    y_true_cls = (targets[:, 1, :] > 0.5).astype(np.int32).ravel()
    y_pred_cls = (probs >= slip_threshold).astype(np.int32).ravel()
    t2_metrics = compute_classification_metrics(y_true_cls, y_pred_cls)

    # Task 3: S4 Regression (window-level mean prediction vs true S4)
    s4_pred = np.mean(preds[:, 2, :], axis=1)
    s4_true = targets[:, 2, 0]
    t3_mae = float(np.mean(np.abs(s4_pred - s4_true)))
    t3_rmse = float(np.sqrt(np.mean((s4_pred - s4_true) ** 2)))

    severe_mask = s4_true >= 0.60
    t3_sev_mae = float(np.mean(np.abs(s4_pred[severe_mask] - s4_true[severe_mask]))) if np.any(severe_mask) else 0.0
    t3_sev_rmse = float(np.sqrt(np.mean((s4_pred[severe_mask] - s4_true[severe_mask]) ** 2))) if np.any(severe_mask) else 0.0

    return {
        "task1_envelope": {"MSE": t1_mse, "MAE": t1_mae},
        "task2_cycle_slip": t2_metrics,
        "task3_s4": {
            "MAE": t3_mae,
            "RMSE": t3_rmse,
            "severe_S4_ge_0.6_MAE": t3_sev_mae,
            "severe_S4_ge_0.6_RMSE": t3_sev_rmse,
        },
    }


def run_batched_inference(model: nn.Module, X: torch.Tensor, batch_size: int = 128) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i : i + batch_size]
            outputs.append(model(batch_x).cpu().numpy())
    return np.concatenate(outputs, axis=0)


# ==============================================================================
# 5. COMPACT RAW BINARY SERIALIZATION & HONEST STORAGE ACCOUNTING
# ==============================================================================
def save_raw_binary_model(layer_dict, raw_bin_path: str):
    """
    Writes a zero-overhead binary file containing:
      - 8-byte magic header ('TCNV22Q8')
      - For each of the 9 layers in deterministic order:
          * int8 weights (1 byte per weight)
          * float32 per-channel scales (4 bytes per output channel)
          * float32 folded biases (4 bytes per output channel)
    """
    layer_order = ["bb_0", "bb_1", "bb_2", "bb_3", "env_0", "slip_0", "slip_2", "s4_0", "s4_2"]
    with open(raw_bin_path, "wb") as f:
        f.write(b"TCNV22Q8")
        for name in layer_order:
            spec = layer_dict[name]
            w_bytes = np.ascontiguousarray(spec["weight_int8"].cpu().numpy(), dtype=np.int8).tobytes()
            s_bytes = np.ascontiguousarray(spec["scale_fp32"].cpu().numpy(), dtype=np.float32).tobytes()
            b_bytes = np.ascontiguousarray(spec["bias_fp32"].cpu().numpy(), dtype=np.float32).tobytes()
            f.write(w_bytes)
            f.write(s_bytes)
            f.write(b_bytes)


def main():
    ckpt_path = "v2_2_repair/models/best_baseline_tcn.pth"
    eval_report_path = "v2_2_repair/models/evaluation_report.json"
    data_dir = "v2_2_repair/data/processed/v2.2_repaired"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Trained checkpoint not found at {ckpt_path}")

    slip_threshold = 0.82
    if os.path.exists(eval_report_path):
        with open(eval_report_path, "r") as f:
            rep = json.load(f)
            slip_threshold = float(rep.get("selected_threshold_from_val", 0.82))

    print("======================================================================")
    print("STAGE 1: LOAD FP32 MODEL & VERIFY BATCHNORM FOLDING EQUIVALENCE")
    print("======================================================================")
    model_fp32 = BaselineTCN(input_channel=3, output_channel=3)
    model_fp32.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model_fp32.eval()

    layers = extract_folded_layers(model_fp32)
    for name, spec in layers.items():
        q_w, s_w = quantize_weight_per_channel(spec["weight_fp32"])
        spec["weight_int8"] = q_w
        spec["scale_fp32"] = s_w

    model_folded_fp32 = QuantizedTCNInference(layers, mode="folded_fp32")
    model_w8a32 = QuantizedTCNInference(layers, mode="w8a32")

    # Numerical Folding Equivalence Check on Validation Split
    X_val, y_val = load_split(data_dir, "val")
    X_test, y_test = load_split(data_dir, "test")

    with torch.no_grad():
        ref_out = model_fp32(X_val[:256])
        fold_out = model_folded_fp32(X_val[:256])
        w8a32_out = model_w8a32(X_val[:256])

    max_fold_diff = float(torch.max(torch.abs(ref_out - fold_out)).item())
    mean_fold_diff = float(torch.mean(torch.abs(ref_out - fold_out)).item())
    max_w8a32_diff = float(torch.max(torch.abs(ref_out - w8a32_out)).item())
    mean_w8a32_diff = float(torch.mean(torch.abs(ref_out - w8a32_out)).item())

    print(f"[CHECK] Unfolded FP32 vs Folded FP32 Max Abs Diff:  {max_fold_diff:.8e} (Mean: {mean_fold_diff:.8e})")
    if max_fold_diff >= 1e-4:
        raise RuntimeError(f"BatchNorm folding equivalence test FAILED: max_diff={max_fold_diff}")
    print("[PASS]  BatchNorm folding is mathematically equivalent to unfolded FP32.")
    print(f"[CHECK] Unfolded FP32 vs Per-Channel W8A32 Max Diff: {max_w8a32_diff:.6f} (Mean: {mean_w8a32_diff:.6f})")

    # Calibrate W8A8 activation scales on first 1,000 validation samples ONLY
    act_scales = calibrate_w8a8_activation_scales(layers, X_val[:1000])
    model_w8a8 = QuantizedTCNInference(layers, mode="w8a8", act_scales=act_scales)

    # Save artifacts
    pth_out_path = "v2_2_repair/models/quantized_folded_tcn.pth"
    raw_bin_path = "v2_2_repair/models/quantized_folded_tcn_raw.bin"
    export_payload = {
        "quantization_scheme": "symmetric_per_output_channel_int8_weights_fp32_bias",
        "slip_threshold": slip_threshold,
        "act_scales_w8a8": act_scales,
        "layers": {
            k: {
                "weight_int8": v["weight_int8"],
                "scale_fp32": v["scale_fp32"],
                "bias_fp32": v["bias_fp32"],
                "in_channels": v["in_channels"],
                "out_channels": v["out_channels"],
                "kernel_size": v["kernel_size"],
                "dilation": v["dilation"],
                "padding": v["padding"],
                "relu": v["relu"],
            }
            for k, v in layers.items()
        },
        # Also save a deterministic validation sample and its exact W8A32 output for C++ verification
        "verification_sample_x": X_test[0].cpu(),
        "verification_sample_y": y_test[0].cpu(),
        "verification_sample_w8a32_out": model_w8a32(X_test[:1])[0].detach().cpu(),
    }
    torch.save(export_payload, pth_out_path)
    save_raw_binary_model(layers, raw_bin_path)

    print("\n======================================================================")
    print("STAGE 2: HONEST STORAGE & RUNTIME MEMORY ACCOUNTING")
    print("======================================================================")
    learnable_params = sum(p.numel() for p in model_fp32.parameters())
    buffer_params = sum(b.numel() for b in model_fp32.buffers() if b.ndim > 0)
    raw_unfolded_fp32_bytes = (learnable_params + buffer_params) * 4

    total_int8_weights = sum(v["weight_int8"].numel() for v in layers.values())
    total_fp32_biases = sum(v["bias_fp32"].numel() for v in layers.values())
    total_fp32_scales = sum(v["scale_fp32"].numel() for v in layers.values())

    int8_weight_bytes = total_int8_weights * 1
    fp32_bias_bytes = total_fp32_biases * 4
    fp32_scale_bytes = total_fp32_scales * 4
    total_raw_quantized_bytes = int8_weight_bytes + fp32_bias_bytes + fp32_scale_bytes

    orig_pth_bytes = os.path.getsize(ckpt_path)
    quant_pth_bytes = os.path.getsize(pth_out_path)
    raw_bin_bytes = os.path.getsize(raw_bin_path)

    # Ping-pong working tensor buffers for C++ execution:
    # bufA (96 * L floats for s4_in concat) + bufB (32 * L floats) + out (3 * L floats) = 131 * L * 4 bytes
    sram_L1000_bytes = 131 * 1000 * 4
    sram_L128_bytes = 131 * 128 * 4

    print(f"Original FP32 .pth File Size (with ZIP overhead): {orig_pth_bytes:,} bytes ({orig_pth_bytes/1024:.2f} KB)")
    print(f"Raw Unfolded FP32 Parameters + BN Buffers:        {raw_unfolded_fp32_bytes:,} bytes ({raw_unfolded_fp32_bytes/1024:.2f} KB)")
    print(f"Exported Quantized .pth File (with test sample):  {quant_pth_bytes:,} bytes ({quant_pth_bytes/1024:.2f} KB)")
    print(f"Raw Binary Model File (.bin, zero ZIP overhead):  {raw_bin_bytes:,} bytes ({raw_bin_bytes/1024:.2f} KB)")
    print(f"  ├─ Raw INT8 Weight Storage ({total_int8_weights:,} weights):     {int8_weight_bytes:,} bytes ({int8_weight_bytes/1024:.2f} KB)")
    print(f"  ├─ Raw FP32 Folded Biases ({total_fp32_biases} channels):      {fp32_bias_bytes:,} bytes ({fp32_bias_bytes/1024:.2f} KB)")
    print(f"  └─ Raw FP32 Per-Channel Scales ({total_fp32_scales} channels): {fp32_scale_bytes:,} bytes ({fp32_scale_bytes/1024:.2f} KB)")
    print(f"Total Embedded Flash Constant Footprint:          {total_raw_quantized_bytes:,} bytes ({total_raw_quantized_bytes/1024:.2f} KB) [{raw_unfolded_fp32_bytes/total_raw_quantized_bytes:.2f}x reduction]")
    print(f"C++ Working Tensor SRAM (L=1000 full window):     {sram_L1000_bytes:,} bytes ({sram_L1000_bytes/1024:.2f} KB)")
    print(f"C++ Working Tensor SRAM (L=128 benchmark slice):  {sram_L128_bytes:,} bytes ({sram_L128_bytes/1024:.2f} KB)")

    print("\n======================================================================")
    print(f"STAGE 3: HELD-OUT TEST SET NUMERICAL VALIDATION (5,000 UNSEEN SAMPLES, th={slip_threshold})")
    print("======================================================================")
    y_test_np = y_test.numpy()
    preds_fp32 = run_batched_inference(model_fp32, X_test)
    preds_w8a32 = run_batched_inference(model_w8a32, X_test)
    preds_w8a8 = run_batched_inference(model_w8a8, X_test)

    eval_fp32 = evaluate_predictions(preds_fp32, y_test_np, slip_threshold)
    eval_w8a32 = evaluate_predictions(preds_w8a32, y_test_np, slip_threshold)
    eval_w8a8 = evaluate_predictions(preds_w8a8, y_test_np, slip_threshold)

    for label, res in [
        ("1. Original Unfolded FP32", eval_fp32),
        ("2. Exported Per-Channel INT8 Weights + FP32 Act (W8A32)", eval_w8a32),
        ("3. Full Integer Weights + Calibrated INT8 Act (W8A8)", eval_w8a8),
    ]:
        t1 = res["task1_envelope"]
        t2 = res["task2_cycle_slip"]
        t3 = res["task3_s4"]
        print(f"\n--- {label} ---")
        print(f"  Task 1 (Envelope)   | MSE: {t1['MSE']:.6f} | MAE: {t1['MAE']:.6f}")
        print(
            f"  Task 2 (Cycle Slip) | Prec: {t2['precision']:.4f} | Rec: {t2['recall']:.4f} | F1: {t2['f1']:.4f} "
            f"| TP={t2['TP']}, FP={t2['FP']}, TN={t2['TN']}, FN={t2['FN']}"
        )
        print(
            f"  Task 3 (S4 Index)   | MAE: {t3['MAE']:.6f} | RMSE: {t3['RMSE']:.6f} "
            f"| Severe (S4>=0.6) MAE: {t3['severe_S4_ge_0.6_MAE']:.6f}, RMSE: {t3['severe_S4_ge_0.6_RMSE']:.6f}"
        )

    audit_report = {
        "folding_equivalence": {
            "max_abs_diff_fp32_vs_folded": max_fold_diff,
            "mean_abs_diff_fp32_vs_folded": mean_fold_diff,
            "max_abs_diff_fp32_vs_w8a32": max_w8a32_diff,
            "mean_abs_diff_fp32_vs_w8a32": mean_w8a32_diff,
        },
        "storage_bytes": {
            "original_fp32_pth_file_bytes": orig_pth_bytes,
            "raw_unfolded_fp32_param_and_buffer_bytes": raw_unfolded_fp32_bytes,
            "exported_quantized_pth_file_bytes": quant_pth_bytes,
            "raw_binary_file_bytes": raw_bin_bytes,
            "raw_int8_weight_bytes": int8_weight_bytes,
            "raw_fp32_bias_bytes": fp32_bias_bytes,
            "raw_fp32_scale_bytes": fp32_scale_bytes,
            "total_flash_constant_bytes": total_raw_quantized_bytes,
            "runtime_working_sram_L1000_bytes": sram_L1000_bytes,
            "runtime_working_sram_L128_bytes": sram_L128_bytes,
        },
        "held_out_test_metrics": {
            "slip_threshold_used": slip_threshold,
            "unfolded_fp32": eval_fp32,
            "quantized_w8a32": eval_w8a32,
            "quantized_w8a8": eval_w8a8,
        },
    }
    report_out = "v2_2_repair/models/quantization_audit_report.json"
    with open(report_out, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\nSaved complete quantization audit report to: {report_out}")


if __name__ == "__main__":
    main()