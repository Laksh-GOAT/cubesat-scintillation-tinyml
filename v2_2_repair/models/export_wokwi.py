import os
import json
import numpy as np
import torch

try:
    from v2_2_repair.models.quantize import QuantizedTCNInference, load_split
except ImportError:
    from quantize import QuantizedTCNInference, load_split


# ==============================================================================
# 1. C-REFERENCE IMPLEMENTATIONS FOR PHASE 4 FULL-ARRAY VERIFICATION
# ==============================================================================
def c_baseline_conv1d(x, w_int8, b_fp32, s_fp32, in_c, out_c, k, d, pad, relu, L):
    out = np.zeros((out_c, L), dtype=np.float32)
    w_3d = w_int8.reshape(out_c, in_c, k).astype(np.float32)
    for oc in range(out_c):
        w_scale = np.float32(s_fp32[oc])
        bias = np.float32(b_fp32[oc])
        for t in range(L):
            acc = bias
            for ic in range(in_c):
                for ki in range(k):
                    t_in = t - pad + ki * d
                    if 0 <= t_in < L:
                        w_val = np.float32(w_3d[oc, ic, ki] * w_scale)
                        acc = np.float32(acc + np.float32(x[ic, t_in]) * w_val)
            if relu and acc < np.float32(0.0):
                acc = np.float32(0.0)
            out[oc, t] = acc
    return out


def c_register_tiled_conv1d(x, w_int8, b_fp32, s_fp32, in_c, out_c, k, d, pad, relu, L):
    """
    Exact Python mirror of the Stage-2 4-OC x 2-T FPU-register-tiled C++ kernel.
    Preserves the exact (ic = 0..in_c-1, ki = 0..k-1) accumulation order per (oc, t).
    """
    out = np.empty((out_c, L), dtype=np.float32)
    w_3d = w_int8.reshape(out_c, in_c, k).astype(np.float32)
    for oc in range(out_c):
        w_scale = np.float32(s_fp32[oc])
        out[oc, :] = np.float32(b_fp32[oc])
        for ic in range(in_c):
            in_row = x[ic]
            for ki in range(k):
                w_val = np.float32(w_3d[oc, ic, ki] * w_scale)
                offset = ki * d - pad
                t_start = -offset if offset < 0 else 0
                t_end = L - offset if offset > 0 else L
                for t in range(t_start, t_end):
                    out[oc, t] = np.float32(out[oc, t] + np.float32(in_row[t + offset]) * w_val)
        if relu:
            np.maximum(out[oc], np.float32(0.0), out=out[oc])
    return out


def c_optimized_s4_0(feats, mean_vec, std_vec, w_int8, b_fp32, s_fp32, L):
    out = np.empty((16, L), dtype=np.float32)
    w_2d = w_int8.reshape(16, 96).astype(np.float32)
    for oc in range(16):
        w_scale = np.float32(s_fp32[oc])
        const_acc = np.float32(b_fp32[oc])
        for c in range(32):
            w_m = np.float32(w_2d[oc, 32 + c] * w_scale)
            w_s = np.float32(w_2d[oc, 64 + c] * w_scale)
            const_acc = np.float32(const_acc + np.float32(mean_vec[c]) * w_m + np.float32(std_vec[c]) * w_s)
        out[oc, :] = const_acc
        for c in range(32):
            w_f = np.float32(w_2d[oc, c] * w_scale)
            in_row = feats[c]
            for t in range(L):
                out[oc, t] = np.float32(out[oc, t] + np.float32(in_row[t]) * w_f)
        np.maximum(out[oc], np.float32(0.0), out=out[oc])
    return out


def run_full_c_pipeline(layers_dict, x_slice_np: np.ndarray, optimized: bool = True) -> np.ndarray:
    L = x_slice_np.shape[-1]
    conv_fn = c_register_tiled_conv1d if optimized else c_baseline_conv1d

    def call_layer(inp, name):
        sp = layers_dict[name]
        return conv_fn(
            inp,
            sp["weight_int8"].cpu().numpy(),
            sp["bias_fp32"].cpu().numpy(),
            sp["scale_fp32"].cpu().numpy(),
            sp["in_channels"],
            sp["out_channels"],
            sp["kernel_size"],
            sp["dilation"],
            sp["padding"],
            sp["relu"],
            L,
        )

    h0 = call_layer(x_slice_np, "bb_0")
    h1 = call_layer(h0, "bb_1")
    h2 = call_layer(h1, "bb_2")
    feats = call_layer(h2, "bb_3")

    out_env = call_layer(feats, "env_0")
    s0 = call_layer(feats, "slip_0")
    out_slip = call_layer(s0, "slip_2")

    feat_mean = np.mean(feats, axis=1, dtype=np.float32)
    feat_std = np.sqrt(np.mean((feats - feat_mean[:, None]) ** 2, axis=1, dtype=np.float32))

    if optimized:
        sp_s4_0 = layers_dict["s4_0"]
        u0 = c_optimized_s4_0(
            feats,
            feat_mean,
            feat_std,
            sp_s4_0["weight_int8"].cpu().numpy(),
            sp_s4_0["bias_fp32"].cpu().numpy(),
            sp_s4_0["scale_fp32"].cpu().numpy(),
            L,
        )
    else:
        s4_in = np.concatenate(
            [feats, np.repeat(feat_mean[:, None], L, axis=1), np.repeat(feat_std[:, None], L, axis=1)],
            axis=0,
        ).astype(np.float32)
        u0 = call_layer(s4_in, "s4_0")

    out_s4 = call_layer(u0, "s4_2")
    return np.concatenate([out_env, out_slip, out_s4], axis=0)


def select_four_test_scenarios(data_dir: str, L_slice: int = 128):
    X_test, y_test = load_split(data_dir, "test")
    s4_all = y_test[:, 2, 0].numpy()
    slip_sums = y_test[:, 1, :L_slice].sum(dim=1).numpy()
    noise_proxy = torch.std(X_test[:, 0, :L_slice] - y_test[:, 0, :L_slice], dim=1).numpy()

    idx_0 = 0
    quiet_candidates = np.where((s4_all < 0.22) & (slip_sums == 0))[0]
    idx_1 = int(quiet_candidates[0]) if len(quiet_candidates) > 0 else 1
    severe_candidates = np.where((s4_all > 0.85) & (noise_proxy > np.median(noise_proxy)))[0]
    idx_2 = int(severe_candidates[0]) if len(severe_candidates) > 0 else 2
    slip_candidates = np.where((slip_sums >= 20) & (s4_all >= 0.60))[0]
    idx_3 = int(slip_candidates[0]) if len(slip_candidates) > 0 else 3

    scenarios = [
        ("Case0_Benchmark_Ref", idx_0),
        ("Case1_Quiet_Normal", idx_1),
        ("Case2_Severe_Scint_Noisy", idx_2),
        ("Case3_Multi_Cycle_Slip", idx_3),
    ]

    selected = []
    for name, idx in scenarios:
        x_sl = X_test[idx, :, :L_slice].contiguous().clone()
        y_sl = y_test[idx, :, :L_slice].contiguous().clone()
        selected.append({"name": name, "test_index": int(idx), "x": x_sl, "y": y_sl})
    return selected


def format_int8_c_array(name: str, arr: np.ndarray) -> str:
    flat = arr.flatten().tolist()
    body = ", ".join(str(int(v)) for v in flat)
    return f"DRAM_ATTR const int8_t {name}[{len(flat)}] = {{{body}}};\n"


def format_float_c_array(name: str, arr: np.ndarray, dram: bool = True) -> str:
    flat = arr.flatten().tolist()
    body = ", ".join(f"{float(v):.8e}f" for v in flat)
    attr = "DRAM_ATTR " if dram else ""
    return f"{attr}const float {name}[{len(flat)}] = {{{body}}};\n"


def main():
    quant_path = "v2_2_repair/models/quantized_folded_tcn.pth"
    data_dir = "v2_2_repair/data/processed/v2.2_repaired"
    if not os.path.exists(quant_path):
        raise FileNotFoundError(f"Missing {quant_path}. Run quantize.py first!")

    payload = torch.load(quant_path, map_location="cpu")
    layers = payload["layers"]
    slip_th = float(payload.get("slip_threshold", 0.82))
    L_slice = 128

    print("======================================================================")
    print("STAGE-2 REGISTER-TILED KERNEL VERIFICATION (PYTHON W8A32 vs C++)")
    print("======================================================================")

    py_runner = QuantizedTCNInference(layers, mode="w8a32")
    py_runner.eval()

    scenarios = select_four_test_scenarios(data_dir, L_slice=L_slice)
    audit_scenarios = []

    for sc in scenarios:
        name = sc["name"]
        x_t = sc["x"]
        y_t = sc["y"]
        x_np = x_t.cpu().numpy().astype(np.float32)

        with torch.no_grad():
            py_out = py_runner(x_t.unsqueeze(0))[0].cpu().numpy()

        c_opt_out = run_full_c_pipeline(layers, x_np, optimized=True)

        py_probs = 1.0 / (1.0 + np.exp(-np.clip(py_out[1], -20.0, 20.0)))
        opt_probs = 1.0 / (1.0 + np.exp(-np.clip(c_opt_out[1], -20.0, 20.0)))
        py_flags = (py_probs >= slip_th).astype(np.int32)
        opt_flags = (opt_probs >= slip_th).astype(np.int32)

        env_max_err = float(np.max(np.abs(py_out[0] - c_opt_out[0])))
        prob_max_err = float(np.max(np.abs(py_probs - opt_probs)))
        s4_max_err = float(np.max(np.abs(py_out[2] - c_opt_out[2])))
        flags_match = bool(np.array_equal(py_flags, opt_flags))

        print(
            f"[{name}] Env MaxErr={env_max_err:.2e} | Prob MaxErr={prob_max_err:.2e} | "
            f"S4 MaxErr={s4_max_err:.2e} | Flags={int(opt_flags.sum())}/{int(py_flags.sum())} (Match={flags_match})"
        )
        if max(env_max_err, prob_max_err, s4_max_err) > 1e-5 or not flags_match:
            raise RuntimeError(f"Numerical tolerance exceeded on {name}!")

        audit_scenarios.append({"name": name, "py_out": py_out, "x_np": x_np})

    layer_order = ["bb_0", "bb_1", "bb_2", "bb_3", "env_0", "slip_0", "slip_2", "s4_0", "s4_2"]
    c_blocks = []
    for lname in layer_order:
        sp = layers[lname]
        c_blocks.append(format_int8_c_array(f"{lname}_w", sp["weight_int8"].cpu().numpy()))
        c_blocks.append(format_float_c_array(f"{lname}_s", sp["scale_fp32"].cpu().numpy(), dram=True))
        c_blocks.append(format_float_c_array(f"{lname}_b", sp["bias_fp32"].cpu().numpy(), dram=True))

    all_x = np.stack([s["x_np"] for s in audit_scenarios], axis=0)
    all_py_out = np.stack([s["py_out"] for s in audit_scenarios], axis=0)
    c_blocks.append(format_float_c_array("test_scenarios_x", all_x, dram=False))
    c_blocks.append(format_float_c_array("py_ref_outputs", all_py_out, dram=False))

    names_init = ", ".join(f'"{s["name"]}"' for s in audit_scenarios)
    c_blocks.append(f"const char* const SCENARIO_NAMES[4] = {{{names_init}}};\n")
    arrays_code = "\n".join(c_blocks)

    sketch_code = f"""// ============================================================================
// CUBESAT V2.2 STAGE-2 EXTREME-OPTIMIZED 9-LAYER TCN ENGINE (W8A32 INT8)
// Stage-2 Optimizations over Stage-1 (1233.6 ms):
//   1. 4-OC x 2-T Hardware FPU Register Tiling (8 accumulators in f0..f7,
//      2 inputs in f8..f9, 4 weights in f10..f13 -> ZERO RAM writes inside loop!)
//   2. Fully Unrolled K=5 and K=3 Dilated Inner Loops (40 MACs / 24 MACs per branch)
//   3. IRAM_ATTR + __attribute__((optimize("O3"), hot)) Zero-Wait-State Code Execution
//   4. 2.5 KB Per-Tile Transposed Weight Scratch Buffer (w_tile[4 * 32 * 5])
// ============================================================================
#pragma GCC optimize ("O3")
#include <Arduino.h>
#include <math.h>

#define L {L_slice}
#define NUM_SCENARIOS 4
#define SLIP_PROB_THRESHOLD {slip_th:.4f}f

{arrays_code}

// Static Working SRAM Buffers:
// bufA (32*128*4 = 16,384 B) + bufB (16,384 B) + w_tile (4*32*5*4 = 2,560 B) + outputs (1,536 B) + stats (256 B)
// Total Buffers = 37,120 B (36.25 KB) + Weights/Scales (16,768 B) = 53,888 B (10.1% of 520 KB ESP32 SRAM)
DRAM_ATTR static float bufA[32 * L];
DRAM_ATTR static float bufB[32 * L];
DRAM_ATTR static float w_tile[32 * 5 * 4];
DRAM_ATTR static float feat_mean[32];
DRAM_ATTR static float feat_std[32];

DRAM_ATTR static float out_env[L];
DRAM_ATTR static float out_slip_logits[L];
DRAM_ATTR static float out_s4[L];

struct LayerProfile {{
    unsigned long bb0_us, bb1_us, bb2_us, bb3_us;
    unsigned long env0_us, slip0_us, slip2_us;
    unsigned long s4_pool_us, s4_0_us, s4_2_us;
    unsigned long total_us;
}};

// ============================================================================
// STAGE-2 KERNEL 1: 4-OC x 2-T REGISTER-TILED & UNROLLED K=5 DILATED CONV1D
// Used by: bb_0 (d=1, pad=2), bb_1 (d=2, pad=4), bb_2 (d=4, pad=8)
// ============================================================================
void IRAM_ATTR __attribute__((optimize("O3"), hot)) conv1d_k5_register_tiled(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int8_t* __restrict__ w,
    const float* __restrict__ s_per_ch,
    const float* __restrict__ b_per_ch,
    int in_c, int out_c, int d, int pad
) {{
    const int d2 = d * 2;
    const int t_int_start = pad;
    const int t_int_end = L - pad; // Exclusive bound for safe interior steps

    for (int oc = 0; oc < out_c; oc += 4) {{
        const float s0 = s_per_ch[oc + 0], s1 = s_per_ch[oc + 1];
        const float s2 = s_per_ch[oc + 2], s3 = s_per_ch[oc + 3];
        const float b0 = b_per_ch[oc + 0], b1 = b_per_ch[oc + 1];
        const float b2 = b_per_ch[oc + 2], b3 = b_per_ch[oc + 3];

        const int stride_oc = in_c * 5;
        const int8_t* __restrict__ w0_src = w + (oc + 0) * stride_oc;
        const int8_t* __restrict__ w1_src = w + (oc + 1) * stride_oc;
        const int8_t* __restrict__ w2_src = w + (oc + 2) * stride_oc;
        const int8_t* __restrict__ w3_src = w + (oc + 3) * stride_oc;

        // Dequantize & transpose 4 output channels into contiguous [ic][ki][4] tile
        float* __restrict__ wt = w_tile;
        for (int idx = 0; idx < stride_oc; idx++) {{
            wt[0] = (float)w0_src[idx] * s0;
            wt[1] = (float)w1_src[idx] * s1;
            wt[2] = (float)w2_src[idx] * s2;
            wt[3] = (float)w3_src[idx] * s3;
            wt += 4;
        }}

        float* __restrict__ out0 = out + (oc + 0) * L;
        float* __restrict__ out1 = out + (oc + 1) * L;
        float* __restrict__ out2 = out + (oc + 2) * L;
        float* __restrict__ out3 = out + (oc + 3) * L;

        // 1. Left & Right Boundary Timesteps (exact branch-checked fallback)
        for (int side = 0; side < 2; side++) {{
            const int tb_start = (side == 0) ? 0 : t_int_end;
            const int tb_end   = (side == 0) ? t_int_start : L;
            for (int t = tb_start; t < tb_end; t++) {{
                float a0 = b0, a1 = b1, a2 = b2, a3 = b3;
                const float* __restrict__ w_ptr = w_tile;
                for (int ic = 0; ic < in_c; ic++) {{
                    const float* __restrict__ in_ic = in + ic * L;
                    for (int ki = 0; ki < 5; ki++) {{
                        const int t_in = t - pad + ki * d;
                        if (t_in >= 0 && t_in < L) {{
                            const float x = in_ic[t_in];
                            a0 += x * w_ptr[0];
                            a1 += x * w_ptr[1];
                            a2 += x * w_ptr[2];
                            a3 += x * w_ptr[3];
                        }}
                        w_ptr += 4;
                    }}
                }}
                out0[t] = (a0 > 0.0f) ? a0 : 0.0f;
                out1[t] = (a1 > 0.0f) ? a1 : 0.0f;
                out2[t] = (a2 > 0.0f) ? a2 : 0.0f;
                out3[t] = (a3 > 0.0f) ? a3 : 0.0f;
            }}
        }}

        // 2. Fast Interior Timesteps: 4-OC x 2-T FPU Register Tile (40 MACs per ic iteration, 0 branches!)
        for (int t = t_int_start; t < t_int_end; t += 2) {{
            float a00 = b0, a01 = b0;
            float a10 = b1, a11 = b1;
            float a20 = b2, a21 = b2;
            float a30 = b3, a31 = b3;

            const float* __restrict__ w_ptr = w_tile;
            const float* __restrict__ in_ptr = in + t;

            for (int ic = 0; ic < in_c; ic++) {{
                float x0, x1;
                // Tap 0 (offset = -d2)
                x0 = in_ptr[-d2]; x1 = in_ptr[-d2 + 1];
                a00 += x0 * w_ptr[0]; a01 += x1 * w_ptr[0];
                a10 += x0 * w_ptr[1]; a11 += x1 * w_ptr[1];
                a20 += x0 * w_ptr[2]; a21 += x1 * w_ptr[2];
                a30 += x0 * w_ptr[3]; a31 += x1 * w_ptr[3];

                // Tap 1 (offset = -d)
                x0 = in_ptr[-d]; x1 = in_ptr[-d + 1];
                a00 += x0 * w_ptr[4]; a01 += x1 * w_ptr[4];
                a10 += x0 * w_ptr[5]; a11 += x1 * w_ptr[5];
                a20 += x0 * w_ptr[6]; a21 += x1 * w_ptr[6];
                a30 += x0 * w_ptr[7]; a31 += x1 * w_ptr[7];

                // Tap 2 (offset = 0)
                x0 = in_ptr[0]; x1 = in_ptr[1];
                a00 += x0 * w_ptr[8];  a01 += x1 * w_ptr[8];
                a10 += x0 * w_ptr[9];  a11 += x1 * w_ptr[9];
                a20 += x0 * w_ptr[10]; a21 += x1 * w_ptr[10];
                a30 += x0 * w_ptr[11]; a31 += x1 * w_ptr[11];

                // Tap 3 (offset = +d)
                x0 = in_ptr[d]; x1 = in_ptr[d + 1];
                a00 += x0 * w_ptr[12]; a01 += x1 * w_ptr[12];
                a10 += x0 * w_ptr[13]; a11 += x1 * w_ptr[13];
                a20 += x0 * w_ptr[14]; a21 += x1 * w_ptr[14];
                a30 += x0 * w_ptr[15]; a31 += x1 * w_ptr[15];

                // Tap 4 (offset = +d2)
                x0 = in_ptr[d2]; x1 = in_ptr[d2 + 1];
                a00 += x0 * w_ptr[16]; a01 += x1 * w_ptr[16];
                a10 += x0 * w_ptr[17]; a11 += x1 * w_ptr[17];
                a20 += x0 * w_ptr[18]; a21 += x1 * w_ptr[18];
                a30 += x0 * w_ptr[19]; a31 += x1 * w_ptr[19];

                w_ptr += 20;
                in_ptr += L;
            }}

            out0[t] = (a00 > 0.0f) ? a00 : 0.0f; out0[t + 1] = (a01 > 0.0f) ? a01 : 0.0f;
            out1[t] = (a10 > 0.0f) ? a10 : 0.0f; out1[t + 1] = (a11 > 0.0f) ? a11 : 0.0f;
            out2[t] = (a20 > 0.0f) ? a20 : 0.0f; out2[t + 1] = (a21 > 0.0f) ? a21 : 0.0f;
            out3[t] = (a30 > 0.0f) ? a30 : 0.0f; out3[t + 1] = (a31 > 0.0f) ? a31 : 0.0f;
        }}
    }}
}}

// ============================================================================
// STAGE-2 KERNEL 2: 4-OC x 2-T REGISTER-TILED & UNROLLED K=3 DILATED CONV1D
// Used by: bb_3 (32->32, k=3, d=8, pad=8), slip_0 (32->16, k=3, d=1, pad=1)
// ============================================================================
void IRAM_ATTR __attribute__((optimize("O3"), hot)) conv1d_k3_register_tiled(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int8_t* __restrict__ w,
    const float* __restrict__ s_per_ch,
    const float* __restrict__ b_per_ch,
    int in_c, int out_c, int d, int pad
) {{
    const int t_int_start = pad;
    const int t_int_end = L - pad;
    const int interior_len = t_int_end - t_int_start;
    const int t_vec_end = t_int_start + (interior_len & ~1);

    for (int oc = 0; oc < out_c; oc += 4) {{
        const float s0 = s_per_ch[oc + 0], s1 = s_per_ch[oc + 1];
        const float s2 = s_per_ch[oc + 2], s3 = s_per_ch[oc + 3];
        const float b0 = b_per_ch[oc + 0], b1 = b_per_ch[oc + 1];
        const float b2 = b_per_ch[oc + 2], b3 = b_per_ch[oc + 3];

        const int stride_oc = in_c * 3;
        const int8_t* __restrict__ w0_src = w + (oc + 0) * stride_oc;
        const int8_t* __restrict__ w1_src = w + (oc + 1) * stride_oc;
        const int8_t* __restrict__ w2_src = w + (oc + 2) * stride_oc;
        const int8_t* __restrict__ w3_src = w + (oc + 3) * stride_oc;

        float* __restrict__ wt = w_tile;
        for (int idx = 0; idx < stride_oc; idx++) {{
            wt[0] = (float)w0_src[idx] * s0;
            wt[1] = (float)w1_src[idx] * s1;
            wt[2] = (float)w2_src[idx] * s2;
            wt[3] = (float)w3_src[idx] * s3;
            wt += 4;
        }}

        float* __restrict__ out0 = out + (oc + 0) * L;
        float* __restrict__ out1 = out + (oc + 1) * L;
        float* __restrict__ out2 = out + (oc + 2) * L;
        float* __restrict__ out3 = out + (oc + 3) * L;

        // 1. Boundary Timesteps
        for (int side = 0; side < 2; side++) {{
            const int tb_start = (side == 0) ? 0 : t_vec_end;
            const int tb_end   = (side == 0) ? t_int_start : L;
            for (int t = tb_start; t < tb_end; t++) {{
                float a0 = b0, a1 = b1, a2 = b2, a3 = b3;
                const float* __restrict__ w_ptr = w_tile;
                for (int ic = 0; ic < in_c; ic++) {{
                    const float* __restrict__ in_ic = in + ic * L;
                    for (int ki = 0; ki < 3; ki++) {{
                        const int t_in = t - pad + ki * d;
                        if (t_in >= 0 && t_in < L) {{
                            const float x = in_ic[t_in];
                            a0 += x * w_ptr[0];
                            a1 += x * w_ptr[1];
                            a2 += x * w_ptr[2];
                            a3 += x * w_ptr[3];
                        }}
                        w_ptr += 4;
                    }}
                }}
                out0[t] = (a0 > 0.0f) ? a0 : 0.0f;
                out1[t] = (a1 > 0.0f) ? a1 : 0.0f;
                out2[t] = (a2 > 0.0f) ? a2 : 0.0f;
                out3[t] = (a3 > 0.0f) ? a3 : 0.0f;
            }}
        }}

        // 2. Fast Interior Timesteps: 4-OC x 2-T FPU Register Tile (24 MACs per ic iteration, 0 branches!)
        for (int t = t_int_start; t < t_vec_end; t += 2) {{
            float a00 = b0, a01 = b0;
            float a10 = b1, a11 = b1;
            float a20 = b2, a21 = b2;
            float a30 = b3, a31 = b3;

            const float* __restrict__ w_ptr = w_tile;
            const float* __restrict__ in_ptr = in + t;

            for (int ic = 0; ic < in_c; ic++) {{
                float x0, x1;
                // Tap 0 (offset = -d)
                x0 = in_ptr[-d]; x1 = in_ptr[-d + 1];
                a00 += x0 * w_ptr[0]; a01 += x1 * w_ptr[0];
                a10 += x0 * w_ptr[1]; a11 += x1 * w_ptr[1];
                a20 += x0 * w_ptr[2]; a21 += x1 * w_ptr[2];
                a30 += x0 * w_ptr[3]; a31 += x1 * w_ptr[3];

                // Tap 1 (offset = 0)
                x0 = in_ptr[0]; x1 = in_ptr[1];
                a00 += x0 * w_ptr[4]; a01 += x1 * w_ptr[4];
                a10 += x0 * w_ptr[5]; a11 += x1 * w_ptr[5];
                a20 += x0 * w_ptr[6]; a21 += x1 * w_ptr[6];
                a30 += x0 * w_ptr[7]; a31 += x1 * w_ptr[7];

                // Tap 2 (offset = +d)
                x0 = in_ptr[d]; x1 = in_ptr[d + 1];
                a00 += x0 * w_ptr[8];  a01 += x1 * w_ptr[8];
                a10 += x0 * w_ptr[9];  a11 += x1 * w_ptr[9];
                a20 += x0 * w_ptr[10]; a21 += x1 * w_ptr[10];
                a30 += x0 * w_ptr[11]; a31 += x1 * w_ptr[11];

                w_ptr += 12;
                in_ptr += L;
            }}

            out0[t] = (a00 > 0.0f) ? a00 : 0.0f; out0[t + 1] = (a01 > 0.0f) ? a01 : 0.0f;
            out1[t] = (a10 > 0.0f) ? a10 : 0.0f; out1[t + 1] = (a11 > 0.0f) ? a11 : 0.0f;
            out2[t] = (a20 > 0.0f) ? a20 : 0.0f; out2[t + 1] = (a21 > 0.0f) ? a21 : 0.0f;
            out3[t] = (a30 > 0.0f) ? a30 : 0.0f; out3[t + 1] = (a31 > 0.0f) ? a31 : 0.0f;
        }}
    }}
}}

// ============================================================================
// STAGE-2 KERNEL 3: 1-OC x 4-T REGISTER-TILED POINTWISE K=1 PROJECTION
// Used by: env_0 (32->1), slip_2 (16->1), s4_2 (16->1)
// ============================================================================
void IRAM_ATTR __attribute__((optimize("O3"), hot)) conv1d_k1_1oc_register_tiled(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int8_t* __restrict__ w,
    const float* __restrict__ s_per_ch,
    const float* __restrict__ b_per_ch,
    int in_c
) {{
    const float w_scale = s_per_ch[0];
    const float bias = b_per_ch[0];
    for (int ic = 0; ic < in_c; ic++) {{
        w_tile[ic] = (float)w[ic] * w_scale;
    }}
    for (int t = 0; t < L; t += 4) {{
        float a0 = bias, a1 = bias, a2 = bias, a3 = bias;
        const float* __restrict__ in_ptr = in + t;
        for (int ic = 0; ic < in_c; ic++) {{
            const float w_val = w_tile[ic];
            a0 += in_ptr[0] * w_val;
            a1 += in_ptr[1] * w_val;
            a2 += in_ptr[2] * w_val;
            a3 += in_ptr[3] * w_val;
            in_ptr += L;
        }}
        out[t + 0] = a0;
        out[t + 1] = a1;
        out[t + 2] = a2;
        out[t + 3] = a3;
    }}
}}

// ============================================================================
// STAGE-2 KERNEL 4: GLOBAL POOLING + 4-OC x 2-T REGISTER-TILED S4_0 (96->16)
// ============================================================================
void IRAM_ATTR __attribute__((optimize("O3"), hot)) compute_s4_pool_opt(const float* __restrict__ feats) {{
    const float inv_L = 1.0f / (float)L;
    for (int c = 0; c < 32; c++) {{
        const float* __restrict__ f_c = feats + c * L;
        float s0 = 0.0f, s1 = 0.0f;
        for (int t = 0; t < L; t += 2) {{
            s0 += f_c[t];
            s1 += f_c[t + 1];
        }}
        const float mean = (s0 + s1) * inv_L;
        float v0 = 0.0f, v1 = 0.0f;
        for (int t = 0; t < L; t += 2) {{
            const float d0 = f_c[t] - mean;
            const float d1 = f_c[t + 1] - mean;
            v0 += d0 * d0;
            v1 += d1 * d1;
        }}
        feat_mean[c] = mean;
        feat_std[c] = sqrtf((v0 + v1) * inv_L);
    }}
}}

void IRAM_ATTR __attribute__((optimize("O3"), hot)) compute_s4_0_register_tiled(
    const float* __restrict__ feats,
    float* __restrict__ s4_hidden
) {{
    for (int oc = 0; oc < 16; oc += 4) {{
        const float s0 = s4_0_s[oc + 0], s1 = s4_0_s[oc + 1];
        const float s2 = s4_0_s[oc + 2], s3 = s4_0_s[oc + 3];
        const int8_t* __restrict__ w0 = s4_0_w + (oc + 0) * 96;
        const int8_t* __restrict__ w1 = s4_0_w + (oc + 1) * 96;
        const int8_t* __restrict__ w2 = s4_0_w + (oc + 2) * 96;
        const int8_t* __restrict__ w3 = s4_0_w + (oc + 3) * 96;

        float cb0 = s4_0_b[oc + 0], cb1 = s4_0_b[oc + 1];
        float cb2 = s4_0_b[oc + 2], cb3 = s4_0_b[oc + 3];

        float* __restrict__ wt = w_tile;
        for (int c = 0; c < 32; c++) {{
            const float m = feat_mean[c];
            const float sd = feat_std[c];
            cb0 += m * ((float)w0[32 + c] * s0) + sd * ((float)w0[64 + c] * s0);
            cb1 += m * ((float)w1[32 + c] * s1) + sd * ((float)w1[64 + c] * s1);
            cb2 += m * ((float)w2[32 + c] * s2) + sd * ((float)w2[64 + c] * s2);
            cb3 += m * ((float)w3[32 + c] * s3) + sd * ((float)w3[64 + c] * s3);

            wt[0] = (float)w0[c] * s0;
            wt[1] = (float)w1[c] * s1;
            wt[2] = (float)w2[c] * s2;
            wt[3] = (float)w3[c] * s3;
            wt += 4;
        }}

        float* __restrict__ out0 = s4_hidden + (oc + 0) * L;
        float* __restrict__ out1 = s4_hidden + (oc + 1) * L;
        float* __restrict__ out2 = s4_hidden + (oc + 2) * L;
        float* __restrict__ out3 = s4_hidden + (oc + 3) * L;

        for (int t = 0; t < L; t += 2) {{
            float a00 = cb0, a01 = cb0;
            float a10 = cb1, a11 = cb1;
            float a20 = cb2, a21 = cb2;
            float a30 = cb3, a31 = cb3;

            const float* __restrict__ w_ptr = w_tile;
            const float* __restrict__ f_ptr = feats + t;
            for (int c = 0; c < 32; c++) {{
                const float x0 = f_ptr[0];
                const float x1 = f_ptr[1];
                a00 += x0 * w_ptr[0]; a01 += x1 * w_ptr[0];
                a10 += x0 * w_ptr[1]; a11 += x1 * w_ptr[1];
                a20 += x0 * w_ptr[2]; a21 += x1 * w_ptr[2];
                a30 += x0 * w_ptr[3]; a31 += x1 * w_ptr[3];
                w_ptr += 4;
                f_ptr += L;
            }}
            out0[t] = (a00 > 0.0f) ? a00 : 0.0f; out0[t + 1] = (a01 > 0.0f) ? a01 : 0.0f;
            out1[t] = (a10 > 0.0f) ? a10 : 0.0f; out1[t + 1] = (a11 > 0.0f) ? a11 : 0.0f;
            out2[t] = (a20 > 0.0f) ? a20 : 0.0f; out2[t + 1] = (a21 > 0.0f) ? a21 : 0.0f;
            out3[t] = (a30 > 0.0f) ? a30 : 0.0f; out3[t + 1] = (a31 > 0.0f) ? a31 : 0.0f;
        }}
    }}
}}

// Uninstrumented Fast Complete 9-Layer Forward Pass in IRAM
void IRAM_ATTR __attribute__((optimize("O3"), hot)) run_tcn_inference_stage2(const float* __restrict__ input_tensor) {{
    conv1d_k5_register_tiled(input_tensor, bufA, bb_0_w, bb_0_s, bb_0_b, 3, 24, 1, 2);
    conv1d_k5_register_tiled(bufA, bufB, bb_1_w, bb_1_s, bb_1_b, 24, 32, 2, 4);
    conv1d_k5_register_tiled(bufB, bufA, bb_2_w, bb_2_s, bb_2_b, 32, 32, 4, 8);
    conv1d_k3_register_tiled(bufA, bufB, bb_3_w, bb_3_s, bb_3_b, 32, 32, 8, 8);

    conv1d_k1_1oc_register_tiled(bufB, out_env, env_0_w, env_0_s, env_0_b, 32);

    conv1d_k3_register_tiled(bufB, bufA, slip_0_w, slip_0_s, slip_0_b, 32, 16, 1, 1);
    conv1d_k1_1oc_register_tiled(bufA, out_slip_logits, slip_2_w, slip_2_s, slip_2_b, 16);

    compute_s4_pool_opt(bufB);
    compute_s4_0_register_tiled(bufB, bufA);
    conv1d_k1_1oc_register_tiled(bufA, out_s4, s4_2_w, s4_2_s, s4_2_b, 16);
}}

void run_stage2_profiled(const float* __restrict__ input_tensor, LayerProfile& prof) {{
    unsigned long t0 = micros(), t1;
    conv1d_k5_register_tiled(input_tensor, bufA, bb_0_w, bb_0_s, bb_0_b, 3, 24, 1, 2);
    t1 = micros(); prof.bb0_us = t1 - t0; t0 = t1;

    conv1d_k5_register_tiled(bufA, bufB, bb_1_w, bb_1_s, bb_1_b, 24, 32, 2, 4);
    t1 = micros(); prof.bb1_us = t1 - t0; t0 = t1;

    conv1d_k5_register_tiled(bufB, bufA, bb_2_w, bb_2_s, bb_2_b, 32, 32, 4, 8);
    t1 = micros(); prof.bb2_us = t1 - t0; t0 = t1;

    conv1d_k3_register_tiled(bufA, bufB, bb_3_w, bb_3_s, bb_3_b, 32, 32, 8, 8);
    t1 = micros(); prof.bb3_us = t1 - t0; t0 = t1;

    conv1d_k1_1oc_register_tiled(bufB, out_env, env_0_w, env_0_s, env_0_b, 32);
    t1 = micros(); prof.env0_us = t1 - t0; t0 = t1;

    conv1d_k3_register_tiled(bufB, bufA, slip_0_w, slip_0_s, slip_0_b, 32, 16, 1, 1);
    t1 = micros(); prof.slip0_us = t1 - t0; t0 = t1;

    conv1d_k1_1oc_register_tiled(bufA, out_slip_logits, slip_2_w, slip_2_s, slip_2_b, 16);
    t1 = micros(); prof.slip2_us = t1 - t0; t0 = t1;

    compute_s4_pool_opt(bufB);
    t1 = micros(); prof.s4_pool_us = t1 - t0; t0 = t1;

    compute_s4_0_register_tiled(bufB, bufA);
    t1 = micros(); prof.s4_0_us = t1 - t0; t0 = t1;

    conv1d_k1_1oc_register_tiled(bufA, out_s4, s4_2_w, s4_2_s, s4_2_b, 16);
    t1 = micros(); prof.s4_2_us = t1 - t0;
    prof.total_us = prof.bb0_us + prof.bb1_us + prof.bb2_us + prof.bb3_us +
                    prof.env0_us + prof.slip0_us + prof.slip2_us +
                    prof.s4_pool_us + prof.s4_0_us + prof.s4_2_us;
}}

void setup() {{
    Serial.begin(115200);
    delay(1000);
    Serial.println("==========================================================================");
    Serial.println("CUBESAT V2.2 STAGE-2 REGISTER-TILED & IRAM UNROLLED 9-LAYER TCN (L=128)");
    Serial.println("Note: Wokwi timing reflects simulated Xtensa LX6 execution, not silicon.");
    Serial.println("==========================================================================");

    LayerProfile prof_s2;
    run_stage2_profiled(test_scenarios_x, prof_s2);

    Serial.println("\\n--- [STAGE-2] REGISTER-TILED + IRAM PER-LAYER PROFILE (SCENARIO 0) ---");
    Serial.printf("  bb_0 (3->24, k=5, d=1):   %8lu us (Legacy:  101963 us | Stage-1:   33938 us)\\n", prof_s2.bb0_us);
    Serial.printf("  bb_1 (24->32, k=5, d=2):  %8lu us (Legacy: 1041628 us | Stage-1:  331506 us)\\n", prof_s2.bb1_us);
    Serial.printf("  bb_2 (32->32, k=5, d=4):  %8lu us (Legacy: 1374783 us | Stage-1:  433558 us)\\n", prof_s2.bb2_us);
    Serial.printf("  bb_3 (32->32, k=3, d=8):  %8lu us (Legacy:  894653 us | Stage-1:  260505 us)\\n", prof_s2.bb3_us);
    Serial.printf("  Head1 env_0 (32->1, k=1): %8lu us (Legacy:   13051 us | Stage-1:    1927 us)\\n", prof_s2.env0_us);
    Serial.printf("  Head2 slip_0 (32->16,k=3):%8lu us (Legacy:  454426 us | Stage-1:  134986 us)\\n", prof_s2.slip0_us);
    Serial.printf("  Head2 slip_2 (16->1, k=1):%8lu us (Legacy:    6628 us | Stage-1:     940 us)\\n", prof_s2.slip2_us);
    Serial.printf("  Head3 s4_pool (mean+std): %8lu us (Legacy:    7047 us | Stage-1:    3358 us)\\n", prof_s2.s4_pool_us);
    Serial.printf("  Head3 s4_0 (96->16, k=1): %8lu us (Legacy:  627817 us | Stage-1:   32266 us)\\n", prof_s2.s4_0_us);
    Serial.printf("  Head3 s4_2 (16->1, k=1):  %8lu us (Legacy:    6628 us | Stage-1:     940 us)\\n", prof_s2.s4_2_us);
    Serial.printf("  STAGE-2 INSTRUMENTED SUM: %8lu us (%.3f ms)\\n", prof_s2.total_us, prof_s2.total_us / 1000.0f);

    Serial.println("\\n--- [PHASE 4] FULL-ARRAY NUMERICAL AUDIT ACROSS 4 TEST SCENARIOS ---");
    for (int s = 0; s < NUM_SCENARIOS; s++) {{
        const float* x_in = test_scenarios_x + s * (3 * L);
        const float* py_ref = py_ref_outputs + s * (3 * L);
        const float* py_env = py_ref;
        const float* py_slip = py_ref + L;
        const float* py_s4 = py_ref + 2 * L;

        run_tcn_inference_stage2(x_in);

        float env_max_e = 0.0f, env_sum_e = 0.0f;
        float prob_max_e = 0.0f, prob_sum_e = 0.0f;
        float s4_max_e = 0.0f, s4_sum_e = 0.0f;
        float s4_mean = 0.0f;
        int opt_flags = 0, py_flags = 0, flag_mismatches = 0;
        bool finite_ok = true;

        for (int t = 0; t < L; t++) {{
            if (!isfinite(out_env[t]) || !isfinite(out_slip_logits[t]) || !isfinite(out_s4[t])) finite_ok = false;
            float ee = fabsf(out_env[t] - py_env[t]);
            if (ee > env_max_e) env_max_e = ee;
            env_sum_e += ee;

            float c_log = constrain(out_slip_logits[t], -20.0f, 20.0f);
            float p_log = constrain(py_slip[t], -20.0f, 20.0f);
            float c_prob = 1.0f / (1.0f + expf(-c_log));
            float p_prob = 1.0f / (1.0f + expf(-p_log));
            float pe = fabsf(c_prob - p_prob);
            if (pe > prob_max_e) prob_max_e = pe;
            prob_sum_e += pe;

            int cf = (c_prob >= SLIP_PROB_THRESHOLD) ? 1 : 0;
            int pf = (p_prob >= SLIP_PROB_THRESHOLD) ? 1 : 0;
            opt_flags += cf;
            py_flags += pf;
            if (cf != pf) flag_mismatches++;

            float se = fabsf(out_s4[t] - py_s4[t]);
            if (se > s4_max_e) s4_max_e = se;
            s4_sum_e += se;
            s4_mean += out_s4[t];
        }}
        s4_mean /= (float)L;

        Serial.printf("[%s] Finite=%s | S4 Mean=%.4f | Env[64]=%.4f | Flags=%d/%d (Mismatch=%d)\\n",
                      SCENARIO_NAMES[s], finite_ok ? "YES" : "NO", s4_mean, out_env[64], opt_flags, py_flags, flag_mismatches);
        Serial.printf("  Full-Array Err vs Python -> Env Max=%.2e (MAE=%.2e) | Prob Max=%.2e (MAE=%.2e) | S4 Max=%.2e (MAE=%.2e)\\n",
                      env_max_e, env_sum_e / L, prob_max_e, prob_sum_e / L, s4_max_e, s4_sum_e / L);
    }}

    Serial.println("\\n--- [PHASE 5] UNINSTRUMENTED MULTI-ITERATION BENCHMARK (20 RUNS AFTER WARM-UP) ---");
    for (int w = 0; w < 3; w++) {{
        run_tcn_inference_stage2(test_scenarios_x + (w % NUM_SCENARIOS) * (3 * L));
    }}

    unsigned long times_us[20];
    uint32_t cycles_arr[20];
    for (int i = 0; i < 20; i++) {{
        const float* x_in = test_scenarios_x + (i % NUM_SCENARIOS) * (3 * L);
        uint32_t c0 = ESP.getCycleCount();
        unsigned long t0 = micros();
        run_tcn_inference_stage2(x_in);
        unsigned long dt = micros() - t0;
        uint32_t dc = ESP.getCycleCount() - c0;
        times_us[i] = dt;
        cycles_arr[i] = dc;
    }}

    for (int i = 0; i < 19; i++) {{
        for (int j = i + 1; j < 20; j++) {{
            if (times_us[j] < times_us[i]) {{
                unsigned long tmp = times_us[i]; times_us[i] = times_us[j]; times_us[j] = tmp;
            }}
        }}
    }}
    unsigned long sum_us = 0;
    uint64_t sum_cycles = 0;
    for (int i = 0; i < 20; i++) {{
        sum_us += times_us[i];
        sum_cycles += cycles_arr[i];
    }}
    unsigned long median_us = (times_us[9] + times_us[10]) / 2;
    float mean_us = (float)sum_us / 20.0f;
    float mean_cycles = (float)sum_cycles / 20.0f;

    Serial.printf("  1. Incomplete V2.2 Proxy Baseline (1-Layer, 9 weights):   ~2937 us (2.937 ms) [INCOMPLETE MODEL]\\n");
    Serial.printf("  2. Complete 9-Layer Legacy Unoptimized Baseline:        4528624 us (4528.624 ms)\\n");
    Serial.printf("  3. Complete 9-Layer Stage-1 Loop-Inverted Engine:       1233580 us (1233.580 ms) [3.67x]\\n");
    Serial.printf("  4. Complete 9-Layer Stage-2 Register-Tiled + IRAM Engine (N=20):\\n");
    Serial.printf("     -> Min:    %lu us (%.3f ms)\\n", times_us[0], times_us[0] / 1000.0f);
    Serial.printf("     -> Median: %lu us (%.3f ms)\\n", median_us, median_us / 1000.0f);
    Serial.printf("     -> Mean:   %.1f us (%.3f ms) [~%.0f CPU cycles, %.2f cycles/MAC]\\n",
                  mean_us, mean_us / 1000.0f, mean_cycles, mean_cycles / 1987584.0f);
    Serial.printf("     -> Max:    %lu us (%.3f ms)\\n", times_us[19], times_us[19] / 1000.0f);
    Serial.printf("     -> Total Speedup vs Legacy 9-Layer TCN:  %.2fx faster\\n", 4528624.0f / mean_us);
    Serial.printf("     -> Stage-2 Speedup vs Stage-1 Engine:    %.2fx faster\\n", 1233580.0f / mean_us);
    Serial.printf("  Static Engine Memory: Weights/Scales=16,768 B | Buffers+Scratch=37,120 B | Total=53,888 B (10.1%% of 520 KB SRAM) | Free Heap=%u B\\n",
                  ESP.getFreeHeap());
    Serial.println("==========================================================================");
}}

static int loop_scenario_idx = 0;

void loop() {{
    const int s = loop_scenario_idx % NUM_SCENARIOS;
    loop_scenario_idx++;
    const float* x_in = test_scenarios_x + s * (3 * L);

    unsigned long t0 = micros();
    run_tcn_inference_stage2(x_in);
    unsigned long infer_us = micros() - t0;

    unsigned long p0 = micros();
    float s4_mean = 0.0f, max_prob = 0.0f;
    int slip_count = 0;
    for (int t = 0; t < L; t++) {{
        s4_mean += out_s4[t];
        float c_log = constrain(out_slip_logits[t], -20.0f, 20.0f);
        float prob = 1.0f / (1.0f + expf(-c_log));
        if (prob > max_prob) max_prob = prob;
        if (prob >= SLIP_PROB_THRESHOLD) slip_count++;
    }}
    s4_mean /= (float)L;
    unsigned long post_us = micros() - p0;

    Serial.printf("[LIVE #%d %s] Infer: %lu us (%.3f ms) | Post: %lu us | S4: %.4f | Env[64]: %.4f | MaxProb: %.4f (Slips: %d)\\n",
                  loop_scenario_idx, SCENARIO_NAMES[s], infer_us, infer_us / 1000.0f, post_us, s4_mean, out_env[64], max_prob, slip_count);
    delay(3000);
}}
"""
    sketch_path = "v2_2_repair/models/wokwi_esp32_sketch.ino"
    with open(sketch_path, "w") as f:
        f.write(sketch_code)
    print(f"-> Wrote Stage-2 Register-Tiled & IRAM-accelerated Wokwi sketch to: {sketch_path}")


if __name__ == "__main__":
    main()