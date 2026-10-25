import os
import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from v2_2_repair.models.baseline_tcn import BaselineTCN
except ImportError:
    from baseline_tcn import BaselineTCN

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_split(base_dir, split_name):
    pattern = os.path.join(base_dir, split_name, "batch_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No batch files found at {pattern}")
    X_list, y_list = [], []
    for f in files:
        d = np.load(f)
        X_list.append(d["X"])
        y_list.append(d["y"])
    X = torch.tensor(np.concatenate(X_list, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.float32)
    return X, y

def compute_classification_metrics(y_true_flat, y_pred_flat):
    tp = int(np.sum((y_true_flat == 1) & (y_pred_flat == 1)))
    fp = int(np.sum((y_true_flat == 0) & (y_pred_flat == 1)))
    tn = int(np.sum((y_true_flat == 0) & (y_pred_flat == 0)))
    fn = int(np.sum((y_true_flat == 1) & (y_pred_flat == 0)))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "precision": float(precision), "recall": float(recall), "f1": float(f1)}

def evaluate_conventional_baselines(X_np, y_np):
    N, _, L = X_np.shape
    kernel = np.ones(25, dtype=np.float32) / 25.0
    env_pred = np.zeros((N, L), dtype=np.float32)
    for i in range(N):
        env_pred[i] = np.convolve(X_np[i, 0], kernel, mode="same")

    t1_mse = float(np.mean((env_pred - y_np[:, 0, :]) ** 2))
    t1_mae = float(np.mean(np.abs(env_pred - y_np[:, 0, :])))

    slip_pred = (np.abs(X_np[:, 2, :]) > 1.15).astype(np.int32).ravel()
    slip_true = (y_np[:, 1, :] > 0.5).astype(np.int32).ravel()
    t2_metrics = compute_classification_metrics(slip_true, slip_pred)

    raw_I = (X_np[:, 0, :] / (env_pred + 1e-5)) ** 2
    s4_base = np.sqrt(np.var(raw_I, axis=1) / (np.mean(raw_I, axis=1) ** 2 + 1e-8))
    s4_true = y_np[:, 2, 0]
    t3_mae = float(np.mean(np.abs(s4_base - s4_true)))
    t3_rmse = float(np.sqrt(np.mean((s4_base - s4_true) ** 2)))

    return {
        "task1_envelope": {"MSE": t1_mse, "MAE": t1_mae},
        "task2_cycle_slip": t2_metrics,
        "task3_s4": {"MAE": t3_mae, "RMSE": t3_rmse},
    }

def evaluate_model(model, dataloader, pos_weight_tensor, threshold=0.5):
    model.eval()
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for bX, by in dataloader:
            out = model(bX)
            loss_env = mse_fn(out[:, 0, :], by[:, 0, :])
            loss_slip = bce_fn(out[:, 1, :], by[:, 1, :])
            loss_s4 = mse_fn(out[:, 2, :], by[:, 2, :])
            loss = loss_env + 0.5 * loss_slip + 2.0 * loss_s4
            total_loss += loss.item() * bX.size(0)
            all_preds.append(out.cpu().numpy())
            all_targets.append(by.cpu().numpy())

    P = np.concatenate(all_preds, axis=0)
    Y = np.concatenate(all_targets, axis=0)

    t1_mse = float(np.mean((P[:, 0, :] - Y[:, 0, :]) ** 2))
    t1_mae = float(np.mean(np.abs(P[:, 0, :] - Y[:, 0, :])))

    probs = 1.0 / (1.0 + np.exp(-np.clip(P[:, 1, :], -20, 20)))
    y_true_cls = (Y[:, 1, :] > 0.5).astype(np.int32).ravel()
    y_pred_cls = (probs >= threshold).astype(np.int32).ravel()
    t2_metrics = compute_classification_metrics(y_true_cls, y_pred_cls)

    s4_pred = np.mean(P[:, 2, :], axis=1)
    s4_true = Y[:, 2, 0]
    t3_mae = float(np.mean(np.abs(s4_pred - s4_true)))
    t3_rmse = float(np.sqrt(np.mean((s4_pred - s4_true) ** 2)))

    severe_mask = s4_true >= 0.60
    t3_severe_mae = float(np.mean(np.abs(s4_pred[severe_mask] - s4_true[severe_mask]))) if np.any(severe_mask) else 0.0
    t3_severe_rmse = float(np.sqrt(np.mean((s4_pred[severe_mask] - s4_true[severe_mask]) ** 2))) if np.any(severe_mask) else 0.0

    return {
        "combined_val_loss": total_loss / len(Y),
        "task1_envelope": {"MSE": t1_mse, "MAE": t1_mae},
        "task2_cycle_slip": t2_metrics,
        "task3_s4": {
            "MAE": t3_mae,
            "RMSE": t3_rmse,
            "severe_S4_ge_0.6_MAE": t3_severe_mae,
            "severe_S4_ge_0.6_RMSE": t3_severe_rmse,
        },
        "raw_probs": probs,
        "raw_y_cls": y_true_cls,
        "s4_pred": s4_pred,
        "s4_true": s4_true,
    }

def train_and_evaluate():
    set_all_seeds(42)
    base_dir = "v2_2_repair/data/processed/v2.2_repaired"
    X_train, y_train = load_split(base_dir, "train")
    X_val, y_val = load_split(base_dir, "val")
    X_test, y_test = load_split(base_dir, "test")

    print(f"Loaded Splits -> Train: {tuple(X_train.shape)}, Val: {tuple(X_val.shape)}, Test: {tuple(X_test.shape)}")

    pos_frac = float((y_train[:, 1, :] > 0.5).float().mean().item())
    pos_weight_val = min(25.0, (1.0 - pos_frac) / max(pos_frac, 1e-4))
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32)
    print(f"Task 2 Train Positive Rate: {pos_frac*100:.2f}% -> Using BCE pos_weight={pos_weight_val:.2f}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    model = BaselineTCN(input_channel=3, output_channel=3)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0015)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_ckpt_path = "v2_2_repair/models/best_baseline_tcn.pth"
    epochs = 15

    print("\n--- TRAINING REPAIRED V2.2 MODEL (CHECKPOINT SELECTED BY VAL LOSS) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bX, by in train_loader:
            optimizer.zero_grad()
            out = model(bX)
            loss_env = mse_fn(out[:, 0, :], by[:, 0, :])
            loss_slip = bce_fn(out[:, 1, :], by[:, 1, :])
            loss_s4 = mse_fn(out[:, 2, :], by[:, 2, :])
            loss = loss_env + 0.5 * loss_slip + 2.0 * loss_s4
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bX.size(0)

        train_loss /= len(X_train)
        val_res = evaluate_model(model, val_loader, pos_weight_tensor, threshold=0.5)
        val_loss = val_res["combined_val_loss"]
        scheduler.step(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            marker = " [BEST VAL CHECKPOINT SAVED]"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val Env MAE: {val_res['task1_envelope']['MAE']:.4f} | "
            f"Val Slip F1: {val_res['task2_cycle_slip']['f1']:.4f} | "
            f"Val S4 MAE: {val_res['task3_s4']['MAE']:.4f}{marker}"
        )

    model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
    val_eval = evaluate_model(model, val_loader, pos_weight_tensor, threshold=0.5)
    best_th, best_f1 = 0.5, -1.0
    for th in [0.35, 0.45, 0.55, 0.65, 0.75, 0.82]:
        m = compute_classification_metrics(val_eval["raw_y_cls"], (val_eval["raw_probs"] >= th).astype(np.int32).ravel())
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
    print(f"\nSelected cycle-slip probability threshold on Validation Set: th={best_th:.2f} (Val F1={best_f1:.4f})")

    test_res = evaluate_model(model, test_loader, pos_weight_tensor, threshold=best_th)
    base_res = evaluate_conventional_baselines(X_test.numpy(), y_test.numpy())

    print("\n======================================================================")
    print("FINAL HELD-OUT TEST SET EVALUATION (1,000 UNSEEN SCENARIOS)")
    print("======================================================================")
    print(f"TASK 1 (Signal Recovery)   | TCN MSE: {test_res['task1_envelope']['MSE']:.6f}, MAE: {test_res['task1_envelope']['MAE']:.4f} "
          f"| Baseline Moving-Avg MSE: {base_res['task1_envelope']['MSE']:.6f}, MAE: {base_res['task1_envelope']['MAE']:.4f}")
    c = test_res["task2_cycle_slip"]
    bc = base_res["task2_cycle_slip"]
    print(f"TASK 2 (Cycle-Slip Class.) | TCN Precision: {c['precision']:.4f}, Recall: {c['recall']:.4f}, F1: {c['f1']:.4f} "
          f"(TP={c['TP']}, FP={c['FP']}, TN={c['TN']}, FN={c['FN']})")
    print(f"                           | Baseline Precision: {bc['precision']:.4f}, Recall: {bc['recall']:.4f}, F1: {bc['f1']:.4f} ")
    print(f"TASK 3 (S4 Estimation)     | TCN MAE: {test_res['task3_s4']['MAE']:.4f}, RMSE: {test_res['task3_s4']['RMSE']:.4f} "
          f"(Severe S4>=0.6 MAE: {test_res['task3_s4']['severe_S4_ge_0.6_MAE']:.4f})")
    print(f"                           | Baseline Detrended-S4 MAE: {base_res['task3_s4']['MAE']:.4f}, RMSE: {base_res['task3_s4']['RMSE']:.4f}")
    print(f"\nSample #0 (Unseen Test)    | True S4: {test_res['s4_true'][0]:.4f} | Repaired TCN Pred S4: {test_res['s4_pred'][0]:.4f}")

    report = {
        "selected_threshold_from_val": best_th,
        "held_out_test_tcn": {
            "task1_envelope": test_res["task1_envelope"],
            "task2_cycle_slip": test_res["task2_cycle_slip"],
            "task3_s4": test_res["task3_s4"],
        },
        "held_out_test_baselines": base_res,
    }
    with open("v2_2_repair/models/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved full test evaluation report to v2_2_repair/models/evaluation_report.json")

if __name__ == "__main__":
    train_and_evaluate()