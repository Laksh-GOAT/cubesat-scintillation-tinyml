import os
import json
import hashlib
import numpy as np

try:
    from data_gen.channel_models import generate_space_weather_payload_signal
except ImportError:
    from channel_models import generate_space_weather_payload_signal

def _array_hash(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()

def generate_repaired_splits(
    base_dir="v2_2_repair/data/processed/v2.2_repaired",
    split_counts=None,
    batch_size=1000,
):
    if split_counts is None:
        # Scaled to 50,000 signals (80/10/10 split)
        split_counts = {
            "train": {"count": 40000, "base_seed": 100000},
            "val":   {"count": 5000, "base_seed": 500000},
            "test":  {"count": 5000, "base_seed": 900000},
        }

    os.makedirs(base_dir, exist_ok=True)
    seen_hashes = set()
    metadata = {"splits": {}, "regeneration_command": "python v2_2_repair/data_gen/generate_dataset.py"}

    for split_name, cfg in split_counts.items():
        total = cfg["count"]
        base_seed = cfg["base_seed"]
        split_dir = os.path.join(base_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"Generating [{split_name}] split: {total} signals (base_seed={base_seed})...")
        s4_values = []
        slip_sample_count = 0

        # Memory safe loop: Processes exactly 1,000 signals per cycle, saves, and clears RAM.
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            X_batch, y_batch, seeds_batch = [], [], []

            for sample_idx in range(start_idx, end_idx):
                sample_seed = base_seed + sample_idx
                rng = np.random.default_rng(sample_seed)
                corrupted, target, meta = generate_space_weather_payload_signal(rng=rng)

                h = _array_hash(corrupted)
                if h in seen_hashes:
                    raise RuntimeError(f"Duplicate signal hash detected at seed {sample_seed}!")
                seen_hashes.add(h)

                X_batch.append(corrupted)
                y_batch.append(target)
                seeds_batch.append(sample_seed)
                s4_values.append(meta["realized_s4_l1"])
                if len(meta["slip_indices"]) > 0:
                    slip_sample_count += 1

            batch_id = start_idx // batch_size
            file_path = os.path.join(split_dir, f"batch_{batch_id}.npz")
            np.savez_compressed(
                file_path,
                X=np.array(X_batch, dtype=np.float32),
                y=np.array(y_batch, dtype=np.float32),
                seeds=np.array(seeds_batch, dtype=np.int64),
            )
            print(f"  -> Saved {split_name}/batch_{batch_id}.npz ({end_idx}/{total})")

        metadata["splits"][split_name] = {
            "num_samples": total,
            "base_seed": base_seed,
            "s4_min": float(np.min(s4_values)),
            "s4_max": float(np.max(s4_values)),
            "s4_mean": float(np.mean(s4_values)),
            "s4_std": float(np.std(s4_values)),
            "signals_with_cycle_slips": int(slip_sample_count),
        }

    meta_path = os.path.join(base_dir, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved split metadata and verification log to {meta_path}")

if __name__ == "__main__":
    generate_repaired_splits()