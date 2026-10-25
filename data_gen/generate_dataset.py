import os
import numpy as np

try:
    from data_gen.channel_models import generate_space_weather_payload_signal
except ImportError:
    from channel_models import generate_space_weather_payload_signal

def create_versioned_dataset(version_tag="v2.2_space_weather_telemetry", total_signals=10000, batch_size=1000):
    output_dir = f"data/processed/{version_tag}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating dataset version [{version_tag}] with {total_signals} multi-task signals...")

    for i in range(0, total_signals, batch_size):
        X_batch = []
        y_batch = []

        for _ in range(batch_size):
            corrupted, target = generate_space_weather_payload_signal()
            X_batch.append(corrupted)  # Shape: (3, 1000)
            y_batch.append(target)     # Shape: (3, 1000)

        batch_idx = i // batch_size
        file_path = os.path.join(output_dir, f"batch_{batch_idx}.npz")
        np.savez_compressed(file_path, X=np.array(X_batch, dtype=np.float32), y=np.array(y_batch, dtype=np.float32))
        print(f"-> Saved {version_tag} batch {batch_idx + 1} ({min(i + batch_size, total_signals)}/{total_signals} signals)")

if __name__ == "__main__":
    create_versioned_dataset()