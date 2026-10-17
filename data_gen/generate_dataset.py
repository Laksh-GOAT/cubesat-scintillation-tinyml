import os
import numpy as np
import scipy.signal as signal

def generate_scintillation_signal(num_samples=5000, s4_index=0.5, sampling_rate=100.0):
    """
    Simulates a GNSS carrier phase signal affected by ionospheric scintillation.
    - s4_index: Amplitude scintillation index (0.2 = mild, 0.8 = severe solar storm).
    - sampling_rate: 100 Hz tracking loop frequency.
    """
    t = np.arange(num_samples) / sampling_rate
    
    # 1. Generate clean carrier phase (e.g., smooth satellite motion doppler shift)
    clean_phase = 0.5 * np.sin(2 * np.pi * 0.1 * t) 
    
    # 2. Generate ionospheric phase scintillation using filtered white noise (power law spectrum)
    # Ionospheric irregularities typically follow a power spectral density (PSD) slope around -2.5 to -3.0
    white_noise = np.random.randn(num_samples)
    b, a = signal.butter(2, 5.0 / (sampling_rate / 2), btype='low') # cutoff at 5 Hz
    phase_jitter = signal.lfilter(b, a, white_noise)
    # Scale phase jitter magnitude based on severity
    phase_jitter = phase_jitter * (s4_index * 1.5)
    
    # 3. Corrupted / Distorted phase seen by the low-cost CubeSat receiver
    corrupted_phase = clean_phase + phase_jitter
    
    return clean_phase, corrupted_phase, phase_jitter

def run_batch_simulation(num_scenarios=500, samples_per_scenario=2000):
    """
    Automates running multiple scintillation scenarios across different S4 severity levels
    and saves them directly into structured dataset arrays.
    """
    print("🚀 Starting Automated Scintillation Data Generation...")
    
    X_data = [] # Input: Corrupted phase (what the cheap receiver sees)
    y_data = [] # Target: True phase error to be filtered out (or clean phase)
    metadata = [] # S4 index tracking for benchmark evaluation later
    
    s4_levels = [0.1, 0.3, 0.5, 0.7, 0.9] # From quiet ionosphere to severe plasma bubbles
    
    for i in range(num_scenarios):
        # Randomly pick an S4 severity level for this scenario
        s4 = np.random.choice(s4_levels)
        
        clean, corrupted, error = generate_scintillation_signal(
            num_samples=samples_per_scenario, 
            s4_index=s4
        )
        
        X_data.append(corrupted)
        y_data.append(clean) # We want the AI to predict the clean signal
        metadata.append(s4)
        
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)
    metadata = np.array(metadata, dtype=np.float32)
    
    # Automatically save into processed data directory for training
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_data[:int(0.8 * num_scenarios)])
    np.save("data/processed/y_train.npy", y_data[:int(0.8 * num_scenarios)])
    np.save("data/processed/X_val.npy", X_data[int(0.8 * num_scenarios):])
    np.save("data/processed/y_val.npy", y_data[int(0.8 * num_scenarios):])
    np.save("data/processed/metadata.npy", metadata)
    
    print("✅ Simulation complete! Dataset successfully saved to `data/processed/`")
    print(f"📊 Training shape: {X_data[:int(0.8 * num_scenarios)].shape}")

if __name__ == "__main__":
    run_batch_simulation()