import numpy as np

def generate_space_weather_payload_signal(num_samples=1000):
    """
    Generates 50 Hz Dual-Band (L1 & L5) GNSS Receiver Observables:
    Inputs (3 Channels):
      - Ch 0: L1 Amplitude with Rytov/Nakagami scintillation, multipath & AWGN
      - Ch 1: L5 Amplitude with heavier 1/f^2 ionospheric dispersion
      - Ch 2: Carrier Phase Tracking Residual with LEO Doppler, Phase Scintillation (up to 0.81 rad), and Cycle Slips
    Targets (3 Channels):
      - Ch 0: Clean Reference Carrier Envelope
      - Ch 1: Exact Cycle Slip Binary Detection Mask (0.0 = Nominal, 1.0 = Phase Jump)
      - Ch 2: True S4 Scintillation Severity Profile (0.1 to 1.2)
    """
    t = np.linspace(0, 1, num_samples)

    # 1. Orbital LEO Doppler Shift & Base Carrier
    doppler_slope = np.random.uniform(-15.0, 15.0)
    instantaneous_freq = 50.0 + (doppler_slope * t)
    base_carrier = np.exp(1j * 2 * np.pi * instantaneous_freq * t)
    clean_envelope = np.abs(base_carrier).astype(np.float32)

    # 2. Space Weather Severity: S4 Index (0.1 quiet to 1.2 extreme storm) & Sigma_Phi (up to 0.81 rad)
    s4_index = np.random.uniform(0.1, 1.2)
    sigma_phi = np.random.uniform(0.05, 0.81) * (s4_index / 1.2)

    # 3. Differential Dual-Frequency Amplitude Scintillation (L5 fades harder than L1)
    fade_l1 = np.clip(np.random.rayleigh(scale=max(0.15, 1.0 - (s4_index * 0.25)), size=num_samples), 0.1, 2.2)
    fade_l5 = np.clip(np.random.rayleigh(scale=max(0.15, 1.0 - (s4_index * 0.35)), size=num_samples), 0.1, 2.5)

    # 4. Multipath Reflections & Thermal Noise (AWGN)
    delay = np.random.randint(5, 25)
    mp_att = np.random.uniform(0.15, 0.5)

    sig_l1 = base_carrier * fade_l1
    sig_l5 = base_carrier * fade_l5

    mp_l1 = np.zeros_like(sig_l1)
    mp_l5 = np.zeros_like(sig_l5)
    mp_l1[delay:] = sig_l1[:-delay] * mp_att
    mp_l5[delay:] = sig_l5[:-delay] * mp_att

    noise_l1 = 0.08 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    noise_l5 = 0.12 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

    obs_l1_amp = np.abs(sig_l1 + mp_l1 + noise_l1).astype(np.float32)
    obs_l5_amp = np.abs(sig_l5 + mp_l5 + noise_l5).astype(np.float32)

    # 5. Carrier Phase Scintillation & Cycle Slip Injection (occurs when S4 > 0.5)
    phase_noise = np.random.normal(0, sigma_phi, size=num_samples).astype(np.float32)
    cycle_slip_mask = np.zeros(num_samples, dtype=np.float32)

    if s4_index > 0.5:
        num_slips = np.random.randint(1, 5)
        slip_indices = np.random.choice(range(50, num_samples - 50), size=num_slips, replace=False)
        for idx in slip_indices:
            # Mark a 10-sample window around the cycle slip for robust TCN detection
            cycle_slip_mask[idx:idx + 10] = 1.0
            # Inject a sudden 1-cycle (normalized) phase discontinuity spike
            phase_noise[idx:idx + 10] += np.random.choice([-1.5, 1.5])

    # Normalize phase observable for stable neural network input
    obs_phase = np.clip(phase_noise, -3.0, 3.0).astype(np.float32)

    # 6. Continuous S4 Severity Target Channel
    s4_target_channel = np.full(num_samples, s4_index, dtype=np.float32)

    # Stack into (3, 1000) tensors
    corrupted_input = np.stack([obs_l1_amp, obs_l5_amp, obs_phase], axis=0)
    target_output = np.stack([clean_envelope, cycle_slip_mask, s4_target_channel], axis=0)

    return corrupted_input, target_output