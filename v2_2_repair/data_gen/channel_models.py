import numpy as np

# Physical GPS Carrier Frequencies (Hz)
F_L1_HZ = 1575.42e6
F_L5_HZ = 1176.45e6
L5_TO_L1_SCINT_SCALE = (F_L1_HZ / F_L5_HZ) ** 0.75  # Amplitude scintillation scaling

def _correlated_nakagami_fading(rng, num_samples, target_s4_regime, corr_len_samples):
    """
    Generates temporally correlated amplitude fading whose intensity I = A^2
    follows a Gamma distribution with shape m = 1 / S4^2 and unit mean.
    """
    m = max(0.65, 1.0 / (target_s4_regime ** 2 + 1e-6))
    raw_intensity = rng.gamma(shape=m, scale=1.0 / m, size=num_samples + corr_len_samples - 1)
    if corr_len_samples > 1:
        kernel = np.hanning(corr_len_samples + 2)[1:-1]
        kernel = kernel / np.sum(kernel)
        smoothed = np.convolve(raw_intensity, kernel, mode="valid")
        emp_mean = np.mean(smoothed)
        emp_std = np.std(smoothed) + 1e-8
        desired_std = emp_mean * target_s4_regime
        intensity = emp_mean + (smoothed - emp_mean) * (desired_std / emp_std)
        intensity = np.clip(intensity, 0.02, 6.0)
    else:
        intensity = np.clip(raw_intensity[:num_samples], 0.02, 6.0)

    amplitude_fade = np.sqrt(intensity).astype(np.float32)
    realized_I = amplitude_fade ** 2
    realized_s4 = float(np.sqrt(np.var(realized_I) / (np.mean(realized_I) ** 2 + 1e-8)))
    return amplitude_fade, realized_s4

def generate_space_weather_payload_signal(num_samples=1000, seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)

    t = np.linspace(0.0, 20.0, num_samples, dtype=np.float32)

    # Task 1 Target: Genuine time-varying clean reference carrier envelope A_clean(t)
    base_level = rng.uniform(0.85, 1.15)
    mod_amp1 = rng.uniform(0.12, 0.28)
    mod_amp2 = rng.uniform(0.05, 0.15)
    freq1 = rng.uniform(0.05, 0.20)
    freq2 = rng.uniform(0.20, 0.45)
    phi1 = rng.uniform(0, 2 * np.pi)
    phi2 = rng.uniform(0, 2 * np.pi)

    clean_envelope = (
        base_level
        + mod_amp1 * np.sin(2 * np.pi * freq1 * t + phi1)
        + mod_amp2 * np.cos(2 * np.pi * freq2 * t + phi2)
    ).astype(np.float32)

    # Task 3 Physics: Correlated Ionospheric Scintillation Fading & Exact Realized S4
    s4_regime = float(rng.uniform(0.08, 1.15))
    corr_len = int(rng.integers(9, 35))

    fade_l1, realized_s4_l1 = _correlated_nakagami_fading(rng, num_samples, s4_regime, corr_len)
    s4_l5_regime = min(1.35, s4_regime * L5_TO_L1_SCINT_SCALE)
    fade_l5, realized_s4_l5 = _correlated_nakagami_fading(rng, num_samples, s4_l5_regime, corr_len)

    doppler_hz = rng.uniform(-12.0, 12.0)
    doppler_rate = rng.uniform(-1.5, 1.5)
    inst_phase = 2 * np.pi * (doppler_hz * t + 0.5 * doppler_rate * (t ** 2))
    unit_carrier = np.exp(1j * inst_phase)

    sig_l1 = clean_envelope * fade_l1 * unit_carrier
    sig_l5 = clean_envelope * fade_l5 * unit_carrier

    # Specular Multipath Reflection
    mp_delay = int(rng.integers(4, 25))
    mp_coeff = float(rng.uniform(0.08, 0.35))
    mp_phase = float(rng.uniform(0, 2 * np.pi))

    mp_l1 = np.zeros_like(sig_l1)
    mp_l5 = np.zeros_like(sig_l5)
    mp_l1[mp_delay:] = sig_l1[:-mp_delay] * mp_coeff * np.exp(1j * mp_phase)
    mp_l5[mp_delay:] = sig_l5[:-mp_delay] * (mp_coeff * 1.1) * np.exp(1j * (mp_phase + 0.4))

    # Thermal Receiver Noise
    noise_std_l1 = float(rng.uniform(0.03, 0.12))
    noise_std_l5 = noise_std_l1 * 1.15
    noise_l1 = noise_std_l1 * (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples))
    noise_l5 = noise_std_l5 * (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples))

    obs_l1_amp = np.abs(sig_l1 + mp_l1 + noise_l1).astype(np.float32)
    obs_l5_amp = np.abs(sig_l5 + mp_l5 + noise_l5).astype(np.float32)

    # Task 2 Physics: Carrier Phase Residual & Exact Cycle-Slip Injection
    sigma_phi = float(rng.uniform(0.04, 0.55) * (0.3 + 0.7 * (realized_s4_l1 / 1.0)))
    raw_phase = rng.normal(0.0, sigma_phi, size=num_samples).astype(np.float32)
    phase_kernel = np.ones(5, dtype=np.float32) / np.sqrt(5.0)
    obs_phase = np.convolve(raw_phase, phase_kernel, mode="same").astype(np.float32)

    cycle_slip_mask = np.zeros(num_samples, dtype=np.float32)
    slip_indices = []
    slip_window = 10

    if realized_s4_l1 > 0.40:
        num_slips = int(rng.integers(1, 5))
        candidates = np.arange(40, num_samples - 40, 25)
        chosen = rng.choice(candidates, size=min(num_slips, len(candidates)), replace=False)
        for idx in sorted(chosen.tolist()):
            slip_indices.append(int(idx))
            cycle_slip_mask[idx : idx + slip_window] = 1.0
            jump_sign = rng.choice([-1.0, 1.0])
            jump_mag = float(rng.uniform(1.15, 2.10))
            obs_phase[idx : idx + slip_window] += jump_sign * jump_mag

    obs_phase = np.clip(obs_phase, -4.0, 4.0).astype(np.float32)
    s4_target_channel = np.full(num_samples, realized_s4_l1, dtype=np.float32)

    corrupted_input = np.stack([obs_l1_amp, obs_l5_amp, obs_phase], axis=0)
    target_output = np.stack([clean_envelope, cycle_slip_mask, s4_target_channel], axis=0)

    meta = {
        "realized_s4_l1": realized_s4_l1,
        "realized_s4_l5": realized_s4_l5,
        "s4_regime_param": s4_regime,
        "slip_indices": slip_indices,
        "slip_window": slip_window,
        "mp_coeff": mp_coeff,
        "noise_std_l1": noise_std_l1,
    }
    validate_sample(corrupted_input, target_output, meta, num_samples)
    return corrupted_input, target_output, meta

def validate_sample(corrupted_input, target_output, meta, num_samples=1000):
    assert corrupted_input.shape == (3, num_samples), f"Bad input shape: {corrupted_input.shape}"
    assert target_output.shape == (3, num_samples), f"Bad target shape: {target_output.shape}"
    assert np.all(np.isfinite(corrupted_input)), "Non-finite values in corrupted_input"
    assert np.all(np.isfinite(target_output)), "Non-finite values in target_output"

    env = target_output[0]
    assert np.min(env) > 0.1 and np.max(env) < 2.5, f"Envelope out of bounds: [{env.min()}, {env.max()}]"
    assert np.std(env) > 0.04, f"Task 1 target is nearly constant (std={np.std(env):.5f})"

    mask = target_output[1]
    unique_vals = np.unique(mask)
    assert np.all(np.isin(unique_vals, [0.0, 1.0])), f"Mask has non-binary values: {unique_vals}"
    expected_ones = len(meta["slip_indices"]) * meta["slip_window"]
    assert int(np.sum(mask)) == expected_ones, f"Mask sum {np.sum(mask)} != expected {expected_ones}"

    s4_chan = target_output[2]
    assert np.allclose(s4_chan, meta["realized_s4_l1"], atol=1e-6), "Task 3 channel mismatch with realized S4"
    assert 0.02 <= meta["realized_s4_l1"] <= 1.60, f"Realized S4 out of range: {meta['realized_s4_l1']}"

if __name__ == "__main__":
    print("Running channel_models.py sanity checks...")
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    x1, y1, m1 = generate_space_weather_payload_signal(rng=rng_a)
    x2, y2, m2 = generate_space_weather_payload_signal(rng=rng_b)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2), "Reproducibility seed check failed!"

    s4_regimes, s4_realized, env_stds = [], [], []
    for s in range(200):
        _, y_s, m_s = generate_space_weather_payload_signal(seed=1000 + s)
        s4_regimes.append(m_s["s4_regime_param"])
        s4_realized.append(m_s["realized_s4_l1"])
        env_stds.append(float(np.std(y_s[0])))

    corr = np.corrcoef(s4_regimes, s4_realized)[0, 1]
    print(f"[PASS] Reproducibility & Shape Checks: X={x1.shape}, y={y1.shape}")
    print(f"[PASS] Task 1 Clean Envelope Mean Std: {np.mean(env_stds):.4f} (Non-constant verified)")
    print(f"[PASS] Task 3 Realized S4 Range: [{min(s4_realized):.3f}, {max(s4_realized):.3f}], Corr with regime r={corr:.4f}")