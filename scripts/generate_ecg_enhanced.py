"""
Enhanced ECG data generator with realistic PQRST morphology.
"""
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ecg"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

FS = 100
DURATION = 10
N_SAMPLES = FS * DURATION  # 1000
N_LEADS = 12
RECORDS_PER_CLASS = 800
CLASSES = {"NORM": 0, "MI": 1, "STTC": 2, "HYP": 3, "CD": 4}


def pqrst_beat(t_beat, hr, amplitude=1.0):
    """Generate a single PQRST complex with realistic morphology."""
    dur = 1.0 / (hr / 60.0)
    beat = np.zeros_like(t_beat)

    # P wave (small rounded bump)
    p_center = 0.1 * dur
    p_width = 0.025 * dur
    beat += amplitude * 0.12 * np.exp(-0.5 * ((t_beat - p_center) / p_width) ** 2)

    # Q wave (small negative dip)
    q_center = 0.17 * dur
    q_width = 0.008 * dur
    beat -= amplitude * 0.08 * np.exp(-0.5 * ((t_beat - q_center) / q_width) ** 2)

    # R wave (tall sharp peak)
    r_center = 0.19 * dur
    r_width = 0.012 * dur
    beat += amplitude * 1.0 * np.exp(-0.5 * ((t_beat - r_center) / r_width) ** 2)

    # S wave (negative dip after R)
    s_center = 0.22 * dur
    s_width = 0.01 * dur
    beat -= amplitude * 0.15 * np.exp(-0.5 * ((t_beat - s_center) / s_width) ** 2)

    # T wave (broad positive bump)
    t_center = 0.35 * dur
    t_width = 0.04 * dur
    beat += amplitude * 0.25 * np.exp(-0.5 * ((t_beat - t_center) / t_width) ** 2)

    return beat


def generate_ecg_signal(n_samples, fs, hr, amplitude=1.0):
    """Generate a single-lead ECG from PQRST beats."""
    t = np.linspace(0, n_samples / fs, n_samples)
    ecg = np.zeros(n_samples)
    beat_dur = 60.0 / hr
    n_beats = int(n_samples / fs / beat_dur) + 2

    for i in range(n_beats):
        start_time = i * beat_dur
        mask = (t >= start_time) & (t < start_time + beat_dur)
        t_local = t[mask] - start_time
        ecg[mask] += pqrst_beat(t_local, hr, amplitude)

    return ecg


def add_baseline_wander(signal, fs, amplitude=0.1, freq=0.2):
    t = np.linspace(0, len(signal) / fs, len(signal))
    return signal + amplitude * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))


def add_noise(signal, snr_db=30):
    power = np.mean(signal ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    return signal + np.sqrt(noise_power) * np.random.randn(len(signal))


def to_12_lead(base_signal, variation=0.15):
    """Generate 12-lead from a base signal with realistic lead-specific variations."""
    leads = np.zeros((len(base_signal), N_LEADS))
    # Approximate lead relationships
    lead_scales = [1.0, 1.1, 0.9, -0.5, 0.4, 0.6, 0.25, 0.5, 0.8, 1.0, 1.2, 1.05]
    lead_offsets = [0, 1, 3, 5, 2, 4, 7, 6, 4, 3, 2, 1]
    for i in range(N_LEADS):
        shifted = np.roll(base_signal, lead_offsets[i])
        leads[:, i] = lead_scales[i] * shifted + variation * np.random.randn(len(base_signal))
        leads[:, i] = add_baseline_wander(leads[:, i], FS, amplitude=np.random.uniform(0.02, 0.08))
    return leads


def generate_normal(n_samples, fs):
    hr = np.random.uniform(58, 90)
    base = generate_ecg_signal(n_samples, fs, hr, amplitude=np.random.uniform(0.8, 1.2))
    base = add_noise(base, snr_db=np.random.uniform(25, 40))
    return to_12_lead(base)


def generate_mi(n_samples, fs):
    hr = np.random.uniform(60, 95)
    base = generate_ecg_signal(n_samples, fs, hr, amplitude=np.random.uniform(0.8, 1.1))
    t = np.linspace(0, n_samples / fs, n_samples)
    # ST elevation
    beat_dur = 60.0 / hr
    st_elevation = np.zeros(n_samples)
    for i in range(int(n_samples / fs / beat_dur) + 2):
        start = i * beat_dur
        st_start = start + 0.22 * beat_dur
        st_end = start + 0.4 * beat_dur
        mask = (t >= st_start) & (t < st_end)
        st_elevation[mask] += np.random.uniform(0.2, 0.5)
    base += st_elevation
    # Pathological Q waves
    for i in range(int(n_samples / fs / beat_dur) + 2):
        start = i * beat_dur
        q_center = start + 0.17 * beat_dur
        q_mask = np.abs(t - q_center) < 0.015 * beat_dur
        base[q_mask] -= np.random.uniform(0.15, 0.3)
    base = add_noise(base, snr_db=np.random.uniform(22, 35))
    return to_12_lead(base, variation=0.2)


def generate_sttc(n_samples, fs):
    hr = np.random.uniform(55, 85)
    base = generate_ecg_signal(n_samples, fs, hr)
    t = np.linspace(0, n_samples / fs, n_samples)
    beat_dur = 60.0 / hr
    # T wave inversion
    for i in range(int(n_samples / fs / beat_dur) + 2):
        start = i * beat_dur
        t_center = start + 0.35 * beat_dur
        t_width = 0.04 * beat_dur
        t_inv = -0.4 * np.exp(-0.5 * ((t - t_center) / t_width) ** 2)
        base += t_inv
    # ST depression
    st_dep = np.zeros(n_samples)
    for i in range(int(n_samples / fs / beat_dur) + 2):
        start = i * beat_dur
        st_start = start + 0.22 * beat_dur
        st_end = start + 0.32 * beat_dur
        mask = (t >= st_start) & (t < st_end)
        st_dep[mask] -= np.random.uniform(0.1, 0.3)
    base += st_dep
    base = add_noise(base, snr_db=np.random.uniform(25, 38))
    return to_12_lead(base, variation=0.18)


def generate_hyp(n_samples, fs):
    hr = np.random.uniform(62, 88)
    amplitude = np.random.uniform(1.6, 2.2)
    base = generate_ecg_signal(n_samples, fs, hr, amplitude=amplitude)
    # Strain pattern (ST depression + T inversion in lateral leads)
    t = np.linspace(0, n_samples / fs, n_samples)
    beat_dur = 60.0 / hr
    for i in range(int(n_samples / fs / beat_dur) + 2):
        start = i * beat_dur
        t_center = start + 0.35 * beat_dur
        t_width = 0.04 * beat_dur
        strain = -0.2 * np.exp(-0.5 * ((t - t_center) / t_width) ** 2)
        base += strain
    base = add_noise(base, snr_db=np.random.uniform(25, 38))
    return to_12_lead(base, variation=0.15)


def generate_cd(n_samples, fs):
    hr = np.random.uniform(45, 75)
    base = generate_ecg_signal(n_samples, fs, hr)
    t = np.linspace(0, n_samples / fs, n_samples)
    beat_dur = 60.0 / hr
    # Widen QRS (add extra component)
    for i in range(int(n_samples / fs / beat_dur) + 2):
        start = i * beat_dur
        r_center = start + 0.19 * beat_dur
        extra_r = 0.35 * np.exp(-0.5 * ((t - (r_center + 0.03 * beat_dur)) / (0.02 * beat_dur)) ** 2)
        base += extra_r
    # Occasional dropped beats
    drop_prob = np.random.uniform(0.03, 0.08)
    for i in range(int(n_samples / fs / beat_dur) + 2):
        if np.random.random() < drop_prob:
            start = i * beat_dur
            mask = (t >= start) & (t < start + beat_dur)
            base[mask] *= 0.1
    base = add_noise(base, snr_db=np.random.uniform(22, 35))
    return to_12_lead(base, variation=0.22)


generators = {
    "NORM": generate_normal,
    "MI": generate_mi,
    "STTC": generate_sttc,
    "HYP": generate_hyp,
    "CD": generate_cd,
}

print("Generating enhanced synthetic ECG training data...")
print(f"  {RECORDS_PER_CLASS} records x {len(CLASSES)} classes = {RECORDS_PER_CLASS * len(CLASSES)} total")

all_signals = []
all_labels = []

for label_name, label_id in CLASSES.items():
    gen = generators[label_name]
    print(f"  Generating {label_name} (class {label_id})...")
    for i in range(RECORDS_PER_CLASS):
        signal = gen(N_SAMPLES, FS)
        all_signals.append(signal)
        all_labels.append(label_id)

X = np.array(all_signals)
y = np.array(all_labels)

perm = np.random.permutation(len(y))
X = X[perm]
y = y[perm]

np.save(DATA_DIR / "ecg_signals.npy", X)
np.save(DATA_DIR / "ecg_labels.npy", y)

print(f"\nSaved to {DATA_DIR}:")
print(f"  ecg_signals.npy: {X.shape}")
print(f"  ecg_labels.npy:  {y.shape}")
print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print("Done!")
