"""
Generate synthetic ECG training data for the Healthcare AI Portal.
Creates 12-lead ECG signals for 5 diagnostic classes (NORM, MI, STTC, HYP, CD).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ecg"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# --- Parameters ---
FS = 100  # sampling frequency (matching PTB-XL 100Hz low-res)
DURATION = 10  # seconds
N_SAMPLES = FS * DURATION  # 1000 samples per record
N_LEADS = 12
RECORDS_PER_CLASS = 600
CLASSES = {"NORM": 0, "MI": 1, "STTC": 2, "HYP": 3, "CD": 4}


def generate_base_ecg(n_samples, fs, heart_rate=72):
    """Generate a single-lead ECG-like signal."""
    t = np.linspace(0, n_samples / fs, n_samples)
    beat_freq = heart_rate / 60.0

    # P wave
    p_wave = 0.15 * np.sin(2 * np.pi * beat_freq * t)
    # QRS complex (sharper)
    qrs = 1.2 * np.exp(-0.5 * ((t % (1 / beat_freq) - 0.15) / 0.02) ** 2)
    # T wave
    t_wave = 0.3 * np.sin(2 * np.pi * beat_freq * t - np.pi / 3)

    ecg = p_wave + qrs + t_wave
    ecg += 0.05 * np.random.randn(n_samples)  # noise
    return ecg


def generate_12_lead(base, variation=0.3):
    """Generate 12 leads from a base signal with realistic variations."""
    leads = np.zeros((len(base), N_LEADS))
    # Standard limb leads and precordial leads with different amplitudes/phases
    lead_scales = [1.0, 1.2, 0.8, -0.6, 0.5, 0.7, 0.3, 0.6, 0.9, 1.1, 1.3, 1.0]
    lead_shifts = [0, 2, 4, 6, 3, 5, 8, 7, 5, 3, 2, 1]
    for i in range(N_LEADS):
        shifted = np.roll(base, lead_shifts[i])
        leads[:, i] = lead_scales[i] * shifted + variation * np.random.randn(len(base))
    return leads


def generate_class_signal(label, n_samples, fs):
    """Generate a 12-lead ECG signal for a specific diagnostic class."""
    hr = np.random.uniform(55, 95)

    if label == "NORM":
        base = generate_base_ecg(n_samples, fs, heart_rate=hr)

    elif label == "MI":
        # ST elevation + Q waves
        base = generate_base_ecg(n_samples, fs, heart_rate=hr)
        t = np.linspace(0, n_samples / fs, n_samples)
        st_elevation = 0.4 * np.sin(2 * np.pi * (hr / 60) * t - np.pi / 4)
        base += st_elevation
        base += 0.15 * np.random.randn(n_samples)

    elif label == "STTC":
        # ST/T wave changes — inverted T waves
        base = generate_base_ecg(n_samples, fs, heart_rate=hr)
        t = np.linspace(0, n_samples / fs, n_samples)
        t_inversion = -0.5 * np.sin(2 * np.pi * (hr / 60) * t - np.pi / 3)
        base += t_inversion

    elif label == "HYP":
        # Hypertrophy — higher voltage QRS
        base = generate_base_ecg(n_samples, fs, heart_rate=hr)
        base *= 1.8
        base += 0.1 * np.random.randn(n_samples)

    elif label == "CD":
        # Conduction disturbance — wider QRS, irregular
        base = generate_base_ecg(n_samples, fs, heart_rate=hr * 0.85)
        t = np.linspace(0, n_samples / fs, n_samples)
        wide_qrs = 0.3 * np.sin(2 * np.pi * 3 * (hr / 60) * t)
        base += wide_qrs
        # Add occasional dropped beats
        drop_mask = np.random.choice([0, 1], size=n_samples, p=[0.05, 0.95])
        base *= drop_mask.astype(float)
        base += 0.08 * np.random.randn(n_samples)

    return generate_12_lead(base)


# --- Generate dataset ---
print("Generating synthetic ECG training data...")
print(f"  {RECORDS_PER_CLASS} records × {len(CLASSES)} classes = {RECORDS_PER_CLASS * len(CLASSES)} total")
print(f"  Signal shape per record: ({N_SAMPLES}, {N_LEADS})")

all_signals = []
all_labels = []

for label_name, label_id in CLASSES.items():
    print(f"  Generating {label_name} (class {label_id})...")
    for i in range(RECORDS_PER_CLASS):
        signal = generate_class_signal(label_name, N_SAMPLES, FS)
        all_signals.append(signal)
        all_labels.append(label_id)

X = np.array(all_signals)
y = np.array(all_labels)

# Shuffle
perm = np.random.permutation(len(y))
X = X[perm]
y = y[perm]

# Save
np.save(DATA_DIR / "ecg_signals.npy", X)
np.save(DATA_DIR / "ecg_labels.npy", y)

print(f"\nSaved to {DATA_DIR}:")
print(f"  ecg_signals.npy: {X.shape}")
print(f"  ecg_labels.npy:  {y.shape}")
print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print("Done!")
