"""
ECG signal preprocessing and feature extraction utilities.
Used by both the training notebooks and the Streamlit app.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=500, order=3):
    """Apply bandpass filter to remove baseline wander and high-freq noise."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0)


def normalize_signal(signal):
    """Z-score normalization of ECG signal."""
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    std[std == 0] = 1  # avoid division by zero
    return (signal - mean) / std


def segment_beats(signal, fs=500, beat_length=250):
    """
    Detect R-peaks and segment individual heartbeats.
    Returns list of fixed-length beat segments.
    """
    # Use lead II (index 1) for R-peak detection if multi-lead
    if signal.ndim > 1:
        detect_signal = signal[:, 1]
    else:
        detect_signal = signal

    # Find R-peaks
    distance = int(0.4 * fs)  # minimum 400ms between beats
    height = np.mean(detect_signal) + 0.5 * np.std(detect_signal)
    peaks, _ = find_peaks(detect_signal, distance=distance, height=height)

    # Extract fixed-length segments centered on each R-peak
    half = beat_length // 2
    beats = []
    for peak in peaks:
        start = peak - half
        end = peak + half
        if start >= 0 and end <= len(signal):
            if signal.ndim > 1:
                beats.append(signal[start:end, :])
            else:
                beats.append(signal[start:end])

    return np.array(beats) if beats else np.array([])


def compute_heart_rate(signal, fs=500):
    """Compute heart rate from R-R intervals."""
    if signal.ndim > 1:
        detect_signal = signal[:, 1]
    else:
        detect_signal = signal

    distance = int(0.4 * fs)
    height = np.mean(detect_signal) + 0.5 * np.std(detect_signal)
    peaks, _ = find_peaks(detect_signal, distance=distance, height=height)

    if len(peaks) < 2:
        return 0.0

    rr_intervals = np.diff(peaks) / fs  # in seconds
    heart_rate = 60.0 / np.mean(rr_intervals)
    return round(heart_rate, 1)


def compute_hrv_features(signal, fs=500):
    """Compute basic HRV (Heart Rate Variability) features."""
    if signal.ndim > 1:
        detect_signal = signal[:, 1]
    else:
        detect_signal = signal

    distance = int(0.4 * fs)
    height = np.mean(detect_signal) + 0.5 * np.std(detect_signal)
    peaks, _ = find_peaks(detect_signal, distance=distance, height=height)

    if len(peaks) < 3:
        return {"sdnn": 0, "rmssd": 0, "mean_rr": 0, "hr_bpm": 0}

    rr = np.diff(peaks) / fs * 1000  # in milliseconds

    return {
        "sdnn": round(np.std(rr), 2),
        "rmssd": round(np.sqrt(np.mean(np.diff(rr) ** 2)), 2),
        "mean_rr": round(np.mean(rr), 2),
        "hr_bpm": round(60000 / np.mean(rr), 1),
    }


def load_ptbxl_record(record_path, fs=500):
    """
    Load a single PTB-XL record using wfdb.
    Returns numpy array of shape (signal_length, 12).
    """
    import wfdb

    record = wfdb.rdrecord(record_path)
    signal = record.p_signal  # shape: (5000, 12) for 500Hz 10-sec recording
    return signal


def prepare_ecg_for_model(signal, fs=500, target_length=1000):
    """
    Full preprocessing pipeline for a single ECG signal.
    1. Bandpass filter
    2. Normalize
    3. Resize to target length
    Returns array ready for model input.
    """
    # Filter
    filtered = bandpass_filter(signal, fs=fs)

    # Normalize
    normalized = normalize_signal(filtered)

    # Resize if needed (simple interpolation)
    if len(normalized) != target_length:
        from scipy.interpolate import interp1d

        x_old = np.linspace(0, 1, len(normalized))
        x_new = np.linspace(0, 1, target_length)
        if normalized.ndim > 1:
            resized = np.zeros((target_length, normalized.shape[1]))
            for i in range(normalized.shape[1]):
                f = interp1d(x_old, normalized[:, i])
                resized[:, i] = f(x_new)
        else:
            f = interp1d(x_old, normalized)
            resized = f(x_new)
        normalized = resized

    return normalized


# Diagnostic label mapping for PTB-XL
DIAGNOSTIC_CLASSES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "HYP": "Hypertrophy",
    "CD": "Conduction Disturbance",
}

SUPERCLASS_MAP = {
    "NORM": 0,
    "MI": 1,
    "STTC": 2,
    "HYP": 3,
    "CD": 4,
}
