"""
Train All Models — Healthcare AI Portal
Uses heart.csv (real data) and generated ECG signals (synthetic).
X-ray model skipped (requires Kaggle download).

Usage: python scripts/train_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_heart_risk_model():
    """Train heart disease risk prediction model on UCI Heart Disease dataset."""
    print("\n" + "=" * 60)
    print("1/2  Training Heart Disease Risk Model")
    print("=" * 60)

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    data_path = PROJECT_ROOT / "data" / "vitals" / "heart.csv"
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} records from heart.csv")

    X = df.drop("target", axis=1).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("  Training Random Forest (200 trees)...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    joblib.dump(rf, MODELS_DIR / "heart_risk.joblib")
    joblib.dump(scaler, MODELS_DIR / "heart_risk_scaler.joblib")
    print(f"  Saved: heart_risk.joblib, heart_risk_scaler.joblib")


def train_ecg_model():
    """Train ECG classification model on generated synthetic data."""
    print("\n" + "=" * 60)
    print("2/2  Training ECG Classification Model")
    print("=" * 60)

    ecg_dir = PROJECT_ROOT / "data" / "ecg"
    X = np.load(ecg_dir / "ecg_signals.npy")
    y = np.load(ecg_dir / "ecg_labels.npy")
    print(f"  Loaded {X.shape[0]} ECG records, shape: {X.shape}")

    # Normalize per-record
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Suppress TF info logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    num_classes = 5
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1], X.shape[2])),
        layers.Conv1D(64, 7, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("  Training 1D CNN (up to 30 epochs, early stopping)...")
    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ],
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Accuracy: {test_acc:.4f}")

    model_path = MODELS_DIR / "ecg_classifier.h5"
    model.save(str(model_path))
    print(f"  Saved: {model_path}")


if __name__ == "__main__":
    print("Healthcare AI Portal — Model Training")
    print("=" * 60)

    train_heart_risk_model()
    train_ecg_model()

    print("\n" + "=" * 60)
    print("All training complete!")
    print(f"Models saved to: {MODELS_DIR}")
    print("=" * 60)
