"""
Train All Models — Healthcare AI Portal
========================================
Run this script to train all models sequentially.
Alternatively, use the individual notebooks in notebooks/ folder.

Usage: python scripts/train_all.py
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
    """
    Train heart disease risk prediction model on UCI Heart Disease dataset.
    This is the fastest model to train — good for testing the pipeline.
    """
    print("\n" + "=" * 60)
    print("Training Heart Disease Risk Model (UCI)")
    print("=" * 60)

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    # Load data
    data_path = PROJECT_ROOT / "data" / "vitals" / "heart.csv"
    if not data_path.exists():
        print(f"  ERROR: {data_path} not found. Run download_data.sh first.")
        return

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} records")

    # Prepare features and target
    X = df.drop("target", axis=1).values
    y = df["target"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'])}")

    # Save model and scaler
    model_path = MODELS_DIR / "heart_risk.joblib"
    scaler_path = MODELS_DIR / "heart_risk_scaler.joblib"
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Model saved: {model_path}")
    print(f"  Scaler saved: {scaler_path}")


def train_ecg_model():
    """
    Train ECG classification model on PTB-XL dataset.
    Uses a 1D CNN for multi-class diagnosis.
    """
    print("\n" + "=" * 60)
    print("Training ECG Classification Model (PTB-XL)")
    print("=" * 60)

    import wfdb
    import ast

    ecg_dir = PROJECT_ROOT / "data" / "ecg"
    meta_path = ecg_dir / "ptbxl_database.csv"

    if not meta_path.exists():
        print(f"  ERROR: {meta_path} not found. Run download_data.sh first.")
        return

    # Load metadata
    df = pd.read_csv(meta_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    print(f"  Loaded {len(df)} ECG records metadata")

    # Load SCP statement mapping
    scp_df = pd.read_csv(ecg_dir / "scp_statements.csv", index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    # Map diagnostic superclasses
    def map_superclass(scp_dict):
        result = set()
        for key in scp_dict:
            if key in scp_df.index:
                result.add(scp_df.loc[key].diagnostic_class)
        return result

    df["diagnostic_superclass"] = df.scp_codes.apply(map_superclass)

    # Filter to single-label records for simplicity
    df["single_label"] = df.diagnostic_superclass.apply(
        lambda x: list(x)[0] if len(x) == 1 else None
    )
    df_filtered = df[df.single_label.notna()].copy()
    print(f"  Single-label records: {len(df_filtered)}")

    # Encode labels
    label_map = {"NORM": 0, "MI": 1, "STTC": 2, "HYP": 3, "CD": 4}
    df_filtered["label"] = df_filtered.single_label.map(label_map)
    df_filtered = df_filtered[df_filtered.label.notna()]

    # Load signals (using 100Hz version for speed)
    print("  Loading ECG signals (100Hz)... This may take a few minutes.")
    signals = []
    labels = []
    max_records = 5000  # Limit for faster training; remove for full dataset

    for i, (idx, row) in enumerate(df_filtered.head(max_records).iterrows()):
        try:
            record = wfdb.rdrecord(str(ecg_dir / row.filename_lr))
            signals.append(record.p_signal)
            labels.append(int(row.label))
        except Exception:
            continue

        if (i + 1) % 1000 == 0:
            print(f"    Loaded {i + 1}/{min(max_records, len(df_filtered))} records")

    X = np.array(signals)  # shape: (N, 1000, 12)
    y = np.array(labels)
    print(f"  Signal array shape: {X.shape}")
    print(f"  Label distribution: {pd.Series(y).value_counts().to_dict()}")

    # Normalize
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build 1D CNN
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    num_classes = len(label_map)

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

    print("\n  Model summary:")
    model.summary()

    # Train
    print("\n  Training...")
    history = model.fit(
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

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Accuracy: {test_acc:.4f}")

    # Save
    model_path = MODELS_DIR / "ecg_classifier.h5"
    model.save(str(model_path))
    print(f"  Model saved: {model_path}")


def train_xray_model():
    """
    Train chest X-ray pneumonia classifier.
    Uses transfer learning with MobileNetV2 for speed.
    """
    print("\n" + "=" * 60)
    print("Training Chest X-Ray Classifier")
    print("=" * 60)

    xray_dir = PROJECT_ROOT / "data" / "xray"
    train_dir = xray_dir / "train"

    if not train_dir.exists():
        print(f"  ERROR: {train_dir} not found.")
        print("  Download the Kaggle Chest X-Ray Pneumonia dataset first.")
        print("  URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        return

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Data generators
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.15,
    )

    train_data = train_gen.flow_from_directory(
        str(train_dir), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="training"
    )

    val_data = train_gen.flow_from_directory(
        str(train_dir), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="validation"
    )

    # Transfer learning with MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze base

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("  Training with MobileNetV2 transfer learning...")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        ],
    )

    # Evaluate on test set if available
    test_dir = xray_dir / "test"
    if test_dir.exists():
        test_gen = ImageDataGenerator(rescale=1.0 / 255)
        test_data = test_gen.flow_from_directory(
            str(test_dir), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode="binary"
        )
        test_loss, test_acc = model.evaluate(test_data, verbose=0)
        print(f"\n  Test Accuracy: {test_acc:.4f}")

    # Save
    model_path = MODELS_DIR / "xray_classifier.h5"
    model.save(str(model_path))
    print(f"  Model saved: {model_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Healthcare AI Portal — Model Training")
    print("=" * 60)

    # Train in order of speed (fastest first)
    train_heart_risk_model()
    train_ecg_model()
    train_xray_model()

    print("\n" + "=" * 60)
    print("All training complete!")
    print(f"Models saved to: {MODELS_DIR}")
    print("Run the app: streamlit run app/streamlit_app.py")
    print("=" * 60)
