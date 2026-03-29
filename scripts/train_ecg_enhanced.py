"""
Enhanced ECG Training with Residual 1D-CNN.
"""
import os
import sys
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Load data
ecg_dir = PROJECT_ROOT / "data" / "ecg"
X = np.load(ecg_dir / "ecg_signals.npy")
y = np.load(ecg_dir / "ecg_labels.npy")
print(f"Loaded {X.shape[0]} ECG records, shape: {X.shape}")

# Normalize per-record
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 5

# Residual block
def residual_block(x, filters, kernel_size=5):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    # Match dimensions for shortcut
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Build model
inp = layers.Input(shape=(X.shape[1], X.shape[2]))
x = layers.GaussianNoise(0.05)(inp)

# Initial conv
x = layers.Conv1D(64, 7, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling1D(2)(x)

# Residual blocks
x = residual_block(x, 64, 5)
x = layers.MaxPooling1D(2)(x)

x = residual_block(x, 128, 5)
x = layers.MaxPooling1D(2)(x)

x = residual_block(x, 128, 3)
x = residual_block(x, 256, 3)

# Head
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inp, x)

# Class weights
from collections import Counter
counts = Counter(y_train)
total = len(y_train)
class_weight = {c: total / (num_classes * cnt) for c, cnt in counts.items()}
print(f"Class weights: {class_weight}")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print("\nTraining Residual 1D-CNN...")
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=50,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6),
    ],
    verbose=1,
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Save
model_path = MODELS_DIR / "ecg_classifier.h5"
model.save(str(model_path))
print(f"Saved: {model_path}")

# Print per-class accuracy
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
from sklearn.metrics import classification_report
class_names = ["NORM", "MI", "STTC", "HYP", "CD"]
print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")
