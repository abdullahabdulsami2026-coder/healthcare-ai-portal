"""
Model loading, saving, and prediction utilities.
Handles loading saved models from disk or Google Drive.
"""

import os
import json
import numpy as np
import joblib


def get_model_path(model_name, models_dir="models"):
    """Get the full path to a saved model file."""
    extensions = [".h5", ".keras", ".pkl", ".joblib", ".json"]
    for ext in extensions:
        path = os.path.join(models_dir, f"{model_name}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_model(model_name, models_dir="models"):
    """
    Load a saved model by name. Supports:
    - Keras/TF models (.h5, .keras)
    - Scikit-learn models (.pkl, .joblib)
    """
    path = get_model_path(model_name, models_dir)
    if path is None:
        raise FileNotFoundError(
            f"No model found for '{model_name}' in {models_dir}/. "
            f"Train the model first using the notebooks."
        )

    ext = os.path.splitext(path)[1]

    if ext in [".h5", ".keras"]:
        from tensorflow import keras
        return keras.models.load_model(path)
    elif ext in [".pkl", ".joblib"]:
        return joblib.load(path)
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def save_model(model, model_name, models_dir="models"):
    """Save a trained model to disk."""
    os.makedirs(models_dir, exist_ok=True)

    # Check if it's a Keras model
    if hasattr(model, "save"):
        path = os.path.join(models_dir, f"{model_name}.h5")
        model.save(path)
    else:
        # Scikit-learn or similar
        path = os.path.join(models_dir, f"{model_name}.joblib")
        joblib.dump(model, path)

    print(f"Model saved to: {path}")
    return path


def predict_with_confidence(model, X, class_names=None):
    """
    Run prediction and return class + confidence scores.
    Works with both Keras and sklearn models.
    """
    # Ensure correct shape
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim == 2 and hasattr(model, "predict_proba"):
        pass  # sklearn expects (n_samples, n_features)
    elif X.ndim == 2:
        X = np.expand_dims(X, axis=0)  # Keras expects batch dim

    # Get probabilities
    if hasattr(model, "predict_proba"):
        # Scikit-learn
        probs = model.predict_proba(X)[0]
    else:
        # Keras
        probs = model.predict(X, verbose=0)[0]

    predicted_class = np.argmax(probs)
    confidence = float(probs[predicted_class])

    result = {
        "predicted_class": int(predicted_class),
        "confidence": round(confidence * 100, 1),
        "all_probabilities": {
            (class_names[i] if class_names else f"Class {i}"): round(float(p) * 100, 1)
            for i, p in enumerate(probs)
        },
    }

    if class_names and predicted_class < len(class_names):
        result["predicted_label"] = class_names[predicted_class]

    return result


def list_available_models(models_dir="models"):
    """List all trained models available in the models directory."""
    if not os.path.exists(models_dir):
        return []

    extensions = {".h5", ".keras", ".pkl", ".joblib"}
    models = []
    for f in os.listdir(models_dir):
        name, ext = os.path.splitext(f)
        if ext in extensions:
            models.append({"name": name, "format": ext, "path": os.path.join(models_dir, f)})

    return models
