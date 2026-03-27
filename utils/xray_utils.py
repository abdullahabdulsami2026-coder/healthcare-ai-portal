"""
Chest X-ray preprocessing and prediction utilities.
"""

import numpy as np
from PIL import Image


# Label names for chest X-ray classification
XRAY_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia", "No Finding"
]

# Simplified binary classification
PNEUMONIA_CLASSES = ["Normal", "Pneumonia"]


def load_and_preprocess_xray(image_path_or_file, target_size=(224, 224)):
    """
    Load and preprocess a chest X-ray image for model input.
    Accepts file path (str) or file-like object (uploaded file).
    """
    if isinstance(image_path_or_file, str):
        img = Image.open(image_path_or_file)
    else:
        img = Image.open(image_path_or_file)

    # Convert to RGB (some X-rays are grayscale)
    img = img.convert("RGB")

    # Resize
    img = img.resize(target_size, Image.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0

    return img_array


def prepare_xray_for_model(img_array):
    """Add batch dimension for model prediction."""
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_grad_cam(model, img_array, class_index, last_conv_layer_name=None):
    """
    Generate Grad-CAM heatmap for explainability.
    Shows which regions of the X-ray the model focused on.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return None

    if last_conv_layer_name is None:
        # Find the last convolutional layer automatically
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        return None

    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()
