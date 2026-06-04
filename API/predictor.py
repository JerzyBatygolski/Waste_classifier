"""
predictor.py
============
Model loading and image preprocessing for the waste classifier API.

Preprocessing pipeline (must match 02_train_model.py exactly):
  1. Letterbox resize to 224x224 (keep aspect ratio, pad with gray)
  2. Scale pixels from [0, 255] to [0.0, 1.0]
  3. Normalize with ImageNet mean and std
  4. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
  5. model(arr, training=False)
"""

import json
import logging
import time
from io import BytesIO
from pathlib import Path

import keras                 # type: ignore
import numpy as np
import tensorflow as tf      # type: ignore
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants - must match train_model.py exactly
# ---------------------------------------------------------------------------

IMG_SIZE       = 224
PADDING_COLOR  = (114, 114, 114)   # neutral gray letterbox padding
IMAGENET_MEAN  = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD   = np.array([0.229, 0.224, 0.225], dtype="float32")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def letterbox_resize(img, target_size, padding_color):
    """
    Resize image keeping aspect ratio, pad remaining space with gray.
    Identical to 01_prepare_dataset.py - must stay in sync.
    """
    img.thumbnail((target_size, target_size), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), padding_color)
    offset_x = (target_size - img.width)  // 2
    offset_y = (target_size - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas


def preprocess_image(img_bytes):
    """
    Full preprocessing pipeline for a raw image (bytes).
    Returns numpy array of shape (1, 224, 224, 3), ready for inference.

    Steps:
      1. Decode bytes -> PIL Image
      2. Convert to RGB (handles RGBA, grayscale, CMYK, etc.)
      3. Letterbox resize to 224x224
      4. Convert to float32 numpy array
      5. Scale to [0.0, 1.0]
      6. Normalize with ImageNet mean and std
      7. Add batch dimension
    """
    img = Image.open(BytesIO(img_bytes))
    img = img.convert("RGB")
    img = letterbox_resize(img, IMG_SIZE, PADDING_COLOR)

    arr = np.array(img, dtype="float32")        # shape: (224, 224, 3)
    arr = arr / 255.0                            # scale to [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD   # normalize
    arr = np.expand_dims(arr, axis=0)            # shape: (1, 224, 224, 3)
    return arr


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class WastePredictor:
    """
    Loads model and class map once at startup, reuses them for every request.
    Loading on each request would be ~3-5s per call - unacceptable for an API.
    """

    def __init__(self, model_path, class_indices_path):
        self.model        = None
        self.idx_to_class = {}
        self._load(model_path, class_indices_path)

    def _load(self, model_path, class_indices_path):
        model_path = Path(model_path)
        class_path = Path(class_indices_path)

        if not model_path.exists():
            raise FileNotFoundError("Model not found: " + str(model_path))
        if not class_path.exists():
            raise FileNotFoundError("Class indices not found: " + str(class_path))

        logger.info("Loading model from: %s", model_path)
        self.model = keras.models.load_model(str(model_path))

        # Wrap in tf.function to compile the graph once at startup.
        # Without this the first real request triggers tracing (~500ms spike).
        self._infer = tf.function(self.model, reduce_retracing=True)

        # Warmup call so the graph is traced before the first real request.
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
        self._infer(dummy, training=False)
        logger.info("Model loaded and warmed up OK")

        with open(class_path, "r", encoding="utf-8") as f:
            class_indices = json.load(f)   # { "battery": 0, "biological": 1, ... }

        # Invert: { 0: "battery", 1: "biological", ... }
        self.idx_to_class = {v: k for k, v in class_indices.items()}
        logger.info("Classes: %s", list(class_indices.keys()))

    def predict_from_array(self, arr):
        """
        Run inference on an already-preprocessed array of shape (1, 224, 224, 3).
        Kept separate from preprocessing so the API layer can distinguish
        decoding errors (422) from inference errors (500).

        Returns dict:
          {
            "class":       "plastic",
            "confidence":  0.9423,
            "all_scores":  { "battery": 0.002, "biological": 0.001, ... },
            "latency_ms":  47.3
          }
        """
        t0 = time.perf_counter()

        # self._infer is a tf.function-wrapped model — graph is already compiled
        # from the warmup call in _load, so no tracing overhead here.
        probs = self._infer(arr, training=False).numpy()[0]   # shape: (num_classes,)

        best_idx   = int(np.argmax(probs))
        best_class = self.idx_to_class[best_idx]
        confidence = float(probs[best_idx])

        all_scores = {
            self.idx_to_class[i]: round(float(p), 6)
            for i, p in enumerate(probs)
        }
        # Sort by score descending for readability
        all_scores = dict(
            sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "class":      best_class,
            "confidence": round(confidence, 6),
            "all_scores": all_scores,
            "latency_ms": latency_ms,
        }

    def predict(self, img_bytes):
        """
        Convenience wrapper: preprocess + infer in one call.
        Latency includes both steps.
        """
        arr = preprocess_image(img_bytes)
        return self.predict_from_array(arr)
