from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.models.classifier import IncrementalClassifier


def load_rgb_image(source: str | Path | Image.Image, image_size: int) -> np.ndarray:
    try:
        if isinstance(source, Image.Image):
            image = source.convert("RGB")
        else:
            with Image.open(source) as opened:
                image = opened.convert("RGB")
        image = image.resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"Could not read image: {error}") from error


def predict_keras(
    model: tf.keras.Model,
    source: str | Path | Image.Image,
    image_size: int,
) -> np.ndarray:
    image = load_rgb_image(source, image_size)
    probabilities = model.predict(image[None, ...], verbose=0)[0]
    return np.asarray(probabilities, dtype=np.float64)


def make_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    def find_backbone(
        candidate: tf.keras.Model,
    ) -> tf.keras.Model | None:
        for layer in candidate.layers:
            if isinstance(layer, tf.keras.Model):
                if layer.name.startswith("mobilenetv2"):
                    return layer
                nested = find_backbone(layer)
                if nested is not None:
                    return nested
        return None

    backbone = find_backbone(model)
    if backbone is None:
        raise ValueError(
            "The selected Keras model does not contain a MobileNetV2 backbone."
        )

    def apply_until_pooling(
        candidate: tf.keras.Model,
        values: tf.Tensor,
    ) -> tuple[tf.Tensor, bool]:
        for layer in candidate.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                return layer(values), True
            if layer is backbone:
                values = layer(values, training=False)
                continue
            if isinstance(layer, tf.keras.Model):
                contains_backbone = find_backbone(layer)
                if contains_backbone is not None:
                    values, found = apply_until_pooling(layer, values)
                    if found:
                        return values, True
            else:
                values = layer(values)
        return values, False

    try:
        features, found_pooling = apply_until_pooling(model, model.inputs[0])
        if not found_pooling:
            raise ValueError(
                "The selected MobileNetV2 model has no global-average-pooling "
                "layer after its backbone."
            )
        return tf.keras.Model(model.inputs, features, name="mobilenet_feature_extractor")
    except (ValueError, KeyError, TypeError) as error:
        if isinstance(error, ValueError) and "global-average-pooling" in str(error):
            raise
        raise ValueError(
            "Could not connect the nested MobileNetV2 feature extractor to the "
            "selected model inputs."
        ) from error


def extract_features(
    extractor: tf.keras.Model,
    records: list[tuple[Path, int]],
    image_size: int,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Cannot extract features from an empty image set.")
    images = np.stack([load_rgb_image(path, image_size) for path, _ in records])
    labels = np.asarray([label for _, label in records], dtype=np.int64)
    features = extractor.predict(images, batch_size=batch_size, verbose=0)
    return np.asarray(features, dtype=np.float32), labels


def predict_incremental(
    extractor: tf.keras.Model,
    classifier: IncrementalClassifier,
    source: str | Path | Image.Image,
    image_size: int,
) -> np.ndarray:
    image = load_rgb_image(source, image_size)
    features = extractor.predict(image[None, ...], verbose=0)
    return classifier.predict_proba(features)[0]
