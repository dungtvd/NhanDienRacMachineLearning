from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import tensorflow as tf


def _remove_null_quantization_config(value: Any) -> bool:
    changed = False
    if isinstance(value, dict):
        if "quantization_config" in value and value["quantization_config"] is None:
            del value["quantization_config"]
            changed = True
        for child in value.values():
            changed = _remove_null_quantization_config(child) or changed
    elif isinstance(value, list):
        for child in value:
            changed = _remove_null_quantization_config(child) or changed
    return changed


def load_keras_model(path: str | Path) -> tf.keras.Model:
    model_path = Path(path)
    with zipfile.ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json"))

    if not _remove_null_quantization_config(config):
        return tf.keras.models.load_model(model_path, compile=False)

    with tempfile.TemporaryDirectory(prefix="waste-model-") as directory:
        compatible_path = Path(directory) / model_path.name
        with zipfile.ZipFile(model_path) as source, zipfile.ZipFile(
            compatible_path, "w"
        ) as compatible:
            for entry in source.infolist():
                contents = (
                    json.dumps(config).encode("utf-8")
                    if entry.filename == "config.json"
                    else source.read(entry.filename)
                )
                compatible.writestr(entry, contents)
        return tf.keras.models.load_model(compatible_path, compile=False)
