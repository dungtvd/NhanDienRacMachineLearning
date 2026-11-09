import json
import zipfile

import numpy as np
import tensorflow as tf

from src.inference.model_loader import load_keras_model


def test_loads_model_with_null_quantization_config(tmp_path) -> None:
    model_path = tmp_path / "model.keras"
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.save(model_path)

    compatible_input = tmp_path / "compatible-input.keras"
    with zipfile.ZipFile(model_path) as source, zipfile.ZipFile(
        compatible_input, "w"
    ) as compatible:
        for entry in source.infolist():
            contents = source.read(entry.filename)
            if entry.filename == "config.json":
                config = json.loads(contents)
                dense_layer = next(
                    layer
                    for layer in config["config"]["layers"]
                    if layer["class_name"] == "Dense"
                )
                dense_layer["config"]["quantization_config"] = None
                contents = json.dumps(config).encode("utf-8")
            compatible.writestr(entry, contents)

    loaded = load_keras_model(compatible_input)
    predictions = loaded(np.zeros((2, 4), dtype=np.float32), training=False)

    assert tuple(predictions.shape) == (2, 3)
    np.testing.assert_allclose(np.sum(predictions.numpy(), axis=1), 1.0)
