from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml

from src.data.dataset_loader import (
    DatasetReport,
    inspect_dataset,
    make_tf_dataset,
    split_records,
)
from src.evaluation.metrics import evaluate_model


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    data_dir = os.environ.get("WASTE_DATASET_DIR")
    models_dir = os.environ.get("WASTE_MODELS_DIR")
    results_dir = os.environ.get("WASTE_RESULTS_DIR")
    staging_dir = os.environ.get("WASTE_STAGING_DIR")
    continual_dir = os.environ.get("WASTE_CONTINUAL_DIR")
    if data_dir:
        config["data"]["dataset_dir"] = data_dir
    if models_dir:
        for model_config in config["models"].values():
            model_config["output_dir"] = models_dir
        if not continual_dir:
            config["continual"]["output_dir"] = str(Path(models_dir) / "continual")
    if results_dir:
        config["results"]["output_dir"] = results_dir
    if staging_dir:
        config["app"]["staging_dir"] = staging_dir
    if continual_dir:
        config["continual"]["output_dir"] = continual_dir
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def print_dataset_report(report: DatasetReport) -> None:
    print(f"Dataset root: {report.dataset_root}")
    print("Valid unique images per class:")
    for name, count in report.class_counts.items():
        print(f"  {name}: {count}")
    print(f"Invalid files: {len(report.invalid_files)}")
    print(f"Exact duplicates skipped: {len(report.duplicate_files)}")
    if report.conflicting_duplicate_groups:
        print(
            "WARNING: exact duplicate image copies with conflicting class labels "
            "were all excluded from training/evaluation:"
        )
        for group in report.conflicting_duplicate_groups:
            print("  " + " <-> ".join(group))
    for error in report.errors:
        print(f"ERROR: {error}")


def run_training(
    config_path: str | Path,
    model_kind: str,
    dataset_dir: str | Path | None = None,
    weights: str | None = None,
    fine_tune: bool | None = None,
) -> tuple[Path, dict]:
    config = load_config(config_path)
    data_config = config["data"]
    training_config = config["training"]
    model_config = config["models"][model_kind]
    class_names = tuple(data_config["class_names"])
    seed = int(config["project"]["seed"])
    set_seed(seed)

    configured_dataset = dataset_dir or data_config["dataset_dir"]
    class_sources = config["data"].get("class_sources")

    report = inspect_dataset(
        configured_dataset,
        class_names,
        class_sources,
    )
    print_dataset_report(report)
    if not report.is_valid:
        raise ValueError("Dataset validation failed; fix the reported class issues.")

    train_records, validation_records, test_records = split_records(
        report.records,
        class_names,
        test_fraction=float(data_config["test_fraction"]),
        validation_fraction=float(data_config["validation_fraction"]),
        seed=seed,
    )
    image_size = int(data_config["image_size"])
    batch_size = int(data_config["batch_size"])
    train_ds = make_tf_dataset(
        train_records, image_size, batch_size, training=True, seed=seed
    )
    validation_ds = make_tf_dataset(validation_records, image_size, batch_size)
    test_ds = make_tf_dataset(test_records, image_size, batch_size)

    if model_kind == "cnn":
        from src.models.cnn import build_cnn

        model = build_cnn(len(class_names), image_size)
    elif model_kind == "mobilenet":
        from src.models.mobilenet import build_mobilenet

        selected_weights = (
            None
            if weights == "none"
            else weights if weights is not None else model_config["weights"]
        )
        model = build_mobilenet(len(class_names), image_size, selected_weights)
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    learning_rate = float(training_config["learning_rate"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    output_dir = Path(model_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = output_dir / f"{model_kind}_{version}.keras"
    checkpoint = output_dir / f"{model_kind}_{version}.checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(training_config["patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    training_started = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=int(training_config["epochs"]),
        callbacks=callbacks,
    )

    do_fine_tune = (
        bool(training_config["fine_tune"])
        if fine_tune is None
        else fine_tune
    )
    if model_kind == "mobilenet" and do_fine_tune:
        backbone = next(
            (
                layer
                for layer in model.layers
                if isinstance(layer, tf.keras.Model)
                and layer.name.startswith("mobilenetv2")
            ),
            None,
        )
        if backbone is None:
            raise ValueError("Could not locate the MobileNetV2 backbone to fine-tune.")
        backbone.trainable = True
        layer_count = int(training_config["fine_tune_layers"])
        layer_count = min(max(layer_count, 1), len(backbone.layers))
        for layer in backbone.layers[:-layer_count]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=float(training_config["fine_tune_learning_rate"])
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_tune_history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=int(training_config["epochs"]),
            callbacks=callbacks,
        )
        history.history.update(
            {
                f"fine_tune_{key}": values
                for key, values in fine_tune_history.history.items()
            }
        )
    training_seconds = time.perf_counter() - training_started

    model.save(artifact)
    metrics = evaluate_model(
        model,
        test_ds,
        class_names,
        config["results"]["output_dir"],
        f"{model_kind}_{version}",
    )
    metrics["training_seconds"] = training_seconds
    metrics_path = Path(config["results"]["output_dir"]) / f"{model_kind}_{version}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    metadata = {
        "model": model_kind,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_names": list(class_names),
        "image_size": image_size,
        "preprocessing": (
            "RGB float pixels 0..255; divide by 255 in the CNN model"
            if model_kind == "cnn"
            else "RGB float pixels 0..255; scale to [-1, 1] in the model"
        ),
        "dataset_root": str(report.dataset_root),
        "class_counts": report.class_counts,
        "split_sizes": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "training_config": training_config,
        "training_seconds": training_seconds,
        "test_metrics": metrics,
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }
    metadata_path = artifact.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    checkpoint.unlink(missing_ok=True)
    print(f"Saved model: {artifact}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    return artifact, metadata
