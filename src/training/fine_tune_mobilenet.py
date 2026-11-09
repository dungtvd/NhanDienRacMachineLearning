from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from src.data.dataset_loader import inspect_dataset, make_tf_dataset, split_records
from src.evaluation.metrics import evaluate_model
from src.inference.model_loader import load_keras_model
from src.training.common import load_config, print_dataset_report, set_seed


def balanced_class_weights(
    train_records: list[tuple[Path, int]],
    class_count: int,
    power: float = 0.5,
) -> tuple[np.ndarray, dict[str, int]]:
    if not 0 <= power <= 1:
        raise ValueError("Class-weight power must be between 0 and 1.")
    labels = np.asarray([label for _, label in train_records], dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("Training records contain labels outside the configured classes.")
    counts = np.bincount(labels, minlength=class_count)
    if np.any(counts == 0):
        raise ValueError("Every class needs training examples for balanced fine-tuning.")
    weights = (len(labels) / (class_count * counts)) ** power
    weights = weights / weights.mean()
    return weights.astype(np.float32), {
        str(index): int(count) for index, count in enumerate(counts)
    }


def with_class_weights(
    dataset: tf.data.Dataset, class_weights: np.ndarray
) -> tf.data.Dataset:
    weights = tf.constant(class_weights, dtype=tf.float32)
    return dataset.map(
        lambda images, labels: (images, labels, tf.gather(weights, labels)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def build_augmented_fine_tuner(
    base_model: tf.keras.Model,
    rotation_factor: float,
    zoom_factor: float,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    backbone = next(
        (
            layer
            for layer in base_model.layers
            if isinstance(layer, tf.keras.Model)
            and layer.name.startswith("mobilenetv2")
        ),
        None,
    )
    if backbone is None:
        raise ValueError("The selected model does not contain a MobileNetV2 backbone.")

    for layer in backbone.layers:
        layer.trainable = False
    trainable_count = min(30, len(backbone.layers))
    for layer in backbone.layers[-trainable_count:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = tf.keras.Input(shape=base_model.input_shape[1:], name="image")
    augmented = tf.keras.layers.RandomFlip("horizontal", name="fine_tune_flip")(inputs)
    augmented = tf.keras.layers.RandomRotation(
        rotation_factor, name="fine_tune_rotation"
    )(augmented)
    augmented = tf.keras.layers.RandomZoom(zoom_factor, name="fine_tune_zoom")(
        augmented
    )
    outputs = base_model(augmented, training=False)
    candidate = tf.keras.Model(inputs, outputs, name="waste_mobilenet_v2_finetuned")
    candidate.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return candidate, backbone


def classification_scores(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: tuple[str, ...],
) -> dict:
    labels = np.concatenate([labels.numpy() for _, labels in dataset])
    predictions = model.predict(dataset, verbose=0).argmax(axis=1)
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(report["accuracy"]),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
    }


def run_balanced_fine_tune(
    config_path: str | Path,
    base_model_path: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    epochs: int | None = None,
) -> tuple[Path, dict]:
    config = load_config(config_path)
    data_config = config["data"]
    training_config = config["training"]
    classes = tuple(data_config["class_names"])
    set_seed(int(config["project"]["seed"]))

    configured_path = base_model_path
    if configured_path is None:
        candidates = sorted(
            Path(config["models"]["mobilenet"]["output_dir"]).glob("mobilenet_*.keras"),
            key=lambda path: path.stat().st_mtime,
        )
        candidates = [
            path for path in candidates if "finetuned" not in path.stem.lower()
        ]
        if not candidates:
            raise FileNotFoundError(
                "No trained MobileNetV2 model found. Train it first or pass "
                "--base-model."
            )
        configured_path = candidates[-1]
    base_path = Path(configured_path)
    if not base_path.is_file():
        raise FileNotFoundError(f"Base model does not exist: {base_path}")

    report = inspect_dataset(dataset_dir or data_config["dataset_dir"], classes)
    print_dataset_report(report)
    if not report.is_valid:
        raise ValueError("Dataset has invalid class structure; resolve errors first.")

    train_records, validation_records, test_records = split_records(
        report.records,
        classes,
        test_fraction=float(data_config["test_fraction"]),
        validation_fraction=float(data_config["validation_fraction"]),
        seed=int(config["project"]["seed"]),
    )
    image_size = int(data_config["image_size"])
    batch_size = int(data_config["batch_size"])
    train_dataset = make_tf_dataset(
        train_records,
        image_size,
        batch_size,
        training=True,
        seed=int(config["project"]["seed"]),
    )
    validation_dataset = make_tf_dataset(
        validation_records, image_size, batch_size
    )
    test_dataset = make_tf_dataset(test_records, image_size, batch_size)

    weights, train_counts = balanced_class_weights(
        train_records,
        len(classes),
        float(training_config["balanced_class_weight_power"]),
    )
    weighted_train_dataset = with_class_weights(train_dataset, weights)

    baseline_model = load_keras_model(base_path)
    baseline_validation = classification_scores(
        baseline_model, validation_dataset, classes
    )
    print(
        "Baseline validation: "
        f"accuracy={baseline_validation['accuracy']:.4f}, "
        f"macro_f1={baseline_validation['f1_macro']:.4f}"
    )

    candidate_model, backbone = build_augmented_fine_tuner(
        baseline_model,
        float(training_config["augmentation_rotation"]),
        float(training_config["augmentation_zoom"]),
    )
    candidate_model.optimizer.learning_rate.assign(
        float(training_config["fine_tune_learning_rate"])
    )

    output_dir = Path(config["models"]["mobilenet"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = output_dir / f"mobilenet_finetuned_{version}.keras"
    checkpoint = output_dir / f"mobilenet_finetuned_{version}.checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(training_config["balanced_finetune_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    started = time.perf_counter()
    history = candidate_model.fit(
        weighted_train_dataset,
        validation_data=validation_dataset,
        epochs=epochs or int(training_config["balanced_finetune_epochs"]),
        callbacks=callbacks,
    )
    training_seconds = time.perf_counter() - started
    validation_metrics = evaluate_model(
        candidate_model,
        validation_dataset,
        classes,
        config["results"]["output_dir"],
        f"mobilenet_finetuned_{version}_validation",
        evaluation_split="validation",
    )
    minimum_f1 = float(training_config["balanced_min_f1_improvement"])
    max_accuracy_drop = float(training_config["balanced_max_accuracy_drop"])
    accepted = (
        validation_metrics["f1_macro"] >= baseline_validation["f1_macro"] + minimum_f1
        and validation_metrics["accuracy"]
        >= baseline_validation["accuracy"] - max_accuracy_drop
    )
    candidate_model.save(artifact)

    metadata = {
        "model": "mobilenet_v2_balanced_fine_tune",
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model_path": str(base_path.resolve()),
        "class_names": list(classes),
        "image_size": image_size,
        "preprocessing": "Inherited from the base MobileNetV2 model.",
        "training_seconds": training_seconds,
        "train_class_counts": {
            classes[int(index)]: count for index, count in train_counts.items()
        },
        "class_weights": {
            classes[index]: float(value) for index, value in enumerate(weights)
        },
        "fine_tuned_backbone_layers": sum(
            layer.trainable for layer in backbone.layers
        ),
        "training_config": {
            "epochs_requested": epochs or training_config["balanced_finetune_epochs"],
            "epochs_completed": len(history.history["loss"]),
            "learning_rate": float(training_config["fine_tune_learning_rate"]),
            "class_weight_power": float(
                training_config["balanced_class_weight_power"]
            ),
            "augmentation_rotation": float(
                training_config["augmentation_rotation"]
            ),
            "augmentation_zoom": float(training_config["augmentation_zoom"]),
        },
        "validation_split_size": len(validation_records),
        "baseline_validation_metrics": baseline_validation,
        "candidate_validation_metrics": validation_metrics,
        "quality_gate": {
            "minimum_macro_f1_improvement": minimum_f1,
            "maximum_accuracy_drop": max_accuracy_drop,
            "passed": accepted,
        },
        "candidate_is_recommended": accepted,
        "test_evaluated": False,
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }
    if accepted:
        test_metrics = evaluate_model(
            candidate_model,
            test_dataset,
            classes,
            config["results"]["output_dir"],
            f"mobilenet_finetuned_{version}",
            evaluation_split="test",
        )
        metadata["test_metrics"] = test_metrics
        metadata["test_evaluated"] = True
    else:
        print(
            "Validation quality gate failed; the existing MobileNetV2 remains "
            "the recommended model."
        )

    metadata_path = artifact.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    checkpoint.unlink(missing_ok=True)
    print(f"Saved candidate model: {artifact}")
    print(f"Saved metadata: {metadata_path}")
    print(
        f"Validation: accuracy={validation_metrics['accuracy']:.4f}, "
        f"macro_f1={validation_metrics['f1_macro']:.4f}; "
        f"quality_gate={'PASSED' if accepted else 'NOT PASSED'}"
    )
    if accepted:
        print(
            "Held-out test evaluated after validation selection: "
            f"accuracy={metadata['test_metrics']['accuracy']:.4f}, "
            f"macro_f1={metadata['test_metrics']['f1_macro']:.4f}"
        )
    return artifact, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an existing MobileNetV2 with moderate class balancing and "
            "select by validation metrics."
        )
    )
    parser.add_argument("--config", default="configs/config_mwcd.yaml")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    arguments = parser.parse_args()
    if arguments.epochs is not None and arguments.epochs < 1:
        parser.error("--epochs must be a positive integer.")
    run_balanced_fine_tune(
        arguments.config,
        arguments.base_model,
        arguments.dataset,
        arguments.epochs,
    )


if __name__ == "__main__":
    main()
