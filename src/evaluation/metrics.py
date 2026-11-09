from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: Sequence[str],
    output_dir: str | Path,
    model_name: str,
    evaluation_split: str = "test",
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = np.concatenate([label.numpy() for _, label in dataset])
    inference_started = time.perf_counter()
    probabilities = model.predict(dataset, verbose=0)
    inference_seconds = time.perf_counter() - inference_started
    predictions = probabilities.argmax(axis=1)

    summary = {
        "model": model_name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(class_names))),
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(class_names)))
        ).tolist(),
        "evaluation_split": evaluation_split,
        "image_count": int(len(labels)),
        "inference_seconds_total": inference_seconds,
        "inference_ms_per_image": inference_seconds * 1000 / len(labels),
    }
    if evaluation_split == "test":
        summary["test_image_count"] = int(len(labels))
    elif evaluation_split == "validation":
        summary["validation_image_count"] = int(len(labels))

    with (output / f"{model_name}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    pd.DataFrame(summary["classification_report"]).transpose().to_csv(
        output / f"{model_name}_classification_report.csv", index=True
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        summary["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(xlabel="Dự đoán", ylabel="Nhãn đúng", title=f"Ma trận nhầm lẫn: {model_name}")
    figure.tight_layout()
    figure.savefig(output / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close(figure)
    return summary
