from __future__ import annotations

import argparse
from pathlib import Path

from src.training.common import load_config
from src.training.continual_learning import ContinualLearningManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an incremental classifier on frozen MobileNetV2 features. "
            "The image backbone is not retrained."
        )
    )
    parser.add_argument("--config", default="configs/config_mwcd.yaml")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    continual = config["continual"]
    manager = ContinualLearningManager(
        continual["output_dir"],
        tuple(data["class_names"]),
        int(data["image_size"]),
        seed=int(config["project"]["seed"]),
        replay_images_per_class=int(continual["replay_images_per_class"]),
        replay_images_per_update=int(continual["replay_images_per_update"]),
        max_old_validation_accuracy_drop=float(
            continual["max_old_validation_accuracy_drop"]
        ),
        minimum_new_validation_accuracy=float(
            continual["minimum_new_validation_accuracy"]
        ),
        test_fraction=float(data["test_fraction"]),
        validation_fraction=float(data["validation_fraction"]),
    )
    dataset_dir = args.dataset or data["dataset_dir"]
    metadata = manager.initialize(
        Path(args.base_model),
        dataset_dir,
        class_sources=data.get("class_sources"),
    )
    print(f"Initialized classifier version: {metadata['version']}")
    print(f"Frozen MobileNetV2: {Path(args.base_model).resolve()}")
    print(f"Dataset: {dataset_dir}")
    print(f"Classes: {', '.join(metadata['class_names'])}")
    print(f"Train/validation/test: {metadata['split_sizes']}")
    print(f"Validation accuracy: {metadata['validation_metrics']['accuracy']:.4f}")
    print(f"Validation macro F1: {metadata['validation_metrics']['f1_macro']:.4f}")
    print(f"Held-out test accuracy: {metadata['test_metrics']['accuracy']:.4f}")
    print(f"Held-out test macro F1: {metadata['test_metrics']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
