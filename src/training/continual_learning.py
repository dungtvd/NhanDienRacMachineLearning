from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data.dataset_loader import inspect_dataset, split_records
from src.data.replay_buffer import ReplayBuffer
from src.inference.model_loader import load_keras_model
from src.inference.predict import extract_features, make_feature_extractor
from src.models.classifier import IncrementalClassifier


def validate_update_records(
    records: list[tuple[Path, int]],
    test_records: list[tuple[Path, int]],
    validation_records: list[tuple[Path, int]] | None = None,
) -> list[tuple[Path, int]]:
    test_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path, _ in test_records
    }
    validation_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path, _ in (validation_records or [])
    }
    seen: dict[str, int] = {}
    unique_records = []
    for path, label in records:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in test_hashes:
            raise ValueError(
                f"Update data contains a fixed test image and would leak test data: {path}"
            )
        if digest in validation_hashes:
            raise ValueError(
                "Update data contains a fixed validation image and would leak "
                f"validation data: {path}"
            )
        previous_label = seen.get(digest)
        if previous_label is not None:
            if previous_label != label:
                raise ValueError(
                    f"Duplicate update image has conflicting labels: {path}"
                )
            continue
        seen[digest] = label
        unique_records.append((path, label))
    return unique_records


class ContinualLearningManager:
    """Versioned continual-learning workflow; only promoted versions become active."""

    def __init__(
        self,
        root: str | Path,
        class_names: tuple[str, ...],
        image_size: int,
        seed: int = 42,
        replay_images_per_class: int = 20,
        replay_images_per_update: int = 60,
        max_old_validation_accuracy_drop: float = 0.03,
        minimum_new_validation_accuracy: float = 0.0,
        test_fraction: float = 0.2,
        validation_fraction: float = 0.2,
    ) -> None:
        if not 0 <= max_old_validation_accuracy_drop <= 1:
            raise ValueError("max_old_validation_accuracy_drop must be between 0 and 1.")
        if not 0 <= minimum_new_validation_accuracy <= 1:
            raise ValueError("minimum_new_validation_accuracy must be between 0 and 1.")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.class_names = tuple(class_names)
        self.image_size = image_size
        self.seed = seed
        self.replay_images_per_class = replay_images_per_class
        self.replay_images_per_update = replay_images_per_update
        self.max_old_validation_accuracy_drop = max_old_validation_accuracy_drop
        self.minimum_new_validation_accuracy = minimum_new_validation_accuracy
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.active_pointer = self.root / "active.json"

    @property
    def initialized(self) -> bool:
        return self.active_pointer.is_file()

    def _active_path(self) -> Path:
        if not self.initialized:
            raise RuntimeError(
                "Continual classifier is not initialized. Train MobileNetV2 and "
                "initialize it from a labeled dataset first."
            )
        with self.active_pointer.open(encoding="utf-8") as file:
            version = json.load(file)["version"]
        directory = self.root / version
        if not directory.is_dir():
            raise FileNotFoundError(f"Active model version is missing: {directory}")
        return directory

    def _read_fixed_records(self, split: str) -> list[tuple[Path, int]]:
        active_dir = self._active_path()
        manifest_path = active_dir / f"fixed_{split}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"The fixed {split} manifest is missing.")
        with manifest_path.open(encoding="utf-8") as file:
            entries = json.load(file)
        records = []
        for item in entries:
            path = Path(item["path"])
            if not path.is_absolute():
                path = active_dir / path
            path = path.resolve()
            try:
                path.relative_to(active_dir.resolve())
            except ValueError as error:
                raise ValueError(
                    f"Fixed {split} manifest path escapes the model version directory."
                ) from error
            records.append((path, int(item["label"])))
        if not records or any(not path.is_file() for path, _ in records):
            raise FileNotFoundError(f"One or more fixed {split} images are unavailable.")
        return records

    def _read_test_records(self) -> list[tuple[Path, int]]:
        return self._read_fixed_records("test")

    def _read_validation_records(self) -> list[tuple[Path, int]]:
        return self._read_fixed_records("validation")

    def _load_extractor(self, base_model_path: str | Path) -> tf.keras.Model:
        model_path = Path(base_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Base MobileNetV2 model does not exist: {model_path}")
        return make_feature_extractor(load_keras_model(model_path))

    def _evaluate(
        self, classifier: IncrementalClassifier, features: np.ndarray, labels: np.ndarray
    ) -> dict:
        predictions = classifier.predict_proba(features).argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "classification_report": classification_report(
                labels,
                predictions,
                labels=list(range(len(self.class_names))),
                target_names=list(self.class_names),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=list(range(len(self.class_names)))
            ).tolist(),
            "sample_count": int(len(labels)),
        }

    def _write_pointer(self, version: str) -> None:
        temporary = self.active_pointer.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump({"version": version}, file)
        temporary.replace(self.active_pointer)

    def _write_version_metadata(self, directory: Path, metadata: dict) -> None:
        metadata_path = directory / "metadata.json"
        temporary = metadata_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        temporary.replace(metadata_path)

    def initialize(
        self,
        base_model_path: str | Path,
        dataset_dir: str | Path,
        force: bool = False,
        class_sources: Mapping[str, Sequence[str]] | None = None,
    ) -> dict:
        operation_started = time.perf_counter()
        if self.initialized and not force:
            raise RuntimeError(
                "A continual classifier is already active. Use the update flow or "
                "explicitly confirm reinitialization."
            )
        report = inspect_dataset(dataset_dir, self.class_names, class_sources)
        if not report.is_valid:
            raise ValueError("Dataset validation failed: " + "; ".join(report.errors))
        train_records, validation_records, test_records = split_records(
            report.records,
            self.class_names,
            test_fraction=self.test_fraction,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
        )
        extractor = self._load_extractor(base_model_path)
        train_features, train_labels = extract_features(
            extractor, train_records, self.image_size
        )
        test_features, test_labels = extract_features(
            extractor, test_records, self.image_size
        )
        validation_features, validation_labels = extract_features(
            extractor, validation_records, self.image_size
        )
        classifier = IncrementalClassifier(
            self.class_names, int(train_features.shape[1]), seed=self.seed
        )
        classifier.partial_fit(train_features, train_labels)
        test_metrics = self._evaluate(classifier, test_features, test_labels)
        validation_metrics = self._evaluate(
            classifier, validation_features, validation_labels
        )

        version = f"v_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        staging = self.root / f".staging_{uuid4().hex}"
        staging.mkdir()
        try:
            classifier.save(staging / "classifier.joblib")
            test_snapshot = staging / "fixed_test"
            test_snapshot.mkdir()
            manifest_entries = []
            for index, (path, label) in enumerate(test_records):
                destination = test_snapshot / f"{index:06d}{path.suffix.lower()}"
                shutil.copyfile(path, destination)
                manifest_entries.append(
                    {"path": (Path("fixed_test") / destination.name).as_posix(), "label": label}
                )
            validation_snapshot = staging / "fixed_validation"
            validation_snapshot.mkdir()
            validation_manifest_entries = []
            for index, (path, label) in enumerate(validation_records):
                destination = validation_snapshot / f"{index:06d}{path.suffix.lower()}"
                shutil.copyfile(path, destination)
                validation_manifest_entries.append(
                    {
                        "path": str(
                            (Path("fixed_validation") / destination.name).as_posix()
                        ),
                        "label": label,
                    }
                )
            replay = ReplayBuffer(
                staging / "replay",
                self.class_names,
                self.replay_images_per_class,
                self.seed,
            )
            replay.add(train_records, source="initial_train")
            metadata = {
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active",
                "class_names": list(self.class_names),
                "image_size": self.image_size,
                "feature_dim": int(train_features.shape[1]),
                "base_model_path": str(Path(base_model_path).resolve()),
                "test_metrics": test_metrics,
                "validation_metrics": validation_metrics,
                "initialization_seconds": time.perf_counter() - operation_started,
                "split_sizes": {
                    "train": len(train_records),
                    "validation": len(validation_records),
                    "test": len(test_records),
                },
                "replay_counts": replay.counts(),
            }
            self._write_version_metadata(staging, metadata)
            with (staging / "fixed_test_manifest.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(manifest_entries, file, indent=2)
            with (staging / "fixed_validation_manifest.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(validation_manifest_entries, file, indent=2)
            destination = self.root / version
            staging.replace(destination)
            self._write_pointer(version)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return metadata

    def active_metadata(self) -> dict:
        directory = self._active_path()
        with (directory / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        if tuple(metadata.get("class_names", ())) != self.class_names:
            raise ValueError(
                "Active model class order differs from the configured class_names. "
                "Restore matching configuration or explicitly reinitialize."
            )
        if int(metadata.get("image_size", -1)) != self.image_size:
            raise ValueError(
                "Active model image_size differs from configuration; restore the "
                "matching configuration before inference or updating."
            )
        return metadata

    def active_classifier(self) -> IncrementalClassifier:
        return IncrementalClassifier.load(self._active_path() / "classifier.joblib")

    @property
    def active_version_dir(self) -> Path:
        return self._active_path()

    def predict_image(self, source: str | Path | Image.Image) -> np.ndarray:
        metadata = self.active_metadata()
        extractor = self._load_extractor(metadata["base_model_path"])
        classifier = self.active_classifier()
        from src.inference.predict import predict_incremental

        return predict_incremental(extractor, classifier, source, self.image_size)

    def update(self, new_records: list[tuple[Path, int]], source: str = "confirmed_update") -> dict:
        operation_started = time.perf_counter()
        if not new_records:
            raise ValueError("No confirmed new images are available for updating.")
        active_dir = self._active_path()
        active_metadata = self.active_metadata()
        test_records = self._read_test_records()
        fixed_validation_records = self._read_validation_records()
        new_records = validate_update_records(
            new_records, test_records, fixed_validation_records
        )
        if not new_records:
            raise ValueError("No unique, non-test images remain in this update batch.")
        unknown = sorted(
            {label for _, label in new_records}
            - set(range(len(self.class_names)))
        )
        if unknown:
            raise ValueError(
                f"New class indices {unknown} are not configured. Reinitialize the "
                "classifier with an expanded class list; labels are never remapped."
            )
        extractor = self._load_extractor(active_metadata["base_model_path"])
        new_features, new_labels = extract_features(
            extractor, new_records, self.image_size
        )
        if new_features.shape[1] != active_metadata["feature_dim"]:
            raise ValueError("New feature dimension does not match the active classifier.")

        rng = random.Random(self.seed)
        indices_by_class = {
            label: [index for index, value in enumerate(new_labels) if value == label]
            for label in range(len(self.class_names))
        }
        validation_indices = []
        training_indices = []
        for label, indices in indices_by_class.items():
            if not indices:
                continue
            if len(indices) < 2:
                raise ValueError(
                    f"Class '{self.class_names[label]}' needs at least 2 confirmed "
                    "images in an update batch to reserve validation data."
                )
            rng.shuffle(indices)
            validation_count = max(1, len(indices) // 5)
            validation_indices.extend(indices[:validation_count])
            training_indices.extend(indices[validation_count:])

        current_classifier = self.active_classifier()
        test_features, test_labels = extract_features(extractor, test_records, self.image_size)
        fixed_validation_features, fixed_validation_labels = extract_features(
            extractor, fixed_validation_records, self.image_size
        )
        before_metrics = self._evaluate(current_classifier, test_features, test_labels)
        before_fixed_validation = self._evaluate(
            current_classifier, fixed_validation_features, fixed_validation_labels
        )
        if validation_indices:
            before_new_validation = self._evaluate(
                current_classifier,
                new_features[validation_indices],
                new_labels[validation_indices],
            )
        else:
            before_new_validation = {"accuracy": None, "sample_count": 0}

        update_records = [new_records[index] for index in training_indices]
        version = f"v_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        staging = self.root / f".staging_{uuid4().hex}"
        staging.mkdir()
        try:
            shutil.copytree(active_dir / "replay", staging / "replay")
            shutil.copytree(active_dir / "fixed_test", staging / "fixed_test")
            shutil.copyfile(
                active_dir / "fixed_test_manifest.json",
                staging / "fixed_test_manifest.json",
            )
            shutil.copytree(
                active_dir / "fixed_validation", staging / "fixed_validation"
            )
            shutil.copyfile(
                active_dir / "fixed_validation_manifest.json",
                staging / "fixed_validation_manifest.json",
            )
            replay = ReplayBuffer(
                staging / "replay",
                self.class_names,
                self.replay_images_per_class,
                self.seed,
            )
            replay_records = replay.sample(self.replay_images_per_update, self.seed)
            if replay_records:
                replay_features, replay_labels = extract_features(
                    extractor, replay_records, self.image_size
                )
            else:
                replay_features = np.empty((0, active_metadata["feature_dim"]), dtype=np.float32)
                replay_labels = np.empty((0,), dtype=np.int64)
            replay.add(update_records, source=source)
            combined_features = np.concatenate(
                (new_features[training_indices], replay_features), axis=0
            )
            combined_labels = np.concatenate(
                (new_labels[training_indices], replay_labels), axis=0
            )
            updated_classifier = IncrementalClassifier.load(
                active_dir / "classifier.joblib"
            )
            updated_classifier.partial_fit(combined_features, combined_labels)
            after_metrics = self._evaluate(updated_classifier, test_features, test_labels)
            after_fixed_validation = self._evaluate(
                updated_classifier,
                fixed_validation_features,
                fixed_validation_labels,
            )
            if validation_indices:
                after_new_validation = self._evaluate(
                    updated_classifier,
                    new_features[validation_indices],
                    new_labels[validation_indices],
                )
            else:
                after_new_validation = {"accuracy": None, "sample_count": 0}
            accepted = (
                after_fixed_validation["accuracy"]
                >= before_fixed_validation["accuracy"]
                - self.max_old_validation_accuracy_drop
                and (
                    after_new_validation["accuracy"] is None
                    or after_new_validation["accuracy"]
                    >= self.minimum_new_validation_accuracy
                )
            )
            updated_classifier.save(staging / "classifier.joblib")
            metadata = {
                **active_metadata,
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active" if accepted else "rejected",
                "parent_version": active_metadata["version"],
                "update_source": source,
                "new_training_images": len(update_records),
                "new_validation_images": len(validation_indices),
                "replay_images_used": len(replay_records),
                "update_seconds": time.perf_counter() - operation_started,
                "replay_counts": replay.counts(),
                "before_test_metrics": before_metrics,
                "after_test_metrics": after_metrics,
                "test_metrics": after_metrics,
                "before_fixed_validation_metrics": before_fixed_validation,
                "after_fixed_validation_metrics": after_fixed_validation,
                "before_new_validation_metrics": before_new_validation,
                "after_new_validation_metrics": after_new_validation,
                "accepted": accepted,
                "acceptance_thresholds": {
                    "max_old_validation_accuracy_drop": self.max_old_validation_accuracy_drop,
                    "minimum_new_validation_accuracy": self.minimum_new_validation_accuracy,
                },
            }
            self._write_version_metadata(staging, metadata)
            destination = self.root / version
            staging.replace(destination)
            if accepted:
                self._write_pointer(version)
            return metadata
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def rollback(self, version: str) -> None:
        destination = self.root / version
        if not (destination / "classifier.joblib").is_file() or not (
            destination / "metadata.json"
        ).is_file():
            raise FileNotFoundError(f"Cannot activate incomplete model version: {version}")
        with (destination / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("status") == "rejected":
            raise ValueError("Rejected model versions cannot be activated.")
        self._write_pointer(version)

    def versions(self) -> list[dict]:
        versions = []
        active = None
        if self.initialized:
            with self.active_pointer.open(encoding="utf-8") as file:
                active = json.load(file)["version"]
        for metadata_path in sorted(self.root.glob("v_*/metadata.json")):
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            metadata["is_active"] = metadata["version"] == active
            versions.append(metadata)
        return versions
