from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier


class IncrementalClassifier:
    """Incremental SGD head over frozen MobileNetV2 image features."""

    def __init__(
        self,
        class_names: Sequence[str],
        feature_dim: int,
        seed: int = 42,
        estimator: SGDClassifier | None = None,
    ) -> None:
        self.class_names = tuple(class_names)
        self.feature_dim = int(feature_dim)
        self.estimator = estimator or SGDClassifier(
            loss="log_loss",
            random_state=seed,
            learning_rate="optimal",
        )
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be non-empty and unique.")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive.")

    @property
    def initialized(self) -> bool:
        return hasattr(self.estimator, "classes_")

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected a 2-D feature matrix with {self.feature_dim} columns; "
                f"received shape {values.shape}."
            )
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Feature matrix must be non-empty and finite.")
        return values

    def _validate_labels(self, labels: np.ndarray, sample_count: int) -> np.ndarray:
        values = np.asarray(labels)
        if values.ndim != 1 or len(values) != sample_count:
            raise ValueError("labels must be 1-D and match the feature count.")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("labels must contain integer class indices.")
        if np.any(values < 0) or np.any(values >= len(self.class_names)):
            raise ValueError("Update contains a class not present in class_names.")
        return values.astype(np.int64, copy=False)

    def partial_fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        if self.initialized:
            current_classes = np.asarray(self.estimator.classes_, dtype=np.int64)
            if not np.array_equal(current_classes, np.arange(len(self.class_names))):
                raise ValueError(
                    "Saved classifier classes do not match the configured class order."
                )
        values = self._validate_features(features)
        targets = self._validate_labels(labels, len(values))
        if not self.initialized:
            observed = set(targets.tolist())
            expected = set(range(len(self.class_names)))
            if observed != expected:
                raise ValueError(
                    "Initial classifier training must contain examples of every "
                    f"configured class; missing indices: {sorted(expected - observed)}."
                )
            self.estimator.partial_fit(values, targets, classes=np.arange(len(self.class_names)))
        else:
            self.estimator.partial_fit(values, targets)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Classifier has not been initialized with training data.")
        values = self._validate_features(features)
        return self.estimator.predict_proba(values)

    def save(self, path: str | Path) -> None:
        if not self.initialized:
            raise RuntimeError("Cannot save an uninitialized classifier.")
        joblib.dump(
            {
                "class_names": self.class_names,
                "feature_dim": self.feature_dim,
                "estimator": self.estimator,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "IncrementalClassifier":
        saved = joblib.load(path)
        classifier = cls(
            saved["class_names"],
            saved["feature_dim"],
            estimator=saved["estimator"],
        )
        if not classifier.initialized:
            raise ValueError("Saved classifier has not been initialized.")
        return classifier

