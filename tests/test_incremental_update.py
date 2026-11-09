import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.data.replay_buffer import ReplayBuffer
from src.models.classifier import IncrementalClassifier
from src.training.fine_tune_mobilenet import balanced_class_weights
from src.training.continual_learning import validate_update_records


def test_classifier_partial_fit_validates_and_round_trips(tmp_path: Path) -> None:
    classifier = IncrementalClassifier(("cardboard", "glass", "metal"), 4, seed=7)
    features = np.asarray(
        [[1, 0, 0, 0], [0.9, 0, 0, 0], [0, 1, 0, 0], [0, 0.9, 0, 0],
         [0, 0, 1, 0], [0, 0, 0.9, 0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    classifier.partial_fit(features, labels)
    classifier.partial_fit(features[:3], labels[:3])

    model_path = tmp_path / "classifier.joblib"
    classifier.save(model_path)
    loaded = IncrementalClassifier.load(model_path)
    assert loaded.class_names == ("cardboard", "glass", "metal")
    assert loaded.feature_dim == 4
    assert loaded.predict_proba(features).shape == (6, 3)

    with pytest.raises(ValueError, match="feature matrix"):
        loaded.partial_fit(np.zeros((2, 5)), np.asarray([0, 1]))
    with pytest.raises(ValueError, match="not present"):
        loaded.partial_fit(np.zeros((1, 4)), np.asarray([3]))


def test_classifier_requires_every_class_on_first_fit(tmp_path: Path) -> None:
    classifier = IncrementalClassifier(("a", "b", "c"), 2)
    with pytest.raises(ValueError, match="every configured class"):
        classifier.partial_fit(np.ones((2, 2)), np.asarray([0, 1]))
    with pytest.raises(RuntimeError, match="uninitialized"):
        classifier.save(tmp_path / "uninitialized.joblib")


def test_replay_buffer_is_bounded_balanced_and_deduplicates(tmp_path: Path) -> None:
    class_names = ("cardboard", "glass")
    source = tmp_path / "source"
    records = []
    for label in range(2):
        for index in range(4):
            path = source / str(label) / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (label * 80, index * 30, 15)).save(path)
            records.append((path, label))

    buffer = ReplayBuffer(tmp_path / "replay", class_names, images_per_class=2, seed=3)
    assert buffer.add(records) == 8
    buffer.add(records)
    assert buffer.counts() == {"cardboard": 2, "glass": 2}
    sample = buffer.sample(3)
    assert len(sample) == 3
    assert len({label for _, label in sample}) == 2
    manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text())
    assert {entry["source"] for entry in manifest} == {"confirmed_update"}


def test_update_rejects_fixed_test_leakage_and_duplicate_conflicting_labels(
    tmp_path: Path,
) -> None:
    test_image = tmp_path / "test.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(test_image)
    test_records = [(test_image, 0)]
    with pytest.raises(ValueError, match="fixed test image"):
        validate_update_records([(test_image, 0)], test_records)

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (8, 8), (9, 8, 7)).save(first)
    first.replace(second)
    Image.new("RGB", (8, 8), (9, 8, 7)).save(first)
    with pytest.raises(ValueError, match="conflicting labels"):
        validate_update_records([(first, 0), (second, 1)], test_records)


def test_update_rejects_fixed_validation_leakage(tmp_path: Path) -> None:
    validation_image = tmp_path / "validation.png"
    Image.new("RGB", (8, 8), (12, 23, 34)).save(validation_image)
    with pytest.raises(ValueError, match="fixed validation image"):
        validate_update_records(
            [(validation_image, 0)],
            [],
            [(validation_image, 0)],
        )


def test_balanced_class_weights_moderate_imbalance() -> None:
    records = []
    records.extend((Path(f"common-{index}.png"), 0) for index in range(100))
    records.extend((Path(f"rare-{index}.png"), 1) for index in range(25))

    weights, counts = balanced_class_weights(records, class_count=2, power=0.5)

    assert counts == {"0": 100, "1": 25}
    assert weights[1] > weights[0]
    assert np.isclose(float(weights.mean()), 1.0)
