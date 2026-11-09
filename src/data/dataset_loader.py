from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DatasetReport:
    dataset_root: Path
    class_names: tuple[str, ...]
    records: tuple[tuple[Path, int], ...]
    class_counts: dict[str, int]
    invalid_files: tuple[str, ...]
    duplicate_files: tuple[str, ...]
    conflicting_duplicate_groups: tuple[tuple[str, ...], ...]
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def locate_class_root(
    dataset_dir: str | Path,
    class_names: Sequence[str],
    class_sources: Mapping[str, Sequence[str]] | None = None,
) -> Path:
    """Find the one directory whose immediate subdirectories are the classes."""
    root = Path(dataset_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    if class_sources is None:
        source_names = tuple(class_names)
    else:
        if set(class_sources) != set(class_names):
            raise ValueError(
                "class_sources must define exactly one source-folder group for "
                "each configured class."
            )
        source_groups = {
            class_name: tuple(class_sources[class_name]) for class_name in class_names
        }
        source_names = tuple(
            source_name
            for class_name in class_names
            for source_name in source_groups[class_name]
        )
        if any(not names for names in source_groups.values()):
            raise ValueError("Every configured class must have at least one source folder.")
        if len(set(source_names)) != len(source_names):
            raise ValueError("A source folder cannot belong to multiple classes.")

    expected = set(source_names)
    candidates = []
    for directory in (root, *sorted((item for item in root.rglob("*") if item.is_dir()))):
        child_dirs = {item.name for item in directory.iterdir() if item.is_dir()}
        if expected.issubset(child_dirs):
            candidates.append(directory)

    if not candidates:
        raise ValueError(
            f"Could not find all class folders {list(class_names)} under {root}. "
            "Check the configured dataset path and extracted archive structure."
        )
    if len(candidates) > 1:
        paths = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple possible class-folder roots found: {paths}")

    class_root = candidates[0]
    actual = {item.name for item in class_root.iterdir() if item.is_dir()}
    unexpected = sorted(actual - expected)
    if unexpected:
        raise ValueError(
            f"Unexpected class folders at {class_root}: {unexpected}. "
            "Update the configured class_names before using these data."
        )
    return class_root


def inspect_dataset(
    dataset_dir: str | Path,
    class_names: Sequence[str],
    class_sources: Mapping[str, Sequence[str]] | None = None,
) -> DatasetReport:
    names = tuple(class_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("class_names must contain unique, non-empty class names.")

    sources = (
        {name: (name,) for name in names}
        if class_sources is None
        else {name: tuple(class_sources[name]) for name in names}
        if set(class_sources) == set(names)
        else None
    )
    if sources is None:
        raise ValueError(
            "class_sources must define exactly one source-folder group for "
            "each configured class."
        )
    class_root = locate_class_root(dataset_dir, names, sources)
    counts = {name: 0 for name in names}
    records_by_hash: dict[str, list[tuple[Path, int]]] = {}
    invalid: list[str] = []
    duplicates: list[str] = []
    conflicts: list[tuple[str, ...]] = []
    errors: list[str] = []

    for label, class_name in enumerate(names):
        for source_name in sources[class_name]:
            class_dir = class_root / source_name
            files = sorted(path for path in class_dir.rglob("*") if path.is_file())

            for path in files:
                if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    invalid.append(str(path))
                    continue
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError):
                    invalid.append(str(path))
                    continue
                records_by_hash.setdefault(digest, []).append((path, label))

    records: list[tuple[Path, int]] = []
    for copies in records_by_hash.values():
        labels = {label for _, label in copies}
        if len(labels) > 1:
            group = tuple(str(path) for path, _ in copies)
            conflicts.append(group)
            duplicates.extend(group)
            continue
        records.append(copies[0])
        counts[names[copies[0][1]]] += 1
        duplicates.extend(str(path) for path, _ in copies[1:])

    for class_name in names:
        if counts[class_name] == 0:
            errors.append(
                f"No valid, unambiguous unique images found for class '{class_name}'."
            )

    return DatasetReport(
        dataset_root=class_root,
        class_names=names,
        records=tuple(sorted(records, key=lambda record: (record[1], str(record[0])))),
        class_counts=counts,
        invalid_files=tuple(invalid),
        duplicate_files=tuple(duplicates),
        conflicting_duplicate_groups=tuple(conflicts),
        errors=tuple(errors),
    )


def split_records(
    records: Sequence[tuple[Path, int]],
    class_names: Sequence[str],
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[Path, int]], ...]:
    """Return reproducible train, validation, and fixed test partitions."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be greater than 0 and less than 1.")
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must contain unique, non-empty class names.")

    class_count = len(class_names)
    labels = np.asarray([label for _, label in records], dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("Dataset contains a label outside the configured classes.")
    counts = np.bincount(labels, minlength=class_count)
    if np.any(counts < 3):
        raise ValueError(
            "Each configured class needs at least 3 valid unique images to create "
            "train, validation, and test splits."
        )

    rng = np.random.default_rng(seed)
    partitions: tuple[list[tuple[Path, int]], ...] = ([], [], [])
    for label in range(class_count):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = len(indices)
        test_size = min(max(1, round(count * test_fraction)), count - 2)
        validation_size = min(
            max(1, round(count * validation_fraction)), count - test_size - 1
        )
        train_size = count - test_size - validation_size
        for partition, start, end in (
            (partitions[0], 0, train_size),
            (partitions[1], train_size, train_size + validation_size),
            (partitions[2], train_size + validation_size, count),
        ):
            partition.extend((records[index][0], label) for index in indices[start:end])

    for partition in partitions:
        rng.shuffle(partition)
    return partitions


def make_tf_dataset(
    records: Sequence[tuple[Path, int]],
    image_size: int,
    batch_size: int,
    training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    if not records:
        raise ValueError("Cannot create a TensorFlow dataset from zero images.")
    if image_size < 1 or batch_size < 1:
        raise ValueError("image_size and batch_size must be positive.")

    paths = [str(path) for path, _ in records]
    labels = [label for _, label in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        contents = tf.io.read_file(path)
        image = tf.image.decode_image(contents, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (image_size, image_size))
        return image, label

    if training:
        dataset = dataset.shuffle(
            len(records), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
