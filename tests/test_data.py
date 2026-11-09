from pathlib import Path

import pytest
from PIL import Image

from src.data.dataset_loader import inspect_dataset, split_records
from src.data.inspect_dataset import create_sample_grid
from src.training.common import load_config


CLASSES = ("cardboard", "glass", "metal")


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def test_discovers_enclosing_folder_and_splits_reproducibly(tmp_path: Path) -> None:
    root = tmp_path / "archive-wrapper" / "dataset-resized"
    for label, name in enumerate(CLASSES):
        for index in range(6):
            make_image(
                root / name / f"{index}.jpg",
                ((label * 50 + index) % 255, label * 30, index * 20),
            )

    report = inspect_dataset(tmp_path, CLASSES)
    assert report.is_valid
    assert report.dataset_root == root
    assert report.class_counts == {name: 6 for name in CLASSES}

    first = split_records(report.records, CLASSES, seed=42)
    second = split_records(report.records, CLASSES, seed=42)
    assert first == second
    assert set(path for path, _ in first[0]).isdisjoint(
        path for part in first[1:] for path, _ in part
    )


def test_inspects_dataset_using_grouped_source_folders(tmp_path: Path) -> None:
    source_groups = {
        "Paper": ("paper", "cardboard"),
        "Glass": ("brown-glass", "green-glass"),
    }
    colors = {
        "paper": (10, 20, 30),
        "cardboard": (40, 50, 60),
        "brown-glass": (70, 80, 90),
        "green-glass": (100, 110, 120),
    }
    for source_name, color in colors.items():
        for index in range(3):
            make_image(
                tmp_path / source_name / f"{index}.jpg",
                tuple((channel + index) % 255 for channel in color),
            )

    report = inspect_dataset(tmp_path, tuple(source_groups), source_groups)

    assert report.is_valid
    assert report.class_counts == {"Paper": 6, "Glass": 6}
    assert {label for _, label in report.records} == {0, 1}


def test_reports_corrupt_and_non_image_files(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for label, name in enumerate(CLASSES):
        for index in range(3):
            make_image(
                root / name / f"{index}.jpg",
                ((label * 50 + index) % 255, label * 30, index * 20),
            )
    (root / "cardboard" / "notes.txt").write_text("not an image", encoding="utf-8")
    (root / "glass" / "broken.png").write_bytes(b"not a png")

    report = inspect_dataset(root, CLASSES)
    assert report.is_valid
    assert len(report.invalid_files) == 2
    assert report.class_counts == {name: 3 for name in CLASSES}


def test_rejects_classes_with_too_few_images(tmp_path: Path) -> None:
    records = []
    for label, name in enumerate(CLASSES):
        for index in range(2):
            path = tmp_path / name / f"{index}.jpg"
            records.append((path, label))

    with pytest.raises(ValueError, match="at least 3"):
        split_records(records, CLASSES)


def test_rejects_identical_image_bytes_with_conflicting_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    for class_name in CLASSES:
        (root / class_name).mkdir(parents=True)
    shared = tmp_path / "shared.png"
    Image.new("RGB", (12, 12), (33, 66, 99)).save(shared)
    (root / CLASSES[0] / "copy.png").write_bytes(shared.read_bytes())
    (root / CLASSES[1] / "copy.png").write_bytes(shared.read_bytes())
    for class_index, class_name in enumerate(CLASSES):
        for index in range(3):
            make_image(
                root / class_name / f"unique-{index}.jpg",
                (class_index * 50 + index + 1, index * 20, class_index * 30),
            )

    report = inspect_dataset(root, CLASSES)
    assert report.is_valid
    assert len(report.conflicting_duplicate_groups) == 1
    assert report.class_counts == {"cardboard": 3, "glass": 3, "metal": 3}
    assert len(report.records) == 9


def test_config_paths_can_be_overridden_with_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "config_mwcd.yaml"
    monkeypatch.setenv("WASTE_DATASET_DIR", str(tmp_path / "custom-data"))
    monkeypatch.setenv("WASTE_MODELS_DIR", str(tmp_path / "custom-models"))
    monkeypatch.setenv("WASTE_RESULTS_DIR", str(tmp_path / "custom-results"))

    config = load_config(config_path)
    assert config["data"]["dataset_dir"] == str(tmp_path / "custom-data")
    assert config["models"]["cnn"]["output_dir"] == str(tmp_path / "custom-models")
    assert config["models"]["mobilenet"]["output_dir"] == str(tmp_path / "custom-models")
    assert config["continual"]["output_dir"] == str(
        tmp_path / "custom-models" / "continual"
    )
    assert config["results"]["output_dir"] == str(tmp_path / "custom-results")


def test_creates_dataset_sample_contact_sheet(tmp_path: Path) -> None:
    records = []
    for label, name in enumerate(CLASSES):
        image_path = tmp_path / name / "sample.png"
        make_image(image_path, (label * 50, 30, 90))
        records.append((image_path, label))
    output = tmp_path / "reports" / "samples.png"

    create_sample_grid(tuple(records), CLASSES, output)

    assert output.is_file()
    assert output.stat().st_size > 0
