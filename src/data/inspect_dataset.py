from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.data.dataset_loader import inspect_dataset
from src.training.common import load_config


def create_sample_grid(records: tuple, class_names: tuple[str, ...], output: Path) -> None:
    selected = []
    per_class = {index: [] for index in range(len(class_names))}
    for path, label in records:
        if len(per_class[label]) < 2:
            per_class[label].append(path)
    for label, paths in per_class.items():
        selected.extend((path, class_names[label]) for path in paths)
    if not selected:
        raise ValueError("No valid images are available for a sample grid.")

    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for axis, (path, class_name) in zip(axes, selected):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(class_name)
        axis.axis("off")
    for axis in axes[len(selected) :]:
        axis.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the configured waste dataset."
    )
    parser.add_argument("--config", default="configs/config_mwcd.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--sample-grid",
        default="results/mwcd_materials/dataset_samples.png",
        help="Output contact sheet path, or use --no-sample-grid.",
    )
    parser.add_argument("--no-sample-grid", action="store_true")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    data_config = config["data"]
    dataset_path = arguments.dataset or data_config["dataset_dir"]
    class_names = tuple(data_config["class_names"])
    try:
        report = inspect_dataset(
            dataset_path,
            class_names,
            data_config.get("class_sources"),
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Dataset root: {report.dataset_root}")
    print(f"Valid, unique images: {len(report.records)}")
    print("Images per class:")
    for class_name, count in report.class_counts.items():
        print(f"  {class_name}: {count}")
    nonzero_counts = [count for count in report.class_counts.values() if count > 0]
    if nonzero_counts:
        ratio = max(nonzero_counts) / min(nonzero_counts)
        print(f"Imbalance ratio (largest/smallest non-empty class): {ratio:.2f}")
    print(f"Invalid files skipped: {len(report.invalid_files)}")
    for path in report.invalid_files[:20]:
        print(f"  INVALID: {path}")
    if len(report.invalid_files) > 20:
        print(f"  ... and {len(report.invalid_files) - 20} more invalid files")
    print(f"Exact duplicate images skipped: {len(report.duplicate_files)}")
    for path in report.duplicate_files[:20]:
        print(f"  DUPLICATE: {path}")
    if len(report.duplicate_files) > 20:
        print(f"  ... and {len(report.duplicate_files) - 20} more duplicates")
    if report.conflicting_duplicate_groups:
        print(
            "WARNING: identical images in different class folders are ambiguous; "
            "all copies were excluded:"
        )
        for group in report.conflicting_duplicate_groups:
            print("  " + " <-> ".join(group))
    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}")
        return 1
    if not arguments.no_sample_grid:
        output = Path(arguments.sample_grid)
        create_sample_grid(report.records, class_names, output)
        print(f"Sample image grid: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
