import argparse

from src.training.common import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the MobileNetV2 waste classifier."
    )
    parser.add_argument("--config", default="configs/config_mwcd.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default=None,
        help="Use ImageNet pretrained weights, or initialize without downloading weights.",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune the configured final backbone layers after head training.",
    )
    args = parser.parse_args()
    weights = "none" if args.weights == "none" else args.weights
    run_training(
        args.config,
        "mobilenet",
        dataset_dir=args.dataset,
        weights=weights,
        fine_tune=args.fine_tune,
    )


if __name__ == "__main__":
    main()
