import argparse

from src.training.common import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNN waste classifier.")
    parser.add_argument("--config", default="configs/config_mwcd.yaml")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    run_training(args.config, "cnn", dataset_dir=args.dataset)


if __name__ == "__main__":
    main()
