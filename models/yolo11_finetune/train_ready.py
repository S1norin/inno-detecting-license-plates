"""Validate fine-tuning inputs without starting training."""

import argparse
from pathlib import Path

from baseline.detection.evaluate import load_dataset

from .config import DEFAULT_CONFIG, read_config, starting_weights


def validate_config(config, data=None):
    document = read_config(config, data=data)
    model = starting_weights(document)
    dataset = Path(document["data"])
    _, paths, _, _, _ = load_dataset(dataset, "train")
    return {"model": str(model), "data": str(dataset.resolve()), "train_images": len(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    print(validate_config(args.config, args.data))


if __name__ == "__main__":
    main()
