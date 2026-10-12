"""Validate fine-tuning inputs without starting training."""

import argparse
from pathlib import Path

import yaml

from baseline.common.weights import ROOT, verified_path
from baseline.detection.evaluate import load_dataset


def validate_config(config, data=None):
    document = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    model = (ROOT / document["model"]).resolve()
    if model != verified_path().resolve():
        raise ValueError("Training model must reference the verified cached detector")
    dataset = Path(data) if data else ROOT / document["data"]
    for name in ("epochs", "batch", "imgsz"):
        if not isinstance(document[name], int) or document[name] <= 0:
            raise ValueError(f"Training {name} must be a positive integer")
    _, paths, _, _, _ = load_dataset(dataset, "train")
    return {"model": str(model), "data": str(dataset.resolve()), "train_images": len(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("train_config.yaml"))
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    print(validate_config(args.config, args.data))


if __name__ == "__main__":
    main()
