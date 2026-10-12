"""Validate and export the CRNN training manifest without training a model."""

import argparse
from pathlib import Path

from .charset import build_charset
from .io import SCALES, dataset_path, load_samples, read_json, timestamp, write_json


def export_training(dataset, charset, scale=1.0):
    expected = build_charset(Path(dataset) / "ocr_train.csv")
    if charset != expected:
        raise ValueError("Charset does not match the current training CSV")

    scales = SCALES if scale == "all" else (scale,)
    records = []
    for current_scale in scales:
        for sample in load_samples(dataset, "train", current_scale):
            crop = dataset_path(dataset, sample["crop_path"])
            if not crop.is_file():
                raise FileNotFoundError(f"Missing training crop: {crop}")
            if set(sample["ground_truth"]) - set(charset["characters"]):
                raise ValueError(f"Training text outside charset: {sample['sample_id']}")
            records.append({**sample, "ctc_target": [charset["character_to_index"][c] for c in sample["ground_truth"]]})

    return {
        "schema_version": 1,
        "dataset_root": str(Path(dataset).resolve()),
        "source_sha256": charset["source_sha256"],
        "split": "train",
        "scale": scale,
        "timestamp": timestamp(),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--charset", type=Path, default=Path("baseline/ocr/charset.json"))
    parser.add_argument("--scale", choices=("1.0", "0.75", "0.5", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scale = args.scale if args.scale == "all" else float(args.scale)
    write_json(args.output, export_training(args.dataset, read_json(args.charset), scale))


if __name__ == "__main__":
    main()
