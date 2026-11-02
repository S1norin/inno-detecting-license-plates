"""Evaluate verified plate weights on CCPD splits at explicit dataset scales."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from baseline.common.weights import DEFAULT_MODEL, ROOT, sha256, verified_path
from baseline.detection.metrics import compute_metrics, tilt_bin

RESULTS = ROOT / "baseline/detection/results"
SETTINGS = {"imgsz": 640, "conf": 0.001, "iou": 0.7, "max_det": 300,
            "device": "cpu", "batch": 1, "half": False, "augment": False, "rect": False}
DATASET_SCALES = (0.5, 0.75, 1.0)


def parse_scales(value):
    scales = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            raise ValueError("Empty scale in --scales")
        scale = float(part)
        if scale not in DATASET_SCALES:
            raise ValueError(f"Unknown scale {part}; dataset scales are 0.5, 0.75, 1.0")
        scales.add(scale)
    return scales


def load_dataset(config, split, scales=frozenset({1.0})):
    config = Path(config).resolve()
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    if document.get("nc") != 1 or document.get("names") not in ({0: "license_plate"}, ["license_plate"]):
        raise ValueError("Dataset must define one license_plate class (class 0)")
    root = Path(document.get("path", config.parent))
    if not root.is_absolute():
        root = config.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root missing: {root}. Run python data/configure_paths.py first.")
    entry = root / str(document[split])
    if scales == frozenset({1.0}) and entry.is_dir():
        raise ValueError("Original-scale runs require the list-form dataset config (dataset_original.yaml)")
    if scales != frozenset({1.0}) and entry.is_file():
        raise ValueError("Scale subset runs require the directory-form dataset config (dataset.yaml)")
    if entry.is_dir():
        enumerated = sorted((path for path in entry.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"),
                            key=lambda path: path.name)
        enumerated = [path.resolve() for path in enumerated]
        listing_digest = hashlib.sha256(
            "\n".join(path.relative_to(root).as_posix() for path in enumerated).encode()).hexdigest()
    else:
        listing = entry
        enumerated = [Path(line.strip()) for line in listing.read_text(encoding="utf-8-sig").splitlines()
                      if line.strip()]
        enumerated = [(p if p.is_absolute() else root / p).resolve() for p in enumerated]
        listing_digest = sha256(listing)

    metadata_path = root / "metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metadata = {}
    for row in rows:
        if row["split"] != split or float(row["scale"]) not in scales:
            continue
        key = Path(row["image_path"]).as_posix()
        if key in metadata:
            raise ValueError(f"Ambiguous metadata join: {key}")
        metadata[key] = row
    if entry.is_dir():
        # The split directory holds every variant; the requested scales select a subset.
        paths = [path for path in enumerated if path.relative_to(root).as_posix() in metadata]
    else:
        paths = enumerated
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Split image list is empty or contains duplicates")
    thresholds_path = root / "run_config.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))["bucket_train_height_thresholds"]
    q1, q2 = thresholds["q33"], thresholds["q67"]
    records, input_hashes = [], {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing split image {path}; prepare images and refresh paths first")
        key = path.relative_to(root).as_posix()
        if key not in metadata:
            if scales == frozenset({1.0}):
                raise ValueError(f"No original-scale {split} metadata for {key}")
            raise ValueError(f"No {split} metadata at the requested scales for {key}")
        row = metadata[key]
        width, height = int(row["width"]), int(row["height"])
        label = root / row["label_path"]
        lines = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1 or len(lines[0]) != 5 or lines[0][0] != "0":
            raise ValueError(f"Expected exactly one class-0 CCPD plate label: {label}")
        x, y, w, h = map(float, lines[0][1:])
        if not np.isfinite([x, y, w, h]).all() or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized box: {label}")
        box = [(x - w / 2) * width, (y - h / 2) * height,
               (x + w / 2) * width, (y + h / 2) * height]
        metadata_box = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
        if not np.allclose(box, metadata_box, atol=0.01, rtol=0):
            raise ValueError(f"Label and metadata disagree: {key}")
        plate_height = float(row["plate_pixel_height"])
        bucket = "small" if plate_height <= q1 else "medium" if plate_height <= q2 else "large"
        if bucket != row["size_bucket"]:
            raise ValueError(f"Size bucket disagrees with train thresholds: {key}")
        tilt = float(row["tilt_magnitude"])
        expected_tilt = max(abs(float(row["tilt_horizontal"])), abs(float(row["tilt_vertical"])))
        if tilt != expected_tilt:
            raise ValueError(f"Tilt metadata disagrees: {key}")
        tilt_bin(tilt)
        records.append({"image_path": key, "width": width, "height": height, "ground_truth": [box],
                        "size_bucket": bucket, "tilt_magnitude": tilt, "scale": float(row["scale"])})
        input_hashes[key] = {"image": sha256(path), "label": sha256(label)}
    if set(metadata) != {r["image_path"] for r in records}:
        if scales == frozenset({1.0}):
            raise ValueError("Split list does not cover all original metadata rows")
        raise ValueError("Split images do not cover all metadata rows for the requested scales")
    fingerprints = {"dataset_config": sha256(config), "split_list": listing_digest,
                    "metadata": sha256(metadata_path), "run_config": sha256(thresholds_path),
                    "images_and_labels": hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()}
    return root, paths, records, fingerprints, thresholds


def configure_cpu():
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def resolve_model(model=None):
    if model is None:
        return DEFAULT_MODEL, verified_path()
    path = Path(model).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model weights missing: {path}")
    return path.name, path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def evaluate(config, split, run_id, validate=False, model=None, scales=frozenset({1.0})):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Run ID must be 1-80 filename-safe characters")
    if validate and scales != frozenset({1.0}):
        raise ValueError("--validate is only supported for the original-scale (1.0) run")
    directory = RESULTS / run_id
    if (directory / f"predictions_{split}.json").exists() or (directory / f"metrics_{split}.json").exists():
        raise FileExistsError(f"Results already exist for {run_id}/{split}; use a new run ID")
    model_id, weight = resolve_model(model)
    root, paths, records, fingerprints, thresholds = load_dataset(config, split, scales)
    configure_cpu()
    from ultralytics import YOLO

    model = YOLO(str(weight))
    if set(model.names) != {0}:
        raise ValueError(f"Checkpoint must be single-class: {model.names}")
    directory.mkdir(parents=True, exist_ok=True)
    input_list = directory / f"inputs_{split}.txt"
    input_list.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    by_path = {path: record for path, record in zip(paths, records, strict=True)}
    seen = set()
    for result in model.predict(source=str(input_list), stream=True, save=False, verbose=False, **SETTINGS):
        path = Path(result.path).resolve()
        if path not in by_path or path in seen:
            raise ValueError(f"Unexpected or duplicate model result: {path}")
        record = by_path[path]
        if result.orig_shape != (record["height"], record["width"]):
            raise ValueError(f"Image dimensions disagree with metadata: {path}")
        classes = result.boxes.cls.cpu().tolist()
        if any(value != 0 for value in classes):
            raise ValueError(f"Unexpected prediction class: {path}")
        xyxy = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        # Sub-pixel model boxes have zero area and can never be true positives.
        keep = [i for i, box in enumerate(xyxy) if box[2] > box[0] and box[3] > box[1]]
        record.update(boxes=[xyxy[i] for i in keep], scores=[scores[i] for i in keep],
                      class_ids=[0] * len(keep))
        seen.add(path)
        if len(seen) % 100 == 0:
            print(f"{split}: {len(seen)}/{len(paths)} images", flush=True)
    if len(seen) != len(records):
        raise ValueError("Inference did not return one record per split image")
    provenance = {
        "schema_version": 1, "model_id": model_id, "weight_sha256": sha256(weight),
        "dataset_config": str(Path(config).resolve()), "dataset_root": str(root), "split": split,
        "scales": sorted(scales),
        "timestamp": datetime.now(UTC).isoformat(), "run_id": run_id, "seed": 42, "cpu_threads": 1,
        "python": sys.version, "packages": {name: importlib.metadata.version(name)
                                            for name in ("torch", "torchvision", "ultralytics", "numpy", "PyYAML",
                                                         "opencv-python", "opencv-python-headless")},
        "inference": SETTINGS, "input_sha256": fingerprints, "size_thresholds": thresholds,
    }
    write_json(directory / f"predictions_{split}.json", {"provenance": provenance, "images": records})
    metrics = {**provenance, **compute_metrics(records)}
    if scales != frozenset({1.0}):
        metrics["by_scale"] = {f"{s:g}": compute_metrics([r for r in records if r["scale"] == s])
                               for s in sorted(scales)}
    if validate:
        # Use a portable generated YAML; never allow Ultralytics to fetch a dataset.
        validation_config = directory / f"dataset_{split}.yaml"
        validation_config.write_text(yaml.safe_dump({"path": str(root), "train": str(input_list),
                                                    "val": str(input_list), "test": str(input_list),
                                                    "nc": 1, "names": {0: "license_plate"}}), encoding="utf-8")
        result = model.val(data=str(validation_config), split=split, workers=0, plots=False,
                           save_json=False, project=str(directory), name=f"sanity_{split}",
                           exist_ok=True, seed=42, deterministic=True, **SETTINGS)
        actual = float(result.box.map50)
        delta = abs(actual - metrics["overall"]["map50"])
        metrics["ultralytics_sanity"] = {"map50": actual, "absolute_difference": delta, "passed": delta <= 0.01}
        write_json(directory / f"metrics_{split}.json", metrics)
        if delta > 0.01:
            raise RuntimeError(f"Custom AP differs from Ultralytics by {delta:.6f} (limit 0.01)")
    else:
        write_json(directory / f"metrics_{split}.json", metrics)
    print(json.dumps({"results": str(directory), "overall": metrics["overall"]}, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/dataset/dataset_original.yaml")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    parser.add_argument("--validate", action="store_true", help="Compare custom AP to Ultralytics val()")
    parser.add_argument("--model", type=Path, default=None,
                        help="Alternative checkpoint path; its SHA-256 is recorded in the metrics JSON")
    parser.add_argument("--scales", default="1.0",
                        help="Comma-separated dataset scales to evaluate (0.5, 0.75, 1.0); default 1.0")
    args = parser.parse_args()
    evaluate(args.data, args.split, args.run_id, args.validate, args.model, parse_scales(args.scales))


if __name__ == "__main__":
    main()
