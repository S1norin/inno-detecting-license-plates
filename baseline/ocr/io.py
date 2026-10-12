"""Dataset and result formats shared by the two recognition protocols."""

import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

MODES = ("none", "gray_contrast", "canny")
PROTOCOLS = ("ocr_only", "end_to_end")
BUCKETS = ("small", "medium", "large")
SCALES = (1.0, 0.75, 0.5)


def timestamp():
    return datetime.now(UTC).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def normalize(text):
    return text.strip().upper()


def dataset_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path


def read_table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_samples(root, split, scale=1.0):
    if split not in ("train", "val", "test") or scale not in SCALES:
        raise ValueError("Invalid split or scale")

    root = Path(root)
    metadata = {}
    for row in read_table(root / "metadata.csv"):
        key = (row["parent_id"], float(row["scale"]))
        if key in metadata:
            raise ValueError(f"Duplicate metadata key: {key}")
        metadata[key] = row

    samples = []
    seen = set()
    for row in read_table(root / f"ocr_{split}.csv"):
        if float(row["scale"]) != scale:
            continue
        key = (row["parent_id"], scale)
        if key in seen:
            raise ValueError(f"Duplicate OCR sample: {key}")
        seen.add(key)
        meta = metadata[key]
        target = normalize(row["plate_text"])
        if row["split"] != split or meta["split"] != split:
            raise ValueError(f"Split mismatch: {key}")
        if not target or target != normalize(meta["plate_text"]):
            raise ValueError(f"Ground-truth mismatch: {key}")
        if row["crop_path"] != meta["crop_path"] or row["size_bucket"] != meta["size_bucket"]:
            raise ValueError(f"Crop/bucket mismatch: {key}")
        if row["size_bucket"] not in BUCKETS:
            raise ValueError(f"Unknown size bucket: {row['size_bucket']}")

        dataset_path(root, row["crop_path"])
        dataset_path(root, meta["image_path"])
        samples.append(
            {
                "sample_id": Path(row["crop_path"]).stem,
                "parent_id": row["parent_id"],
                "scale": scale,
                "size_bucket": row["size_bucket"],
                "ground_truth": target,
                "crop_path": row["crop_path"],
                "image_path": meta["image_path"],
            }
        )

    if not samples:
        raise ValueError(f"No samples for {split}, scale={scale}")
    return samples


def validate_raw(raw, *, complete=True):
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported raw-result schema")
    meta = raw["metadata"]
    if meta["protocol"] not in PROTOCOLS or meta["mode"] not in MODES:
        raise ValueError("Unknown recognition protocol or preprocessing mode")
    if meta["split"] not in ("val", "test") or meta["scale"] not in SCALES:
        raise ValueError("Invalid evaluation split or scale")
    for field in ("engine", "weight_digests", "dataset_digests", "timestamp"):
        if not meta.get(field):
            raise ValueError(f"Missing traceability field: {field}")

    seen = set()
    for record in raw["records"]:
        if record["sample_id"] in seen:
            raise ValueError("Duplicate raw sample")
        seen.add(record["sample_id"])
        if record["size_bucket"] not in BUCKETS or not normalize(record["ground_truth"]):
            raise ValueError("Invalid bucket or empty ground truth")
        if record["status"] not in ("read", "unread", "no_detection"):
            raise ValueError("Unknown recognition status")
        text = record["text"]
        probabilities = record["char_confidences"]
        if len(probabilities) != len(text):
            raise ValueError("One confidence is required for every decoded character")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("Character probabilities must be finite and in [0, 1]")
        if (record["status"] == "read") != bool(text):
            raise ValueError("Recognition text/status mismatch")

    if complete and len(seen) != meta["sample_count"]:
        raise ValueError("Raw cache is incomplete; finish inference before evaluating")
    return raw
