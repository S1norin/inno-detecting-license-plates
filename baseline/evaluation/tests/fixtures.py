"""Synthetic results trees for the evaluation stack tests.

Builds a small, internally consistent dataset + detection + OCR results tree
using the project's own metric functions, so every fixture number is
hand-checkable and all cross-file digests are real.
"""

import csv
import hashlib
import json
from pathlib import Path

from baseline.detection.metrics import compute_metrics
from baseline.ocr.confidence import sweep
from baseline.ocr.io import MODES, digest
from baseline.ocr.metrics import evaluate

DETECTION_MODEL = "yolo11s"
DETECTION_WEIGHT_SHA = "95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381"
EASYOCR_WEIGHT_SHA = "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17"
PREPROCESSING_SHA = "ab" * 32
TIMESTAMP = "2025-01-01T00:00:00+00:00"

GT_BOX = [10.0, 10.0, 50.0, 30.0]

# parent_id: (split, size_bucket, tilt, plate_text)
SAMPLES = {
    "p1": ("val", "large", 5.0, "AB123CD"),
    "p2": ("val", "medium", 15.0, "X5Y6Z7W"),
    "p3": ("val", "small", 25.0, "Q9R8S7T"),
    "p4": ("val", "large", 30.0, "M4N3K2J"),
    "p9": ("val", "medium", 12.0, "Z9Y8X7W"),
    "p10": ("val", "small", 3.0, "B7C6D5E"),
    "p11": ("val", "large", 28.0, "F4G3H2J"),
    "p12": ("val", "medium", 18.0, "K1L2M3N"),
    "p5": ("test", "small", 10.0, "T1ABC99"),
    "p6": ("test", "medium", 5.0, "T2DEF88"),
    "p7": ("test", "large", 22.0, "T3GHI77"),
    "p8": ("test", "small", 40.0, "T4JKL66"),
}

# Detection predictions per (split, parent_id): list of (box, score)
DETECTION_BOXES = {
    "p1": [([10.0, 10.0, 50.0, 30.0], 0.9)],
    "p2": [([12.0, 11.0, 52.0, 31.0], 0.8)],
    "p3": [([11.0, 10.0, 49.0, 30.0], 0.7)],
    "p4": [([10.0, 12.0, 50.0, 28.0], 0.6)],
    "p9": [([10.0, 10.0, 50.0, 30.0], 0.85)],
    "p10": [([10.0, 10.0, 50.0, 30.0], 0.75)],
    "p11": [([10.0, 10.0, 50.0, 30.0], 0.7)],
    "p12": [([10.0, 10.0, 50.0, 30.0], 0.65)],
    # test: one miss, one low-IoU, one multi-box, one clean
    "p5": [([80.0, 80.0, 95.0, 90.0], 0.15)],
    "p6": [([1.0, 10.0, 31.0, 30.0], 0.55)],
    "p7": [([10.0, 10.0, 50.0, 30.0], 0.9), ([15.0, 10.0, 55.0, 30.0], 0.8)],
    "p8": [([10.0, 10.0, 50.0, 30.0], 0.95)],
}

# OCR outcomes per (protocol, mode, parent_id): (status, text, char_confidences)
OCR_OUTCOMES = {
    "ocr_only": {
        "none": {
            "p1": ("read", "AB123CD", [0.9] * 7),
            "p2": ("read", "X5Y6Z7W", [0.9] * 7),
            "p3": ("read", "Q1R2S7T", [0.85, 0.7, 0.85, 0.7, 0.9, 0.9, 0.9]),
            "p4": ("unread", "", []),
            "p9": ("read", "Z9Y8X7W", [0.9] * 7),
            "p10": ("unread", "", []),
            "p11": ("read", "F4H2J", [0.6] * 5),
            "p12": ("read", "K1L2M3N", [0.4] * 7),
        },
        "gray_contrast": {
            "p1": ("read", "AB123CD", [0.9] * 7),
            "p2": ("read", "X5Y6Z7W", [0.9] * 7),
            "p3": ("read", "Q9R8S7T", [0.9] * 7),
            "p4": ("unread", "", []),
            "p9": ("read", "Z98X71", [0.9] * 6),
            "p10": ("read", "B7C6D5E", [0.8] * 7),
            "p11": ("read", "F4G3H2", [0.8] * 6),
            "p12": ("read", "K1L2M3N", [0.9] * 7),
        },
        "canny": {
            "p1": ("read", "AB133CD", [0.9, 0.9, 0.7, 0.6, 0.9, 0.9, 0.9]),
            "p2": ("read", "X5Y6Z7W", [0.8] * 7),
            "p3": ("unread", "", []),
            "p4": ("read", "M4N3K2J", [0.9] * 7),
            "p9": ("read", "Z9Y8X71", [0.85] * 7),
            "p10": ("read", "B7C6D5E", [0.8] * 7),
            "p11": ("unread", "", []),
            "p12": ("read", "K1L2M3X", [0.8] * 7),
        },
    },
    "end_to_end": {
        "none": {
            "p1": ("read", "AB123CD", [0.9] * 7),
            "p2": ("read", "X5Y6Z7W", [0.9] * 7),
            "p3": ("no_detection", "", []),
            "p4": ("read", "M4N3K2", [0.5, 0.4, 0.4, 0.5, 0.5, 0.4]),
            "p9": ("read", "Z9Y8X7W", [0.85] * 7),
            "p10": ("no_detection", "", []),
            "p11": ("read", "F4G3H2", [0.6] * 6),
            "p12": ("read", "K1L2M3N", [0.5] * 7),
        },
        "gray_contrast": {
            "p1": ("read", "AB123CD", [0.9] * 7),
            "p2": ("read", "X5Y6Z7W", [0.9] * 7),
            "p3": ("no_detection", "", []),
            "p4": ("unread", "", []),
            "p9": ("read", "Z9Y8X71", [0.9] * 7),
            "p10": ("read", "B7C6D5E", [0.4] * 7),
            "p11": ("unread", "", []),
            "p12": ("read", "K1L2M3N", [0.9] * 7),
        },
        "canny": {
            "p1": ("read", "AB123CD", [0.9] * 7),
            "p2": ("read", "X5Y6Z7W", [0.8] * 7),
            "p3": ("no_detection", "", []),
            "p4": ("read", "M4N3K2J", [0.6] * 7),
            "p9": ("read", "Z9Y8X7W", [0.85] * 7),
            "p10": ("read", "B7C6D1", [0.7] * 6),
            "p11": ("unread", "", []),
            "p12": ("read", "K1L2M3X", [0.8] * 7),
        },
    },
}

# Test-split outcomes use the selected mode only; the generator writes them for
# every mode and the test report picks the selected one.
OCR_TEST_OUTCOMES = {
    "ocr_only": {
        "none": {
            "p5": ("read", "T1ABC99", [0.9] * 7),
            "p6": ("read", "T2DEF88", [0.9] * 7),
            "p7": ("read", "T5GHI77", [0.85, 0.7, 0.9, 0.7, 0.9, 0.9, 0.9]),
            "p8": ("unread", "", []),
        },
        "gray_contrast": {
            "p5": ("read", "T1ABC99", [0.9] * 7),
            "p6": ("read", "T2DEF88", [0.9] * 7),
            "p7": ("read", "T5GHI77", [0.85, 0.7, 0.9, 0.7, 0.9, 0.9, 0.9]),
            "p8": ("unread", "", []),
        },
        "canny": {
            "p5": ("read", "T1ABC99", [0.9] * 7),
            "p6": ("read", "T2DEF88", [0.9] * 7),
            "p7": ("read", "T5GHI77", [0.85, 0.7, 0.9, 0.7, 0.9, 0.9, 0.9]),
            "p8": ("unread", "", []),
        },
    },
    "end_to_end": {
        "none": {
            "p5": ("read", "T1ABC99", [0.9] * 7),
            "p6": ("read", "T2DEF88", [0.9] * 7),
            "p7": ("no_detection", "", []),
            "p8": ("read", "T4JKL66", [0.9] * 7),
        },
        "gray_contrast": {
            "p5": ("read", "T1ABC99", [0.9] * 7),
            "p6": ("read", "T2DEF88", [0.9] * 7),
            "p7": ("read", "T3GHI77", [0.8] * 7),
            "p8": ("read", "T4JKL66", [0.9] * 7),
        },
        "canny": {
            "p5": ("read", "T1ABC99", [0.9] * 7),
            "p6": ("read", "T2DEF88", [0.9] * 7),
            "p7": ("no_detection", "", []),
            "p8": ("read", "T4JKL66", [0.9] * 7),
        },
    },
}


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _dataset(root):
    dataset = root / "dataset"
    meta_header = [
        "image_path", "label_path", "crop_path", "source_relpath", "split", "parent_id",
        "scale", "width", "height", "x1", "y1", "x2", "y2", "plate_pixel_height",
        "size_bucket", "original_bucket", "tilt_horizontal", "tilt_vertical",
        "tilt_magnitude", "plate_text",
    ]
    meta_rows = []
    ocr_rows = {"val": [], "test": []}
    for parent_id, (split, bucket, tilt, plate) in SAMPLES.items():
        name = f"{parent_id}_s100"
        x1, y1, x2, y2 = (int(value) for value in GT_BOX)
        meta_rows.append(
            {
                "image_path": f"images/{split}/{name}.jpg",
                "label_path": f"labels/{split}/{name}.txt",
                "crop_path": f"crops_gt/{split}/{name}.png",
                "source_relpath": f"CCPD2019/ccpd_base/{name}.jpg",
                "split": split,
                "parent_id": parent_id,
                "scale": "1.0",
                "width": "100",
                "height": "100",
                "x1": str(x1),
                "y1": str(y1),
                "x2": str(x2),
                "y2": str(y2),
                "plate_pixel_height": "20",
                "size_bucket": bucket,
                "original_bucket": bucket,
                "tilt_horizontal": str(int(tilt)),
                "tilt_vertical": "0",
                "tilt_magnitude": str(tilt),
                "plate_text": plate,
            }
        )
        ocr_rows[split].append(
            {
                "crop_path": f"crops_gt/{split}/{name}.png",
                "plate_text": plate,
                "split": split,
                "parent_id": parent_id,
                "scale": "1.0",
                "size_bucket": bucket,
            }
        )

    ocr_header = ["crop_path", "plate_text", "split", "parent_id", "scale", "size_bucket"]
    _write_csv(dataset / "metadata.csv", meta_header, meta_rows)
    _write_csv(dataset / "ocr_val.csv", ocr_header, ocr_rows["val"])
    _write_csv(dataset / "ocr_test.csv", ocr_header, ocr_rows["test"])
    return dataset


def _detection(root):
    directory = root / "detection"
    provenance_base = {"model_id": DETECTION_MODEL, "timestamp": TIMESTAMP}
    for split in ("val", "test"):
        records = []
        for parent_id, (sample_split, bucket, tilt, _plate) in SAMPLES.items():
            if sample_split != split:
                continue
            boxes = [box for box, _score in DETECTION_BOXES[parent_id]]
            scores = [score for _box, score in DETECTION_BOXES[parent_id]]
            records.append(
                {
                    "image_path": f"images/{split}/{parent_id}_s100.jpg",
                    "width": 100,
                    "height": 100,
                    "ground_truth": [GT_BOX],
                    "size_bucket": bucket,
                    "tilt_magnitude": tilt,
                    "boxes": boxes,
                    "scores": scores,
                    "class_ids": [0] * len(boxes),
                }
            )
        provenance = {"split": split, **provenance_base}
        _write_json(directory / f"predictions_{split}.json", {"provenance": provenance, "images": records})
        _write_json(directory / f"metrics_{split}.json", {**provenance, **compute_metrics(records)})
    return directory


def _ocr_records(protocol, split, mode):
    outcomes = OCR_TEST_OUTCOMES[protocol][mode] if split == "test" else OCR_OUTCOMES[protocol][mode]
    records = []
    for parent_id, (sample_split, bucket, _tilt, plate) in SAMPLES.items():
        if sample_split != split:
            continue
        status, text, confidences = outcomes[parent_id]
        records.append(
            {
                "sample_id": f"{parent_id}_s100",
                "size_bucket": bucket,
                "ground_truth": plate,
                "status": status,
                "text": text,
                "char_confidences": confidences,
            }
        )
    return records


def _ocr_metadata(protocol, split, mode, dataset_digests):
    metadata = {
        "protocol": protocol,
        "split": split,
        "mode": mode,
        "scale": 1.0,
        "sample_count": sum(1 for entry in SAMPLES.values() if entry[0] == split),
        "engine": "easyocr",
        "weight_digests": {"craft_mlt_25k.pth": EASYOCR_WEIGHT_SHA},
        "dataset_digests": dataset_digests,
        "timestamp": TIMESTAMP,
        "preprocessing_sha256": PREPROCESSING_SHA,
    }
    if protocol == "end_to_end":
        metadata["detector"] = {"model_id": DETECTION_MODEL, "weight_sha256": DETECTION_WEIGHT_SHA}
    return metadata


def _contract(metadata):
    contract = {field: metadata[field] for field in ("protocol", "scale", "engine", "weight_digests")}
    contract["metadata_sha256"] = metadata["dataset_digests"]["metadata.csv"]
    contract["preprocessing_sha256"] = metadata["preprocessing_sha256"]
    if metadata["protocol"] == "end_to_end":
        contract["detector"] = {field: metadata["detector"][field] for field in ("model_id", "weight_sha256")}
    return contract


def _selection_for_protocol(protocol, raw_files, dataset_digests, score_function="mean"):
    candidates = {}
    for mode in MODES:
        records = _ocr_records(protocol, "val", mode)
        policy = sweep(records, score_function)
        raw_path = raw_files[mode]
        candidates[mode] = {
            "mode": mode,
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": digest(raw_path),
            "contract": _contract(_ocr_metadata(protocol, "val", mode, dataset_digests)),
            "validation_csv_sha256": dataset_digests["ocr_val.csv"],
            "sample_ids": sorted(record["sample_id"] for record in records),
            **policy,
            "metrics": evaluate(records, policy["selected_tau"], score_function),
        }
    best = max(
        candidates.values(),
        key=lambda candidate: (
            max(point["objective"] for point in candidate["curve"]),
            -candidate["selected_tau"],
            -MODES.index(candidate["mode"]),
        ),
    )
    return {
        "mode": best["mode"],
        "tau": best["selected_tau"],
        "score_function": score_function,
        "contract": best["contract"],
        "candidates": candidates,
    }


def _ocr(root, dataset):
    dataset_digests = {
        "ocr_val.csv": _sha256(dataset / "ocr_val.csv"),
        "ocr_test.csv": _sha256(dataset / "ocr_test.csv"),
        "metadata.csv": _sha256(dataset / "metadata.csv"),
    }
    directory = root / "ocr"
    protocols = {}
    for protocol in ("ocr_only", "end_to_end"):
        folder = directory / protocol
        raw_files = {}
        for mode in MODES:
            metadata = _ocr_metadata(protocol, "val", mode, dataset_digests)
            records = _ocr_records(protocol, "val", mode)
            raw_files[mode] = folder / f"raw_{protocol}_val_{mode}.json"
            _write_json(raw_files[mode], {"schema_version": 1, "metadata": metadata, "records": records})

        selection = _selection_for_protocol(protocol, raw_files, dataset_digests)
        _write_json(folder / f"selection_{protocol}_val.json", {
            "schema_version": 1,
            "selection_split": "val",
            "timestamp": TIMESTAMP,
            "protocols": {protocol: selection},
        })

        mode = selection["mode"]
        tau = selection["tau"]
        records = _ocr_records(protocol, "test", mode)
        metadata = _ocr_metadata(protocol, "test", mode, dataset_digests)
        report = {
            "schema_version": 1,
            "metadata": {
                **metadata,
                "timestamp": TIMESTAMP,
                "score_function": selection["score_function"],
                "tau": tau,
                "selection_timestamp": TIMESTAMP,
            },
            "metrics": evaluate(records, tau, selection["score_function"]),
        }
        _write_json(folder / f"report_{protocol}_test.json", report)
        protocols[protocol] = {"mode": mode, "tau": tau}
    return directory, protocols


def build(root):
    """Materialize the full fixture tree under root; return the key paths."""
    root = Path(root)
    dataset = _dataset(root)
    detection = _detection(root)
    ocr, selections = _ocr(root, dataset)
    return {
        "root": root,
        "dataset": dataset,
        "detection": detection,
        "ocr": ocr,
        "selections": selections,
        "model": DETECTION_MODEL,
    }
