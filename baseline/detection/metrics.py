"""Metrics from durable single-class predictions, with no model dependency."""

import argparse
import json
from pathlib import Path

import numpy as np

SIZE_GROUPS = ("small", "medium", "large")
TILT_GROUPS = ("0-9", "10-19", "20+")


def tilt_bin(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid tilt magnitude: {value}")
    return "0-9" if value < 10 else "10-19" if value < 20 else "20+"


def box_iou(truth, predictions):
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1, 4)
    low = np.maximum(truth[:, None, :2], predictions[None, :, :2])
    high = np.minimum(truth[:, None, 2:], predictions[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=2)
    areas_truth = np.maximum(truth[:, 2:] - truth[:, :2], 0).prod(axis=1)
    areas_pred = np.maximum(predictions[:, 2:] - predictions[:, :2], 0).prod(axis=1)
    union = areas_truth[:, None] + areas_pred[None, :] - intersection
    return intersection / np.maximum(union, 1e-7)


def match(truth, predictions, threshold=0.5):
    """Use the Ultralytics 8.3.162 validator's default one-to-one assignment."""
    iou = box_iou(truth, predictions)
    correct = np.zeros(iou.shape[1], dtype=bool)
    matches = np.array(np.nonzero(iou >= threshold)).T
    if len(matches) > 1:
        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    if len(matches):
        correct[matches[:, 1]] = True
    return correct


def average_precision(correct, scores, n_truth):
    if not n_truth:
        return None
    if not len(scores):
        return 0.0
    correct = np.asarray(correct)[np.argsort(-np.asarray(scores))]
    tp = np.cumsum(correct)
    fp = np.cumsum(~correct)
    recall = tp / (n_truth + 1e-16)
    precision = tp / (tp + fp)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    grid = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(grid, mrec, mpre), grid))


def summarize(records, confidence=0.25, iou=0.5):
    scores, correct = [], []
    n_truth = n_predictions = tp = fp = 0
    for record in records:
        truth = record["ground_truth"]
        predictions = record["boxes"]
        image_scores = np.asarray(record["scores"], dtype=float)
        if len(predictions) != len(image_scores):
            raise ValueError("Box and score counts differ")
        if not np.isfinite(image_scores).all() or np.any((image_scores < 0) | (image_scores > 1)):
            raise ValueError("Prediction scores must be finite and in [0, 1]")
        scores.extend(image_scores.tolist())
        correct.extend(match(truth, predictions, iou).tolist())
        selected = np.asarray(predictions, dtype=float).reshape(-1, 4)[image_scores >= confidence]
        image_tp = int(match(truth, selected, iou).sum())
        tp += image_tp
        fp += len(selected) - image_tp
        n_truth += len(truth)
        n_predictions += len(predictions)

    fn = n_truth - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "image_count": len(records), "ground_truth_count": n_truth, "prediction_count": n_predictions,
        "tp": tp, "fp": fp, "fn": fn, "map50": average_precision(correct, scores, n_truth),
        "precision": precision if n_truth else None, "recall": recall if n_truth else None,
        "f1": f1 if n_truth else None,
    }


def compute_metrics(records, confidence=0.25, iou=0.5):
    if not 0 <= confidence <= 1 or not 0 < iou <= 1:
        raise ValueError("Invalid confidence or IoU threshold")
    paths = [r["image_path"] for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate prediction image identifiers")
    for record in records:
        if record["size_bucket"] not in SIZE_GROUPS:
            raise ValueError(f"Unknown size bucket: {record['size_bucket']}")
        for key in ("ground_truth", "boxes"):
            boxes = np.asarray(record[key], dtype=float)
            if boxes.size and (boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all()):
                raise ValueError(f"Invalid {key} boxes")
            if boxes.size and np.any(boxes[:, 2:] <= boxes[:, :2]):
                raise ValueError(f"Degenerate {key} boxes")
        tilt_bin(record["tilt_magnitude"])
    return {
        "iou_threshold": iou, "confidence_threshold": confidence,
        "overall": summarize(records, confidence, iou),
        "by_size": {group: summarize([r for r in records if r["size_bucket"] == group], confidence, iou)
                    for group in SIZE_GROUPS},
        "by_tilt": {group: summarize([r for r in records if tilt_bin(r["tilt_magnitude"]) == group], confidence, iou)
                    for group in TILT_GROUPS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = {**artifact["provenance"], **compute_metrics(artifact["images"])}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
