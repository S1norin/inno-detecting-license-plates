"""Recognition measures using the conventions in issue #7."""

import math

from .confidence import accepts
from .io import BUCKETS, normalize

TILT_BINS = ("0-9", "10-19", "20+")


def tilt_bin(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Tilt must be finite")
    if value < 10:
        return "0-9"
    if value < 20:
        return "10-19"
    return "20+"


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, character in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (character != other)))
        previous = current
    return previous[-1]


def summarize(records, tau, function="mean"):
    count = len(records)
    accepted = exact = tp = fp = fn = 0
    character_total = 0.0
    for record in records:
        target = normalize(record["ground_truth"])
        if not target:
            raise ValueError("Empty ground truth")
        if not accepts(record, tau, function):
            fn += 1
            continue

        predicted = normalize(record["text"])
        distance = edit_distance(predicted, target)
        accepted += 1
        exact += predicted == target
        character_total += 1 - distance / max(len(predicted), len(target))
        if distance <= 1:
            tp += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "count": count,
        "accepted": accepted,
        "coverage": accepted / count if count else 0.0,
        "character_accuracy": character_total / count if count else 0.0,
        "plate_exact_match": exact / count if count else 0.0,
        "accepted_exact_accuracy": exact / accepted if accepted else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate(records, tau, function="mean"):
    result = {
        "overall": summarize(records, tau, function),
        "by_size_bucket": {
            bucket: summarize([r for r in records if r["size_bucket"] == bucket], tau, function) for bucket in BUCKETS
        },
    }
    if all("tilt_magnitude" in record for record in records):
        result["by_tilt_bin"] = {
            bin_name: summarize([r for r in records if tilt_bin(r["tilt_magnitude"]) == bin_name], tau, function)
            for bin_name in TILT_BINS
        }
    return result
