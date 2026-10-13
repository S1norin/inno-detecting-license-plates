"""Failure-case extraction with deterministic ranking and cause tags.

Detection failures come from the frozen predictions of both splits;
recognition failures come from the validation cache of each protocol's
selected mode only, so no test-split sample ever appears in the report.
"""

from baseline.ocr.confidence import accepts
from baseline.ocr.io import PROTOCOLS, join_tilt, normalize
from baseline.ocr.metrics import edit_distance

IOU_THRESHOLD = 0.5
LOW_CONFIDENCE = 0.25
TOP_N = 5


def _box_iou(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _detection_case(record):
    truth = record["ground_truth"]
    boxes = record["boxes"]
    scores = record["scores"]
    matched = 0
    best_iou = 0.0
    best_score = 0.0
    for gt in truth:
        hit = False
        for box, score in zip(boxes, scores, strict=True):
            iou = _box_iou(gt, box)
            if iou > best_iou:
                best_iou, best_score = iou, score
            if iou >= IOU_THRESHOLD:
                hit = True
        matched += hit

    tags = []
    if matched < len(truth):
        tags.append("miss")
    if 0.0 < best_iou < IOU_THRESHOLD:
        tags.append("low_iou")
    if len(boxes) > len(truth):
        tags.append("multiple_detections")
    if boxes and best_score < LOW_CONFIDENCE:
        tags.append("low_confidence")
    if not tags:
        return None
    return {
        "image_path": record["image_path"],
        "size_bucket": record["size_bucket"],
        "tilt_magnitude": record["tilt_magnitude"],
        "tags": tags,
        "details": {
            "ground_truth_count": len(truth),
            "prediction_count": len(boxes),
            "matched": matched,
            "best_iou": round(best_iou, 4),
            "best_score": best_score,
        },
    }


def _detection_top(records):
    cases = [case for case in (_detection_case(r) for r in records) if case]
    cases.sort(key=lambda c: (c["details"]["matched"] >= c["details"]["ground_truth_count"],
                              c["details"]["best_iou"], c["details"]["best_score"], c["image_path"]))
    return cases[:TOP_N]


def _recognition_case(record, tau, function):
    gt = normalize(record["ground_truth"])
    text = normalize(record["text"])
    status = record["status"]
    if status == "no_detection":
        tags, distance, accepted = ["no_detection"], len(gt), False
    elif status == "unread":
        tags, distance, accepted = ["unread"], len(gt), False
    else:
        accepted = accepts(record, tau, function)
        distance = edit_distance(text, gt)
        if not accepted:
            tags = ["abstained"]
        elif distance > 1:
            tags = ["substitution"]
            if len(text) < len(gt):
                tags.append("missing_chars")
            if len(text) > len(gt):
                tags.append("extra_chars")
        else:
            return None
    return {
        "sample_id": record["sample_id"],
        "size_bucket": record["size_bucket"],
        "tilt_magnitude": record["tilt_magnitude"],
        "tags": tags,
        "details": {
            "ground_truth": gt,
            "text": text,
            "status": status,
            "distance": distance,
            "accepted": accepted,
            "tau": tau,
        },
    }


def _recognition_top(records, tau, function):
    cases = [case for case in (_recognition_case(r, tau, function) for r in records) if case]
    cases.sort(key=lambda c: (-c["details"]["distance"], c["sample_id"]))
    return cases[:TOP_N]


def build_failures(dataset, detection, ocr):
    failure = {
        "detection": {
            split: _detection_top(detection.predictions[split]) for split in ("val", "test")
        },
        "recognition": {},
    }
    for protocol in PROTOCOLS:
        mode = ocr.selected_mode[protocol]
        tau = ocr.selected_tau[protocol]
        function = ocr.selection[protocol]["score_function"]
        records = join_tilt(ocr.raw[(protocol, mode)]["records"], dataset.root)
        failure["recognition"][protocol] = {
            "mode": mode,
            "tau": tau,
            "score_function": function,
            "cases": _recognition_top(records, tau, function),
        }
    return failure
