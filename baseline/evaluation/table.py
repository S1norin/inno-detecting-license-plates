"""Consolidated comparison table built from frozen stage results.

Every row is computed from the inputs as-is: detection metrics come from the
frozen metrics files, recognition rows come from the frozen validation caches
(recomputed with the shared metrics functions) and the frozen test reports.
Nothing in this module runs inference.
"""

import hashlib
import json

from baseline.ocr.io import MODES, PROTOCOLS, join_tilt
from baseline.ocr.metrics import evaluate

ABLATION_TAU = 0.50
DETECTION_KEYS = (
    "image_count",
    "ground_truth_count",
    "prediction_count",
    "tp",
    "fp",
    "fn",
    "map50",
    "precision",
    "recall",
    "f1",
)
DELTA_KEYS = (
    "character_accuracy",
    "plate_exact_match",
    "accepted_exact_accuracy",
    "coverage",
    "precision",
    "recall",
    "f1",
)


def _detection_section(metrics):
    overall = metrics["overall"]
    return {
        "overall": {key: overall[key] for key in DETECTION_KEYS},
        "by_size": metrics["by_size"],
        "by_tilt": metrics["by_tilt"],
    }


def _delta(later, earlier):
    return {key: later[key] - earlier[key] for key in DELTA_KEYS}


def _ocr_section(raw, dataset_root, tau, score_function):
    records = join_tilt(raw["records"], dataset_root)
    return evaluate(records, tau, score_function)


def build_table(dataset, detection, ocr):
    e1 = {split: _detection_section(detection.metrics[split]) for split in ("val", "test")}

    e2 = {}
    for protocol in PROTOCOLS:
        policy = ocr.selection[protocol]
        mode = ocr.selected_mode[protocol]
        tau = ocr.selected_tau[protocol]
        e2[protocol] = {
            "selected": {
                "mode": mode,
                "tau": tau,
                "score_function": policy["score_function"],
            },
            "val": _ocr_section(ocr.raw[(protocol, mode)], dataset.root, tau, policy["score_function"]),
            "test": ocr.test_report[protocol]["metrics"],
        }
    e2b = e2["end_to_end"]
    e2b["delta_vs_ocr_only"] = {
        "val": _delta(e2b["val"]["overall"], e2["ocr_only"]["val"]["overall"]),
        "test": _delta(e2b["test"]["overall"], e2["ocr_only"]["test"]["overall"]),
    }

    e3 = {}
    none_overall = None
    for mode in MODES:
        section = _ocr_section(ocr.raw[("ocr_only", mode)], dataset.root, ABLATION_TAU, "mean")
        if mode == "none":
            none_overall = section["overall"]
        section["delta_vs_none"] = _delta(section["overall"], none_overall) if mode != "none" else None
        e3[mode] = section

    e6 = {
        protocol: {
            "tau": ocr.selected_tau[protocol],
            "accuracy": e2[protocol]["test"]["overall"]["accepted_exact_accuracy"],
            "coverage": e2[protocol]["test"]["overall"]["coverage"],
        }
        for protocol in PROTOCOLS
    }

    return {
        "e1_detection": e1,
        "e2_ocr_only": e2["ocr_only"],
        "e2b_end_to_end": e2b,
        "e3_ablation": e3,
        "e6_threshold": e6,
    }


def run_id(dataset, detection, ocr):
    digests = {"dataset": dataset.digests, "detection": detection.digests, "ocr": ocr.digests}
    payload = json.dumps(digests, sort_keys=True).encode("utf-8")
    return "eval-" + hashlib.sha256(payload).hexdigest()[:16]


def make_document(dataset, detection, ocr, timestamp):
    return {
        "run_id": run_id(dataset, detection, ocr),
        "timestamp": timestamp,
        "inputs": {
            "dataset": {
                "path": str(dataset.root),
                "digests": dataset.digests,
                "sample_count": dataset.sample_count,
            },
            "detection": {
                "path": str(detection.directory),
                "model": detection.model_id,
                "digests": detection.digests,
            },
            "ocr": {
                "path": str(ocr.directory),
                "engine": ocr.engine,
                "digests": ocr.digests,
            },
        },
        "table": build_table(dataset, detection, ocr),
    }
