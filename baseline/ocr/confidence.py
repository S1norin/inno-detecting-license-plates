"""Plate confidence scores and validation-only threshold selection."""

import math

from .io import normalize

SCORE_FUNCTIONS = ("mean", "product", "sumlog")
THRESHOLDS = tuple(value / 100 for value in range(50, 100))


def plate_score(probabilities, function="mean"):
    if function not in SCORE_FUNCTIONS:
        raise ValueError(f"Unknown score function: {function}")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Character probabilities must be finite and in [0, 1]")
    if not probabilities:
        return -math.inf if function == "sumlog" else 0.0
    if function == "mean":
        return sum(probabilities) / len(probabilities)
    if function == "product":
        return math.prod(probabilities)
    return sum(math.log(p) if p > 0 else -math.inf for p in probabilities)


def accepts(record, tau, function="mean"):
    if not 0 <= tau <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    if record["status"] != "read" or not normalize(record["text"]):
        return False
    threshold = math.log(tau) if function == "sumlog" and tau > 0 else tau
    if function == "sumlog" and tau == 0:
        threshold = -math.inf
    return plate_score(record["char_confidences"], function) >= threshold


def sweep(records, function="mean"):
    if not records:
        raise ValueError("Cannot select a threshold from an empty validation set")

    curve = []
    for tau in THRESHOLDS:
        accepted = [record for record in records if accepts(record, tau, function)]
        correct = sum(normalize(r["text"]) == normalize(r["ground_truth"]) for r in accepted)
        accuracy = correct / len(accepted) if accepted else 0.0
        coverage = len(accepted) / len(records)
        curve.append(
            {
                "tau": tau,
                "accuracy": accuracy,
                "coverage": coverage,
                "objective": accuracy + coverage - 1,
                "accepted": len(accepted),
            }
        )

    selected = max(curve, key=lambda point: (point["objective"], -point["tau"]))
    return {"score_function": function, "selected_tau": selected["tau"], "curve": curve}
