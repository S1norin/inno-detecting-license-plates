"""Validate the fixed original-scale fine-tuning configuration."""

import re
from pathlib import Path

import yaml

from baseline.common.weights import ROOT, verified_path

DEFAULT_CONFIG = Path(__file__).with_name("train_config.yaml")
OUTPUT_ROOT = ROOT / "models/yolo11_finetune/results"
FIELDS = {"model", "data", "epochs", "batch", "imgsz", "patience", "seed", "deterministic", "project", "name"}


def read_config(path=DEFAULT_CONFIG, *, data=None, epochs=None, name=None):
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != FIELDS:
        raise ValueError(f"Training config must contain exactly: {', '.join(sorted(FIELDS))}")
    document = dict(document)
    for key, value in (("data", data), ("epochs", epochs), ("name", name)):
        if value is not None:
            document[key] = value

    for key in ("epochs", "batch", "imgsz", "patience"):
        if type(document[key]) is not int or document[key] <= 0:
            raise ValueError(f"Training {key} must be a positive integer")
    if document["epochs"] > 100 or document["imgsz"] != 640:
        raise ValueError("Use at most 100 epochs and image size 640")
    if type(document["seed"]) is not int or document["seed"] != 42 or document["deterministic"] is not True:
        raise ValueError("Training requires seed 42 and deterministic: true")
    if not isinstance(document["name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", document["name"]):
        raise ValueError("Training name must be 1-80 filename-safe characters")

    for key in ("model", "data", "project"):
        if not isinstance(document[key], (str, Path)) or not str(document[key]).strip():
            raise ValueError(f"Training {key} must be a nonempty local path")
        document[key] = str((ROOT / document[key]).resolve())
    if not Path(document["project"]).is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Training project must be inside models/yolo11_finetune/results")
    return document


def starting_weights(document):
    weight = verified_path()
    if Path(document["model"]) != weight.resolve():
        raise ValueError("Training model must reference the verified cached detector")
    return weight
