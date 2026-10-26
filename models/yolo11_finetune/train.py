"""Fine-tune the pinned detector on one CUDA GPU; never evaluate test."""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from baseline.common.weights import sha256
from baseline.detection.evaluate import load_dataset

from .config import DEFAULT_CONFIG, read_config, starting_weights

MAP50 = "metrics/mAP50(B)"
EXPECTED_COUNTS = {"train": 3500, "val": 700}


def write_record(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_gpu(device):
    import torch

    if type(device) is not int or device < 0:
        raise ValueError("Choose one nonnegative CUDA device index")
    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU; use the separate requirements-gpu.txt environment")
    if torch.version.cuda != "12.4":
        raise RuntimeError("Use the pinned CUDA 12.4 PyTorch wheels from requirements-gpu.txt")

    return {"hostname": socket.gethostname(), "platform": platform.platform(), "processor": platform.processor(),
            "python": sys.version, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu_index": device, "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device))}


def environment_versions():
    required = {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "ultralytics": "8.3.162",
                "numpy": "2.1.3", "PyYAML": "6.0.2", "opencv-python": "4.11.0.86",
                "opencv-python-headless": "4.11.0.86"}
    versions = {name: importlib.metadata.version(name) for name in required}
    for name, expected in required.items():
        if versions[name] != expected:
            raise RuntimeError(f"Training requires {name}=={expected}, found {versions[name]}")
    return {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()}


def training_inputs(config, directory):
    loaded = {split: load_dataset(config, split) for split in ("train", "val")}
    root = loaded["train"][0]
    if loaded["val"][0] != root:
        raise ValueError("Training and validation must use the same dataset root")
    train, val = loaded["train"][1], loaded["val"][1]
    if set(train) & set(val):
        raise ValueError("Training and validation image paths overlap")
    for split, (_, paths, _, _, _) in loaded.items():
        if len(paths) != EXPECTED_COUNTS[split]:
            raise ValueError(f"Expected {EXPECTED_COUNTS[split]} original-scale {split} images, found {len(paths)}")

    document = {"path": str(root), "nc": 1, "names": {0: "license_plate"}}
    provenance = {}
    for split, (_, paths, _, fingerprints, _) in loaded.items():
        listing = directory / f"{split}.txt"
        listing.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
        document[split] = str(listing)
        provenance[split] = {"count": len(paths), "fingerprints": fingerprints, "list_sha256": sha256(listing)}

    dataset = directory / "dataset_train_val.yaml"
    dataset.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return dataset, provenance


class TrainingRecord:
    def __init__(self, path, config, start_digest, machine, versions, inputs):
        self.path = path
        self.started = time.monotonic()
        self.document = {"schema_version": 1, "status": "started", "started_at": datetime.now(UTC).isoformat(),
                         "start_weight_sha256": start_digest, "requested_config": config, "config": None,
                         "selection_metric": MAP50, "tie_break": "first maximum", "history": [],
                         "selected_epoch": None, "best_val_map50": None, "epochs_completed": 0,
                         "library_versions": versions, "machine": machine, "inputs": inputs,
                         "reproducibility_scope": "same machine and pinned environment only"}
        write_record(self.path, self.document)

    def on_start(self, trainer):
        self.document["status"] = "running"
        self.document["config"] = json.loads(json.dumps(vars(trainer.args), default=str, allow_nan=False))
        self.document["optimizer"] = {"name": type(trainer.optimizer).__name__, "parameter_groups": [
            {key: group[key] for key in ("lr", "initial_lr", "momentum", "weight_decay") if key in group}
            for group in trainer.optimizer.param_groups]}
        write_record(self.path, self.document)

    def on_epoch(self, trainer):
        epoch = trainer.epoch + 1
        history = self.document["history"]
        # Ultralytics also invokes this callback after final best.pt validation.
        if history and epoch == history[-1]["epoch"]:
            return
        if epoch != len(history) + 1:
            raise RuntimeError("Training epochs must be recorded consecutively")
        score = float(trainer.metrics[MAP50])
        if not math.isfinite(score) or not 0 <= score <= 1 or score != float(trainer.fitness):
            raise RuntimeError("Checkpoint fitness must equal finite validation mAP@0.5")
        history.append({"epoch": epoch, "val_map50": score, "learning_rates": dict(trainer.lr)})
        if self.document["selected_epoch"] is None or score > self.document["best_val_map50"]:
            self.document.update(selected_epoch=epoch, best_val_map50=score)
        self.document["epochs_completed"] = len(history)
        write_record(self.path, self.document)

    def complete(self, checkpoint):
        if not self.document["history"] or self.document["config"] is None:
            raise RuntimeError("Training finished without a complete configuration and validation history")
        self.document.update(status="completed", finished_at=datetime.now(UTC).isoformat(),
                             duration_seconds=time.monotonic() - self.started,
                             checkpoint=checkpoint.name, final_checkpoint_sha256=sha256(checkpoint))
        write_record(self.path, self.document)

    def fail(self, error):
        self.document.update(status="failed", finished_at=datetime.now(UTC).isoformat(),
                             duration_seconds=time.monotonic() - self.started,
                             error=f"{type(error).__name__}: {error}")
        write_record(self.path, self.document)


def train(config=DEFAULT_CONFIG, *, data=None, epochs=None, name=None, device=0):
    machine = require_gpu(device)
    versions = environment_versions()
    document = read_config(config, data=data, epochs=epochs, name=name)
    weight = starting_weights(document)
    directory = Path(document["project"]) / document["name"]
    directory.mkdir(parents=True, exist_ok=False)
    dataset, inputs = training_inputs(document["data"], directory)
    checkpoint_directory = directory / "weights"
    checkpoint_directory.mkdir()
    record = TrainingRecord(checkpoint_directory / "training_record.json", document, sha256(weight),
                            machine, versions, inputs)

    try:
        from ultralytics import YOLO

        from .trainer import Map50Trainer

        model = YOLO(str(weight))
        if set(model.names) != {0}:
            raise ValueError("Starting checkpoint must be a single-class detector")
        model.add_callback("on_train_start", record.on_start)
        model.add_callback("on_fit_epoch_end", record.on_epoch)
        options = {key: value for key, value in document.items() if key not in {"model", "data"}}
        model.train(trainer=Map50Trainer, data=str(dataset), device=device, workers=0, val=True, split="val",
                    save=True, exist_ok=True, resume=False, pretrained=str(weight), amp=False, plots=False, **options)
        checkpoint = Path(model.trainer.best)
        if checkpoint != checkpoint_directory / "best.pt":
            raise RuntimeError("Trainer wrote the selected checkpoint outside the reserved run directory")
        record.complete(checkpoint)
    except BaseException as error:
        record.fail(error)
        raise
    return record.document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--epochs", type=int, help="Override the epoch budget, e.g. 2 for a smoke run")
    parser.add_argument("--name", help="Fresh run name; existing output directories are refused")
    parser.add_argument("--device", type=int, default=0, help="One CUDA device index")
    args = parser.parse_args()
    try:
        result = train(args.config, data=args.data, epochs=args.epochs, name=args.name, device=args.device)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
