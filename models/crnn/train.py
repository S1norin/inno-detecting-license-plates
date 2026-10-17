"""Deterministic CPU CRNN training on original-scale crops."""

import argparse
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from baseline.ocr.charset import build_charset
from baseline.ocr.ctc import decode_greedy
from baseline.ocr.export_train import export_training
from baseline.ocr.io import dataset_path, digest, load_samples, read_json, timestamp, write_json
from baseline.ocr.metrics import edit_distance

from .model import CRNN, ModelConfig, save_checkpoint, transform_image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHARSET = ROOT / "baseline/ocr/charset.json"
DEFAULT_EPOCHS = 50


class CropDataset(Dataset):
    def __init__(self, root, samples, charset, *, encode_targets=True):
        self.root = Path(root)
        self.samples = samples
        self.charset = charset
        self.targets = []
        for sample in samples:
            if sample["scale"] != 1.0:
                raise ValueError("CRNN training/validation uses original-scale crops only")
            target = [charset["character_to_index"][char] for char in sample["ground_truth"]] if encode_targets else []
            required_steps = len(target) + sum(left == right for left, right in zip(target, target[1:], strict=False))
            if encode_targets and (not target or required_steps > 16):
                raise ValueError(f"CTC target cannot fit 16 steps: {sample['sample_id']}")
            path = dataset_path(self.root, sample["crop_path"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing crop: {path}")
            self.targets.append(torch.tensor(target, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = dataset_path(self.root, sample["crop_path"])
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot decode crop: {path}")
        return transform_image(image), self.targets[index], sample["ground_truth"]


def assemble_data(dataset, charset_path=DEFAULT_CHARSET):
    dataset = Path(dataset)
    charset = read_json(charset_path)
    if charset != build_charset(dataset / "ocr_train.csv"):
        raise ValueError("Charset does not match build_charset(ocr_train.csv)")
    training = export_training(dataset, charset, scale=1.0)["records"]
    validation = load_samples(dataset, "val", 1.0)
    return (CropDataset(dataset, training, charset),
            CropDataset(dataset, validation, charset, encode_targets=False), charset)


def collate_batch(batch):
    images, targets, texts = zip(*batch, strict=True)
    return torch.stack(images), torch.cat(targets), torch.tensor([len(t) for t in targets], dtype=torch.long), texts


def configure_cpu(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)


def decode_logits(logits, charset):
    probabilities, indices = logits.softmax(dim=2).max(dim=2)
    characters = [""] + charset["characters"]
    return [decode_greedy(index.tolist(), probability.tolist(), characters)
            for index, probability in zip(indices, probabilities, strict=True)]


def validate_epoch(model, loader, charset):
    model.eval()
    character_accuracy = 0.0
    exact = count = 0
    with torch.inference_mode():
        for images, _, _, texts in loader:
            predictions = decode_logits(model(images), charset)
            for predicted, target in zip(predictions, texts, strict=True):
                text = predicted["text"]
                character_accuracy += 1 - edit_distance(text, target) / max(len(text), len(target))
                exact += text == target
                count += 1
    return {"val_character_accuracy": character_accuracy / count,
            "val_plate_exact_match": exact / count}


def better_epoch(candidate, best):
    return best is None or candidate > best


def dataset_digests(root, training, validation):
    hashes = {filename: digest(Path(root) / filename)
              for filename in ("metadata.csv", "ocr_train.csv", "ocr_val.csv")}
    for name, dataset in (("train", training), ("val", validation)):
        crops = {sample["crop_path"]: digest(dataset_path(root, sample["crop_path"])) for sample in dataset.samples}
        content = json.dumps(crops, sort_keys=True, separators=(",", ":")).encode()
        hashes[f"{name}_original_crops"] = hashlib.sha256(content).hexdigest()
    return hashes


def train_model(dataset, output, *, charset_path=DEFAULT_CHARSET, epochs=DEFAULT_EPOCHS,
                batch_size=32, learning_rate=1e-3, seed=42):
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output}. Use a new run directory.")
    started = time.perf_counter()
    configure_cpu(seed)
    training, validation, charset = assemble_data(dataset, charset_path)
    hashes = dataset_digests(dataset, training, validation)
    model = CRNN(ModelConfig(num_classes=len(charset["characters"]) + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=False, fused=False)
    criterion = nn.CTCLoss(blank=0)
    training_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0,
                                 collate_fn=collate_batch, generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0,
                                   collate_fn=collate_batch, generator=torch.Generator().manual_seed(seed))
    config = {
        "architecture": asdict(model.config), "parameter_count": sum(p.numel() for p in model.parameters()),
        "epochs": epochs, "batch_size": batch_size, "scale": 1.0, "num_workers": 0,
        "optimizer": {"name": "Adam", "learning_rate": learning_rate, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0, "amsgrad": False, "foreach": False, "fused": False},
        "loss": {"name": "CTCLoss", "blank": 0, "reduction": "mean", "zero_infinity": False},
        "transform": {"grayscale": True, "preserve_aspect_ratio": True, "padding": "black right/bottom",
                      "normalization": "(pixel / 255 - 0.5) / 0.5"},
        "epoch_selection": "maximum validation character accuracy; ties select earlier epoch",
    }
    provenance = {
        "config": config, "seed": seed, "threads": 1, "device": "cpu", "deterministic_algorithms": True,
        "library_versions": {name: version(name)
                             for name in ("torch", "torchvision", "numpy", "opencv-python-headless")},
        "python": platform.python_version(),
        "machine": {"platform": platform.platform(), "architecture": platform.machine()},
        "dataset_digests": hashes, "charset_sha256": digest(charset_path),
        "sample_counts": {"train": len(training), "val": len(validation)},
    }
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    best_accuracy = None
    selected_epoch = None
    history, durations = [], []
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        for images, targets, lengths, _ in training_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            input_lengths = torch.full((len(images),), logits.shape[1], dtype=torch.long)
            loss = criterion(logits.log_softmax(2).transpose(0, 1).contiguous(), targets, input_lengths, lengths)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite CTC loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        metrics = {"epoch": epoch, "train_ctc_loss": total_loss / len(training),
                   **validate_epoch(model, validation_loader, charset)}
        history.append(metrics)
        if better_epoch(metrics["val_character_accuracy"], best_accuracy):
            best_accuracy = metrics["val_character_accuracy"]
            selected_epoch = epoch
            save_checkpoint(checkpoint, model, charset,
                            {**provenance, "epoch": epoch, "val_character_accuracy": best_accuracy,
                             "validation_history": list(history)})
        durations.append(time.perf_counter() - epoch_started)
        record = {"schema_version": 1, **provenance, "dataset_root": str(Path(dataset).resolve()),
                  "per_epoch": history, "epoch_durations_seconds": durations, "completed_epochs": epoch,
                  "selected_epoch": selected_epoch, "best_val_character_accuracy": best_accuracy,
                  "checkpoint_sha256": digest(checkpoint), "checkpoint": "checkpoint.pt",
                  "timestamp": timestamp(), "duration_seconds": time.perf_counter() - started}
        write_json(output / "training_record.json", record)
        print(json.dumps({**metrics, "duration_seconds": durations[-1], "selected_epoch": selected_epoch}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/dataset")
    parser.add_argument("--charset", type=Path, default=DEFAULT_CHARSET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    record = train_model(args.dataset, args.output, charset_path=args.charset, epochs=args.epochs,
                         batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
    print(json.dumps({"checkpoint_sha256": record["checkpoint_sha256"], "selected_epoch": record["selected_epoch"]}))


if __name__ == "__main__":
    main()
