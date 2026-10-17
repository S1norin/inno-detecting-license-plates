"""Digest-verified CRNN adapter for the shared recognition harness."""

import re
from importlib.metadata import version
from pathlib import Path

import cv2
import torch

from baseline.common.weights import verify
from baseline.ocr.ctc import decode_greedy
from baseline.ocr.io import digest

from .model import load_checkpoint, transform_image


class CRNNEngine:
    def __init__(self, checkpoint_path, expected_sha256):
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
            raise ValueError("An expected lowercase checkpoint SHA-256 is required")
        checkpoint_path = verify(Path(checkpoint_path), expected_sha256)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cv2.setNumThreads(1)
        self.model, checkpoint = load_checkpoint(checkpoint_path)
        if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] < 1:
            raise ValueError("Checkpoint must contain a selected training epoch")
        self.characters = [""] + checkpoint["charset"]["characters"]
        self.weight_digests = {"crnn_checkpoint.pt": expected_sha256}
        architecture = dict(checkpoint["architecture"])
        architecture["cnn_channels"] = list(architecture["cnn_channels"])
        self.identity = {
            "name": "crnn", "device": "cpu", "threads": 1, "decoder": "greedy",
            "character_confidence": "max probability in each collapsed CTC run",
            "input_geometry": {"channels": 1, "height": 48, "width": 256,
                               "resize": "aspect-preserving", "padding": "black right/bottom",
                               "normalization": "(pixel / 255 - 0.5) / 0.5"},
            "architecture": architecture,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "selected_epoch": checkpoint["epoch"], "charset_content_sha256": checkpoint["charset_content_sha256"],
            "torch_version": version("torch"), "torchvision_version": version("torchvision"),
            "numpy_version": version("numpy"), "adapter_sha256": digest(Path(__file__)),
            "model_sha256": digest(Path(__file__).with_name("model.py")),
            "ctc_sha256": digest(Path(__file__).resolve().parents[2] / "baseline/ocr/ctc.py"),
        }

    def recognize(self, image):
        with torch.inference_mode():
            logits = self.model(transform_image(image).unsqueeze(0))
            probabilities = logits.softmax(dim=2)
            values, indices = probabilities.max(dim=2)
        return decode_greedy(indices[0].tolist(), values[0].tolist(), self.characters)
