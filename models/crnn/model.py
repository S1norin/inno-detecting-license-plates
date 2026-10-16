"""Small grayscale CRNN, aspect-preserving input transform, and checkpoints."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    input_height: int = 48
    input_width: int = 256
    cnn_channels: tuple[int, ...] = (32, 64, 128, 128)
    projection_size: int = 256
    hidden_size: int = 128
    recurrent_layers: int = 2
    num_classes: int = 59

    def __post_init__(self):
        if (self.input_height, self.input_width) != (48, 256):
            raise ValueError("CRNN input geometry must be H48/W256")
        if len(self.cnn_channels) != 4 or any(c <= 0 for c in self.cnn_channels):
            raise ValueError("Four positive CNN widths are required")
        if min(self.projection_size, self.hidden_size, self.recurrent_layers) < 1 or self.num_classes < 2:
            raise ValueError("Invalid CRNN architecture sizes")


class CRNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        layers = []
        channels = 1
        for width in self.config.cnn_channels:
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width),
                           nn.ReLU(), nn.MaxPool2d(2)])
            channels = width
        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Linear(channels * 3, self.config.projection_size)
        self.rnn = nn.LSTM(self.config.projection_size, self.config.hidden_size,
                           num_layers=self.config.recurrent_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.num_classes)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 48, 256):
            raise ValueError("Expected a batch of (1, 48, 256) grayscale inputs")
        features = self.cnn(images).permute(0, 3, 1, 2).flatten(2)
        sequence, _ = self.rnn(self.projection(features))
        return self.classifier(sequence)


def transform_image(image):
    """Fit inside H48/W256, pad right/bottom with black, normalize to [-1, 1]."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty uint8 crop")
    if image.ndim == 3 and image.shape[2] in (3, 4):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError("Expected grayscale, BGR, or BGRA input")
    height, width = image.shape
    scale = min(48 / height, 256 / width)
    shape = (min(256, max(1, round(width * scale))), min(48, max(1, round(height * scale))))
    resized = cv2.resize(image, shape, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((48, 256), dtype=np.uint8)
    canvas[:shape[1], :shape[0]] = resized
    return torch.from_numpy(canvas.astype(np.float32) / 127.5 - 1).unsqueeze(0)


def charset_digest(charset):
    content = json.dumps(charset, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_charset(charset):
    characters = charset["characters"]
    if not characters or len(set(characters)) != len(characters) or any(len(c) != 1 for c in characters):
        raise ValueError("Invalid charset characters")
    if charset["ctc_blank_index"] != 0 or charset["character_to_index"] != {
        char: index for index, char in enumerate(characters, 1)
    }:
        raise ValueError("Charset must use contiguous indices starting at 1 with blank 0")


def save_checkpoint(path, model, charset, metadata):
    validate_charset(charset)
    if model.config.num_classes != len(charset["characters"]) + 1:
        raise ValueError("Model class count does not match charset")
    reserved = {"schema_version", "architecture", "state_dict", "charset", "charset_content_sha256"}
    if reserved & metadata.keys():
        raise ValueError("Checkpoint metadata overrides reserved fields")
    payload = {"schema_version": 1, "architecture": asdict(model.config), "state_dict": model.state_dict(),
               "charset": charset, "charset_content_sha256": charset_digest(charset), **metadata}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A file object keeps the internal ZIP archive name independent of output directory/name.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported CRNN checkpoint schema")
    validate_charset(payload["charset"])
    if payload["charset_content_sha256"] != charset_digest(payload["charset"]):
        raise ValueError("Checkpoint charset digest mismatch")
    config = ModelConfig(**payload["architecture"])
    if config.num_classes != len(payload["charset"]["characters"]) + 1:
        raise ValueError("Checkpoint class count does not match charset")
    model = CRNN(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload
