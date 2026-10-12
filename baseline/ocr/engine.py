"""Offline EasyOCR 1.7.2 with probabilities extracted from CTC logits."""

import math
from importlib.metadata import version
from pathlib import Path

from baseline.common.weights import verify

from .ctc import decode_greedy
from .io import digest, read_json

WEIGHT_FILES = ("craft_mlt_25k.pth", "zh_sim_g2.pth")


def verify_weights(directory, manifest):
    entries = read_json(manifest)
    if isinstance(entries, dict):
        entries = entries.get("models", entries)
    if isinstance(entries, dict):
        entries = list(entries.values())
    if not isinstance(entries, list):
        raise ValueError("Weight manifest must contain a list or a mapping of model entries")

    hashes = {}
    for filename in WEIGHT_FILES:
        matches = [entry for entry in entries if Path(entry["filename"]).name == filename]
        if len(matches) != 1:
            raise ValueError(f"Manifest must contain exactly one entry for {filename}")
        expected = matches[0]["sha256"].lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid pinned sha256 for {filename}")
        path = Path(directory) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing offline weight: {path}; obtain it through baseline/common first")
        verify(path, expected)
        hashes[filename] = expected
    return hashes


class EasyOCREngine:
    def __init__(self, weights_directory, manifest):
        self.weight_digests = verify_weights(weights_directory, manifest)
        if version("easyocr") != "1.7.2":
            raise RuntimeError("The CTC adapter requires easyocr==1.7.2")

        import easyocr
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.reader = easyocr.Reader(
            ["ch_sim", "en"],
            gpu=False,
            model_storage_directory=str(Path(weights_directory).resolve()),
            user_network_directory=str(Path(weights_directory).resolve() / "user_network"),
            download_enabled=False,
            quantize=False,
            verbose=False,
        )
        self.reader.recognizer.eval()
        self.identity = {
            "name": "easyocr",
            "version": "1.7.2",
            "languages": ["ch_sim", "en"],
            "device": "cpu",
            "quantize": False,
            "decoder": "greedy",
            "character_confidence": "max probability in each collapsed CTC run",
            "contrast_retry": False,
            "torch_version": version("torch"),
            "torchvision_version": version("torchvision"),
            "numpy_version": version("numpy"),
            "adapter_sha256": digest(Path(__file__)),
            "ctc_sha256": digest(Path(__file__).with_name("ctc.py")),
        }

    def recognize(self, image):
        import cv2
        import torch
        from easyocr.config import imgH
        from easyocr.recognition import AlignCollate, ListDataset
        from easyocr.utils import get_image_list

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        horizontal, free = self.reader.detect(image)
        regions, width = get_image_list(horizontal[0], free[0], gray, model_height=imgH)
        if not regions:
            return {"text": "", "char_confidences": []}

        regions.sort(key=lambda item: (min(p[1] for p in item[0]), min(p[0] for p in item[0])))
        crops = ListDataset([region[1] for region in regions])
        images = AlignCollate(imgH=imgH, imgW=width)([crops[i] for i in range(len(crops))])
        ignored = [i + 1 for i, char in enumerate(self.reader.character) if char not in self.reader.lang_char]

        with torch.inference_mode():
            dummy_text = torch.zeros((len(regions), int(width / 10) + 1), dtype=torch.long)
            logits = self.reader.recognizer(images, dummy_text)
            probabilities = logits.softmax(dim=2)
            probabilities[:, :, ignored] = 0
            totals = probabilities.sum(dim=2, keepdim=True)
            if not torch.isfinite(totals).all() or (totals <= 0).any():
                raise RuntimeError("Recognizer emitted invalid probability mass")
            probabilities /= totals
            values, indices = probabilities.max(dim=2)

        results = [
            decode_greedy(index.tolist(), value.tolist(), self.reader.converter.character)
            for index, value in zip(indices, values, strict=True)
        ]
        text = "".join(result["text"] for result in results)
        confidences = [p for result in results for p in result["char_confidences"]]
        if any(not math.isfinite(p) for p in confidences):
            raise RuntimeError("Recognizer emitted a nonfinite confidence")
        return {"text": text, "char_confidences": confidences}
