"""YOLO detection adapter.

Canonical inference mirrors baseline/detection/evaluate.py:
imgsz 640, iou 0.7, CPU, no half precision. The confidence floor is a
parameter: 0.001 reproduces the reported mAP runs; higher values are for
interactive display only.
"""

import time
from dataclasses import dataclass

from ultralytics import YOLO

from .config import DIGEST_FINETUNED, DIGEST_FROZEN, WEIGHT_FINETUNED, WEIGHT_FROZEN, YOLO_CANONICAL
from .weights_io import verify_weight


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]
    elapsed_ms: float
    image_size: tuple[int, int]  # (width, height) as fed to the model


class YOLODetector:
    """One loaded checkpoint and a thin predict wrapper."""

    def __init__(self, name: str, weight_path, digest: str):
        self.name = name
        verify_weight(weight_path, digest)
        self.model = YOLO(str(weight_path))

    def detect(self, image_bgr, conf: float = 0.25) -> DetectionResult:
        """Run inference on a BGR image; boxes are in source-image pixels."""
        start = time.perf_counter()
        results = self.model.predict(
            source=image_bgr,
            stream=True,
            save=False,
            verbose=False,
            conf=conf,
            **YOLO_CANONICAL,
        )
        detections = []
        for result in results:
            h, w = image_bgr.shape[:2]
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        x1=max(0.0, x1),
                        y1=max(0.0, y1),
                        x2=min(float(w), x2),
                        y2=min(float(h), y2),
                        confidence=float(box.conf[0]),
                    )
                )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(
            detections=tuple(detections),
            elapsed_ms=elapsed_ms,
            image_size=(image_bgr.shape[1], image_bgr.shape[0]),
        )


def get_detectors() -> dict[str, YOLODetector]:
    """Load both checkpoints (digest-verified). Caller should cache this."""
    return {
        "frozen": YOLODetector("Frozen YOLO11s (pretrained)", WEIGHT_FROZEN, DIGEST_FROZEN),
        "finetuned": YOLODetector("Fine-tuned YOLO11s", WEIGHT_FINETUNED, DIGEST_FINETUNED),
    }
