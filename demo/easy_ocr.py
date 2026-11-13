"""EasyOCR recognition adapter built on the canonical baseline engine.

Reuses baseline/ocr/engine.py (easyocr 1.7.2, offline weights, ch_sim + en,
greedy CTC decode) so the live demo reads plates with exactly the baseline
OCR configuration.
"""

import time
from dataclasses import dataclass, field

from baseline.ocr.engine import EasyOCREngine

from .config import EASYOCR_MANIFEST, EASYOCR_WEIGHTS_DIR


@dataclass(frozen=True)
class EasyOCRResult:
    text: str
    char_confidences: tuple[float, ...] = field(default_factory=tuple)
    elapsed_ms: float = 0.0


class EasyOCRReader:
    def __init__(self):
        self.engine = EasyOCREngine(EASYOCR_WEIGHTS_DIR, EASYOCR_MANIFEST)

    def recognize(self, crop_bgr) -> EasyOCRResult:
        start = time.perf_counter()
        out = self.engine.recognize(crop_bgr)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return EasyOCRResult(
            text=out["text"],
            char_confidences=tuple(out["char_confidences"]),
            elapsed_ms=elapsed_ms,
        )
