"""CRNN recognition adapter built on the canonical repo implementation.

Uses models/crnn/engine.py (CRNNEngine) so the demo runs exactly the
checkpoint's own architecture, charset, transform, and greedy CTC decoder.
"""

import time
from dataclasses import dataclass, field

from models.crnn.engine import CRNNEngine

from .config import DIGEST_CRNN, WEIGHT_CRNN


@dataclass(frozen=True)
class Recognition:
    text: str
    char_confidences: tuple[float, ...] = field(default_factory=tuple)
    elapsed_ms: float = 0.0

    @property
    def min_confidence(self) -> float | None:
        return min(self.char_confidences) if self.char_confidences else None


class CRNNRecognizer:
    def __init__(self):
        self.engine = CRNNEngine(WEIGHT_CRNN, DIGEST_CRNN)
        self.characters = self.engine.characters
        self.identity = self.engine.identity

    def recognize(self, crop_bgr_or_gray) -> Recognition:
        start = time.perf_counter()
        out = self.engine.recognize(crop_bgr_or_gray)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Recognition(
            text=out["text"],
            char_confidences=tuple(out["char_confidences"]),
            elapsed_ms=elapsed_ms,
        )
