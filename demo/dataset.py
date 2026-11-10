"""Dataset access: metadata joins, image loading, GT boxes, scale variants."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from .config import DATASET_DIR, METADATA_CSV


@dataclass(frozen=True)
class Sample:
    """One dataset variant (one image + its annotation)."""

    sample_id: str  # e.g. 722482ce4334c84e_s100
    split: str  # train / val / test
    scale: float  # 1.0 / 0.75 / 0.5
    parent_id: str
    image_path: Path
    crop_path: Path
    source_relpath: str
    width: int
    height: int
    x1: int
    y1: int
    x2: int
    y2: int
    plate_pixel_height: int
    size_bucket: str
    original_bucket: str
    tilt_horizontal: int
    tilt_vertical: int
    tilt_magnitude: int
    plate_text: str

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def load_metadata() -> pd.DataFrame:
    frame = pd.read_csv(METADATA_CSV, encoding="utf-8-sig")
    frame = frame.set_index("image_path", drop=False)
    return frame


def rows_for(frame: pd.DataFrame, split: str, scale: float = 1.0) -> list[Sample]:
    sub = frame[(frame["split"] == split) & (frame["scale"] == scale)]
    samples = []
    for _, row in sub.iterrows():
        rel = row["image_path"]
        samples.append(
            Sample(
                sample_id=Path(rel).stem,
                split=row["split"],
                scale=row["scale"],
                parent_id=row["parent_id"],
                image_path=DATASET_DIR / rel,
                crop_path=DATASET_DIR / row["crop_path"],
                source_relpath=row["source_relpath"],
                width=int(row["width"]),
                height=int(row["height"]),
                x1=int(row["x1"]),
                y1=int(row["y1"]),
                x2=int(row["x2"]),
                y2=int(row["y2"]),
                plate_pixel_height=int(row["plate_pixel_height"]),
                size_bucket=row["size_bucket"],
                original_bucket=row["original_bucket"],
                tilt_horizontal=int(row["tilt_horizontal"]),
                tilt_vertical=int(row["tilt_vertical"]),
                tilt_magnitude=int(row["tilt_magnitude"]),
                plate_text=row["plate_text"],
            )
        )
    return samples


def variants_of(frame: pd.DataFrame, parent_id: str) -> dict[float, Sample]:
    """All scale variants of one source photo, keyed by scale."""
    sub = frame[frame["parent_id"] == parent_id]
    return {float(row["scale"]): _row_to_sample(row) for _, row in sub.iterrows()}


def _row_to_sample(row) -> Sample:
    rel = row["image_path"]
    return Sample(
        sample_id=Path(rel).stem,
        split=row["split"],
        scale=float(row["scale"]),
        parent_id=row["parent_id"],
        image_path=DATASET_DIR / rel,
        crop_path=DATASET_DIR / row["crop_path"],
        source_relpath=row["source_relpath"],
        width=int(row["width"]),
        height=int(row["height"]),
        x1=int(row["x1"]),
        y1=int(row["y1"]),
        x2=int(row["x2"]),
        y2=int(row["y2"]),
        plate_pixel_height=int(row["plate_pixel_height"]),
        size_bucket=row["size_bucket"],
        original_bucket=row["original_bucket"],
        tilt_horizontal=int(row["tilt_horizontal"]),
        tilt_vertical=int(row["tilt_vertical"]),
        tilt_magnitude=int(row["tilt_magnitude"]),
        plate_text=row["plate_text"],
    )


def load_image(path: Path):
    """Load a BGR uint8 image; None if unreadable."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def load_crop(sample: Sample):
    return load_image(sample.crop_path)
