"""Unit tests for the PlateLab demo (no model weights required)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo import results  # noqa: E402
from demo.dataset import load_metadata, variants_of  # noqa: E402
from demo.viz import crop_box, draw_box, draw_detections, gray_to_bgr  # noqa: E402
from demo.weights_io import sha256_of  # noqa: E402


def _make_image() -> np.ndarray:
    image = np.full((400, 600, 3), 30, dtype=np.uint8)
    image[100:200, 150:450] = (230, 240, 245)
    return image


class TestCanonicalResults:
    """Values must match docs/md/final_results.md exactly."""

    def test_detection_test(self):
        assert results.DETECTION_TEST["frozen"] == {
            "map50": 0.9073,
            "precision": 0.9408,
            "recall": 0.8938,
            "f1": 0.9167,
        }
        assert results.DETECTION_TEST["finetuned"] == {
            "map50": 0.9949,
            "precision": 0.9466,
            "recall": 0.9975,
            "f1": 0.9714,
        }

    def test_detection_val(self):
        assert results.DETECTION_VAL["frozen"]["map50"] == 0.9245
        assert results.DETECTION_VAL["finetuned"]["map50"] == 0.9949

    def test_recognition_test(self):
        rows = {(r["engine"], r["protocol"]): r for r in results.RECOGNITION_TEST}
        assert rows[("EasyOCR (frozen baseline)", "ocr_only")]["char_acc"] == 0.3474
        assert rows[("EasyOCR (frozen baseline)", "end_to_end")]["char_acc"] == 0.3240
        assert rows[("CRNN", "ocr_only")] == {
            "engine": "CRNN",
            "protocol": "ocr_only",
            "char_acc": 0.7593,
            "exact": 267,
            "f1": 0.7271,
            "note": "",
        }
        final = rows[("CRNN", "end_to_end + fine-tuned YOLO11s")]
        assert final["char_acc"] == 0.7661
        assert final["exact"] == 296
        assert final["f1"] == 0.7372

    def test_preprocessing_val(self):
        assert results.PREPROCESSING_VAL == {
            "none": 0.7558,
            "gray_contrast": 0.7469,
            "canny": 0.4575,
        }

    def test_scale_recognition(self):
        assert results.SCALE_RECOGNITION["ocr_only"][1.0] == (0.7593, 267)
        assert results.SCALE_RECOGNITION["ocr_only"][0.5] == (0.7610, 268)
        assert results.SCALE_RECOGNITION["end_to_end + fine-tuned"][1.0] == (0.7661, 296)
        assert results.SCALE_RECOGNITION["end_to_end + frozen"][0.5] == (0.6721, 144)

    def test_failure_cases(self):
        assert len(results.FAILURE_CASES) == 6
        ids = {c["sample_id"] for c in results.FAILURE_CASES}
        assert ids == {
            "6f3c772ca888fa92_s100",
            "b996022cc9a5ee07_s100",
            "356b4ca635beeca7_s100",
            "588e616d9e7cec6c_s100",
            "261f3b723deb67ac_s100",
            "bfe070b4b5543916_s100",
        }

    def test_runtime_m3(self):
        assert results.RUNTIME_M3["frozen"]["detection_median_ms"] == 77.8
        assert results.RUNTIME_M3["finetuned"]["detection_median_ms"] == 80.2
        assert results.RUNTIME_M3["crnn_median_ms"] == 5.7
        assert results.RUNTIME_M3["end_to_end_ms"] == "83-86"


class TestDataset:
    @pytest.fixture(scope="class")
    def frame(self):
        return load_metadata()

    def test_row_count(self, frame):
        assert len(frame) == 15_000

    def test_split_counts(self, frame):
        counts = frame["split"].value_counts()
        assert counts["train"] == 10_500
        assert counts["val"] == 2_100
        assert counts["test"] == 2_400

    def test_scale_variants(self, frame):
        parent = frame["parent_id"].iloc[0]
        variants = variants_of(frame, parent)
        assert len(variants) == 3
        assert {v.scale for v in variants.values()} == {1.0, 0.75, 0.5}
        splits = {v.split for v in variants.values()}
        assert len(splits) == 1  # anti-leakage invariant

    def test_boxes_in_pixel_space(self, frame):
        for col in ("x1", "y1", "x2", "y2"):
            assert (frame[col] >= 0).all()
        assert (frame["x2"] <= frame["width"]).all()
        assert (frame["y2"] <= frame["height"]).all()
        assert (frame["x2"] > frame["x1"]).all()
        assert (frame["y2"] > frame["y1"]).all()


class TestViz:
    def test_crop_box(self):
        image = _make_image()
        crop = crop_box(image, (150, 100, 450, 200))
        assert crop.shape == (100, 300, 3)

    def test_crop_box_clamps(self):
        image = _make_image()
        crop = crop_box(image, (-10, -10, 650, 420))
        assert crop.shape == (400, 600, 3)

    def test_draw_box(self):
        image = _make_image()
        shown = draw_box(image, (150, 100, 450, 200), (0, 200, 0), "GT")
        assert shown.shape == image.shape
        assert (shown[100:200, 150:450] != image[100:200, 150:450]).any()

    def test_draw_detections(self):
        from demo.yolo import Detection

        image = _make_image()
        shown = draw_detections(
            image,
            [
                Detection(0.9, 150, 100, 450, 200),
                Detection(0.4, 10, 10, 200, 300),
                Detection(0.2, 100, 50, 300, 250),
            ],
            (0, 0, 255),
        )
        assert shown.shape == image.shape

    def test_gray_to_bgr(self):
        gray = np.zeros((10, 20), dtype=np.uint8)
        bgr = gray_to_bgr(gray)
        assert bgr.shape == (10, 20, 3)


class TestDemoInteractions:
    def test_plate_text_search_and_literal_query(self, monkeypatch):
        import pandas as pd

        from demo.sections import dataset_explorer, pipeline

        frame = pd.DataFrame([
            {"split": "test", "scale": 1.0, "image_path": "images/abc.jpg", "plate_text": "皖A12345"},
        ])
        for section, select in ((dataset_explorer, dataset_explorer._select_sample), (pipeline, pipeline._pick_sample)):
            monkeypatch.setattr(section, "_frame", lambda: frame)
            monkeypatch.setattr(section, "_row_to_sample", lambda row: row["plate_text"])
            assert select("test", 1.0, "皖A123") == "皖A12345"
            assert select("test", 1.0, "[") is None

    def test_failure_check_does_not_assert_unverified_false_positives(self):
        from demo.sections.failures import _check
        from demo.yolo import Detection

        case = {"stage": "detection", "detector": "finetuned", "title": "False positives"}
        box = Detection(0, 0, 10, 10, 0.8)
        ok, detail = _check(case, {"dets": {"finetuned": [box, box]}})
        assert not ok
        assert "box matching" in detail


class TestWeights:
    def test_sha256_of(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        assert sha256_of(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_known_digests(self):
        from demo.config import DIGEST_CRNN, DIGEST_FINETUNED, DIGEST_FROZEN

        assert DIGEST_FINETUNED == "0a4f080fe3749b8be145101468a0dd98455cf50117d2acd37575f33e0d261055"
        assert DIGEST_CRNN == "92d1fbeae19a81ddf3ae498a40e4416ed619ad9667736c592ccbb899e59ca2cc"
        assert len(DIGEST_FROZEN) == 64
