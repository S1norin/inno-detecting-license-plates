"""Timing probe: engine selection and deterministic sampling."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from baseline.evaluation import timing
from baseline.ocr.io import digest, read_json
from models.crnn.model import CRNN, ModelConfig, save_checkpoint
from models.crnn.tests import test_data


class BuildOcrEngineTests(unittest.TestCase):
    def setUp(self):
        fixture = test_data.TrainingDataTests(methodName="test_originals_ctc_targets_and_unseen_validation_character")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        charset = read_json(fixture.charset)
        model = CRNN(ModelConfig(num_classes=len(charset["characters"]) + 1))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier.bias[1] = 10
        self.checkpoint = self.root / "probe-checkpoint.pt"
        save_checkpoint(self.checkpoint, model, charset, {"epoch": 2})
        self.sha256 = digest(self.checkpoint)
        self._deterministic = torch.are_deterministic_algorithms_enabled()
        self._threads = torch.get_num_threads()

    def tearDown(self):
        torch.use_deterministic_algorithms(self._deterministic)
        torch.set_num_threads(self._threads)

    def test_crnn_manifest_resolution_when_checkpoint_omitted(self):
        entry = {"filename": "probe-checkpoint.pt", "sha256": self.sha256}
        with patch("baseline.common.weights.model_entry", return_value=entry), \
             patch("baseline.common.weights.verified_path", return_value=self.checkpoint) as resolved:
            engine = timing._build_ocr_engine("crnn", None, None, Path("models.json"))
        resolved.assert_called_once_with("crnn-full-training", manifest=Path("models.json"))
        self.assertEqual(engine.weight_digests, {"crnn_checkpoint.pt": self.sha256})

    def test_crnn_explicit_checkpoint_verified_by_own_digest(self):
        engine = timing._build_ocr_engine("crnn", None, self.checkpoint, Path("models.json"))
        self.assertEqual(engine.weight_digests, {"crnn_checkpoint.pt": self.sha256})

    def test_crnn_digest_mismatch_refused(self):
        with patch("models.crnn.engine.verify", side_effect=ValueError("expected 0000 actual 1111")), \
             self.assertRaisesRegex(ValueError, "expected.*actual"):
            timing._build_ocr_engine("crnn", None, self.checkpoint, Path("models.json"))

    def test_easyocr_stays_the_default(self):
        with patch("baseline.ocr.engine.EasyOCREngine") as easyocr:
            engine = timing._build_ocr_engine("easyocr", Path("weights"), None, Path("models.json"))
        easyocr.assert_called_once_with(Path("weights"), Path("models.json"))
        self.assertIs(engine, easyocr.return_value)


class SamplingTests(unittest.TestCase):
    def dataset(self):
        rows = {}
        for _split, parents in (("val", ("a", "b")), ("test", ("c", "d"))):
            for parent in parents:
                rows[(parent, 1.0)] = {"image_path": f"{parent}_s100.jpg", "crop_path": f"{parent}_s100.jpg"}
        return SimpleNamespace(
            sample_ids={"val": [f"{p}_s100" for p in ("a", "b")],
                        "test": [f"{p}_s100" for p in ("c", "d")]},
            rows=rows,
        )

    def test_sample_images_are_deterministic_per_seed(self):
        first = timing.sample_images(self.dataset(), 3, 42)
        second = timing.sample_images(self.dataset(), 3, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({sample["sample_id"] for sample in first}), 3)

    def test_sample_images_cover_both_splits(self):
        samples = timing.sample_images(self.dataset(), 4, 42)
        self.assertEqual({sample["split"] for sample in samples}, {"val", "test"})

    def test_stage_stats_report_median_p95_and_count(self):
        stats = timing.stage_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["median_ms"], 3.0)
        self.assertEqual(stats["p95_ms"], 5.0)


if __name__ == "__main__":
    unittest.main()
