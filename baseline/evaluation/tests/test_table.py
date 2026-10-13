"""Consolidated table tests against hand-computed fixture values."""

import json
import tempfile
import unittest
from pathlib import Path

from baseline.evaluation.inputs import load_all
from baseline.evaluation.table import make_document
from baseline.evaluation.tests.fixtures import build

FIXED_TS = "2026-01-01T00:00:00+00:00"


def _doc(root):
    dataset, detection, ocr = load_all(root / "dataset", root / "detection", root / "ocr")
    return make_document(dataset, detection, ocr, FIXED_TS)


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        build(cls.root)
        cls.doc = _doc(cls.root)
        cls.table = cls.doc["table"]

    def test_detection_rows(self):
        val = self.table["e1_detection"]["val"]["overall"]
        self.assertEqual((val["tp"], val["fp"], val["fn"]), (8, 0, 0))
        self.assertEqual(val["f1"], 1.0)
        test = self.table["e1_detection"]["test"]["overall"]
        self.assertEqual((test["tp"], test["fp"], test["fn"]), (2, 1, 2))
        self.assertEqual(test["precision"], 2 / 3)
        self.assertEqual(test["recall"], 0.5)
        self.assertAlmostEqual(test["f1"], 4 / 7)
        # test tilts: p6=5 (0-9), p5=10 (10-19), p7=22, p8=40 (20+)
        by_tilt = self.table["e1_detection"]["test"]["by_tilt"]
        self.assertEqual(by_tilt["0-9"]["ground_truth_count"], 1)
        self.assertEqual(by_tilt["10-19"]["ground_truth_count"], 1)
        self.assertEqual(by_tilt["20+"]["ground_truth_count"], 2)

    def test_ocr_only_selected_rows(self):
        section = self.table["e2_ocr_only"]
        self.assertEqual(section["selected"], {"mode": "gray_contrast", "tau": 0.5, "score_function": "mean"})
        val = section["val"]["overall"]
        self.assertEqual(val["coverage"], 7 / 8)
        self.assertEqual(val["plate_exact_match"], 5 / 8)
        self.assertEqual(val["accepted_exact_accuracy"], 5 / 7)
        self.assertEqual((val["tp"], val["fp"], val["fn"]), (6, 1, 1))
        self.assertAlmostEqual(val["character_accuracy"], 46 / 56)
        test = section["test"]["overall"]
        self.assertEqual(test["coverage"], 0.75)
        self.assertEqual(test["plate_exact_match"], 0.5)
        self.assertEqual(test["accepted_exact_accuracy"], 2 / 3)

    def test_end_to_end_rows_and_delta(self):
        section = self.table["e2b_end_to_end"]
        self.assertEqual(section["selected"]["mode"], "none")
        val = section["val"]["overall"]
        self.assertEqual(val["coverage"], 5 / 8)
        self.assertEqual(val["plate_exact_match"], 0.5)
        self.assertEqual(val["precision"], 1.0)
        self.assertAlmostEqual(val["character_accuracy"], 34 / 56)
        delta = section["delta_vs_ocr_only"]["test"]
        self.assertEqual(delta["coverage"], 0.0)
        self.assertEqual(delta["plate_exact_match"], 0.25)
        self.assertAlmostEqual(delta["accepted_exact_accuracy"], 1 / 3)
        self.assertAlmostEqual(delta["character_accuracy"], 0.75 - 20 / 28)

    def test_ablation_rows(self):
        e3 = self.table["e3_ablation"]
        self.assertIsNone(e3["none"]["delta_vs_none"])
        self.assertEqual(e3["none"]["overall"]["coverage"], 5 / 8)
        self.assertAlmostEqual(e3["none"]["overall"]["character_accuracy"], 31 / 56)
        self.assertEqual(e3["gray_contrast"]["overall"]["coverage"], 7 / 8)
        self.assertEqual(e3["gray_contrast"]["delta_vs_none"]["coverage"], 0.25)
        self.assertEqual(e3["canny"]["overall"]["coverage"], 6 / 8)
        self.assertAlmostEqual(e3["canny"]["delta_vs_none"]["character_accuracy"], 39 / 56 - 31 / 56)

    def test_threshold_rows(self):
        e6 = self.table["e6_threshold"]
        self.assertEqual(e6["ocr_only"], {"tau": 0.5, "accuracy": 2 / 3, "coverage": 0.75})
        self.assertEqual(e6["end_to_end"], {"tau": 0.5, "accuracy": 1.0, "coverage": 0.75})

    def test_inputs_provenance(self):
        inputs = self.doc["inputs"]
        self.assertEqual(set(inputs["dataset"]["digests"]), {"metadata.csv", "ocr_val.csv", "ocr_test.csv"})
        self.assertEqual(inputs["detection"]["model"], "yolo11s")
        self.assertEqual(inputs["ocr"]["engine"], "easyocr")
        self.assertEqual(len(self.doc["run_id"]), 21)
        self.assertTrue(self.doc["run_id"].startswith("eval-"))

    def test_rerun_is_byte_identical_apart_from_timestamp(self):
        first = json.dumps(self.doc, sort_keys=True, indent=2)
        second = _doc(self.root)
        second["timestamp"] = self.doc["timestamp"]
        self.assertEqual(json.dumps(second, sort_keys=True, indent=2), first)


if __name__ == "__main__":
    unittest.main()
