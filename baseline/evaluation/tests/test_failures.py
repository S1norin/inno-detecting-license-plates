"""Failure-case extraction tests against the planted fixture failures."""

import tempfile
import unittest
from pathlib import Path

from baseline.evaluation.failures import build_failures
from baseline.evaluation.inputs import load_all
from baseline.evaluation.tests.fixtures import build


class FailuresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        build(cls.root)
        dataset, detection, ocr = load_all(cls.root / "dataset", cls.root / "detection", cls.root / "ocr")
        cls.failures = build_failures(dataset, detection, ocr)

    def test_detection_val_is_clean(self):
        self.assertEqual(self.failures["detection"]["val"], [])

    def test_detection_test_cases_and_order(self):
        cases = self.failures["detection"]["test"]
        self.assertEqual([c["image_path"] for c in cases],
                         ["images/test/p5_s100.jpg", "images/test/p6_s100.jpg", "images/test/p7_s100.jpg"])
        by_path = {c["image_path"]: c for c in cases}
        self.assertEqual(by_path["images/test/p5_s100.jpg"]["tags"], ["miss"])
        self.assertEqual(by_path["images/test/p5_s100.jpg"]["details"]["best_iou"], 0.0)
        self.assertEqual(by_path["images/test/p6_s100.jpg"]["tags"], ["miss", "low_iou", "low_confidence"])
        self.assertEqual(by_path["images/test/p6_s100.jpg"]["details"]["best_score"], 0.15)
        self.assertAlmostEqual(by_path["images/test/p6_s100.jpg"]["details"]["best_iou"], 420 / 980, delta=1e-4)
        self.assertEqual(by_path["images/test/p7_s100.jpg"]["tags"], ["multiple_detections"])
        self.assertEqual(by_path["images/test/p7_s100.jpg"]["details"]["prediction_count"], 2)

    def test_recognition_ocr_only_cases(self):
        section = self.failures["recognition"]["ocr_only"]
        self.assertEqual(section["mode"], "gray_contrast")
        self.assertEqual([c["sample_id"] for c in section["cases"]], ["p4_s100", "p9_s100"])
        by_id = {c["sample_id"]: c for c in section["cases"]}
        self.assertEqual(by_id["p4_s100"]["tags"], ["unread"])
        self.assertEqual(by_id["p9_s100"]["tags"], ["substitution", "missing_chars"])
        self.assertEqual(by_id["p9_s100"]["details"]["ground_truth"], "Z9Y8X7W")
        self.assertEqual(by_id["p9_s100"]["details"]["text"], "Z98X71")

    def test_recognition_end_to_end_cases(self):
        section = self.failures["recognition"]["end_to_end"]
        self.assertEqual(section["mode"], "none")
        self.assertEqual([c["sample_id"] for c in section["cases"]],
                         ["p10_s100", "p3_s100", "p4_s100"])
        by_id = {c["sample_id"]: c for c in section["cases"]}
        self.assertEqual(by_id["p3_s100"]["tags"], ["no_detection"])
        self.assertEqual(by_id["p10_s100"]["tags"], ["no_detection"])
        self.assertEqual(by_id["p4_s100"]["tags"], ["abstained"])

    def test_no_test_samples_in_recognition(self):
        test_ids = {"p5_s100", "p6_s100", "p7_s100", "p8_s100"}
        for protocol in ("ocr_only", "end_to_end"):
            for case in self.failures["recognition"][protocol]["cases"]:
                self.assertNotIn(case["sample_id"], test_ids)

    def test_at_least_three_cases_per_stage(self):
        detection_total = len(self.failures["detection"]["val"]) + len(self.failures["detection"]["test"])
        recognition_total = sum(len(s["cases"]) for s in self.failures["recognition"].values())
        self.assertGreaterEqual(detection_total, 3)
        self.assertGreaterEqual(recognition_total, 3)


if __name__ == "__main__":
    unittest.main()
