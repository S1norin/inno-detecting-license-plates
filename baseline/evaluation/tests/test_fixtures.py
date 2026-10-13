"""Smoke tests for the synthetic fixture trees."""

import tempfile
import unittest
from pathlib import Path

from baseline.evaluation.tests.fixtures import build
from baseline.ocr.io import MODES, PROTOCOLS, read_json, validate_raw


class FixtureTreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.fixture = build(Path(cls.temporary.name))

    def test_dataset_files(self):
        dataset = self.fixture["dataset"]
        for name in ("metadata.csv", "ocr_val.csv", "ocr_test.csv"):
            self.assertTrue((dataset / name).is_file(), name)

    def test_detection_files(self):
        detection = self.fixture["detection"]
        for split in ("val", "test"):
            predictions = read_json(detection / f"predictions_{split}.json")
            metrics = read_json(detection / f"metrics_{split}.json")
            self.assertEqual(len(predictions["images"]), 8 if split == "val" else 4)
            self.assertEqual(metrics["split"], split)
            overall = metrics["overall"]
            self.assertEqual(overall["tp"] + overall["fn"], overall["ground_truth_count"])

    def test_detection_test_matching_literals(self):
        overall = read_json(self.fixture["detection"] / "metrics_test.json")["overall"]
        # Hand-checkable: p5 miss (score 0.15 < 0.25, not selected), p6
        # low-IoU partial overlap (0.43, no match), p7 one TP + one FP, p8 TP
        self.assertEqual(overall["tp"], 2)
        self.assertEqual(overall["fp"], 2)
        self.assertEqual(overall["fn"], 2)

    def test_detection_val_all_matched(self):
        overall = read_json(self.fixture["detection"] / "metrics_val.json")["overall"]
        self.assertEqual(overall["tp"], 8)
        self.assertEqual(overall["fp"], 0)
        self.assertEqual(overall["fn"], 0)

    def test_raw_caches_validate(self):
        for protocol in PROTOCOLS:
            for mode in MODES:
                raw = read_json(self.fixture["ocr"] / protocol / f"raw_{protocol}_val_{mode}.json")
                validate_raw(raw)

    def test_selections_and_test_reports(self):
        for protocol in PROTOCOLS:
            selection = read_json(self.fixture["ocr"] / protocol / f"selection_{protocol}_val.json")
            policy = selection["protocols"][protocol]
            self.assertEqual(set(policy["candidates"]), set(MODES))
            self.assertEqual(policy["mode"], self.fixture["selections"][protocol]["mode"])
            report = read_json(self.fixture["ocr"] / protocol / f"report_{protocol}_test.json")
            self.assertEqual(report["metadata"]["mode"], policy["mode"])
            self.assertEqual(report["metadata"]["tau"], policy["tau"])

    def test_sample_counts(self):
        for protocol in PROTOCOLS:
            for mode in MODES:
                raw = read_json(self.fixture["ocr"] / protocol / f"raw_{protocol}_val_{mode}.json")
                self.assertEqual(raw["metadata"]["sample_count"], len(raw["records"]))

    def test_join_tilt_attaches_metadata_values(self):
        from baseline.ocr.io import join_tilt

        raw = read_json(self.fixture["ocr"] / "ocr_only" / "raw_ocr_only_val_none.json")
        before = [dict(record) for record in raw["records"]]
        joined = join_tilt(raw["records"], self.fixture["dataset"])
        self.assertEqual(raw["records"], before, "join_tilt must not mutate its input")
        values = {record["sample_id"]: record["tilt_magnitude"] for record in joined}
        self.assertEqual(values["p1_s100"], 5.0)
        self.assertEqual(values["p4_s100"], 30.0)

    def test_ocr_tilt_bins_hand_computed(self):
        from baseline.ocr.io import join_tilt
        from baseline.ocr.metrics import evaluate

        raw = read_json(self.fixture["ocr"] / "ocr_only" / "raw_ocr_only_val_none.json")
        records = join_tilt(raw["records"], self.fixture["dataset"])
        by_tilt = evaluate(records, 0.5)["by_tilt_bin"]
        # ocr_only/none val at tau 0.5:
        #   0-9:  p1=5 (exact read), p10=3 (unread)
        #   10-19: p2=15, p9=12 (exact reads), p12=18 (conf 0.4, abstained)
        #   20+:  p3=25 (read, d=2), p4=30 (unread), p11=28 (read, d=3)
        self.assertEqual(by_tilt["0-9"]["count"], 2)
        self.assertEqual(by_tilt["0-9"]["accepted"], 1)
        self.assertEqual(by_tilt["0-9"]["character_accuracy"], 0.5)
        self.assertEqual(by_tilt["10-19"]["count"], 3)
        self.assertEqual(by_tilt["10-19"]["accepted"], 2)
        self.assertEqual(by_tilt["10-19"]["coverage"], 2 / 3)
        self.assertEqual(by_tilt["20+"]["count"], 3)
        self.assertEqual(by_tilt["20+"]["accepted"], 2)
        self.assertEqual(by_tilt["20+"]["plate_exact_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
