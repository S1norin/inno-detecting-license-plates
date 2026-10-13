"""Input loading and provenance validation tests."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from baseline.evaluation.inputs import InputError, load_all
from baseline.evaluation.tests.fixtures import DETECTION_MODEL, build


def _load(root):
    return load_all(root / "dataset", root / "detection", root / "ocr")


def _mutate(root, relative, mutator):
    path = root / relative
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    mutator(value)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


class InputLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        build(cls.root)

    def _tampered_root(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "tree"
        shutil.copytree(self.root, root)
        return root

    def test_valid_fixtures_load(self):
        dataset, detection, ocr = _load(self.root)
        self.assertEqual(detection.model_id, DETECTION_MODEL)
        self.assertEqual(ocr.engine, "easyocr")
        self.assertEqual(dataset.sample_count, {"val": 8, "test": 4})
        self.assertEqual(len(detection.predictions["val"]), 8)
        self.assertEqual(len(ocr.raw), 6)
        for protocol in ("ocr_only", "end_to_end"):
            self.assertIn(protocol, ocr.selection)
            self.assertEqual(ocr.test_report[protocol]["metadata"]["split"], "test")

    def test_missing_file_named(self):
        root = self._tampered_root()
        (root / "detection" / "metrics_test.json").unlink()
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("metrics_test.json", str(error.exception))

    def test_missing_ocr_file_named(self):
        root = self._tampered_root()
        (root / "ocr" / "end_to_end" / "selection_end_to_end_val.json").unlink()
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("selection_end_to_end_val.json", str(error.exception))

    def test_wrong_detection_split(self):
        root = self._tampered_root()
        _mutate(root, "detection/metrics_test.json", lambda doc: doc.__setitem__("split", "val"))
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("split", str(error.exception))

    def test_detection_model_mismatch(self):
        root = self._tampered_root()
        _mutate(root, "detection/metrics_val.json", lambda doc: doc.__setitem__("model_id", "yolo11n"))
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("model", str(error.exception))

    def test_detection_sample_set_mismatch(self):
        root = self._tampered_root()
        def drop_last(doc):
            doc["images"].pop()
        _mutate(root, "detection/predictions_test.json", drop_last)
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("sample set", str(error.exception))

    def test_ocr_dataset_digest_mismatch(self):
        root = self._tampered_root()
        def bad_digest(doc):
            doc["metadata"]["dataset_digests"]["ocr_val.csv"] = "0" * 64
        _mutate(root, "ocr/ocr_only/raw_ocr_only_val_none.json", bad_digest)
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("ocr_val.csv", str(error.exception))

    def test_ocr_sample_count_mismatch(self):
        root = self._tampered_root()
        path = "ocr/ocr_only/raw_ocr_only_val_canny.json"
        _mutate(root, path, lambda doc: doc["metadata"].__setitem__("sample_count", 7))
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("incomplete", str(error.exception))
        self.assertIn("raw_ocr_only_val_canny.json", str(error.exception))

    def test_ocr_sample_count_differs_from_dataset(self):
        root = self._tampered_root()
        def drop_record(doc):
            doc["records"].pop()
            doc["metadata"]["sample_count"] = len(doc["records"])
        _mutate(root, "ocr/ocr_only/raw_ocr_only_val_canny.json", drop_record)
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("sample_count", str(error.exception))

    def test_ocr_engine_mismatch(self):
        root = self._tampered_root()
        path = "ocr/end_to_end/raw_end_to_end_val_none.json"
        _mutate(root, path, lambda doc: doc["metadata"].__setitem__("engine", "paddleocr"))
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("engine", str(error.exception))

    def test_e2e_detector_model_mismatch(self):
        root = self._tampered_root()
        def bad_detector(doc):
            doc["metadata"]["detector"]["model_id"] = "yolo11n"
        _mutate(root, "ocr/end_to_end/raw_end_to_end_val_gray_contrast.json", bad_detector)
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("detector model", str(error.exception))

    def test_ocr_raw_protocol_mismatch(self):
        root = self._tampered_root()
        path = "ocr/ocr_only/raw_ocr_only_val_none.json"
        _mutate(root, path, lambda doc: doc["metadata"].__setitem__("protocol", "end_to_end"))
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("protocol", str(error.exception))

    def test_selection_raw_digest_mismatch(self):
        root = self._tampered_root()
        def bad_selection(doc):
            doc["protocols"]["ocr_only"]["candidates"]["none"]["raw_sha256"] = "0" * 64
        _mutate(root, "ocr/ocr_only/selection_ocr_only_val.json", bad_selection)
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("raw digest", str(error.exception))

    def test_test_report_mode_mismatch(self):
        root = self._tampered_root()
        def bad_mode(doc):
            doc["metadata"]["mode"] = "canny"
        _mutate(root, "ocr/end_to_end/report_end_to_end_test.json", bad_mode)
        with self.assertRaises(InputError) as error:
            _load(root)
        self.assertIn("mode", str(error.exception))

    def test_digests_recorded(self):
        dataset, detection, ocr = _load(self.root)
        self.assertEqual(set(dataset.digests), {"metadata.csv", "ocr_val.csv", "ocr_test.csv"})
        self.assertEqual(len(detection.digests), 4)
        self.assertEqual(len(ocr.digests), 10)
        for value in list(dataset.digests.values()) + list(detection.digests.values()) + list(ocr.digests.values()):
            self.assertEqual(len(value), 64)


if __name__ == "__main__":
    unittest.main()
