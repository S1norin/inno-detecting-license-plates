import unittest
from pathlib import Path
from unittest.mock import patch

from baseline.ocr.io import digest, read_json
from models.crnn.model import load_checkpoint
from models.crnn.tests import test_data
from models.crnn.train import better_epoch, train_model


class TrainingLoopTests(unittest.TestCase):
    def setUp(self):
        fixture = test_data.TrainingDataTests(methodName="test_originals_ctc_targets_and_unseen_validation_character")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.charset = fixture.charset

    def test_repeatable_training_and_complete_record(self):
        options = {"charset_path": self.charset, "epochs": 2, "batch_size": 1}
        first = train_model(self.root, self.root / "first", **options)
        second = train_model(self.root, self.root / "second", **options)
        self.assertEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])
        self.assertEqual(first["per_epoch"], second["per_epoch"])
        self.assertEqual(first["checkpoint_sha256"], digest(self.root / "first/checkpoint.pt"))
        for field in ("config", "seed", "threads", "library_versions", "dataset_digests", "charset_sha256",
                      "selected_epoch", "checkpoint_sha256", "per_epoch"):
            self.assertIn(field, read_json(self.root / "first/training_record.json"))
        self.assertEqual(first["completed_epochs"], 2)
        self.assertEqual(first["sample_counts"], {"train": 1, "val": 1})
        self.assertEqual(first["threads"], 1)
        self.assertEqual(first["device"], "cpu")
        self.assertNotIn("ocr_test.csv", first["dataset_digests"])

    def test_tied_validation_keeps_earlier_epoch(self):
        with patch("models.crnn.train.validate_epoch", return_value={"val_character_accuracy": 0.25,
                                                                    "val_plate_exact_match": 0.0}):
            record = train_model(self.root, self.root / "tied", charset_path=self.charset, epochs=2, batch_size=1)
        self.assertEqual(record["selected_epoch"], 1)
        _, checkpoint = load_checkpoint(self.root / "tied/checkpoint.pt")
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertFalse(better_epoch(0.25, 0.25))
        self.assertTrue(better_epoch(0.26, 0.25))

    def test_refuses_output_overwrite_and_invalid_configuration(self):
        train_model(self.root, self.root / "existing", charset_path=self.charset, epochs=1, batch_size=1)
        with self.assertRaises(FileExistsError):
            train_model(self.root, self.root / "existing", charset_path=self.charset)
        with self.assertRaises(ValueError):
            train_model(self.root, self.root / "invalid", epochs=0)

    def test_training_never_reads_test_csv_or_crops(self):
        original_open = Path.open
        original_read = __import__("cv2").imread

        def guarded_open(path, *args, **kwargs):
            if path.name == "ocr_test.csv" or "test" in path.parts:
                raise AssertionError("Training read test data")
            return original_open(path, *args, **kwargs)

        def guarded_read(path, *args, **kwargs):
            if "test" in Path(path).parts:
                raise AssertionError("Training read test crop")
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch("models.crnn.train.cv2.imread", guarded_read):
            train_model(self.root, self.root / "guarded", charset_path=self.charset, epochs=1, batch_size=1)


if __name__ == "__main__":
    unittest.main()
