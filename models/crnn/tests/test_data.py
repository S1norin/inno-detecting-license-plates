import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from baseline.ocr.charset import build_charset
from baseline.ocr.io import write_json
from models.crnn.train import assemble_data, collate_batch


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = []
        for split in ("train", "val"):
            rows = []
            for scale in (1.0, 0.75, 0.5):
                row = {"parent_id": split, "split": split, "scale": scale,
                       "crop_path": f"crops_gt/{split}/a_s{round(scale * 100)}.png",
                       "plate_text": "皖A12345" if split == "train" else "皖Z12345", "size_bucket": "small"}
                rows.append(row)
                self.metadata.append({**row, "image_path": f"images/{split}/a.jpg", "tilt_magnitude": 0})
                path = self.root / row["crop_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                self.assertTrue(cv2.imwrite(str(path), np.full((20, 60, 3), 100, dtype=np.uint8)))
            self.table(f"ocr_{split}.csv", rows)
        self.table("metadata.csv", self.metadata)
        self.charset = self.root / "charset.json"
        write_json(self.charset, build_charset(self.root / "ocr_train.csv"))

    def table(self, filename, rows):
        with (self.root / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_originals_ctc_targets_and_unseen_validation_character(self):
        training, validation, charset = assemble_data(self.root, self.charset)
        self.assertEqual((len(training), len(validation)), (1, 1))
        self.assertNotIn("Z", charset["characters"])
        images, targets, lengths, texts = collate_batch([training[0], training[0]])
        self.assertEqual(tuple(images.shape), (2, 1, 48, 256))
        self.assertEqual(lengths.tolist(), [7, 7])
        self.assertEqual(targets.numel(), 14)
        self.assertEqual(texts, ("皖A12345", "皖A12345"))
        self.assertEqual(validation[0][2], "皖Z12345")
        self.assertTrue(torch.all(targets > 0))

    def test_charset_mismatch_refused(self):
        charset = build_charset(self.root / "ocr_train.csv")
        charset["characters"].append("Z")
        write_json(self.charset, charset)
        with self.assertRaisesRegex(ValueError, "Charset"):
            assemble_data(self.root, self.charset)

    def test_assembly_never_opens_test_csv_or_crops(self):
        original_open = Path.open
        paths = []

        def guarded(path, *args, **kwargs):
            paths.append(path)
            if path.name == "ocr_test.csv" or "test" in path.relative_to(self.root).parts:
                raise AssertionError("Test data must not be opened")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            training, validation, _ = assemble_data(self.root, self.charset)
            training[0]
            validation[0]
        self.assertTrue(paths)


if __name__ == "__main__":
    unittest.main()
