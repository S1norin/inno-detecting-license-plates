import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from baseline.detection.evaluate import load_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "dataset_original.yaml"
        self.config.write_text(yaml.safe_dump({"path": str(self.root), "val": "val.txt", "nc": 1,
                                              "names": {0: "license_plate"}}))
        (self.root / "run_config.json").write_text(json.dumps({"bucket_train_height_thresholds":
                                                             {"q33": 83, "q67": 112}}))
        self.rows = []
        for index, (height, bucket, tilt) in enumerate([(83, "small", 9), (84, "medium", 10),
                                                       (112, "medium", 19), (113, "large", 20)]):
            image = f"image{index}_s100.jpg"
            label = f"label{index}.txt"
            (self.root / image).write_bytes(b"preflight fixture; decoding is tested by inference")
            (self.root / label).write_text(f"0 0.25 {height / 400} 0.5 {height / 200}\n")
            self.rows.append({"image_path": image, "label_path": label, "scale": 1, "split": "val",
                              "width": 200, "height": 200, "x1": 0, "y1": 0, "x2": 100, "y2": height,
                              "plate_pixel_height": height, "size_bucket": bucket,
                              "tilt_magnitude": tilt, "tilt_horizontal": tilt, "tilt_vertical": 0})
        self.write_metadata()
        (self.root / "val.txt").write_text("".join(f"{self.root / row['image_path']}\n" for row in self.rows))

    def write_metadata(self):
        with (self.root / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_size_boundaries_and_input_fingerprints(self):
        _, paths, records, hashes, _ = load_dataset(self.config, "val")
        self.assertEqual(len(paths), 4)
        self.assertEqual([r["size_bucket"] for r in records], ["small", "medium", "medium", "large"])
        self.assertTrue(all(len(digest) == 64 for digest in hashes.values()))
        self.assertEqual(load_dataset(self.config, "val")[3], hashes)

    def test_missing_image(self):
        (self.root / self.rows[0]["image_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing split image"):
            load_dataset(self.config, "val")

    def test_duplicate_split_entry(self):
        listing = self.root / "val.txt"
        listing.write_text(listing.read_text() + str(self.root / self.rows[0]["image_path"]) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            load_dataset(self.config, "val")

    def test_ambiguous_metadata(self):
        self.rows.append(self.rows[0].copy())
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            load_dataset(self.config, "val")

    def test_non_original_scale_rejected(self):
        self.rows[0]["scale"] = 0.5
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "No original-scale"):
            load_dataset(self.config, "val")

    def test_label_metadata_disagreement(self):
        (self.root / self.rows[0]["label_path"]).write_text("0 0.5 0.5 0.5 0.5\n")
        with self.assertRaisesRegex(ValueError, "disagree"):
            load_dataset(self.config, "val")


if __name__ == "__main__":
    unittest.main()
