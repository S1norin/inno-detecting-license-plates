import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import yaml

from baseline.common.weights import sha256
from baseline.detection import evaluate as evaluate_module


class ModelOptionTests(unittest.TestCase):
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
        for index, height in enumerate((83, 113)):
            image = f"image{index}_s100.jpg"
            label = f"label{index}.txt"
            (self.root / image).write_bytes(b"fixture image bytes")
            (self.root / label).write_text(f"0 0.25 {height / 400} 0.5 {height / 200}\n")
            self.rows.append({"image_path": image, "label_path": label, "scale": 1, "split": "val",
                              "width": 200, "height": 200, "x1": 0, "y1": 0, "x2": 100, "y2": height,
                              "plate_pixel_height": height,
                              "size_bucket": "small" if height <= 83 else "large",
                              "tilt_magnitude": 0, "tilt_horizontal": 0, "tilt_vertical": 0})
        with (self.root / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        (self.root / "val.txt").write_text("".join(f"{self.root / row['image_path']}\n" for row in self.rows))
        self.weight = self.root / "supplied.pt"
        self.weight.write_bytes(b"supplied checkpoint bytes")
        self._deterministic = torch.are_deterministic_algorithms_enabled()
        self._threads = torch.get_num_threads()

    def tearDown(self):
        torch.use_deterministic_algorithms(self._deterministic)
        torch.set_num_threads(self._threads)

    def fake_model(self):
        model = MagicMock()
        model.names = {0: "license_plate"}
        results = []
        for row in self.rows:
            boxes = MagicMock()
            boxes.cls = torch.tensor([0.0])
            boxes.xyxy = torch.tensor([[0.0, 0.0, 100.0, float(row["plate_pixel_height"])]])
            boxes.conf = torch.tensor([0.9])
            result = MagicMock()
            result.path = str(self.root / row["image_path"])
            result.orig_shape = (row["height"], row["width"])
            result.boxes = boxes
            results.append(result)
        model.predict.return_value = iter(results)
        return model

    def run_evaluate(self, run_id, **kwargs):
        with patch.object(evaluate_module, "RESULTS", self.root / "results"), \
             patch("ultralytics.YOLO", return_value=self.fake_model()) as yolo:
            metrics = evaluate_module.evaluate(self.config, "val", run_id, **kwargs)
        return metrics, yolo

    def test_default_uses_verified_pinned_weights(self):
        pinned = self.root / "pinned.pt"
        pinned.write_bytes(b"pinned checkpoint bytes")
        with patch.object(evaluate_module, "verified_path", return_value=pinned):
            metrics, yolo = self.run_evaluate("default-run")
        self.assertEqual(metrics["model_id"], evaluate_module.DEFAULT_MODEL)
        self.assertEqual(metrics["weight_sha256"], sha256(pinned))
        yolo.assert_called_once_with(str(pinned))

    def test_supplied_model_digest_recorded_without_manifest(self):
        metrics, yolo = self.run_evaluate("supplied-run", model=self.weight)
        self.assertEqual(metrics["model_id"], "supplied.pt")
        self.assertEqual(metrics["weight_sha256"], sha256(self.weight))
        yolo.assert_called_once_with(str(self.weight))
        self.assertGreater(metrics["overall"]["map50"], 0.9)

    def test_missing_supplied_model_refused(self):
        with self.assertRaisesRegex(FileNotFoundError, "Model weights missing"):
            evaluate_module.evaluate(self.config, "val", "missing-run", model=self.root / "absent.pt")

    def test_degenerate_model_boxes_are_filtered(self):
        model = self.fake_model()
        results = list(model.predict.return_value)
        results[0].boxes.cls = torch.tensor([0.0, 0.0])
        results[0].boxes.xyxy = torch.tensor([[0.0, 0.0, 100.0, 83.0], [50.0, 0.0, 50.0, 40.0]])
        results[0].boxes.conf = torch.tensor([0.9, 0.8])
        model.predict.return_value = iter(results)
        with patch.object(evaluate_module, "RESULTS", self.root / "results"), \
             patch("ultralytics.YOLO", return_value=model):
            metrics = evaluate_module.evaluate(self.config, "val", "degenerate-run", model=self.weight)
        self.assertEqual(metrics["overall"]["prediction_count"], 2)


if __name__ == "__main__":
    unittest.main()
