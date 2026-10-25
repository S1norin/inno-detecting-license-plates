import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from baseline.common.weights import sha256
from models.yolo11_finetune import config, train
from models.yolo11_finetune.trainer import Map50Stopping, Map50Trainer, Map50Validator


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.file = self.root / "config.yaml"
        self.document = yaml.safe_load(config.DEFAULT_CONFIG.read_text())

    def read(self, **changes):
        self.file.write_text(yaml.safe_dump({**self.document, **changes}))
        return config.read_config(self.file)

    def test_valid_defaults_and_two_epoch_smoke(self):
        result = self.read()
        self.assertEqual(result["patience"], 20)
        self.assertEqual(config.read_config(self.file, epochs=2, name="smoke")["epochs"], 2)

    def test_positive_integer_settings(self):
        for key in ("epochs", "batch", "imgsz", "patience"):
            for value in (0, -1, True, 1.5, "16", None, float("nan")):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                    self.read(**{key: value})

    def test_protocol_seed_image_size_and_budget(self):
        for changes in ({"seed": True}, {"seed": 43}, {"deterministic": False}, {"imgsz": 320}, {"epochs": 101}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.read(**changes)

    def test_unknown_missing_and_invalid_paths(self):
        for changes in ({"resume": True}, {"project": "/tmp/untracked-training"}, {"name": "../other"},
                        {"model": None}, {"data": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.read(**changes)
        self.file.write_text("epochs: 2\n")
        with self.assertRaises(ValueError):
            config.read_config(self.file)

    def test_corrupt_cached_start_weights_fail_before_model_load(self):
        checkpoint = self.root / "pretrained.pt"
        checkpoint.write_bytes(b"corrupted")
        expected = "0" * 64
        manifest = self.root / "models.json"
        manifest.write_text(json.dumps([{"name": "yolo11s-license-plate", "filename": checkpoint.name,
                                         "url": "https://example.com/weights", "sha256": expected}]))
        from baseline.common.weights import verified_path

        self.file.write_text(yaml.safe_dump(self.document))
        with patch.object(train, "require_gpu", return_value={}), \
                patch.object(train, "environment_versions", return_value={}), \
                patch.object(config, "verified_path", side_effect=lambda: verified_path(cache=self.root,
                                                                                       manifest=manifest)), \
                patch("ultralytics.YOLO") as yolo, self.assertRaisesRegex(ValueError, "checksum mismatch"):
            train.train(self.file)
        yolo.assert_not_called()

    def test_other_checkpoint_path_rejected(self):
        document = self.read(model="other.pt")
        with patch.object(config, "verified_path", return_value=self.root / "pinned.pt"), \
                self.assertRaisesRegex(ValueError, "verified cached detector"):
            config.starting_weights(document)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.config = self.root / "dataset_original.yaml"
        self.config.write_text(yaml.safe_dump({"path": str(self.root), "train": "train.txt", "val": "val.txt",
                                              "test": "must-not-read.txt", "nc": 1, "names": {0: "license_plate"}}))
        (self.root / "run_config.json").write_text(
            json.dumps({"bucket_train_height_thresholds": {"q33": 83, "q67": 112}}))
        self.rows = []
        for split in ("train", "val"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            image = f"images/{split}/original_s100.jpg"
            label = f"labels/{split}/original_s100.txt"
            (self.root / image).write_bytes(b"original image fixture")
            (self.root / label).write_text("0 0.25 0.2075 0.5 0.415\n")
            (self.root / f"{split}.txt").write_text(f"{self.root / image}\n")
            self.rows.append({"image_path": image, "label_path": label, "scale": 1, "split": split,
                              "width": 200, "height": 200, "x1": 0, "y1": 0, "x2": 100, "y2": 83,
                              "plate_pixel_height": 83, "size_bucket": "small", "tilt_magnitude": 0,
                              "tilt_horizontal": 0, "tilt_vertical": 0})
        self.write_metadata()

    def write_metadata(self):
        with (self.root / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_generated_yaml_has_only_verified_original_train_and_val(self):
        with patch.object(train, "EXPECTED_COUNTS", {"train": 1, "val": 1}):
            dataset, inputs = train.training_inputs(self.config, self.output)
        document = yaml.safe_load(dataset.read_text())
        self.assertNotIn("test", document)
        self.assertNotIn("download", document)
        self.assertEqual(inputs["train"]["count"], 1)
        self.assertEqual(inputs["val"]["count"], 1)
        self.assertEqual(Path(document["train"]).read_text(), (self.root / "train.txt").read_text())
        self.assertFalse((self.root / "must-not-read.txt").exists())

    def test_incomplete_training_split_refused(self):
        with self.assertRaisesRegex(ValueError, "Expected 3500"):
            train.training_inputs(self.config, self.output)

    def test_scaled_sample_refused(self):
        self.rows[0]["scale"] = 0.5
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "original-scale"):
            train.training_inputs(self.config, self.output)

    def test_validation_image_cannot_enter_train_list(self):
        (self.root / "train.txt").write_text((self.root / "val.txt").read_text())
        with self.assertRaisesRegex(ValueError, "original-scale train metadata"):
            train.training_inputs(self.config, self.output)


class SelectionTests(unittest.TestCase):
    def test_map50_not_library_weighted_fitness(self):
        validator = object.__new__(Map50Validator)
        # mAP50 wins at the first epoch while the library's mixed score wins at the second.
        for map50, mixed in ((0.9, 0.4), (0.8, 0.7)):
            with patch("ultralytics.models.yolo.detect.DetectionValidator.get_stats",
                       return_value={train.MAP50: map50, "fitness": mixed}):
                self.assertEqual(validator.get_stats()["fitness"], map50)

    def test_invalid_validation_score_refused(self):
        for score in (float("nan"), -0.1, 1.1):
            with patch("ultralytics.models.yolo.detect.DetectionValidator.get_stats",
                       return_value={train.MAP50: score}), self.assertRaises(ValueError):
                object.__new__(Map50Validator).get_stats()

    def test_patience_and_zero_plateau(self):
        for sequence in ([0.0] * 4, [0.8, 0.9, 0.8, 0.9, 0.8]):
            with self.subTest(sequence=sequence):
                stopper = Map50Stopping(patience=3)
                decisions = [stopper(i, score) for i, score in enumerate(sequence, 1)]
                self.assertFalse(any(decisions[:-1]))
                self.assertTrue(decisions[-1])

    def test_best_checkpoint_preserves_first_peak_on_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = object.__new__(Map50Trainer)
            trainer.best = Path(directory) / "best.pt"

            def library_save():
                if trainer.fitness == trainer.best_fitness:
                    trainer.best.write_bytes(f"epoch-{trainer.epoch}".encode())

            with patch("ultralytics.models.yolo.detect.DetectionTrainer.save_model", side_effect=library_save):
                for epoch, score in enumerate([0.0, 0.0, 0.9, 0.8, 0.9], 1):
                    trainer.epoch, trainer.fitness = epoch, score
                    trainer.best_fitness = max(getattr(trainer, "best_fitness", 0.0), score)
                    trainer.save_model()
            self.assertEqual(trainer.best.read_bytes(), b"epoch-3")


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "training_record.json"
        self.record = train.TrainingRecord(self.path, {"epochs": 4, "patience": 20}, "a" * 64,
                                           {"gpu_name": "fixture GPU"}, {"ultralytics": "8.3.162"},
                                           {"train": {"count": 3500}})
        self.trainer = SimpleNamespace(args=SimpleNamespace(lr0=0.01, warmup_epochs=3.0, seed=42, patience=20),
                                       optimizer=SimpleNamespace(param_groups=[{"lr": 0.002, "initial_lr": 0.002}]),
                                       lr={"lr/pg0": 0.002})

    def test_complete_record_history_selection_and_checkpoint_digest(self):
        self.record.on_start(self.trainer)
        for epoch, score in enumerate((0.4, 0.9, 0.9, 0.7)):
            self.trainer.epoch = epoch
            self.trainer.metrics = {train.MAP50: score}
            self.trainer.fitness = score
            self.record.on_epoch(self.trainer)
        self.record.on_epoch(self.trainer)  # final_eval must not append a fifth epoch
        checkpoint = self.root / "best.pt"
        checkpoint.write_bytes(b"selected checkpoint")
        self.record.complete(checkpoint)
        record = json.loads(self.path.read_text())
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["selected_epoch"], 2)
        self.assertEqual(record["epochs_completed"], 4)
        self.assertEqual([item["val_map50"] for item in record["history"]], [0.4, 0.9, 0.9, 0.7])
        self.assertEqual(record["final_checkpoint_sha256"], sha256(checkpoint))
        for key in ("start_weight_sha256", "requested_config", "config", "optimizer", "library_versions",
                    "machine", "inputs", "started_at", "finished_at", "duration_seconds", "reproducibility_scope"):
            self.assertTrue(record[key], key)
        self.assertEqual(record["config"]["lr0"], 0.01)

    def test_fitness_mismatch_refused(self):
        self.trainer.epoch, self.trainer.fitness = 0, 0.4
        self.trainer.metrics = {train.MAP50: 0.9}
        with self.assertRaisesRegex(RuntimeError, "fitness"):
            self.record.on_epoch(self.trainer)

    def test_empty_history_cannot_be_success(self):
        with self.assertRaisesRegex(RuntimeError, "without a complete"):
            self.record.complete(self.root / "missing.pt")

    def test_failure_record_is_not_reported_as_completed(self):
        self.record.fail(RuntimeError("interrupted"))
        record = json.loads(self.path.read_text())
        self.assertEqual(record["status"], "failed")
        self.assertIn("interrupted", record["error"])
        self.assertNotIn("final_checkpoint_sha256", record)


class WrapperTests(unittest.TestCase):
    def test_cpu_refusal_before_any_dataset_read(self):
        with patch("torch.cuda.is_available", return_value=False), \
                patch.object(train, "training_inputs") as inputs, self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
            train.train()
        inputs.assert_not_called()

    def test_single_gpu_only_and_pinned_cuda(self):
        for device in (-1, True, "cpu", "0,1"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                train.require_gpu(device)
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=1), \
                patch("torch.version.cuda", "12.1"), \
                self.assertRaisesRegex(RuntimeError, "CUDA 12.4"):
            train.require_gpu(0)

    def test_wrapper_passes_guarded_inputs_and_writes_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight = root / "start.pt"
            weight.write_bytes(b"verified fixture")
            document = {"model": str(weight), "data": str(root / "source.yaml"), "project": str(root),
                        "name": "run", "epochs": 2, "patience": 20, "imgsz": 640, "batch": 16,
                        "seed": 42, "deterministic": True}
            model = Mock(names={0: "license_plate"})
            callbacks = {}
            model.add_callback.side_effect = lambda key, value: callbacks.update({key: value})

            def fake_training(**options):
                trainer = SimpleNamespace(args=SimpleNamespace(**options),
                                          optimizer=SimpleNamespace(param_groups=[{"lr": 0.01}]), lr={"lr/pg0": 0.01},
                                          best=root / "run/weights/best.pt")
                model.trainer = trainer
                callbacks["on_train_start"](trainer)
                for epoch, score in enumerate([0.7, 0.8]):
                    trainer.epoch, trainer.fitness, trainer.metrics = epoch, score, {train.MAP50: score}
                    callbacks["on_fit_epoch_end"](trainer)
                trainer.best.write_bytes(b"best validation checkpoint")

            model.train.side_effect = fake_training
            with patch.object(train, "require_gpu", return_value={"gpu_name": "fixture"}), \
                    patch.object(train, "environment_versions", return_value={"ultralytics": "8.3.162"}), \
                    patch.object(train, "read_config", return_value=document), \
                    patch.object(train, "starting_weights", return_value=weight), \
                    patch.object(train, "training_inputs", return_value=(root / "train_val.yaml", {"train": {}})), \
                    patch("ultralytics.YOLO", return_value=model):
                record = train.train()
                with self.assertRaises(FileExistsError):
                    train.train()
            options = model.train.call_args.kwargs
            self.assertIs(options["trainer"], Map50Trainer)
            self.assertTrue(options["val"])
            self.assertFalse(options["amp"])
            self.assertFalse(options["resume"])
            self.assertEqual(options["split"], "val")
            self.assertEqual(record["selected_epoch"], 2)
            self.assertEqual(record["start_weight_sha256"], sha256(weight))
            self.assertEqual(record["status"], "completed")


if __name__ == "__main__":
    unittest.main()
