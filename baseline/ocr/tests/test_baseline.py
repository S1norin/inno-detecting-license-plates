import argparse
import copy
import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline.ocr.charset import build_charset
from baseline.ocr.confidence import accepts, plate_score, sweep
from baseline.ocr.ctc import decode_greedy
from baseline.ocr.end_to_end import load_predictions, select_box
from baseline.ocr.engine import WEIGHT_FILES, EasyOCREngine, verify_weights
from baseline.ocr.evaluate import evaluate_test, evaluate_validation, model_contract, select_validation
from baseline.ocr.export_train import export_training
from baseline.ocr.io import digest, load_samples, validate_raw, write_json
from baseline.ocr.metrics import edit_distance, evaluate
from baseline.ocr.run import run_inference
from baseline.ocr.setup import combined_pins, prepare


def record(text="皖A12345", confidences=None, status=None, sample_id="sample", bucket="medium"):
    return {
        "sample_id": sample_id,
        "size_bucket": bucket,
        "ground_truth": "皖A12345",
        "text": text,
        "char_confidences": confidences if confidences is not None else [0.9] * len(text),
        "status": status or ("read" if text else "unread"),
    }


def raw(mode="none", split="val", records=None):
    records = records if records is not None else [record()]
    return {
        "schema_version": 1,
        "metadata": {
            "protocol": "ocr_only",
            "split": split,
            "mode": mode,
            "scale": 1.0,
            "sample_count": len(records),
            "engine": {"name": "synthetic"},
            "weight_digests": {"synthetic": "a" * 64},
            "dataset_digests": {"metadata.csv": "b" * 64, f"ocr_{split}.csv": "c" * 64},
            "timestamp": "2026-10-12T00:00:00+00:00",
        },
        "records": records,
    }


class MetricTests(unittest.TestCase):
    def test_hand_computed_plate_metrics_and_abstention(self):
        records = [record(), record("皖A12346", sample_id="one_error"), record("皖A123XX", sample_id="wrong")]
        records.append(record("", sample_id="missing", status="no_detection"))
        result = evaluate(records, 0.5)
        overall = result["overall"]
        self.assertEqual((overall["tp"], overall["fp"], overall["fn"]), (2, 1, 1))
        self.assertAlmostEqual(overall["character_accuracy"], 9 / 14)
        self.assertEqual(overall["plate_exact_match"], 0.25)
        self.assertEqual(overall["coverage"], 0.75)
        self.assertAlmostEqual(overall["f1"], 2 / 3)
        self.assertEqual(result["by_size_bucket"]["small"]["count"], 0)

    def test_normalization_and_insertion_distance(self):
        self.assertEqual(edit_distance("ABC", "ABXC"), 1)
        self.assertEqual(evaluate([record(" 皖a12345 ")], 0.5)["overall"]["plate_exact_match"], 1)

    def test_scores_and_log_threshold_domain(self):
        self.assertAlmostEqual(plate_score([0.8, 0.5]), 0.65)
        self.assertAlmostEqual(plate_score([0.8, 0.5], "product"), 0.4)
        candidate = record("AB", [0.8, 0.8])
        for tau in (0.5, 0.7, 0.99):
            self.assertEqual(accepts(candidate, tau, "product"), accepts(candidate, tau, "sumlog"))
        self.assertFalse(accepts(record("", []), 0))
        self.assertFalse(accepts(record("A", [0]), 0.5, "sumlog"))
        with self.assertRaises(ValueError):
            plate_score([float("nan")])

    def test_threshold_ties_choose_lower_tau(self):
        policy = sweep([record(confidences=[1.0] * 7)])
        self.assertEqual(policy["selected_tau"], 0.5)
        self.assertEqual(len(policy["curve"]), 50)
        with self.assertRaises(ValueError):
            sweep([])

    def test_ctc_repeated_symbols_and_blank_boundaries(self):
        result = decode_greedy([0, 1, 1, 0, 1, 2, 2], [0.9, 0.6, 0.8, 0.7, 0.5, 0.9, 0.8], ["_", "A", "B"])
        self.assertEqual(result, {"text": "AAB", "char_confidences": [0.8, 0.5, 0.9]})
        self.assertEqual(decode_greedy([0, 0], [1, 1], ["_"])["text"], "")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def cache(self, value):
        path = self.root / f"{value['metadata']['mode']}.json"
        write_json(path, value)
        return path

    def test_select_validation_then_test_without_reocr(self):
        paths = [self.cache(raw(mode)) for mode in ("none", "gray_contrast", "canny")]
        selection = select_validation(paths)
        self.assertEqual(selection["protocols"]["ocr_only"]["mode"], "none")
        report = evaluate_test(raw(split="test"), selection)
        self.assertEqual(report["metrics"]["overall"]["plate_exact_match"], 1)
        self.assertEqual(report["metadata"]["tau"], 0.5)

        changed = raw(split="test")
        changed["metadata"]["weight_digests"]["synthetic"] = "d" * 64
        with self.assertRaises(ValueError):
            evaluate_test(changed, selection)

    def test_no_test_tuning_or_partial_ablation(self):
        with self.assertRaises(ValueError):
            select_validation([self.cache(raw(split="test"))])
        with self.assertRaises(ValueError):
            select_validation([self.cache(raw())])

    def test_validation_report_has_traceability_and_binds_to_selected_cache(self):
        paths = [self.cache(raw(mode)) for mode in ("none", "gray_contrast", "canny")]
        selection = select_validation(paths)
        report = evaluate_validation(raw(), selection, digest(paths[0]))
        self.assertEqual(report["metadata"]["split"], "val")
        self.assertEqual(report["metadata"]["tau"], 0.5)
        self.assertTrue(report["metadata"]["selected_mode"])
        with self.assertRaises(ValueError):
            evaluate_validation(raw(), selection, "d" * 64)

    def test_reject_mixed_validation_samples(self):
        paths = [self.cache(raw(mode)) for mode in ("none", "gray_contrast")]
        paths.append(self.cache(raw("canny", records=[record(sample_id="different")])))
        with self.assertRaises(ValueError):
            select_validation(paths)

    def test_reject_incomplete_or_malformed_cache(self):
        value = raw()
        value["metadata"]["sample_count"] = 2
        with self.assertRaises(ValueError):
            validate_raw(value)
        validate_raw(value, complete=False)
        value = raw(records=[record(confidences=[0.9])])
        with self.assertRaises(ValueError):
            validate_raw(value)

    def test_hash_verified_weights_fail_closed(self):
        entries = []
        for filename in WEIGHT_FILES:
            path = self.root / filename
            path.write_bytes(b"synthetic test weight")
            entries.append({"name": filename, "filename": f"easyocr/{filename}", "sha256": digest(path)})
        manifest = self.root / "models.json"
        write_json(manifest, {"models": entries})
        self.assertEqual(len(verify_weights(self.root, manifest)), 2)
        (self.root / WEIGHT_FILES[0]).write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "expected.*actual"):
            verify_weights(self.root, manifest)

    def test_engine_disables_downloads_and_switches_recognizer_to_eval(self):
        options = []
        evaluations = []
        reader = types.SimpleNamespace(recognizer=types.SimpleNamespace(eval=lambda: evaluations.append(True)))
        easyocr = types.ModuleType("easyocr")
        easyocr.Reader = lambda languages, **kwargs: options.append(kwargs) or reader
        torch = types.ModuleType("torch")
        torch.set_num_threads = lambda count: None
        torch.use_deterministic_algorithms = lambda enabled: None
        with (
            patch.dict(sys.modules, {"easyocr": easyocr, "torch": torch}),
            patch("baseline.ocr.engine.verify_weights", return_value={"test": "a" * 64}),
            patch("baseline.ocr.engine.version", return_value="1.7.2"),
        ):
            EasyOCREngine(self.root, self.root / "models.json")
        self.assertEqual(evaluations, [True])
        self.assertFalse(options[0]["download_enabled"])
        self.assertFalse(options[0]["gpu"])
        self.assertFalse(options[0]["quantize"])

    def test_clamping_no_detection_and_deterministic_box_choice(self):
        self.assertEqual(
            select_box([[-3.2, 1.4, 200, 15.2], [1, 1, 3, 3]], [0.9, 0.9], 100, 20), ((0, 1, 100, 16), 0.9)
        )
        self.assertIsNone(select_box([], [], 100, 20))
        self.assertIsNone(select_box([[110, 1, 120, 3]], [0.9], 100, 20))

    def test_predictions_require_all_images_and_pixel_coordinates(self):
        path = self.root / "predictions.json"
        value = {
            "schema_version": 1,
            "metadata": {"box_format": "xyxy_pixels", "model_id": "test", "weight_sha256": "a" * 64},
            "records": [{"image_path": "images/val/a.jpg", "boxes": [], "scores": []}],
        }
        write_json(path, value)
        samples = [{"image_path": "images/val/a.jpg"}]
        _, predictions = load_predictions(path, samples)
        self.assertEqual(predictions[samples[0]["image_path"]]["boxes"], [])
        with self.assertRaises(ValueError):
            load_predictions(path, samples + [{"image_path": "images/val/b.jpg"}])

    def test_native_detector_predictions_preserve_traceability_and_reject_wrong_inputs(self):
        path = self.root / "native.json"
        value = {
            "provenance": {
                "schema_version": 1,
                "model_id": "yolo11s-license-plate",
                "weight_sha256": "a" * 64,
                "split": "val",
                "inference": {"conf": 0.001, "imgsz": 640},
                "input_sha256": {"metadata": "b" * 64},
            },
            "images": [{"image_path": "images/val/a.jpg", "boxes": [[1, 2, 5, 6]], "scores": [0.9], "class_ids": [0]}],
        }
        write_json(path, value)
        samples = [{"image_path": "images/val/a.jpg"}]
        metadata, predictions = load_predictions(path, samples, split="val", metadata_sha256="b" * 64)
        self.assertEqual(metadata["inference"]["conf"], 0.001)
        self.assertEqual(predictions["images/val/a.jpg"]["boxes"], [[1, 2, 5, 6]])
        with self.assertRaisesRegex(ValueError, "different split"):
            load_predictions(path, samples, split="test")
        with self.assertRaisesRegex(ValueError, "different dataset"):
            load_predictions(path, samples, metadata_sha256="c" * 64)
        value["images"][0]["class_ids"] = [1]
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "single-class"):
            load_predictions(path, samples)

    def test_ocr_setup_preserves_shared_pins_and_defaults_to_inspection(self):
        pins = combined_pins()
        expected_torch = "2.6.0" if sys.platform == "darwin" else "2.6.0+cpu"
        self.assertEqual(pins["torch"], expected_torch)
        self.assertEqual(pins["matplotlib"], "3.10.3")
        self.assertEqual(pins["easyocr"], "1.7.2")
        with patch("baseline.ocr.setup.ensure_environment") as install:
            prepare()
            install.assert_not_called()


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        metadata = []
        for split in ("train", "val", "test"):
            rows = []
            for scale in (1.0, 0.75, 0.5):
                sample = {
                    "parent_id": f"parent_{split}",
                    "split": split,
                    "scale": scale,
                    "crop_path": f"crops_gt/{split}/a_{scale}.png",
                    "plate_text": "皖A12345",
                    "size_bucket": "medium",
                    "tilt_magnitude": 5.0,
                }
                rows.append(sample)
                metadata.append({**sample, "image_path": f"images/{split}/a_{scale}.jpg"})
                for field in ("crop_path", "image_path"):
                    path = self.root / metadata[-1][field]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"test image placeholder")
            self.table(f"ocr_{split}.csv", rows)
        self.table("metadata.csv", metadata)

    def table(self, filename, rows):
        with (self.root / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_original_scale_is_default_and_charset_is_training_only(self):
        self.assertEqual(len(load_samples(self.root, "val")), 1)
        charset = build_charset(self.root / "ocr_train.csv")
        self.assertEqual(set(charset["characters"]), set("皖A12345"))
        self.assertEqual(len(export_training(self.root, charset, "all")["records"]), 3)
        with self.assertRaises(ValueError):
            build_charset(self.root / "ocr_val.csv")
        (self.root / "crops_gt/train/a_1.0.png").unlink()
        with self.assertRaises(FileNotFoundError):
            export_training(self.root, charset)

    def test_split_and_path_leakage_are_rejected(self):
        rows = [
            {
                "parent_id": "parent_val",
                "split": "test",
                "scale": 1.0,
                "crop_path": "../outside.png",
                "plate_text": "皖A12345",
                "size_bucket": "medium",
            }
        ]
        self.table("ocr_val.csv", rows)
        with self.assertRaises(ValueError):
            load_samples(self.root, "val")

    def inference_args(self):
        return argparse.Namespace(
            dataset=self.root,
            split="val",
            scale=1.0,
            protocol="ocr_only",
            mode="none",
            predictions=None,
            selection=None,
            output=self.root / "raw.json",
        )

    def test_cache_resume_does_not_recognize_completed_samples(self):
        engine = types.SimpleNamespace(identity={"name": "synthetic"}, weight_digests={"test": "a" * 64})
        calls = []
        engine.recognize = lambda image: calls.append(image) or {"text": "皖A12345", "char_confidences": [0.9] * 7}
        cv2 = types.ModuleType("cv2")
        cv2.imread = lambda path: "synthetic image"
        preprocessing = types.ModuleType("data.preprocessing")
        preprocessing.preprocess_plate = lambda image, mode: image
        args = self.inference_args()

        with patch.dict(sys.modules, {"cv2": cv2, "data.preprocessing": preprocessing}):
            cache = run_inference(args, engine)
            run_inference(args, engine)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(cache["records"]), 1)

            args.mode = "canny"
            with self.assertRaises(ValueError):
                run_inference(args, engine)
            args.mode = "none"
            (self.root / "crops_gt/val/a_1.0.png").write_bytes(b"changed input")
            with self.assertRaises(ValueError):
                run_inference(args, engine)

    def test_test_inference_requires_selection_and_reserves_one_cache(self):
        engine = types.SimpleNamespace(identity={"name": "synthetic"}, weight_digests={"test": "a" * 64})
        engine.recognize = lambda image: {"text": "皖A12345", "char_confidences": [0.9] * 7}
        cv2 = types.ModuleType("cv2")
        cv2.imread = lambda path: "synthetic image"
        preprocessing = types.ModuleType("data.preprocessing")
        preprocessing.preprocess_plate = lambda image, mode: image
        args = self.inference_args()
        args.split = "test"

        with patch.dict(sys.modules, {"cv2": cv2, "data.preprocessing": preprocessing}):
            with self.assertRaises(ValueError):
                run_inference(args, engine)
            validation = raw()
            validation["metadata"]["engine"] = engine.identity
            validation["metadata"]["weight_digests"] = engine.weight_digests
            validation["metadata"]["dataset_digests"]["metadata.csv"] = digest(self.root / "metadata.csv")
            validation["metadata"]["preprocessing_sha256"] = digest(Path("data/preprocessing.py"))
            contract = model_contract(validation["metadata"])
            selection = {
                "schema_version": 1,
                "selection_split": "val",
                "timestamp": "test",
                "protocols": {"ocr_only": {"mode": "none", "tau": 0.5, "score_function": "mean", "contract": contract}},
            }
            args.selection = self.root / "selection.json"
            write_json(args.selection, selection)
            run_inference(args, engine)
            run_inference(args, engine)
            args.output = self.root / "another_test.json"
            with self.assertRaises(ValueError):
                run_inference(args, engine)

    def test_end_to_end_cache_records_no_detection_without_running_ocr(self):
        class Image:
            shape = (100, 200, 3)

            def __getitem__(self, item):
                return self

        calls = []
        engine = types.SimpleNamespace(identity={"name": "synthetic"}, weight_digests={"test": "a" * 64})
        engine.recognize = lambda image: calls.append(image) or {"text": "皖A12345", "char_confidences": [0.9] * 7}
        cv2 = types.ModuleType("cv2")
        cv2.imread = lambda path: Image()
        preprocessing = types.ModuleType("data.preprocessing")
        preprocessing.preprocess_plate = lambda image, mode: image
        args = self.inference_args()
        args.protocol = "end_to_end"
        args.predictions = self.root / "predictions.json"
        prediction = {
            "schema_version": 1,
            "metadata": {"box_format": "xyxy_pixels", "model_id": "test", "weight_sha256": "b" * 64},
            "records": [{"image_path": "images/val/a_1.0.jpg", "boxes": [], "scores": []}],
        }
        write_json(args.predictions, prediction)

        with patch.dict(sys.modules, {"cv2": cv2, "data.preprocessing": preprocessing}):
            cache = run_inference(args, engine)
            self.assertEqual(cache["records"][0]["status"], "no_detection")
            self.assertEqual(calls, [])
            prediction["records"][0].update(boxes=[[-3, 10, 205, 80]], scores=[0.8])
            write_json(args.predictions, prediction)
            args.output = self.root / "with_box.json"
            cache = run_inference(args, engine)
            self.assertEqual(cache["records"][0]["detector_box"], [0, 10, 200, 80])
            self.assertEqual(cache["records"][0]["status"], "read")
            self.assertEqual(len(calls), 1)

    def test_metadata_join_rejects_ground_truth_changes(self):
        sample = load_samples(self.root, "val")[0]
        changed = copy.deepcopy(sample)
        changed["plate_text"] = "京A99999"
        changed["split"] = "val"
        self.table("ocr_val.csv", [changed])
        with self.assertRaises(ValueError):
            load_samples(self.root, "val")


if __name__ == "__main__":
    unittest.main()
