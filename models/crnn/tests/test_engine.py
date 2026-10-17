import argparse
import io
import json
import math
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from baseline.ocr.io import digest, read_json, validate_raw
from baseline.ocr.run import main, run_inference
from models.crnn.engine import CRNNEngine
from models.crnn.model import CRNN, ModelConfig, save_checkpoint
from models.crnn.tests import test_data


class EngineTests(unittest.TestCase):
    def setUp(self):
        fixture = test_data.TrainingDataTests(methodName="test_originals_ctc_targets_and_unseen_validation_character")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        charset = read_json(fixture.charset)
        model = CRNN(ModelConfig(num_classes=len(charset["characters"]) + 1))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier.bias[1] = 10
        self.checkpoint = self.root / "checkpoint.pt"
        save_checkpoint(self.checkpoint, model, charset, {"epoch": 2})
        self.sha256 = digest(self.checkpoint)

    def test_digest_mismatch_refused_before_model_load(self):
        with patch("models.crnn.engine.load_checkpoint") as load:
            with self.assertRaisesRegex(ValueError, "expected.*actual"):
                CRNNEngine(self.checkpoint, "0" * 64)
            load.assert_not_called()
        with self.assertRaises(ValueError):
            CRNNEngine(self.checkpoint, "")

    def test_character_confidences_and_json_stable_identity(self):
        engine = CRNNEngine(self.checkpoint, self.sha256)
        for image in (np.zeros((24, 100, 3), dtype=np.uint8), np.zeros((48, 200), dtype=np.uint8)):
            result = engine.recognize(image)
            self.assertTrue(result["text"])
            self.assertEqual(len(result["text"]), len(result["char_confidences"]))
            self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in result["char_confidences"]))
        for field in ("name", "decoder", "character_confidence", "input_geometry", "architecture", "selected_epoch",
                      "adapter_sha256", "model_sha256", "ctc_sha256", "torch_version"):
            self.assertIn(field, engine.identity)
        self.assertEqual(engine.identity, json.loads(json.dumps(engine.identity)))
        self.assertEqual(engine.weight_digests, {"crnn_checkpoint.pt": self.sha256})

    def test_shared_runner_cache_validates_and_resumes(self):
        engine = CRNNEngine(self.checkpoint, self.sha256)
        args = argparse.Namespace(dataset=self.root, split="val", scale=1.0, protocol="ocr_only", mode="none",
                                  predictions=None, selection=None, output=self.root / "raw.json")
        raw = run_inference(args, engine)
        validate_raw(raw)
        with patch.object(engine, "recognize", side_effect=AssertionError("Completed crop recognized again")):
            run_inference(args, engine)
        self.assertEqual(read_json(args.output)["metadata"]["engine"]["name"], "crnn")


class RunnerCLITests(unittest.TestCase):
    def arguments(self):
        return ["--protocol", "ocr_only", "--split", "val", "--mode", "none", "--output", "raw.json"]

    def test_easyocr_is_unchanged_default(self):
        with patch("baseline.ocr.run.EasyOCREngine") as easyocr, patch("baseline.ocr.run.run_inference") as run:
            main(self.arguments())
        easyocr.assert_called_once_with(Path("baseline/weights/easyocr"), Path("baseline/common/models.json"))
        self.assertIs(run.call_args.args[1], easyocr.return_value)

    def test_crnn_requires_checkpoint_and_expected_digest(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main([*self.arguments(), "--engine", "crnn"])
        self.assertEqual(error.exception.code, 2)

    def test_crnn_selection_constructs_requested_engine(self):
        with patch("models.crnn.engine.CRNNEngine") as crnn, patch("baseline.ocr.run.run_inference") as run:
            main([*self.arguments(), "--engine", "crnn", "--crnn-checkpoint", "checkpoint.pt",
                  "--crnn-expected-sha256", "a" * 64])
        crnn.assert_called_once_with(Path("checkpoint.pt"), "a" * 64)
        self.assertIs(run.call_args.args[1], crnn.return_value)


if __name__ == "__main__":
    unittest.main()
