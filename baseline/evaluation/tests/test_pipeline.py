"""End-to-end pipeline test on the fixture, offline (--no-probe)."""

import sys
import tempfile
import unittest
from pathlib import Path

from baseline.evaluation import run
from baseline.evaluation.tests.fixtures import build


class PipelineTest(unittest.TestCase):
    def _run_once(self, base, out):
        args = run.build_parser().parse_args([
            "--dataset", str(base / "dataset"),
            "--detection-results", str(base / "detection"),
            "--ocr-results", str(base / "ocr"),
            "--out", str(out),
            "--no-probe",
            "--timestamp", "2026-07-08T12:00:00+00:00",
        ])
        return run.run(args)

    def test_offline_run_writes_all_artifacts_and_stays_stdlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build(base)
            out = self._run_once(base, base / "results")
            self.assertTrue((out / "table.json").is_file())
            self.assertTrue((out / "failures.json").is_file())
            self.assertTrue((out / "REPORT.md").is_file())
            self.assertTrue((out / "figures" / "baseline_comparison.png").is_file())
            self.assertFalse((out / "timing.json").exists())

            report = (out / "REPORT.md").read_text(encoding="utf-8")
            for heading in ("E1 — Detection", "E2 — OCR-only", "E2b — End-to-end",
                            "E3 — Preprocessing ablation", "E6 —", "Failure cases", "Notes"):
                self.assertIn(heading, report)
            self.assertIn("miss, low_iou, low_confidence", report)
            self.assertIn("no_detection", report)

            for module in ("torch", "ultralytics", "easyocr", "cv2"):
                self.assertNotIn(module, sys.modules, f"--no-probe run imported {module}")

    def test_rerun_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build(base)
            out = self._run_once(base, base / "results")
            first = {path.name: path.read_bytes() for path in out.rglob("*") if path.is_file()}

            out2 = self._run_once(base, base / "results_second")
            second = {path.name: path.read_bytes() for path in out2.rglob("*") if path.is_file()}

            self.assertEqual(set(first), set(second))
            for name in first:
                self.assertEqual(first[name], second[name], f"{name} differs between reruns")


if __name__ == "__main__":
    unittest.main()
