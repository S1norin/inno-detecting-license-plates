"""Timing probe tests that avoid the heavy inference imports."""

import tempfile
import unittest
from pathlib import Path

from baseline.evaluation.inputs import load_all
from baseline.evaluation.tests.fixtures import build
from baseline.evaluation.timing import sample_images, stage_stats


class ProbeSamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        build(cls.root)
        cls.dataset, _detection, _ocr = load_all(cls.root / "dataset", cls.root / "detection", cls.root / "ocr")

    def test_deterministic_for_fixed_seed(self):
        first = sample_images(self.dataset, 8, 42)
        second = sample_images(self.dataset, 8, 42)
        self.assertEqual([s["sample_id"] for s in first], [s["sample_id"] for s in second])

    def test_seed_changes_selection(self):
        a = [s["sample_id"] for s in sample_images(self.dataset, 6, 1)]
        b = [s["sample_id"] for s in sample_images(self.dataset, 6, 2)]
        self.assertNotEqual(a, b)

    def test_clamped_to_available_samples(self):
        samples = sample_images(self.dataset, 100, 42)
        self.assertEqual(len(samples), 12)

    def test_only_original_scale(self):
        for sample in sample_images(self.dataset, 12, 42):
            self.assertIn(sample["split"], ("val", "test"))
            self.assertIn("_s100", sample["crop_path"])


class StageStatsTest(unittest.TestCase):
    def test_median_p95_mean_hand_computed(self):
        times = [10.0, 20.0, 30.0, 40.0, 100.0]
        stats = stage_stats(times)
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["median_ms"], 30.0)
        self.assertEqual(stats["p95_ms"], 100.0)
        self.assertEqual(stats["mean_ms"], 40.0)
        self.assertAlmostEqual(stats["total_s"], 0.2)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            stage_stats([])


if __name__ == "__main__":
    unittest.main()
