import unittest

import numpy as np

from baseline.detection.metrics import box_iou, compute_metrics, match, tilt_bin


def record(boxes=None, scores=None, size="small", tilt=0, image="image.jpg"):
    return {"image_path": image, "ground_truth": [[0, 0, 10, 10]], "boxes": boxes or [],
            "scores": scores or [], "size_bucket": size, "tilt_magnitude": tilt}


class MetricTests(unittest.TestCase):
    def test_perfect_detection_and_empty_groups(self):
        result = compute_metrics([record([[0, 0, 10, 10]], [0.9])])
        self.assertEqual(result["overall"]["f1"], 1)
        self.assertAlmostEqual(result["overall"]["map50"], 0.995)
        self.assertIsNone(result["by_size"]["medium"]["map50"])
        self.assertEqual(result["by_size"]["medium"]["image_count"], 0)

    def test_missed_plate(self):
        result = compute_metrics([record()])["overall"]
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))
        self.assertEqual(result["map50"], 0)
        self.assertEqual(result["f1"], 0)

    def test_duplicate_detection_is_false_positive(self):
        result = compute_metrics([record([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8])])["overall"]
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_nonoverlap_is_false_positive(self):
        result = compute_metrics([record([[20, 20, 30, 30]], [0.9])])["overall"]
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_low_confidence_is_kept_for_ap_but_not_f1(self):
        result = compute_metrics([record([[0, 0, 10, 10]], [0.1])])["overall"]
        self.assertGreater(result["map50"], 0.99)
        self.assertEqual(result["f1"], 0)
        boundary = compute_metrics([record([[0, 0, 10, 10]], [0.25])])["overall"]
        self.assertEqual(boundary["tp"], 1)

    def test_one_to_one_and_iou_boundary(self):
        truth = [[0, 0, 10, 10], [0, 0, 10, 10]]
        self.assertEqual(int(match(truth, [[0, 0, 10, 10]]).sum()), 1)
        iou = box_iou([[0, 0, 10, 10]], [[0, 0, 5, 10]])[0, 0]
        self.assertEqual(iou, 0.5)
        self.assertTrue(match([[0, 0, 10, 10]], [[0, 0, 5, 10]], 0.5)[0])
        self.assertFalse(match([[0, 0, 10, 10]], [[0, 0, 4.9, 10]])[0])

    def test_bins_and_counts(self):
        self.assertEqual([tilt_bin(x) for x in [0, 9, 10, 19, 20, 90]],
                         ["0-9", "0-9", "10-19", "10-19", "20+", "20+"])
        rows = [record(size=size, tilt=tilt, image=str(index))
                for index, (size, tilt) in enumerate(zip(["small", "medium", "large"], [0, 10, 20], strict=True))]
        result = compute_metrics(rows)
        for dimension in ("by_size", "by_tilt"):
            self.assertEqual(sum(group["image_count"] for group in result[dimension].values()), 3)

    def test_ties_are_repeatable_and_duplicates_rejected(self):
        rows = [record([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.9])]
        self.assertEqual(compute_metrics(rows), compute_metrics(rows))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compute_metrics(rows + rows)

    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            compute_metrics([record([[0, 0, 10, 10]], [np.nan])])


if __name__ == "__main__":
    unittest.main()
