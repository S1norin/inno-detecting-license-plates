import json
import tempfile
import unittest
from pathlib import Path

from models.yolo11_finetune import compare


def metrics_document(model_id="a.pt", split="val", map50=0.9):
    def group(count, offset=0.0):
        return {"image_count": count, "ground_truth_count": count, "prediction_count": count,
                "tp": count, "fp": 0, "fn": 0, "map50": map50 + offset,
                "precision": 0.95, "recall": 0.9, "f1": 0.92}
    return {"model_id": model_id, "weight_sha256": "0" * 64, "split": split,
            "overall": group(700),
            "by_size": {"small": group(240), "medium": group(230), "large": group(230)},
            "by_tilt": {"0-9": group(300), "10-19": group(200), "20+": group(200)}}


class CompareTests(unittest.TestCase):
    def test_scopes_fields_and_deltas(self):
        pretrained = metrics_document("pre.pt", map50=0.9)
        finetuned = metrics_document("fine.pt", map50=0.95)
        comparison = compare.build_comparison(pretrained, finetuned)
        self.assertEqual(list(comparison), ["overall", "size/small", "size/medium", "size/large",
                                            "tilt/0-9", "tilt/10-19", "tilt/20+"])
        row = comparison["overall"]
        self.assertEqual(row["image_count"], 700)
        self.assertEqual(row["pretrained"]["map50"], 0.9)
        self.assertAlmostEqual(row["finetuned"]["map50"], 0.95)
        self.assertAlmostEqual(row["delta"]["map50"], 0.05)
        self.assertEqual(set(row["pretrained"]), {"map50", "precision", "recall", "f1"})

    def test_split_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "Split mismatch"):
            compare.build_comparison(metrics_document(split="val"), metrics_document(split="test"))

    def test_image_count_mismatch_refused(self):
        finetuned = metrics_document()
        finetuned["by_size"]["small"]["image_count"] = 100
        with self.assertRaisesRegex(ValueError, "Image count differs"):
            compare.build_comparison(metrics_document(), finetuned)

    def test_load_metrics_requires_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.json"
            path.write_text(json.dumps({"overall": {}}))
            with self.assertRaisesRegex(ValueError, "missing"):
                compare.load_metrics(path)

    def test_markdown_table_covers_every_scope_and_metric(self):
        comparison = compare.build_comparison(metrics_document(), metrics_document())
        table = compare.markdown_table(comparison)
        self.assertEqual(table.count("\n"), 1 + 7 * 4)
        for field in compare.METRIC_FIELDS:
            self.assertIn(f"| {field} ", table)


if __name__ == "__main__":
    unittest.main()
