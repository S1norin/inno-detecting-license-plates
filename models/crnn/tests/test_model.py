import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baseline.ocr.io import digest
from models.crnn.model import CRNN, ModelConfig, load_checkpoint, save_checkpoint, transform_image


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_default_shape_and_parameter_budget(self):
        model = CRNN().eval()
        with torch.inference_mode():
            result = model(torch.randn(2, 1, 48, 256))
        self.assertEqual(tuple(result.shape), (2, 16, 59))
        self.assertTrue(1_000_000 < sum(p.numel() for p in model.parameters()) < 1_200_000)

    def test_aspect_preserving_transform_and_padding(self):
        result = transform_image(np.full((24, 72, 3), 255, dtype=np.uint8))
        self.assertEqual(tuple(result.shape), (1, 48, 256))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.all(result[:, :, :144] == 1))
        self.assertTrue(torch.all(result[:, :, 144:] == -1))
        wide = transform_image(np.full((10, 1000), 255, dtype=np.uint8))
        self.assertTrue(torch.all(wide[:, :3] == 1))
        self.assertTrue(torch.all(wide[:, 3:] == -1))
        with self.assertRaises(ValueError):
            transform_image(np.empty((0, 5), dtype=np.uint8))

    def test_checkpoint_roundtrip_and_stable_bytes(self):
        charset = {"characters": ["A", "皖"], "ctc_blank_index": 0, "character_to_index": {"A": 1, "皖": 2}}
        model = CRNN(ModelConfig(num_classes=3)).eval()
        image = torch.randn(1, 1, 48, 256)
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "first.pt", Path(folder) / "second.pt"
            save_checkpoint(first, model, charset, {"epoch": 2})
            save_checkpoint(second, model, charset, {"epoch": 2})
            self.assertEqual(digest(first), digest(second))
            loaded, payload = load_checkpoint(first)
            self.assertEqual(payload["epoch"], 2)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(image), loaded(image)))


if __name__ == "__main__":
    unittest.main()
