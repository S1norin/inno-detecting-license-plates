"""Original-scale CRNN data assembly."""

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from baseline.ocr.charset import build_charset
from baseline.ocr.export_train import export_training
from baseline.ocr.io import dataset_path, load_samples, read_json

from .model import transform_image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHARSET = ROOT / "baseline/ocr/charset.json"


class CropDataset(Dataset):
    def __init__(self, root, samples, charset, *, encode_targets=True):
        self.root = Path(root)
        self.samples = samples
        self.charset = charset
        self.targets = []
        for sample in samples:
            if sample["scale"] != 1.0:
                raise ValueError("CRNN training/validation uses original-scale crops only")
            target = [charset["character_to_index"][char] for char in sample["ground_truth"]] if encode_targets else []
            required_steps = len(target) + sum(left == right for left, right in zip(target, target[1:], strict=False))
            if encode_targets and (not target or required_steps > 16):
                raise ValueError(f"CTC target cannot fit 16 steps: {sample['sample_id']}")
            path = dataset_path(self.root, sample["crop_path"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing crop: {path}")
            self.targets.append(torch.tensor(target, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = dataset_path(self.root, sample["crop_path"])
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot decode crop: {path}")
        return transform_image(image), self.targets[index], sample["ground_truth"]


def assemble_data(dataset, charset_path=DEFAULT_CHARSET):
    dataset = Path(dataset)
    charset = read_json(charset_path)
    if charset != build_charset(dataset / "ocr_train.csv"):
        raise ValueError("Charset does not match build_charset(ocr_train.csv)")
    training = export_training(dataset, charset, scale=1.0)["records"]
    validation = load_samples(dataset, "val", 1.0)
    return (CropDataset(dataset, training, charset),
            CropDataset(dataset, validation, charset, encode_targets=False), charset)


def collate_batch(batch):
    images, targets, texts = zip(*batch, strict=True)
    return torch.stack(images), torch.cat(targets), torch.tensor([len(t) for t in targets], dtype=torch.long), texts
