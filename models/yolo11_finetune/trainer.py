"""Small adapters for mAP@0.5 selection in Ultralytics 8.3.162."""

import math
import os
from copy import copy
from pathlib import Path

import yaml
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils.torch_utils import EarlyStopping

MAP50 = "metrics/mAP50(B)"


class Map50Validator(DetectionValidator):
    def get_stats(self):
        stats = super().get_stats()
        score = float(stats[MAP50])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Validation mAP@0.5 must be finite and in [0, 1]")
        stats["fitness"] = score
        return stats


class Map50Stopping(EarlyStopping):
    def __call__(self, epoch, fitness):
        # The pinned stopper treats best_fitness == 0 as an uninitialized run.
        return super().__call__(epoch, -1.0 if fitness == 0 else fitness)


class Map50Trainer(DetectionTrainer):
    def get_dataset(self):
        # This YAML is generated from validated train/val lists by train.py.
        # Avoid the generic loader's font download and test/download fields.
        data = yaml.safe_load(Path(self.args.data).read_text(encoding="utf-8"))
        if set(data) != {"path", "train", "val", "nc", "names"}:
            raise ValueError("Use the generated train/val-only training dataset")
        return {**data, "path": Path(data["path"]), "yaml_file": self.args.data, "channels": 3}

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return Map50Validator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self.stopper = Map50Stopping(patience=self.args.patience)

    def save_model(self):
        # The library overwrites best.pt on equal fitness; keep the first peak.
        tied = self.best.exists() and getattr(self, "_saved_map50", None) == self.fitness == self.best_fitness
        previous = self.best.read_bytes() if tied else None
        try:
            super().save_model()
        finally:
            if previous is not None:
                temporary = self.best.with_suffix(".tie.tmp")
                temporary.write_bytes(previous)
                os.replace(temporary, self.best)
        if self.fitness == self.best_fitness:
            self._saved_map50 = self.fitness
