# YOLO11s plate detection baseline

Run all commands from the repository root. Evaluation uses the pretrained
`morsetechlab/yolov11-license-plate-detection` small detector, without training.
The immutable revision, filename, and SHA-256 are in `baseline/common/models.json`.

## Prepare the data and environment

Use the repository's documented data preparation steps to produce the ignored
`data/dataset/images/` files, or use the prepared dataset supplied by the team.
For the primary experiment, there must be 3,500 train, 700 validation, and 800
test original images. Refresh absolute paths after cloning or moving data.
Do not change the splits, scales, or train-derived size thresholds.

Install uv explicitly if it is unavailable, then create the separate baseline
environment. The setup command supports Windows and Unix and finds Python 3.12:

```text
python -m pip install uv
python -m baseline.common.env
python -m baseline.common.env
```

The second setup is a no-op. Existing mismatched or missing pins are installed
with uv; the data environment and its requirements are not modified. The CPU
Torch wheels are requested from the PyTorch CPU index. Ultralytics requires
`opencv-python`, so both OpenCV distributions are pinned to the same version.

Windows PowerShell:

```powershell
$baselinePython = '.\.venv-baseline\Scripts\python.exe'
& $baselinePython data/configure_paths.py
& $baselinePython -m baseline.common.weights fetch
& $baselinePython -m baseline.common.weights verify
& $baselinePython -m unittest discover -s baseline/tests -v
```

Unix:

```sh
./.venv-baseline/bin/python data/configure_paths.py
./.venv-baseline/bin/python -m baseline.common.weights fetch
./.venv-baseline/bin/python -m baseline.common.weights verify
./.venv-baseline/bin/python -m unittest discover -s baseline/tests -v
```

Fetching is explicit. Evaluation only verifies an existing local file, and
never downloads missing or corrupt weights. On a checksum mismatch, the error
names both digests. Remove a corrupt cached file explicitly before refetching.

## Evaluate and check reproducibility

Windows PowerShell:

```powershell
& $baselinePython -m baseline.detection.evaluate --split val --run-id baseline --validate
& $baselinePython -m baseline.detection.evaluate --split test --run-id baseline
& $baselinePython -m baseline.detection.evaluate --split val --run-id repeat
& $baselinePython -m baseline.detection.evaluate --split test --run-id repeat
& $baselinePython -m baseline.detection.metrics baseline/detection/results/baseline/predictions_val.json --output baseline/detection/results/baseline/recomputed_val.json
& $baselinePython -m baseline.detection.train_ready
```

Unix:

```sh
./.venv-baseline/bin/python -m baseline.detection.evaluate --split val --run-id baseline --validate
./.venv-baseline/bin/python -m baseline.detection.evaluate --split test --run-id baseline
./.venv-baseline/bin/python -m baseline.detection.evaluate --split val --run-id repeat
./.venv-baseline/bin/python -m baseline.detection.evaluate --split test --run-id repeat
./.venv-baseline/bin/python -m baseline.detection.metrics baseline/detection/results/baseline/predictions_val.json --output baseline/detection/results/baseline/recomputed_val.json
./.venv-baseline/bin/python -m baseline.detection.train_ready
```

Use a new run ID when rerunning: existing split results are never overwritten.
Omit `--run-id` to generate a unique timestamped ID. Use `--data /path/to/dataset_original.yaml`
for a prepared dataset elsewhere; its YAML root and lists must point to that
location. Validation rejects stale paths, missing images or labels, duplicate
images, ambiguous metadata, and non-original-scale records.

Outputs are `baseline/detection/results/<run_id>/predictions_<split>.json` and
`metrics_<split>.json`. Predictions contain one record per image, including
empty detections, with original pixel `xyxy` boxes, scores, class IDs, ground
truth, and the joined difficulty metadata. Ground truth is read from YOLO labels
and checked against `metadata.csv`.

Metrics include overall and size/tilt groups, sample counts, TP/FP/FN, mAP@0.5,
precision, recall, and F1. They record the model digest, dataset/config/split,
UTC timestamp, input fingerprints, runtime versions, and inference settings.
The metrics command recomputes them without model inference.

## Metric and inference definitions

- Matching is single-class, one-to-one at IoU >= 0.5, following the pinned
  Ultralytics validator assignment. Confidence-ranked AP uses its 101-point
  interpolated precision envelope; with one class AP equals mAP.
- Precision, recall, and F1 use confidence >= 0.25. IoU 0.5 and confidence 0.25
  are separate thresholds. Low-score predictions remain available for AP.
- Size groups come from metadata and the train-derived thresholds in
  `run_config.json`: small <= 83 px plate height, medium <= 112 px, large > 112 px.
- Tilt uses `max(abs(tilt_horizontal), abs(tilt_vertical))`, with bins `[0,10)`,
  `[10,20)`, and `[20,infinity)`. Groups are assigned by image ground truth;
  all false positives on an image belong to that image's group.
- Empty groups have zero counts and null metrics. Images with no predictions
  remain in denominators and contribute missed plates.
- Inference uses CPU, one Torch thread, FP32, seed 42, deterministic operations,
  batch 1, image size 640, confidence floor 0.001, NMS IoU 0.7, and max 300 boxes.
  Rectangular padding and augmentation are disabled in both inference and `val()`.

`--validate` performs an independent Ultralytics `val()` pass and fails when
overall mAP@0.5 differs by more than 0.01. Check settings on validation, then
freeze them before test evaluation. A second test run checks reproducibility
only. Compare prediction `images` and the metric `overall`, `by_size`, and
`by_tilt` fields; timestamps and run IDs intentionally differ. Reproducibility
is expected on the same pinned environment and machine, not across CPU hardware.

## Fine-tuning readiness

`train_config.yaml` supplies cached weights, the original dataset, and explicit
starting hyperparameters. `train_ready` verifies the cached digest, YAML, and
train paths without starting training. These defaults are preparation for a
later experiment, not tuned parameters.

Lint from the root with `uv run ruff check baseline/` (or
`python -m uv run ruff check baseline/` when uv is not on PATH).
