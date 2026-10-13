# Design

## Context

`baseline/detection/` is empty. The dataset under `data/dataset/` is complete:
`dataset_original.yaml` (scale 1.0, primary), images, YOLO labels, GT crops,
and `metadata.csv` (size bucket, tilt per variant). The project is a uv
project (Python 3.12) whose main `.venv` is reserved for the data pipeline
(`data/requirements.txt`). See proposal.md for motivation.

## Goals / Non-Goals

**Goals:**
- One-command detection baseline: a script that sets up its environment,
  obtains hash-verified weights, and writes reproducible metrics.
- A shared `baseline/common/` environment/weights module usable unchanged by
  the recognition change and later fine-tuning.
- Per-image predictions as a durable intermediate artifact, so bucket/tilt
  breakdowns and later failure analysis do not require re-running inference.

**Non-Goals:**
- No fine-tuning (a later change consumes the train config).
- No scale-variant evaluation (explicit scale experiments use `dataset.yaml`
  later; primary is original scale only).
- No changes to `data/` or the main venv.

## Decisions

### D1: Dedicated baseline venv (`.venv-baseline`)
The ensure step creates and manages `.venv-baseline` at the repo root with
`uv venv` / `uv pip install --python .venv-baseline`.

- Why: torch + ultralytics are ~1 GB and pull their own numpy/opencv pins
  that would conflict with or bloat the data-pipeline env (numpy 2.5.3,
  opencv-headless).
- Alternative rejected: installing into the main `.venv` — couples the data
  pipeline to model-stack versions.
- The venv is gitignored; manifests live in the repo, so it is fully
  reconstructible.

### D2: Per-stage pinned requirements files
Each stage owns a manifest: `baseline/requirements-detection.txt`
(exact `==` pins). `baseline/common/env.py` parses it, compares against the
installed versions in `.venv-baseline`, and installs only what is missing or
mismatched.

- Why per-stage: the OCR stage will add easyocr; detection should not install
  it.
- Why exact pins without `--require-hashes`: wheel-level hashes for the whole
  dependency tree are brittle to maintain; exact version pins already give a
  reproducible resolution. sha256 verification is applied where it matters
  most — the model weights (D3).

### D3: Weight manifest with sha256 (`baseline/common/models.json`)
Entries: `{name, url, filename, sha256}`. `weights.py` downloads to
`baseline/weights/<filename>` (gitignored), verifies sha256 with hashlib, and
fails loudly on mismatch. Ultralytics loads the local file path directly.

- Why: this is what actually pins the baseline model artifact; a silently
  swapped weight file would invalidate every comparison.
- The YOLOv11s plate detector comes from
  `huggingface.co/morsetechlab/yolov11-license-plate-detection`; the exact
  file name and its sha256 are recorded in `models.json` during the first
  verified download (task).
- Alternative rejected: letting ultralytics auto-download — bypasses
  verification and pins nothing.

### D4: Per-image predictions, then custom metrics
Run inference over all images of a split and store per-image predictions
(image id, boxes, scores). A metrics module joins predictions with
`metadata.csv` (size_bucket, tilt) and computes:

- matching: single-class IoU matching at IoU 0.5
- overall: mAP@0.5, precision, recall, F1@0.5 (detection confidence
  threshold fixed at the model default 0.25, recorded in output)
- breakdowns: the same metrics per size bucket (dataset's train-derived
  small/medium/large) and per tilt bin (fixed 10-degree bins on
  `tilt_magnitude`: 0-9, 10-19, 20+)

- Why not rely on ultralytics' built-in `val()` alone: it gives overall mAP
  but no per-bucket/per-tilt breakdowns (E4/E5 need them), and storing
  predictions enables failure-case dumps later.
- Alternative rejected: calling `val()` per bucket with filtered yaml files —
  duplicates work and scatters evaluation logic.

### D5: Run-stamped output directory
Each run writes to `baseline/detection/results/<run_id>/` (run_id =
UTC timestamp): `metrics_val.json`, `metrics_test.json`,
`predictions_val.json`, `predictions_test.json`. JSON records carry model id,
weight sha256, dataset config, split, and timestamp per the traceability
requirement.

### D6: Deterministic evaluation
Evaluation forces CPU inference (`device='cpu'`) so re-runs on the same
machine are bit-identical. Runtime cost is acceptable (a few thousand images,
one model, once per split).

### D7: Fine-tuning readiness
`baseline/detection/train_config.yaml` declares the model (cached weight
path), dataset (`dataset_original.yaml`), and default hyperparameters
(epochs, imgsz 640, batch, lr). No training code ships in this change; the
later change consumes this config.

## Module layout

```
baseline/
├── common/
│   ├── __init__.py
│   ├── env.py              # venv ensure + pinned install
│   ├── weights.py          # download + sha256 verify
│   └── models.json         # weight manifest (urls + sha256)
├── requirements-detection.txt
├── detection/
│   ├── __init__.py
│   ├── evaluate.py         # entry point: --split {val,test}
│   ├── metrics.py          # predictions + metadata -> metrics
│   ├── train_config.yaml   # E1 prep (consumed by a later change)
│   ├── results/            # gitignored, run-stamped
│   └── README.md
└── weights/                # gitignored, hash-verified caches
```

## Risks / Trade-offs

- [Large first-run download (torch wheel + weights)] → documented in
  README; uv caches wheels, so only the first setup is slow.
- [Exact HF weight file name unknown until first download] → first task
  verifies the file, records URL + sha256 in `models.json`; nothing else
  depends on the name.
- [MPS/GPU non-determinism could break the deterministic re-run scenario] →
  D6 forces CPU for evaluation.
- [Per-image prediction files are a few MB per split] → fine locally; they
  are gitignored and regenerable.
- [Single-class IoU matching reimplemented in `metrics.py`] → kept minimal
  (greedy match at fixed IoU); validated against ultralytics' built-in mAP
  on the same split as a sanity check (task).

## Migration Plan

New code only; no existing behavior changes. Rollback = delete
`baseline/common/`, `baseline/detection/`, `baseline/weights/`,
`.venv-baseline`, and the two `.gitignore` lines.

## Open Questions

- Exact torch/ultralytics versions to pin: pick the newest mutually
  compatible releases at implementation time and freeze them in the manifest.
- Tilt bin edges (0-9/10-19/20+) are a reasonable default; if the metadata
  distribution suggests finer bins, adjust in `metrics.py` — does not change
  specs or tasks.
