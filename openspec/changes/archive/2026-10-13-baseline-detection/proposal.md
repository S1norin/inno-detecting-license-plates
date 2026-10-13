# Proposal

## Why

The project plan (docs/CV_Plan.pdf) requires a detection baseline before any
fine-tuning or end-to-end evaluation: the pretrained
`morsetechlab/yolov11-license-plate-detection` (YOLOv11s) detector must be
run on our CCPD subset and its performance quantified per plate size bucket
and tilt bin (E1 baseline, E4/E5). `baseline/detection/` is empty and the repo
has no mechanism to obtain pinned model dependencies and weights
reproducibly.

## What Changes

- New shared baseline environment module (`baseline/common/`):
  - self-installs pinned dependencies into a dedicated venv (separate from
    the data-pipeline `.venv`), creating it if missing
  - downloads model weights from pinned URLs and verifies sha256; a hash
    mismatch is a hard failure, never a silent use
- New detection baseline (`baseline/detection/`):
  - loads the pinned YOLOv11s plate detector (weights cached locally,
    hash-verified)
  - evaluates on the val and test splits of `data/dataset/`
    (primary: `dataset_original.yaml`, original scale only)
  - reports mAP@0.5 and F1@IoU=0.5 overall, per size bucket
    (small/medium/large) and per tilt bin
  - writes machine-readable metrics (JSON) plus a short results note
- Fine-tuning preparation only: a YOLO training config for the train split is
  in place so a later change can fine-tune (E1); no training happens in this
  change.

## Capabilities

### New Capabilities

- `baseline-environment`: self-managed, pinned baseline environment —
  dependency installation into a dedicated venv and hash-verified model
  weight acquisition shared by all baseline stages.
- `baseline-detection`: detection baseline behavior — pinned YOLOv11s
  plate detector evaluation on val/test with size-bucket and tilt-bin
  breakdowns, producing reproducible metrics.

### Modified Capabilities

(None — no specs exist yet.)

## Impact

- New code: `baseline/common/`, `baseline/detection/`.
- New dependencies, installed only into the dedicated baseline venv:
  PyTorch, ultralytics (versions pinned in the baseline manifests); the main
  project `.venv` and `data/requirements.txt` are untouched.
- `.gitignore`: baseline venv and cached model weights are gitignored.
- Read-only use of `data/dataset/` (yaml configs, images, labels,
  `metadata.csv`). No data pipeline changes.
- Downstream: `baseline-recognition` (end-to-end path) and the later
  fine-tuning change build on this change's environment module and metrics.
