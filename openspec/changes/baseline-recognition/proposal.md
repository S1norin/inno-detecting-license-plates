# Proposal

## Why

The project plan (docs/CV_Plan.pdf) requires a recognition baseline on two
protocols before CRNN work: OCR-only (ground-truth crops, so detection errors
do not leak into recognition) and end-to-end (crops from the Stage 1
detector), plus a confidence policy that decides "read" vs. "abstain" and is
reported as an accuracy-coverage curve (E2, E2b, E6). `baseline/ocr/` is
empty and no OCR code exists in the repo.

## What Changes

- Reuses the shared environment/weights module from the
  `baseline-detection` change (`baseline/common/`) and adds a pinned
  recognizer (EasyOCR, per the plan's stack) with hash-verified model
  weights to that mechanism.
- New OCR baseline (`baseline/ocr/`):
  - engine wrapper that returns the plate text and per-character
    confidences
  - **Path A (OCR-only)**: reads `crops_gt/` + `ocr_{train,val,test}.csv`
    directly
  - **Path B (end-to-end)**: runs the baseline detector on full images,
    crops from predicted boxes, then OCR
  - preprocessing ablation hook: none / gray+contrast / Canny, reusing
    `data/preprocessing.py`
  - confidence policy: per-plate score (mean per-character probability by
    default; product and sum-of-log-probabilities as alternatives),
    threshold τ tuned on val, read/abstain decision, accuracy-coverage
    curve; test reported once with the chosen τ
  - metrics: character accuracy, plate exact match, and P/R/F1 based on
    edit distance, per split and per size bucket
- CRNN preparation only: a character-set module and a training-data export
  from `ocr_train.csv`, so a later change can train a CRNN on this dataset;
  no CRNN training happens here.

## Capabilities

### New Capabilities

- `baseline-recognition`: recognition baseline behavior — OCR of plate
  crops under both the OCR-only and end-to-end protocols, with a
  confidence-based read/abstain policy and reproducible metrics.

### Modified Capabilities

(None. The `baseline-environment` capability is consumed, not changed; its
weight manifest gains the recognizer entry as part of this change's tasks.)

## Impact

- New code: `baseline/ocr/`.
- New dependency in the baseline venv: EasyOCR (torch is already pinned by
  the detection change); `baseline/common/models.json` gains the recognizer
  weight entries.
- Depends on the `baseline-detection` change: the shared env/weights module
  and the detector used by the end-to-end path.
- Read-only use of `data/dataset/` (crops, ocr CSVs, images,
  `preprocessing.py`). No data pipeline changes.
- Downstream: CRNN training change consumes the charset module and data
  export; end-to-end evaluation and failure analysis consume the
  per-protocol metrics.
