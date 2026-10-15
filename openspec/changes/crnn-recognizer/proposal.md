# Proposal

## Why

The recognition baseline (EasyOCR) is the system's clear weak spot: 0.35 character accuracy and 0 exact plates on the test split. The project plan (experiment E2) names a plate-specific CRNN recognizer as the second approach, and the 20 Oct "second approach" milestone requires a second model working. The data side was already prepared by the baseline-recognition change (training charset, CTC decode, validated 10,500-crop manifest); only the model, training, and evaluation remain.

## What Changes

- Add a CRNN recognizer (CNN backbone + BiLSTM + CTC) trained on the 3,500 original-scale (1.0) training crops against the 58-character training charset plus CTC blank.
- Training runs on CPU only, 1 thread, seed 42, deterministic mode, so the checkpoint is bit-reproducible from scratch.
- Epoch is selected on the validation split by character accuracy (ties: earlier epoch); the existing confidence-threshold sweep and test-once policy are unchanged.
- Add a `CRNNEngine` implementing the existing engine contract (`identity`, `weight_digests`, `recognize`) and an `--engine {easyocr,crnn}` selection in the OCR inference runner, so both protocols (OCR-only, end-to-end) and all three preprocessing modes work through the unmodified evaluation harness.
- Report CRNN metrics per protocol, size bucket, and tilt bin alongside the EasyOCR baseline numbers in a compact report and the progress report comparison table.

## Capabilities

### New Capabilities

- `crnn-recognition`: plate-specific CRNN/CTC recognizer: deterministic CPU training on original-scale train crops, validation-based epoch selection, and evaluation under both recognition protocols through the shared engine contract.

### Modified Capabilities

None. The `baseline-recognition` harness (protocols, preprocessing ablation, confidence policy, metrics, traceability) is reused unchanged; only the code that constructs the engine gains a selection flag.

## Impact

- New code: top-level `models/` package with `models/crnn/` (model, training, engine) and unit tests; `baseline/` keeps its meaning as the home of the two baselines and their shared infrastructure.
- Modified code: `baseline/ocr/run.py` (engine selection), `baseline/ocr/README.md`, `AGENTS.md` (model code location convention).
- Data: reads `data/dataset/` (crops, `ocr_*.csv`, `metadata.csv`) and `baseline/ocr/charset.json`; writes nothing to the dataset.
- Artifacts: checkpoint and training/evaluation logs in gitignored `models/crnn/results/`; compact tracked report in `models/crnn/`.
- Environment: no new packages; uses the existing pinned `torch 2.6.0+cpu` in the shared baseline environment.
- Deadline: produces the comparison evidence for the 20 Oct second-approach milestone.
