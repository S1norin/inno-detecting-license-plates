# Design

## Context

See proposal.md for motivation. Current state that shapes the approach:

- The frozen baseline is a pinned pretrained YOLO11s checkpoint
  (`baseline/weights/license-plate-finetune-v1s.pt`, SHA-256 verified via
  `baseline/common/models.json`), evaluated by
  `baseline/detection/evaluate.py` with fixed CPU inference settings
  (1 thread, FP32, seed 42, 640px, conf floor 0.001, NMS 0.7). Its numbers
  (val mAP@0.5 0.9245, test 0.9073) are the comparison reference.
- `baseline/detection/train_config.yaml` and `train_ready.py` exist as
  fine-tuning preparation; no code in the baseline starts training.
- The CRNN end-to-end protocol consumes detector predictions as a JSON file
  (`baseline/ocr/run.py --protocol end_to_end --predictions ...
  --engine crnn`), so an end-to-end re-run is CLI orchestration, not new code.
- Training must happen on a separate GPU machine; this repository has no GPU.
  The dataset is 1.8GB; the checkpoint is ~19MB and device-agnostic.
- Ultralytics 8.3.162 and Torch 2.6.0 are already pinned in
  `.venv-baseline` (CPU wheels).

## Goals / Non-Goals

**Goals:**

- A trained, val-selected checkpoint with a complete machine-readable
  training record.
- Apples-to-apples val/test numbers for pretrained vs fine-tuned under the
  existing harness, including size/tilt breakdowns.
- One end-to-end CRNN re-run with the better detector, reported against the
  previous end-to-end numbers.
- A `REPORT.md` a reader can use without running anything.

**Non-Goals:**

- No scale-variant (0.75/0.5) training — reserved for a later scale
  experiment.
- No changes to metric definitions, the shared recognition harness, the CRNN
  model, or the dataset (splits, seed, scales, thresholds untouched).
- No hyperparameter search. One training configuration, chosen a priori.
- No cross-machine reproducibility guarantee.
- No architecture changes, augmentation, or distillation.

## Decisions

### D1: Train on the GPU machine, evaluate in this repo's CPU harness

The fine-tuned checkpoint (plus the training record) is the only artifact that
crosses the machine boundary. All evaluation — detection val/test and the
end-to-end re-run — runs here in `.venv-baseline`.

Rationale: the frozen baseline's numbers were produced by this harness with
fixed CPU inference settings; evaluating the new checkpoint in the same
harness makes the comparison exact and reuses the validated `--validate`
cross-check. YOLO checkpoints are device-agnostic, so GPU-trained weights
load cleanly on CPU.

Alternative considered: evaluate on the GPU machine too. Rejected — it means
pinning a second evaluation environment and comparing numbers across
inference setups.

### D2: Code lives in `models/yolo11_finetune/`

Following the `models/crnn/` pattern. `train_ready.py` and
`train_config.yaml` move from `baseline/detection/` (behavior preserved;
config output directory becomes `models/yolo11_finetune/results`, and the
config gains the patience setting). `baseline/detection/README.md` gets a
pointer in its fine-tuning section. The `baseline-detection` main spec is
behavioral and has no file paths, so nothing there changes. `models/` code
importing `baseline/` helpers is already established (`models/crnn/engine.py`).

### D3: Wrap Ultralytics `train()`; no custom training loop

`train.py` is a thin CLI wrapper around
`ultralytics.YOLO(verified_weights).train(data=..., ...)`, reading
`train_config.yaml`. The pinned version's default fitness is
`0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95`, not the mAP@0.5 required by this change.
A small `DetectionValidator` adapter sets fitness to validation mAP@0.5;
`DetectionTrainer` continues to run the library's training loop, checkpoint
serialization and early stopping. The adapter preserves the first checkpoint
on tied maxima (the library otherwise overwrites it), and a thin stopper adapter
allows a zero-mAP plateau to exhaust patience instead of being treated as an
uninitialized run indefinitely. These cases are covered by CPU unit tests.

The wrapper generates verified train/val-only lists and a YAML without a test
entry or download script. The trainer consumes this local description directly,
avoiding the generic dataset loader's automatic font download. The full resolved
library arguments and actual optimizer learning rates are saved in the training
record, which is updated atomically after every epoch and on failure.

Alternative considered: a custom loop (CRNN-style). Rejected — YOLO's loss,
anchors, and NMS live inside Ultralytics; a custom loop would diverge from
the pinned validator and buy nothing observable.

### D4: Training configuration

Image size 640, max 100 epochs, `patience` 20, batch 16, seed 42,
Ultralytics default learning rate and warmup (recorded in the training
record), no data augmentation beyond Ultralytics defaults. The
pretrained model is already plate-specific and strong (0.91 test mAP), so
fine-tuning is expected to converge well inside 100 epochs; patience caps the
run. GPU makes the full budget cheap, so no reduced-epoch variant is
justified.

Training uses FP32 (`amp=False`) to avoid the library's AMP-check download of
an unrelated model. One explicit CUDA device and `workers=0` keep execution on
a single machine; existing run directories and resume are refused. These
execution choices are included in the resolved configuration. A two-epoch
smoke run can override only the epoch budget and run name; it must use a
separate output directory from the full experiment.

### D5: Evaluating the fine-tuned checkpoint

`baseline/detection/evaluate.py` gains a `--model` path option (default: the
pinned pretrained model, current behavior unchanged). When `--model` is
given, the digest-verification of the pinned manifest is replaced by
recording the SHA-256 of the supplied file in the metrics JSON — the
existing "model digest" field keeps every number traceable to exact weights.
Both detectors are thus evaluated by one code path, one settings block.

### D6: GPU training environment

A new `models/yolo11_finetune/requirements-gpu.txt` pins the same versions as
the baseline (ultralytics 8.3.162, Torch 2.6.0, torchvision, numpy, opencv,
opencv-python-headless) with CUDA wheels for Torch. It is a separate
environment from `.venv-baseline`; the data environment and the CPU baseline
environment are not modified. The GPU machine also runs
`data/configure_paths.py` after unzipping the dataset, exactly as on a fresh
clone.

### D7: End-to-end re-run is orchestration only

1. Evaluate the winning detector (higher val mAP@0.5) on val and test with
   `baseline/detection/evaluate.py` → `predictions_{val,test}.json`.
2. Feed them to `baseline/ocr/run.py --protocol end_to_end --engine crnn`
   with the existing CRNN checkpoint, selecting mode and tau on val, test
   once — the same frozen-policy flow the CRNN evaluation used.
3. Report against the previous end-to-end numbers (frozen detector) in
   `models/yolo11_finetune/REPORT.md`.

No new Python is required for this step.

## Risks / Trade-offs

- [Modest expected gain] The pretrained model is already plate-finetuned on a
  similar distribution → the comparison is honest either way; if the
  pretrained wins, the end-to-end re-run uses it and there is no regression,
  only a documented negative result.
- [GPU non-determinism] Ultralytics training on CUDA is not bit-reproducible
  → fixed seed, same-machine claim only, metrics reported at the project's
  standard precision (see spec: Reproducibility scope).
- [Remote training is a manual step] The full run happens on a machine this
  repo does not control → mitigate with a 2-epoch smoke run first, the
  `train_ready` pre-flight, and the complete training record as the
  machine-readable receipt.
- [`--model` weakens the pinned-weights default] → default behavior is
  unchanged; every non-default evaluation records the exact weight digest it
  used in its metrics JSON.
- [Long run interrupted on the remote machine] → Ultralytics checkpoints are
  per-epoch; a rerun with the same config is the recovery path, and the
  training record states how many epochs completed.

## Migration Plan

No deployment. Work happens on a `feat/yolo11-finetune` branch:

1. Move `train_ready.py` / `train_config.yaml` to `models/yolo11_finetune/`,
   add `train.py`, tests, requirements, update the baseline README; run the
   baseline test suite and ruff.
2. On the GPU machine: set up the pinned environment, run `train_ready`,
   smoke run (2 epochs), full run.
3. Return `best.pt` + training record to `models/yolo11_finetune/results/`
   (gitignored; digests in REPORT.md).
4. Evaluate, re-run end-to-end, write `REPORT.md` here.

Rollback is deleting the branch; nothing tracked outside the branch changes
behavior.

## Open Questions

- Exact final hyperparameters if the 2-epoch smoke run reveals instability
  (e.g. patience value) — answerable on the GPU machine without touching the
  spec or task breakdown.
- Whether to commit a small figure (e.g. per-size mAP comparison) in
  REPORT.md — decided when writing the report.
