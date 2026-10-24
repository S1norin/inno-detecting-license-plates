# Proposal

## Why

Detection is the project's remaining open experiment (E1): the detector is still
the frozen pretrained YOLO11s baseline (test mAP@0.5 0.9073, recall 0.8938,
weakest on small plates), and the progress report names "fine-tune YOLO11 on the
train split and compare against the frozen detector; re-run end-to-end
recognition with the better detector" as the next stage. All preparation already
exists: the pinned weights are cached and digest-verified, the original-scale
dataset is ready, and `train_ready` validates the inputs. This change performs
the fine-tuning, the comparison, and the end-to-end re-run.

## What Changes

- New `models/yolo11_finetune/` module that fine-tunes the pinned pretrained
  YOLO11s on the 3,500 original-scale train images on a separate GPU machine,
  selects the best checkpoint on validation mAP@0.5 with early stopping, and
  writes a machine-readable training record.
- `baseline/detection/train_ready.py` and `baseline/detection/train_config.yaml`
  move to `models/yolo11_finetune/`; the config's output directory moves to
  `models/yolo11_finetune/results` and gains the early-stopping setting.
- The fine-tuned checkpoint is evaluated on validation and test in this
  repository's existing CPU detection harness with the same inference settings
  as the frozen baseline, and compared against it overall and per size bucket
  and tilt bin.
- The CRNN end-to-end recognition protocol is re-run with the better detector:
  preprocessing mode and confidence threshold selected on validation, test
  reported once.
- `models/yolo11_finetune/REPORT.md` documents the training, the
  pretrained-vs-fine-tuned comparison, and the end-to-end re-run.

## Capabilities

### New Capabilities

- `yolo11-finetune`: plate detector fine-tuning — trains the pinned pretrained
  YOLO11s on the original-scale train split on a GPU machine, evaluates the
  resulting checkpoint against the frozen baseline under the shared detection
  harness, and re-runs end-to-end recognition with the better detector.

### Modified Capabilities

(None. The `baseline-detection` requirements are unchanged — its
fine-tuning-readiness promise is consumed, not modified — and the CRNN
recognition and shared evaluation capabilities are used as-is.)

## Impact

- New: `models/yolo11_finetune/` (`train.py`, `train_config.yaml`,
  `train_ready.py`, `REPORT.md`, `tests/`, gitignored `results/`), plus a
  requirements file pinning the GPU training environment.
- Moved: `baseline/detection/train_ready.py`,
  `baseline/detection/train_config.yaml` (behavior preserved; output directory
  and early stopping updated).
- Modified: `baseline/detection/README.md` (fine-tuning-readiness section now
  points at `models/yolo11_finetune/`); the detection evaluation entry point
  gains the ability to evaluate the fine-tuned checkpoint.
- Reused unchanged: `baseline/detection/evaluate.py` metrics logic, the CRNN
  engine and shared recognition harness for the end-to-end re-run.
- No dataset, split, seed, scale-factor, or size-threshold changes; no new
  packages in the data or baseline evaluation environments.
