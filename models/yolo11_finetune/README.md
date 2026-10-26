# YOLO11s fine-tuning

This module implements setup and training code for issues #25 and #26.
GPU training, detector comparison and end-to-end evaluation are subsequent
tasks in `openspec/changes/yolo11-finetune/tasks.md`. No fine-tuned accuracy is
claimed before those experiments run.

Run commands from the repository root. Keep the CPU baseline environment for
baseline evaluation. Training requires one NVIDIA GPU and a separate Python
3.12 environment with CUDA 12.4 PyTorch wheels, pinned in `requirements-gpu.txt`.
The wheel pair follows the [PyTorch 2.6.0 installation matrix](https://pytorch.org/get-started/previous-versions/#v260).
No packages or weights are installed by the training entry point.

## Prepare the GPU machine

Transfer the repository, the prepared dataset and the pinned pretrained weight
file to the GPU machine using the team's normal transfer procedure. Extract
the dataset under `data/dataset/`; cache the pretrained checkpoint under
`baseline/weights/`. Refresh paths on that machine. Do not move images between
splits or include scaled variants in training.

Example environment setup on Linux (installing dependencies is an explicit step):

```sh
uv venv --python 3.12 models/yolo11_finetune/.venv-gpu
uv pip install --python models/yolo11_finetune/.venv-gpu/bin/python -r models/yolo11_finetune/requirements-gpu.txt
models/yolo11_finetune/.venv-gpu/bin/python data/configure_paths.py
models/yolo11_finetune/.venv-gpu/bin/python -m models.yolo11_finetune.train_ready
```

Readiness also works without a GPU using
`./.venv-baseline/bin/python -m models.yolo11_finetune.train_ready`.
It verifies the cached starting-weight digest, config and 3,500 training inputs
without training. The training wrapper additionally verifies all 700 original
validation inputs, counts and train/val disjointness. It reuses the baseline
dataset checks, including original-scale metadata joins and image/label hashes.
The generated training YAML includes only verified train/val lists, never test.

## Smoke run and full run

On the GPU machine, after readiness succeeds:

```sh
models/yolo11_finetune/.venv-gpu/bin/python -m models.yolo11_finetune.train --epochs 2 --name smoke --device 0
models/yolo11_finetune/.venv-gpu/bin/python -m models.yolo11_finetune.train --name finetune --device 0
```

Inspect the smoke receipt and checkpoint before starting the full run. Defaults
come from `train_config.yaml`: image size 640, at most 100 epochs, patience 20,
batch 16, seed 42, deterministic mode. Learning rate, warmup and augmentations
use pinned Ultralytics defaults; the actual optimizer and learning rates are
recorded. Execution uses FP32, one explicit CUDA device and zero loader workers.
AMP is disabled to avoid its extra model download. There is no CPU fallback,
automatic weight fetching, resume or reuse of an existing run directory. For
a rerun, choose a fresh name. `--data` accepts a prepared original-scale YAML
at another local location; `--config` selects an equivalent validated config.

The pinned library normally selects a mixed mAP fitness. Our small validator
adapter makes both best-checkpoint selection and patience use **validation
mAP@0.5**. Ties preserve the first maximum. A zero-mAP plateau also exhausts
patience. The training loop and checkpoint format remain Ultralytics code.

## Outputs and traceability

Generated outputs are ignored under `models/yolo11_finetune/results/<name>/`:

- `train.txt`, `val.txt` and `dataset_train_val.yaml`: exact sanitized inputs.
- `weights/best.pt`: validation-selected checkpoint, finalized by Ultralytics.
- `weights/training_record.json`: atomic machine-readable receipt next to it.
- Native Ultralytics artifacts, including `last.pt`, `results.csv` and `args.yaml`.

The receipt records the verified starting-weight SHA-256, requested config,
resolved library config (including lr/warmup), actual optimizer parameter groups,
per-epoch validation mAP@0.5 and learning rates, first best epoch, installed
library versions, machine/GPU details, dataset fingerprints, timing and final
checkpoint SHA-256. Epochs are one-based. The extra final-validation callback
does not append a duplicate epoch. A training exception marks the receipt as
`failed`; only a completed history plus an existing best checkpoint yields
`status: completed`. This status describes completion of the configured budget
or early stopping, not completion of later comparison/evaluation tasks.

Return `best.pt` and its receipt together; verify the checkpoint SHA-256 against
`final_checkpoint_sha256` before later evaluation. No test evaluation is run by
this module. Reproducibility is scoped to the same machine and pinned
environment; GPU repeatability has not yet been measured.

## Evaluating the fine-tuned checkpoint

Evaluation uses the frozen baseline harness with its `--model` option, which
records the supplied file's SHA-256 in the metrics JSON in place of the
pinned-manifest check. The default (no `--model`) remains the pinned
pretrained detector:

```sh
./.venv-baseline/bin/python -m baseline.detection.evaluate --split val --run-id finetuned-val --validate \
  --model models/yolo11_finetune/results/finetune/weights/best.pt
```

## Verification without GPU training

```sh
./.venv-baseline/bin/python -m unittest discover -s models/yolo11_finetune/tests -v
uv run ruff check .
```

Tests cover config restrictions, corrupt starting weights, train/val isolation,
map50 selection, ties, zero-score patience, training-record completeness,
failure status, output reuse and CPU refusal. Running the training CLI in the
CPU baseline environment must exit with an NVIDIA CUDA GPU requirement before
reading training data. Mocked wrapper tests exercise orchestration; they do not
substitute for the separate real two-epoch GPU smoke task.
