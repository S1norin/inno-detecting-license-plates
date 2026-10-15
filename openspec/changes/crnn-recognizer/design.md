# Design

## Context

The recognition harness in `baseline/ocr/` is engine-agnostic: `run.py` calls any object exposing `identity`, `weight_digests`, and `recognize(image) -> {"text", "char_confidences"}`, and everything downstream (raw caches, threshold sweep, test-once reservation, metrics, tilt/size breakdowns) works from those records. Today `run.py` hardcodes `EasyOCREngine` in `main()`. The CRNN data side already exists: `charset.json` (58 train-only characters, CTC blank at index 0), `ctc.py` (greedy decode with per-character confidence), and `export_train.py` (validated crop manifest with `ctc_target` indices). The shared baseline environment has `torch 2.6.0+cpu` and `torchvision 0.21.0`; no new packages are allowed.

## Goals / Non-Goals

**Goals:**

- Train a small CRNN/CTC recognizer on the 3,500 original-scale training crops, deterministically on CPU, producing a bit-reproducible checkpoint.
- Select the epoch on validation character accuracy and evaluate through the unchanged shared harness under both protocols and all three preprocessing modes.
- Produce the CRNN-vs-EasyOCR comparison for the 20 Oct second-approach milestone.

**Non-Goals:**

- No fine-tuning of the detector; end-to-end crops come from the existing frozen YOLO predictions.
- No scaled-variant training (0.75/0.5 stay reserved for the later scale experiment).
- No GPU/MPS support, no device flag, no data augmentation beyond the fixed input resize.
- No changes to the harness, metrics, confidence policy, or the EasyOCR engine.

## Decisions

### D1: Package layout `models/crnn/`

New top-level `models/` package (with `__init__.py`) containing the `models/crnn/` subpackage with three modules plus `__init__.py`:

- `model.py` — architecture, input transform, checkpoint save/load.
- `train.py` — CLI training loop, validation, epoch selection, training record.
- `engine.py` — `CRNNEngine` implementing the shared engine contract.

Rationale: `baseline/` keeps its meaning — the two off-the-shelf baselines and the shared infrastructure they were built with — while the team's own models live in `models/`, mirroring the report's baseline-vs-second-approach framing. The shared harness stays in `baseline/ocr/` and is imported across packages (`from baseline.ocr.ctc import decode_greedy`); a nested `models/crnn/.gitignore` ignores `results/`, matching the `baseline/ocr/` pattern, and `AGENTS.md` gains a line pointing non-baseline model code to `models/`. Alternative (placing the CRNN under `baseline/ocr/crnn/` for proximity to the harness) was rejected: the second condition should not sit inside `baseline/`, and `models/` also gives E1 (YOLO fine-tuning) and the scale experiments a home later.

### D2: Architecture and input geometry

```
input:  1 x 48 x 256   grayscale, (x/255 - 0.5)/0.5, aspect-preserving
                            height 48, right-padded to width 256
CNN:    conv3x3 1->32   + BN + ReLU + maxpool2   (24 x 128)
        conv3x3 32->64  + BN + ReLU + maxpool2   (12 x 64)
        conv3x3 64->128 + BN + ReLU + maxpool2   (6 x 32)
        conv3x3 128->128+ BN + ReLU + maxpool2   (3 x 16)
sequence: (B,128,3,16) -> (B,16,384) -> Linear(384,256)
RNN:    BiLSTM x2, hidden 256 -> 512
output: Linear(512, 59)   (58 chars + blank, blank=0 per charset.json)
loss:   nn.CTCLoss(blank=0), per-sample target lengths
```

About 1.1M parameters. Sequence length 16 comfortably exceeds the longest plate (7 characters). Right padding preserves left-to-right character order. Alternatives: taller 64px input (more data for small plates, ~2x compute) and pretrained backbones (breaks the CPU-only, no-download constraint); both rejected for the deadline and single-factor comparison.

### D3: Training loop

- Data: `load_samples(dataset, "train", 1.0)` (3,500 crops), charset verified equal to `build_charset(ocr_train.csv)` before start (same guard `export_train.py` uses).
- Images loaded on the fly (`cv2.imread` + resize), `num_workers=0`, batch 32; avoids a separate tensor cache and keeps the dataset the single source of truth.
- `torch.manual_seed(42)`, `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)`, Adam lr 1e-3, fixed epoch count (default 50; final count fixed after the first-epoch timing probe and recorded in the training record).
- Every epoch: greedy decode of the full 700-sample validation split with the existing `decode_greedy`; character accuracy computed with the same `1 - edit_distance / max(len)` formula as `metrics.py`.
- Keep the best checkpoint by validation character accuracy, ties to the earlier epoch. No early stopping, no scheduler: selection is a pure post-hoc rule, which keeps the loop deterministic and auditable.

### D4: Checkpoint and training record

`torch.save` of a dict: `state_dict`, `charset_sha256`, architecture config, `epoch`, `val_character_accuracy`, `seed`, thread count, torch/torchvision versions, dataset and CSV digests, per-epoch validation metrics list. Saved as `checkpoint.pt` next to `training_record.json` in a gitignored `models/crnn/results/RUN/` directory. `torch.save` is deterministic for a given state dict on a fixed torch build, so same-machine reruns are bit-identical; the torch version is recorded so that claim is bounded.

### D5: Engine contract

`CRNNEngine(checkpoint_path, expected_sha256)`:

- Verifies the checkpoint digest on construction and refuses mismatches (mirrors `EasyOCREngine.verify_weights`).
- `identity`: name `crnn`, decoder `greedy`, character-confidence definition, input geometry, architecture summary, selected epoch, adapter/ctc file digests, torch version.
- `weight_digests`: `{"crnn_checkpoint.pt": sha256}`.
- `recognize(image)`: grayscale (convert if 3-channel), resize/pad/normalize, forward under `torch.inference_mode()` with 1 thread and deterministic mode, softmax, then the existing `decode_greedy` — so per-character confidences follow the exact same convention as the EasyOCR adapter (max probability in each collapsed CTC run).

Because `model_contract` in `evaluate.py` keys on `engine` and `weight_digests`, a CRNN test run can never accidentally consume an EasyOCR selection file, and vice versa.

### D6: `run.py` integration

Add `--engine {easyocr,crnn}` (default `easyocr`, preserving existing behavior) plus `--crnn-checkpoint` and `--crnn-expected-sha256` (required only when `--engine crnn`). `main()` constructs the matching engine; `run_inference` is unchanged. The engine identity flows into the raw cache metadata automatically.

### D7: Evaluation workflow

1. `python -m models.crnn.train ...` -> checkpoint + record.
2. Validation inference, 6 passes: `python -m baseline.ocr.run --protocol {ocr_only,end_to_end} --split val --mode {none,gray_contrast,canny} --engine crnn ...` (end-to-end passes take the existing frozen detector `--predictions`).
3. `python -m baseline.ocr.evaluate select --raw <6 val caches> --output selection.json` (requires all three modes per protocol, as with the baseline).
4. Test inference once per protocol with `--selection`, then `evaluate val/test` reports and the selection plot.
5. Comparison against EasyOCR reuses the already-computed baseline test metrics; no baseline rerun.

### D8: Reporting

Compact tracked `models/crnn/REPORT.md` (config, training curve summary, per-protocol metrics, comparison table) plus the CRNN rows added to `docs/progress_report.md` and `.tex` for the milestone. Raw caches, checkpoints, and plots stay in gitignored `models/crnn/results/`.

## Risks / Trade-offs

- [CPU training time exceeds the week's budget] -> First task is a 1-epoch timing probe; if 1 thread is too slow, raise the batch to 64 (fewer, larger steps) before considering 4 fixed threads; the epoch count is fixed after the probe and recorded.
- [Overfitting: 1.1M parameters on 3,500 crops] -> The model is deliberately small; the per-epoch validation curve is recorded, and the documented fallback is hidden 256 -> 128 (~0.5M params).
- [Small plates (<=83 px) lose detail when resized to height 48] -> Expected to show in the per-bucket metrics; that is a finding for the report, not a defect to fix in this change.
- [Canny/gray_contrast modes are semantically odd for a trained model] -> Kept only because `select_validation` requires all three modes; the selection is expected to pick `none`, and the ablation numbers are reported, not hidden.
- [torch.save bit-identity depends on the torch build] -> Bounded by recording the exact torch version and machine in the training record; the reproducibility claim is "same environment", matching how the rest of the repo pins runs.

## Open Questions

None. The epoch count is the only number that depends on the timing probe, and fixing it does not change the specs, the approach, or the task breakdown.
