# CRNN implementation and training verification

This report covers issues #16–#18 (model/data, deterministic training, and
engine integration). Full 50-epoch training and the final protocol comparison
are task group 4 of the CRNN change and are not part of these three issues.
No test inference or test-based model selection was performed here.

## Model and data

The grayscale CRNN uses four Conv/BN/ReLU/pool blocks with widths
32/64/128/128, a 384-to-256 sequence projection, two bidirectional LSTM
layers with hidden size 256, and a 512-to-59 classifier. It has **2,999,419
parameters** and returns `(B, 16, 59)` logits. The user explicitly selected
the original hidden size 256, superseding issue #16's inconsistent ~1.1M
estimate. The design and tests use the correct ~3M parameter count.

Inputs preserve aspect ratio within H48/W256, use black right padding
(and bottom padding for exceptionally wide crops), and normalize grayscale
pixels to [-1, 1]. Checkpoints embed the complete training-only charset,
its content digest, the architecture, and selected epoch metadata.

The supplied archive provided the 3,500 train and 700 validation original-scale
crops. The existing CSV/metadata joins validate splits, scales, text, and crop
paths. Charset equality is checked against `build_charset(ocr_train.csv)`;
there are 58 characters, indexed 1–58, with CTC blank 0. Every training target
was checked to lie in that vocabulary and fit the 16-step CTC sequence.
Validation may contain unseen characters; those remain in the validation
ground truth without expanding the training charset. No scaled variants
are used, and no tracked file under `data/dataset/` was modified.

## Timing and fixed full-run configuration

On 16 October 2026, the hidden-size-256 probe on the real data measured **57.0776
seconds per epoch**, including full validation, and **66.7246 seconds total**,
including setup and input fingerprinting. The final full-training count is
fixed at **50 epochs** (the CLI default), roughly **47.6 minutes** based on
the probe; this is an estimate, not a completed full-run measurement.

Configuration: CPU only; one Torch/OpenCV thread; seed 42; deterministic
algorithms; batch 32; Adam lr 0.001, betas (0.9, 0.999), eps 1e-8, no weight
decay; CTC blank 0; no scheduler, early stopping, or augmentation. Runtime:
Python 3.12.10, Torch 2.6.0+cpu, Torchvision 0.21.0+cpu, NumPy 2.1.3,
OpenCV 4.11.0, AMD Ryzen 7 7700X. No Python packages or pretrained assets were
added. OpenSpec CLI 1.14.1 was installed as a separate workspace development tool.

Each epoch records training CTC loss, greedy-decoded validation character
accuracy (the shared edit-distance formula), and plate exact match. The
highest validation character accuracy selects the checkpoint; ties retain
the earlier epoch. Training reads only train/validation CSVs and crops.
The shared metadata table is fingerprinted and used for split joins.

The machine-readable record includes architecture/input/optimizer/loss
configuration, requested and completed epoch counts, seed, thread count,
runtime versions and machine, CSV/metadata/charset and original-crop digests,
all epoch metrics, selected epoch, timing, and the final checkpoint digest.
Wall-clock timing, output paths, and timestamps are excluded from checkpoint
bytes. Saving through a file object fixes the internal Torch ZIP archive name.

## Two-run reproducibility check

Two independent two-epoch CLI runs used all 3,500 training and 700 validation
originals with identical configuration. Both checkpoint files have SHA-256:

`0b5aae42ad39c911868e48cc33f66986a54f0f18af42d6f25c2fedc2864ab998`

Their complete per-epoch numerical metrics match exactly:

| Epoch | Mean training CTC loss | Val character accuracy | Val exact match |
|---|---:|---:|---:|
| 1 | 2.8945722376 | 0.2655102041 | 0.0000000000 |
| 2 | 2.6018353865 | 0.2655102041 | 0.0000000000 |

Both select epoch 1, correctly retaining the earlier epoch on a validation tie. These short runs verify the training machinery and are not
the final trained CRNN or a baseline-comparison result. Same-machine byte
identity is bounded by the recorded pinned environment and CPU configuration.

The probe and both reproducibility runs are under `results/timing-probe-h256/`,
`results/reproducibility-h256-a/`, and `results/reproducibility-h256-b/`, each containing
`checkpoint.pt` and `training_record.json`. All generated artifacts are
gitignored. Unit tests also check ties, output-overwrite refusal, charset
mismatch, unseen validation characters, and that training never opens the
test CSV or a test crop.
