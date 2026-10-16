# CRNN implementation and training verification

This report covers issues #16–#18 (model/data, deterministic training, and
engine integration). Full 50-epoch training and the final protocol comparison
are task group 4 of the CRNN change and are not part of these three issues.
No test inference or test-based model selection was performed here.

## Model and data

The grayscale CRNN uses four Conv/BN/ReLU/pool blocks with widths
32/64/128/128, a 384-to-256 sequence projection, two bidirectional LSTM
layers with hidden size 128, and a 256-to-59 classifier. It has **1,145,211
parameters** and returns `(B, 16, 59)` logits. The draft's hidden size 256
would give 2,999,419 parameters, so size 128 meets issue #16's approximately
1.1M target; the design has been corrected accordingly.

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

On 16 October 2026, the one-epoch probe on the real data measured **52.6166
seconds per epoch**, including full validation, and **80.5894 seconds total**,
including setup and input fingerprinting. The final full-training count is
fixed at **50 epochs** (the CLI default), roughly **43.8 minutes** based on
the probe; this is an estimate, not a completed full-run measurement.

Configuration: CPU only; one Torch/OpenCV thread; seed 42; deterministic
algorithms; batch 32; Adam lr 0.001, betas (0.9, 0.999), eps 1e-8, no weight
decay; CTC blank 0; no scheduler, early stopping, or augmentation. Runtime:
Python 3.12.10, Torch 2.6.0+cpu, Torchvision 0.21.0+cpu, NumPy 2.1.3,
OpenCV 4.11.0, AMD Ryzen 7 7700X. No new packages or pretrained assets were
installed or downloaded for this work.

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

`f63e98d015aecd6f0eec8d6c4e5f2afdea105932b9f993be690d34da99057dac`

Their complete per-epoch numerical metrics match exactly:

| Epoch | Mean training CTC loss | Val character accuracy | Val exact match |
|---|---:|---:|---:|
| 1 | 3.0633293640 | 0.2655102041 | 0.0000000000 |
| 2 | 2.6226538051 | 0.2655102041 | 0.0000000000 |

Both select epoch 1, correctly breaking the validation tie in favor of the
earlier epoch. These short runs verify the training machinery and are not
the final trained CRNN or a baseline-comparison result. Same-machine byte
identity is bounded by the recorded pinned environment and CPU configuration.

The probe and both reproducibility runs are under `results/timing-probe/`,
`results/reproducibility-a/`, and `results/reproducibility-b/`, each containing
`checkpoint.pt` and `training_record.json`. All generated artifacts are
gitignored. Unit tests also check ties, output-overwrite refusal, charset
mismatch, unseen validation characters, and that training never opens the
test CSV or a test crop.

## Engine and runner integration

The digest-verified `CRNNEngine` uses the shared `decode_greedy` and returns
one finite confidence per emitted character. Its JSON-stable identity records
architecture, geometry, selected epoch, vocabulary/source digests, and runtime
versions; checkpoint content is the weight identity. A synthetic nonblank
model tests the CTC/confidence contract and cache resume. The selected real
short-run checkpoint was also exercised on three actual validation crops:
confidence lengths and finiteness passed, the standard raw cache passed
`validate_raw`, and repeating inference resumed that cache successfully.

`baseline.ocr.run` adds `--engine {easyocr,crnn}` and the two required CRNN
checkpoint flags. EasyOCR is the unchanged default; the shared
`run_inference`, evaluation, and test-once machinery are unchanged. New tests
cover digest mismatch, identity fields, raw-cache validity/resume, required
flags, and EasyOCR default construction. Existing OCR tests pass. CLI help
matches the documented training and evaluation commands.

Final checks: 99 existing baseline tests and 16 new CRNN tests passed;
`uv run ruff check .` and `git diff --check` passed. Tracked dataset files
are unchanged. The OpenSpec CLI is not installed on this machine, so
`openspec validate crnn-recognizer` remains an unavailable tooling check.
