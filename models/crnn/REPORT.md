# CRNN recognizer: training and evaluation

This report covers the `crnn-recognizer` change (issues #16–#19): the
model, deterministic training, engine integration, and the final
evaluation against the EasyOCR baseline. The 50-epoch full run, the six
validation passes, and the frozen-policy test runs below are the final
CRNN results; no test-based model selection was performed.

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

Training uses the 3,500 train and 700 validation original-scale crops.
The existing CSV/metadata joins validate splits, scales, text, and crop
paths. Charset equality is checked against `build_charset(ocr_train.csv)`;
there are 58 characters, indexed 1–58, with CTC blank 0. Every training
target was checked to lie in that vocabulary and fit the 16-step CTC
sequence. Validation may contain unseen characters; those remain in the
validation ground truth without expanding the training charset. No scaled
variants are used, and no tracked file under `data/dataset/` was modified.

## Timing and fixed full-run configuration

On 16 October 2026, the hidden-size-256 probe on the real data measured
**57.0776 seconds per epoch**, including full validation, and **66.7246
seconds total**, including setup and input fingerprinting (Ryzen 7 7700X
machine). The final full-training count is fixed at **50 epochs** (the CLI
default).

The completed full run on this machine (Apple M3, macOS arm64) took
**2,271.9 seconds total (37.9 minutes)** for 50 epochs: mean 45.37 s per
epoch, min 43.70 s, max 50.25 s, including per-epoch validation. The 1-epoch
probe and the two reproducibility runs remain the short
implementation-verification runs described below.

Configuration: CPU only; one Torch/OpenCV thread; seed 42; deterministic
algorithms; batch 32; Adam lr 0.001, betas (0.9, 0.999), eps 1e-8, no weight
decay; CTC blank 0; no scheduler, early stopping, or augmentation. Runtime:
Python 3.12.12, Torch 2.6.0+cpu, Torchvision 0.21.0+cpu, NumPy 2.1.3,
OpenCV 4.11.0.86. No Python packages or pretrained assets were added.

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
The probe and both reproducibility runs are under
`results/timing-probe-h256/`, `results/reproducibility-h256-a/`, and
`results/reproducibility-h256-b/`, each containing `checkpoint.pt` and
`training_record.json`. All generated artifacts are gitignored. Unit tests
also check ties, output-overwrite refusal, charset mismatch, unseen
validation characters, and that training never opens the test CSV or a test
crop.

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
`run_inference`, evaluation, and test-once machinery are unchanged. Tests
cover digest mismatch, identity fields, raw-cache validity/resume, required
flags, and EasyOCR default construction. Existing OCR tests pass. CLI help
matches the documented training and evaluation commands.

## Full 50-epoch training

The full run used the fixed configuration above on 17 October 2026
(`results/full-training/`, Apple M3). Training loss fell below 0.02 by
epoch 43 while validation character accuracy kept slowly rising; the
selected epoch is **43**, with validation character accuracy
**0.7430102041** and plate exact match **0.3242857143** (227/700). The
`training_record.json` contains the complete configuration, all 50 per-epoch
metrics, and the checkpoint digest
`540cba378cc0ff1dee948a5995ff576467562f6dfe00512db2fd94ca7124747f`, which
matches the saved `checkpoint.pt` byte-for-byte.

Training curve summary (selected epoch in bold):

| Epoch | Train CTC loss | Val character accuracy | Val exact match |
|---:|---:|---:|---:|
| 1 | 2.894178 | 0.265510 | 0.000000 |
| 5 | 2.532073 | 0.320408 | 0.000000 |
| 10 | 1.671775 | 0.587764 | 0.030000 |
| 15 | 0.874973 | 0.709872 | 0.237143 |
| 20 | 0.493841 | 0.729668 | 0.264286 |
| 25 | 0.306229 | 0.724031 | 0.245714 |
| 30 | 0.126811 | 0.732755 | 0.290000 |
| 35 | 0.058419 | 0.728061 | 0.281429 |
| 40 | 0.100174 | 0.723673 | 0.267143 |
| **43** | **0.013210** | **0.743010** | **0.324286** |
| 45 | 0.004473 | 0.739719 | 0.312857 |
| 50 | 0.001855 | 0.742832 | 0.314286 |

## Evaluation under both protocols

Six validation inference passes (ocr_only and end_to-end, each in
none/gray_contrast/canny; end-to-end with the frozen YOLO predictions from
`baseline/detection/results/review-macos/predictions_val.json`) all pass
`validate_raw` with 700 samples. `evaluate select` over the six caches
selected **`none` with tau 0.50** (mean confidence) for **both protocols**.
Test inference then ran once per protocol under the saved selection
(800 samples each), with the test-once reservation binding each policy to
its cache.

Validation ablation, each mode at its own validation-selected threshold:

| Protocol | Mode | Tau | Accepted / 700 | Coverage | Character accuracy | Exact plates | Plate F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ocr_only | none | 0.50 | 700 | 1.0000 | 0.7430 | 227 | 0.6854 |
| ocr_only | gray_contrast | 0.50 | 699 | 0.9986 | 0.7264 | 190 | 0.6780 |
| ocr_only | canny | 0.50 | 690 | 0.9857 | 0.4673 | 6 | 0.1481 |
| end_to_end | none | 0.50 | 688 | 0.9829 | 0.6549 | 109 | 0.5263 |
| end_to_end | gray_contrast | 0.50 | 685 | 0.9786 | 0.6436 | 94 | 0.5075 |
| end_to_end | canny | 0.50 | 678 | 0.9686 | 0.3984 | 0 | 0.0716 |

Frozen-policy test results (selected mode `none`, tau 0.50):

| Protocol | Samples | Accepted | Coverage | Character accuracy | Exact plates | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ocr_only | 800 | 800 | 1.0000 | 0.7538 | 269 / 800 | 0.5500 | 1.0000 | 0.7097 |
| end_to_end | 800 | 781 | 0.9762 | 0.6561 | 133 / 800 | 0.3624 | 0.9371 | 0.5226 |

Test by plate size (CRNN, selected policy):

| Protocol | Size | Samples | Character accuracy | Exact plates | F1 |
|---|---|---:|---:|---:|---:|
| ocr_only | small | 261 | 0.8635 | 115 | 0.8652 |
| ocr_only | medium | 277 | 0.8222 | 121 | 0.7770 |
| ocr_only | large | 262 | 0.5722 | 33 | 0.3976 |
| end_to_end | small | 261 | 0.7564 | 54 | 0.6371 |
| end_to_end | medium | 277 | 0.7275 | 61 | 0.6355 |
| end_to_end | large | 262 | 0.4806 | 18 | 0.2177 |

Test by tilt bin (CRNN, selected policy):

| Protocol | Tilt | Samples | Character accuracy | Exact plates | F1 |
|---|---|---:|---:|---:|---:|
| ocr_only | 0-9 | 167 | 0.7765 | 39 | 0.6902 |
| ocr_only | 10-19 | 63 | 0.5130 | 1 | 0.1739 |
| ocr_only | 20+ | 570 | 0.7738 | 229 | 0.7555 |
| end_to_end | 0-9 | 167 | 0.6738 | 18 | 0.4749 |
| end_to_end | 10-19 | 63 | 0.4340 | 1 | 0.1194 |
| end_to_end | 20+ | 570 | 0.6754 | 114 | 0.5696 |

### CRNN versus EasyOCR (test split)

| Protocol | Engine | Coverage | Character accuracy | Exact plates | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ocr_only | EasyOCR | 0.7300 | 0.3474 | 0 / 800 | 0.0377 | 0.0924 | 0.0535 |
| ocr_only | CRNN | 1.0000 | 0.7538 | 269 / 800 | 0.5500 | 1.0000 | 0.7097 |
| end_to_end | EasyOCR | 0.7025 | 0.3240 | 0 / 800 | 0.0267 | 0.0593 | 0.0368 |
| end_to_end | CRNN | 0.9762 | 0.6561 | 133 / 800 | 0.3624 | 0.9371 | 0.5226 |

EasyOCR rows are the frozen baseline numbers from
`baseline/ocr/REPORT.md`; no baseline rerun was performed.

## Findings

- The plate-specific CRNN is a large improvement over the EasyOCR baseline
  under both protocols: test character accuracy 0.7538 vs 0.3474 (ocr_only)
  and 0.6561 vs 0.3240 (end_to_end), with 269 and 133 exact plates where
  EasyOCR read zero.
- The end-to-end gap to OCR-only (0.6561 vs 0.7538) is the detector floor:
  19 test samples abstained (17 missing detections plus 2 unread crops), and
  cropped boxes lose context the GT crop keeps. The 19 abstentions are the
  17 missing detections plus 2 low-confidence rejections.
- Large plates (width > 112 px) degrade sharply (ocr_only 0.5722 vs 0.8635
  for small): the fixed H48 input downsamples them hard. This is the
  expected small-input finding from the design, not a harness defect.
- The 10–19 degree tilt bin is the weakest (ocr_only 0.5130, 1 exact plate);
  0–9 and 20+ are far stronger.
- `gray_contrast` is close to `none` for the trained model, while `canny`
  collapses (6 exact plates on validation). The selection correctly picks
  `none` for both protocols, as the design anticipated.
- Coverage is near 1.0: the CRNN emits text for almost every crop, so the
  tau 0.50 gate barely rejects. Confidence is not calibrated correctness;
  exact-plate precision (0.5500) is the honest read-quality number.

## Artifacts and verification

Local, ignored outputs under `models/crnn/results/`:

- `full-training/checkpoint.pt` + `training_record.json`: the final 50-epoch
  run; the checkpoint digest in the record matches the file.
- `evaluation/ocr_only/` and `evaluation/end_to_end/`: the six
  `raw_val_<mode>.json` caches, the two `raw_test_none.json` caches,
  `metrics_val_<mode>.json` for all six, `metrics_test_none.json` for both
  protocols, and the `test_inference_<protocol>.json` reservations.
- `evaluation/selection.json` (both protocols, all six candidate modes with
  curves) and `evaluation/selection.png`.

Every table value above was read from those saved JSON reports or from the
existing EasyOCR baseline report. No test inference was rerun, no policy was
changed after observing test results, and no tracked dataset file was
modified.
