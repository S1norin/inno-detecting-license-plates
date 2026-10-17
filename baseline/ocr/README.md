# Recognition baseline

Work for issues #6 and #7. The OCR branch includes the infrastructure
and detector from PRs #8 and #9. OCR additions live under `baseline/ocr/`, apart
from `baseline/requirements-ocr.txt`. See [INTEGRATION.md](INTEGRATION.md)
for the completed integration and [REPORT.md](REPORT.md) for measured results.

## Current verification

The 48 combined baseline tests cover hand-computed recognition metrics, CTC
decoding, validation selection, cache resume, hash mismatches, missing
detections, crop clamping, training vocabulary and test-run guards.
The committed `charset.json` is generated from the actual `ocr_train.csv`.
Integration tests read the detector's native `provenance`/`images` schema,
preserve its inference settings and reject wrong split/metadata inputs.

The real shared environment and an offline EasyOCR smoke test were verified.
The supplied archive's dataset tables match the repository byte for byte;
all 30,000 prepared images/crops are extracted locally. Training export was
validated for all 10,500 crops. Both protocols were evaluated on all three
validation modes, with one frozen-policy test run each. Recognition is weak;
see the measured results in [REPORT.md](REPORT.md). Synthetic tests verify the
implementation and do not measure model accuracy.

A fresh clone still needs the prepared images/crops and explicit weight fetches;
large inputs, weights, environments and raw caches remain gitignored.

## Independent checks

From the repository root with Python 3.12:

```bash
python -m unittest discover -s baseline/ocr/tests -v
python -m baseline.ocr.charset
uv run ruff check .
```

Character indices start at 1; CTC blank is 0. The vocabulary contains exactly
the characters seen in training, restricted to CCPD province characters,
Latin uppercase letters and digits. No val/test text is used to construct it.

After the prepared crops are available, export every training crop:

```bash
python -m baseline.ocr.export_train --scale all \
  --output baseline/ocr/results/train_manifest.json
```

This fails if any crop is missing or the vocabulary differs from training.
Use `--scale 1.0` to export originals only; OCR evaluation defaults to originals.

## Environment and weights

Use the `.venv-baseline` provided by #4, separate from the data environment.
Obtain permission before installing packages or downloading weights, per
`AGENTS.md`. Inspect requirements without installing anything:

```bash
python -m baseline.ocr.setup
```

After installation is approved, create/update the shared environment:

```bash
python -m baseline.ocr.setup --install
```

Setup combines the detector and OCR requirements into one constrained install
and rejects conflicting pins. It retains #4's torch 2.6.0+cpu / torchvision
0.21.0+cpu pair, NumPy 2.1.3 and matplotlib 3.10.3. EasyOCR's additional
dependencies are resolved within those constraints. `matplotlib` is only needed
to plot. The setup command without `--install` remains read-only.

After permission, fetch the two pinned EasyOCR assets:

```bash
.venv-baseline/bin/python -m baseline.common.weights fetch easyocr-craft
.venv-baseline/bin/python -m baseline.common.weights fetch easyocr-zh-sim-g2
.venv-baseline/bin/python -m baseline.common.weights verify easyocr-craft
.venv-baseline/bin/python -m baseline.common.weights verify easyocr-zh-sim-g2
```

The shared downloader separately verifies archive and extracted-file SHA-256
values pinned in `baseline/common/models.json`, then promotes only the named
weight into `baseline/weights/easyocr/`. Initial receipts were checked against
EasyOCR 1.7.2's upstream MD5 values before SHA-256 pinning. The reader verifies
both files before loading them and sets `download_enabled=False`. There are
no automatic installs or downloads during inference.

## Inference and validation selection

Use the baseline Python for all model commands. The examples below use the
POSIX environment path; on Windows use `.venv-baseline/Scripts/python.exe`.

Run all three preprocessing modes on the validation split:

```bash
for mode in none gray_contrast canny; do
  .venv-baseline/bin/python -m baseline.ocr.run \
    --protocol ocr_only --split val --mode "$mode" \
    --output "baseline/ocr/results/ocr_only/raw_ocr_only_val_${mode}.json"
done
```

For end-to-end, repeat with `--protocol end_to_end`, add
`--predictions baseline/detection/results/RUN/predictions_val.json`, and
use a separate output folder. #5's native saved results are accepted directly;
no conversion or detector rerun is needed when that file is available.

Plan for roughly 0.5–2 seconds per crop on CPU, plus model loading and atomic
cache writes; Canny and parallel jobs can take longer. Each protocol/mode has
700 validation crops at scale 1.0, and the selected test run has 800 samples.
The full validation ablation processes 4,200 inputs across six runs.

Each raw file contains every selected image, decoded text and one CTC
probability per character, GT text, size bucket, status, input-image hash,
model/version/weight hashes, dataset-table hashes, protocol, mode and scale.
A missing detection becomes `no_detection` with empty text, never a skipped
sample. A missing/unreadable image is a hard error. Files are checkpointed
atomically after every sample; rerunning the same command resumes incomplete
caches and reuses completed records. Changed inputs or parameters are rejected.

Select the mode and threshold without another OCR pass:

```bash
python -m baseline.ocr.evaluate select \
  --raw baseline/ocr/results/ocr_only/raw_ocr_only_val_none.json \
        baseline/ocr/results/ocr_only/raw_ocr_only_val_gray_contrast.json \
        baseline/ocr/results/ocr_only/raw_ocr_only_val_canny.json \
  --output baseline/ocr/results/ocr_only/accuracy_coverage_val.json

.venv-baseline/bin/python -m baseline.ocr.evaluate plot \
  --selection baseline/ocr/results/ocr_only/accuracy_coverage_val.json \
  --output baseline/ocr/results/ocr_only/accuracy_coverage_val.png

for mode in none gray_contrast canny; do
  python -m baseline.ocr.evaluate val \
    --raw "baseline/ocr/results/ocr_only/raw_ocr_only_val_${mode}.json" \
    --selection baseline/ocr/results/ocr_only/accuracy_coverage_val.json \
    --output "baseline/ocr/results/ocr_only/metrics_ocr_only_val_${mode}.json"
done
```

Pass the three end-to-end caches to `select` separately when they are ready.
Selection is per protocol, using the same rule. It requires all three modes,
the same validation samples, model weights, scale and dataset metadata.

## Final test reporting

Read the selected `mode` in the saved policy and run **that mode only** on test.
For example, if `none` was selected:

```bash
.venv-baseline/bin/python -m baseline.ocr.run \
  --protocol ocr_only --split test --mode none \
  --selection baseline/ocr/results/ocr_only/accuracy_coverage_val.json \
  --output baseline/ocr/results/ocr_only/raw_ocr_only_test_none.json

python -m baseline.ocr.evaluate test \
  --raw baseline/ocr/results/ocr_only/raw_ocr_only_test_none.json \
  --selection baseline/ocr/results/ocr_only/accuracy_coverage_val.json \
  --output baseline/ocr/results/ocr_only/metrics_ocr_only_test_none.json
```

Repeat for end-to-end with its selected policy and test detector predictions.
Test inference requires the validation policy and verifies the model, scale,
dataset metadata and selected mode. A reservation beside each policy binds one
test cache to each protocol. Interrupted runs resume that cache; another cache
or changed policy is rejected. Saved policies/reports cannot be overwritten.
These guards prevent accidental reruns; they are not a replacement for the
team agreement to report test once. Do not copy policies into new folders or
delete reservations to tune against test outcomes.

## CRNN model training and engine selection

Issues #16–#18 add the team's CRNN under `models/crnn/`. This reuses the
existing pinned CPU environment; no new packages or pretrained downloads
are needed. EasyOCR remains the default when `--engine` is omitted.

Prepare the existing crop dataset first. Training verifies the committed
charset against `build_charset(ocr_train.csv)` and uses only the 3,500
scale-1.0 train crops. The 700 validation originals select the epoch by
greedy-decoded character accuracy, with ties going to the earlier epoch.
No test CSV or test crop is read by training. Unknown validation characters
remain in the ground truth without expanding the training vocabulary.

From the repository root, run the fixed full configuration when ready for
the full-training task group:

```bash
.venv-baseline/bin/python -m models.crnn.train \
  --dataset data/dataset --epochs 50 \
  --output models/crnn/results/full-training
```

Use `.venv-baseline/Scripts/python.exe` on Windows. To use the team's
prepared dataset elsewhere, replace `--dataset data/dataset` with its root.
That root must contain `metadata.csv`, `ocr_train.csv`, `ocr_val.csv`, and
the original `crops_gt/train/` and `crops_gt/val/` files.

Training is CPU-only, single-threaded and deterministic, with seed 42,
batch 32 and Adam lr 0.001. It exposes `--epochs`, `--batch-size`,
`--learning-rate`, `--seed`, `--charset` and `--dataset`, but no device flag.
There is no early stopping or scheduler. A nonempty output directory is
rejected; choose a new run directory instead of overwriting artifacts.
The selected model is `checkpoint.pt`; `training_record.json` contains
configuration, input digests, every epoch's metrics, selected epoch, and
checkpoint SHA-256. See `models/crnn/REPORT.md` for the timing probe and
the completed two-run reproducibility check. The 50-epoch run is separate
from those short implementation-verification runs.

Use the expected digest from the trusted training record for inference:

```bash
CRNN_SHA=$(.venv-baseline/bin/python -c \
  "import json; print(json.load(open('models/crnn/results/full-training/training_record.json'))['checkpoint_sha256'])")

.venv-baseline/bin/python -m baseline.ocr.run \
  --protocol ocr_only --split val --mode none \
  --engine crnn \
  --crnn-checkpoint models/crnn/results/full-training/checkpoint.pt \
  --crnn-expected-sha256 "$CRNN_SHA" \
  --output models/crnn/results/evaluation/ocr_only/raw_val_none.json
```

On PowerShell, obtain the same digest with:

```powershell
$crnnSha = (Get-Content -Raw models/crnn/results/full-training/training_record.json | ConvertFrom-Json).checkpoint_sha256
.\.venv-baseline\Scripts\python.exe -m baseline.ocr.run `
  --protocol ocr_only --split val --mode none --engine crnn `
  --crnn-checkpoint models/crnn/results/full-training/checkpoint.pt `
  --crnn-expected-sha256 $crnnSha `
  --output models/crnn/results/evaluation/ocr_only/raw_val_none.json
```

Both checkpoint flags are required for `--engine crnn`. The adapter verifies
SHA-256 before loading, uses the same greedy CTC collapse/confidence convention,
and emits the existing raw-cache schema. Its identity includes architecture,
input geometry, selected epoch, vocabulary digest, source digests, and runtime
versions. Resume checks work unchanged; model/weight identities also prevent
a CRNN cache or selection policy from being mixed with EasyOCR artifacts.

For the later full evaluation, repeat validation inference for all three
modes and both protocols, adding the frozen detector `--predictions` for
`end_to_end`. Reuse the selection/reporting commands above with the CRNN raw
cache paths and separate CRNN policy/output directories. Test inference still
requires that saved validation policy and still reserves one cache per protocol.
Do not rerun the EasyOCR baseline or change the shared selection/metric rules.

Run CRNN and existing OCR checks in the baseline environment:

```bash
.venv-baseline/bin/python -m unittest discover -s models/crnn/tests -v
.venv-baseline/bin/python -m unittest discover -s baseline/ocr/tests -v
```

## Confidence and metrics

`mean` is the default plate confidence. `product` and `sumlog` are alternatives.
The probability threshold grid is 0.50–0.99 in steps of 0.01. For `sumlog`,
compare against `log(tau)` so the threshold is in the correct domain; this is
equivalent to product gating without underflow. Empty OCR results always abstain.

For each threshold, accuracy is exact-match accuracy among accepted plates,
and coverage is accepted plates / all plates. Choose the threshold maximizing
accuracy + coverage - 1; ties use lower tau, then mode order
`none`, `gray_contrast`, `canny`. Zero accepted plates have accuracy 0.

The report includes overall and small/medium/large breakdowns:

- Character accuracy: mean of `1 - edit_distance / max(pred_length, gt_length)`;
  abstentions count as 0.
- Plate exact match: exact accepted plates / all plates.
- Accepted exact accuracy and coverage, reported separately.
- Plate P/R/F1 per #7: TP = accepted with edit distance <= 1;
  FP = accepted with distance > 1; FN = abstained or no detection.
  Wrong reads are **not also FN** under this project's stated convention.
  This is a project-specific metric; do not interpret recall alone as read accuracy.

Text normalization trims surrounding whitespace and uppercases Latin letters.
The three size buckets retain the training-derived assignments in the CSV.

## EasyOCR confidence adapter

EasyOCR 1.7.2's public result contains a region confidence, not native
per-character confidences. The engine uses its CRAFT text regions and
recognizer, extracts normalized CTC probabilities, and greedily collapses
repeated symbols/blanks. Each emitted character receives the maximum
probability in its consecutive CTC run. A blank separates repeated characters.
Regions are concatenated in top-to-bottom, left-to-right order. All regions of
one plate share a recognition batch/width; native CPU `readtext()` can recognize
regions separately, so decoded text is not guaranteed to be identical.

This is an explicit adapter, not a claim that `readtext()` supplies character
probabilities. CPU inference uses one torch thread, deterministic algorithms
and no dynamic quantization. EasyOCR's automatic contrast retry is disabled
so preprocessing modes remain explicit. The adapter is pinned to 1.7.2;
real validation crops were checked before reporting metrics.

Reference: [EasyOCR recognition implementation](https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/easyocr/recognition.py).
