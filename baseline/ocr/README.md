# Recognition baseline

Work for issues #6 and #7. All new code lives under `baseline/ocr/`, apart
from `baseline/requirements-ocr.txt`. No detector, shared environment or
dataset pipeline files are changed. See [INTEGRATION.md](INTEGRATION.md)
for the exact inputs needed from issues #4 and #5.

## Current verification

The dependency-free tests cover hand-computed recognition metrics, CTC
decoding, validation selection, cache resume, hash mismatches, missing
detections, crop clamping, training vocabulary and test-run guards.
The committed `charset.json` is generated from the actual `ocr_train.csv`.

Real EasyOCR inference, preprocessing ablations and final dataset metrics
have **not** been run yet. A fresh clone contains LFS pointers for the source
photos and does not contain the regenerated images or crops. Model weights
and the shared baseline environment are also absent. Do not interpret the
synthetic test results as model accuracy or close #6/#7 on their basis.

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
`AGENTS.md`. Once the shared environment is ready, add our pinned dependencies:

```bash
uv pip install --python .venv-baseline/bin/python -r baseline/requirements-ocr.txt
```

EasyOCR also needs a compatible torch/torchvision pair, OpenCV, NumPy, Pillow
and its other dependencies. Resolve that pair with the owner of #4; this branch
does not choose a competing torch version. `matplotlib` is only needed to plot.

Place `craft_mlt_25k.pth` and `zh_sim_g2.pth` in `baseline/weights/easyocr/`.
Their **actual, pinned SHA-256 values** must be in `baseline/common/models.json`.
The reader verifies both local files before loading them and sets
`download_enabled=False`. There are no automatic installs or downloads.

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
`--predictions baseline/detection/results/RUN/predictions_val_ocr.json`, and
use a separate output folder. The interchange format is in INTEGRATION.md.
It can be produced from the detector's saved predictions without rerunning it.

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
Regions are concatenated in top-to-bottom, left-to-right order.

This is an explicit adapter, not a claim that `readtext()` supplies character
probabilities. CPU inference uses one torch thread, deterministic algorithms
and no dynamic quantization. EasyOCR's automatic contrast retry is disabled
so preprocessing modes remain explicit. The adapter is pinned to 1.7.2;
verify it on real crops before reporting metrics.

Reference: [EasyOCR recognition implementation](https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/easyocr/recognition.py).
