# Recognition baseline results

Evaluated locally on 12 October 2026 for issues [#6](https://github.com/S1norin/inno-detecting-license-plates/issues/6) and [#7](https://github.com/S1norin/inno-detecting-license-plates/issues/7).
Both protocols and all validation ablations are complete. The pretrained EasyOCR
baseline is weak on this custom CCPD subset; the evaluation infrastructure works,
but these results do not establish useful plate-reading accuracy.

## Data and frozen protocol

The supplied prepared archive has metadata and split tables byte-identical to the
repository. All 15,000 images and 15,000 GT crops were extracted locally. Original
scale only: train 3,500, validation 700, test 800; split assignments, seed 42 and
scales were preserved. Size thresholds remain train-derived: small <= 83 px,
medium > 83 and <= 112 px, large > 112 px.

- OCR-only reads the original GT crops, isolating recognition from detection.
- End-to-end reads full images and chooses the highest-score saved YOLO box,
  clamps it to the image and rounds outwards. The detector inference floor is
  0.001; no additional 0.25 cutoff is applied before OCR. An absent box remains
  in the sample denominator as `no_detection`.
- Validation evaluates `none`, `gray_contrast` and `canny` using the unchanged
  `data.preprocessing.preprocess_plate` implementation.
- Mean character confidence is swept over tau 0.50–0.99 by 0.01. Selection
  maximizes accepted exact accuracy + coverage - 1; ties choose lower tau, then
  mode order. Both protocols select `none`, mean confidence, tau **0.50**.
- Each protocol runs test once with its validation-selected policy. No engine,
  preprocessing or threshold change was made after observing test results.

## Validation ablation

Each row reports the candidate mode at its own validation-selected threshold.
Character accuracy and exact match include all 700 samples; confidence-rejected
and unread plates contribute zero. Coverage measures accepted reads, not correctness.

| Protocol | Mode | Tau | Accepted / 700 | Coverage | Character accuracy | Exact plates | Plate F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ocr_only | none | 0.50 | 517 | 0.7386 | 0.3580 | 3 | 0.0609 |
| ocr_only | gray_contrast | 0.50 | 400 | 0.5714 | 0.2693 | 1 | 0.0365 |
| ocr_only | canny | 0.50 | 88 | 0.1257 | 0.0328 | 0 | 0.0000 |
| end_to_end | none | 0.50 | 494 | 0.7057 | 0.3279 | 1 | 0.0392 |
| end_to_end | gray_contrast | 0.50 | 408 | 0.5829 | 0.2620 | 0 | 0.0226 |
| end_to_end | canny | 0.50 | 81 | 0.1157 | 0.0303 | 0 | 0.0000 |

## Frozen-policy test results

| Protocol | Samples | Accepted | Coverage | Character accuracy | Exact plates | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ocr_only | 800 | 584 | 0.7300 | 0.3474 | 0 / 800 | 0.0377 | 0.0924 | 0.0535 |
| end_to_end | 800 | 562 | 0.7025 | 0.3240 | 0 / 800 | 0.0267 | 0.0593 | 0.0368 |

### Test by plate size

| Protocol | Size | Samples | Coverage | Character accuracy | Exact plates | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ocr_only | small | 261 | 0.7280 | 0.3481 | 0 | 0.0263 | 0.0658 | 0.0376 |
| ocr_only | medium | 277 | 0.7653 | 0.4001 | 0 | 0.0472 | 0.1333 | 0.0697 |
| ocr_only | large | 262 | 0.6947 | 0.2910 | 0 | 0.0385 | 0.0805 | 0.0520 |
| end_to_end | small | 261 | 0.6820 | 0.3207 | 0 | 0.0169 | 0.0349 | 0.0227 |
| end_to_end | medium | 277 | 0.7581 | 0.3719 | 0 | 0.0238 | 0.0694 | 0.0355 |
| end_to_end | large | 262 | 0.6641 | 0.2767 | 0 | 0.0402 | 0.0737 | 0.0520 |

### Counts and metric interpretation

- ocr_only: TP=22, FP=562, FN=216; raw statuses: read=743, unread=57.
- end_to_end: TP=15, FP=547, FN=238; raw statuses: read=736, no_detection=17, unread=47.

TP means an accepted read with edit distance <= 1; FP means an accepted read
with distance > 1; FN means abstained or unread. Wrong reads are not also FN
under issue #7's convention. Consequently, recall is not the fraction of all
plates correctly read. Exact match is exact accepted plates / all samples.
Character accuracy is mean `1 - distance / max(pred_length, gt_length)`, with
abstentions contributing zero. Normalization trims whitespace and uppercases
Latin characters, preserving CJK. Mean confidence is not calibrated correctness:
a wrong plate can pass tau 0.50 despite one very uncertain character.

## Detector integration check

Max's infrastructure/detector commits `340f17f6` and `d99a428f` were merged
locally without conflicts. Fresh local YOLO predictions use his fixed inference
settings and reproduce the numerical detector summary in his committed report.

| Split | mAP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| val | 0.9245 | 0.9402 | 0.8986 | 0.9189 |
| test | 0.9073 | 0.9408 | 0.8938 | 0.9167 |

Detection P/R/F1 here use score >= 0.25 and matching IoU >= 0.5; AP uses the
0.001 prediction floor. These detector metrics have different definitions from
the recognition edit-distance metrics above.

## Validation error examples

Examples below come from validation OCR-only `none`, not test-based tuning.

| Sample | Ground truth | Raw OCR | Observation |
|---|---|---|---|
| `ee22be0fe8e36f86_s100` | 皖AF9671 | empty | No recognized text region/read. |
| `27e45a18bf0bc792_s100` | 皖A357D9 | `#435709` | Province/letter substitutions; confidently read digits do not guarantee an exact plate. |
| `65b47491f34d2552_s100` | 皖AC321C | ` 03210` | Missing/replaced province and letters; surrounding space is trimmed for metrics. |

The GT-crop protocol already has very low exact accuracy, so recognition is a
major limitation independently of YOLO. Canny has much lower coverage and
character accuracy on validation. A future recognizer experiment must tune on
train/validation; these frozen test outputs must not be reused to choose changes.

## Runtime, weights and adapter

Python 3.12.14; EasyOCR 1.7.2; torch 2.6.0+cpu; torchvision 0.21.0+cpu;
NumPy 2.1.3; OpenCV 4.11.0.86; matplotlib 3.10.3; Ultralytics 8.3.162.
OCR uses CPU, one torch thread, deterministic algorithms, greedy CTC, no dynamic
quantization and no automatic contrast retry. Models are verified before loading
and inference disables downloads. Exact package pins are in the shared requirements.

EasyOCR's public API returns region confidence, not native character confidence.
The adapter extracts normalized CTC probabilities and assigns each emitted
character the maximum probability of its consecutive collapsed run. Regions
are concatenated top-to-bottom, left-to-right, using one batch/shared width per
plate. Native CPU `readtext()` can batch differently, so identical text is not
guaranteed: a 12-crop validation comparison matched 11 crops, with one differing
in its first character. These are results of the documented adapter.

| Asset | SHA-256 |
|---|---|
| `license-plate-finetune-v1s.pt` | `95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381` |
| `craft_mlt_25k.pth` | `4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17` |
| `zh_sim_g2.pth` | `cb678fdef09d651e7763ca551ad790dc89f0b2e3d2a640484330e338fb574c7a` |

Archive hashes and upstream EasyOCR receipts are also pinned in the manifest.
Raw caches include engine/CTC/preprocessing hashes and per-input image hashes.
Dataset metadata SHA-256:
`dec477828b64942bfc45b095bb9b867ab33bdcefa46d4a97903a099d7b371552`.

## Artifacts and verification

Local, ignored outputs:

- `baseline/ocr/results/{ocr_only,end_to_end}/raw_<protocol>_val_<mode>.json`
  and `metrics_<protocol>_val_<mode>.json`: all six complete validation runs.
- In each protocol folder: `accuracy_coverage_val.json`, its PNG plot, the
  selected `raw_<protocol>_test_none.json` and `metrics_<protocol>_test_none.json`,
  plus the test-inference reservation binding the frozen policy to its cache.
- `baseline/ocr/results/accuracy_coverage_val.{png,svg}`: combined six curves;
  `accuracy_coverage_val_detail.{png,svg}`: separate panels zoomed to 0–2.7%
  exact accuracy; `summary.json`: compact metrics; `run_context.json`: run checks.
- `baseline/detection/results/ocr-local-{val,test}/`: predictions and detector metrics.
- `baseline/ocr/results/train_manifest.json`: all 10,500 validated training crops.

The tracked training-only `charset.json` has 58 characters plus CTC blank index 0.
No CRNN training was performed. All 48 combined baseline tests pass, including
hand-computed metrics, cache contracts, offline weight verification and guarded
test inference. Ruff passes for the whole repository. A repeated environment
setup is a no-op. Reopening the completed OCR-only validation cache reused it
without changing its digest or rerunning recognition.

Metric/selection JSON files carry engine, weight digests, protocol, mode, split,
score function, tau and timestamps. Reports bind raw and selection SHA-256 values.
Every validation cache has 700 distinct samples; every test cache has 800;
no unread or missing-detection sample is dropped.

Code and this compact report are on `feat/ocr-recognition-baseline` in
[PR #10](https://github.com/S1norin/inno-detecting-license-plates/pull/10).
It targets detector PR #9; merge dependencies first and retarget to `main`
before the final merge. Raw data, weights and inference caches stay local. See
[README.md](README.md) for reproduction commands and
[INTEGRATION.md](INTEGRATION.md) for the shared environment and detector contract.
