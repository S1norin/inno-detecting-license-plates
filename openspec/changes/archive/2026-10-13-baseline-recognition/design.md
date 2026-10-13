# Design

## Context

The `baseline-detection` change provides `baseline/common/` (pinned venv
management, sha256-verified weights) and a hash-verified YOLOv11s detector.
`data/dataset/` provides `crops_gt/` + `ocr_{train,val,test}.csv`
(crop_path, plate_text, split, size_bucket) and `preprocessing.py`
(`preprocess_plate(image, mode)` with none / gray_contrast / canny). See
proposal.md for motivation.

## Goals / Non-Goals

**Goals:**
- One-command OCR baseline for both protocols (OCR-only, end-to-end) and all
  three preprocessing modes, with per-character confidences.
- A confidence policy with a deterministic threshold-selection rule and an
  accuracy-coverage curve.
- Cached raw recognition results so metric/τ recomputation never re-runs OCR.
- CRNN readiness: charset + validated training export.

**Non-Goals:**
- No CRNN training (later change consumes the charset/export).
- No detector changes; end-to-end uses the baseline detector as-is.
- No multi-plate images (CCPD subset is single-plate).
- No GPU; CPU inference only (deterministic, matches detection change).

## Decisions

### D1: EasyOCR as the engine
`easyocr.Reader(['ch_sim', 'en'])` — supports Chinese province characters and
Latin/digits, and returns per-character confidences, which the confidence
policy requires.

- Why: the plan's stack names EasyOCR; per-char confidences are native.
- Alternative rejected: PaddleOCR — faster and standard for CN plates, but
  adds a second deep-learning framework (PaddlePaddle) alongside torch and
  deviates from the written plan.
- Constraint: EasyOCR is unmaintained (1.7.2); exact pin + smoke test in
  tasks (see Risks).

### D2: Intercept EasyOCR's weight auto-download
EasyOCR downloads its detection and recognition weight files on first
`Reader()` construction unless they already exist in `model_directory`. The
weights module pre-downloads both files into `baseline/weights/easyocr/`
(hash-verified, entries in `models.json`), and the reader is constructed with
`model_directory` pointing there — no network access at inference time.

- Exact file names/URLs are recorded during the first verified download
  (task); the mechanism does not depend on the names.

### D3: Shared pipeline, two input adapters
```
                 +------------------+        +------------------+
 Path A: csv --->| GT crops         |        | predicted-box    |<--- Path B: images
                 | (crops_gt)       |        | crops (detector) |        + baseline detector
                 +--------+---------+        +---------+--------+
                          +--------------+---+
                                         v
                              preprocess (none|gray_contrast|canny)
                                         v
                              EasyOCR wrapper -> (text, [char confs])
                                         v
                              raw results cache (per crop)
                                         v
                              score -> tau gate -> metrics
```
- `engine.py`: `recognize(crop) -> (text, char_confs)`; wraps EasyOCR,
  normalizes output (strip, uppercase Latin).
- `ocr_only.py`: iterates `ocr_<split>.csv`, loads GT crops.
- `end_to_end.py`: runs the baseline detector on each full image, takes the
  highest-score box, crops it (clamped to image bounds); no box → recorded
  as unread, run continues.
- `run.py`: CLI entry (`--protocol`, `--split`, `--mode`, `--score-fn`).

### D4: Score functions and threshold selection
- Score functions over per-character probabilities p₁..pₙ:
  `mean` (default), `product`, `sumlog` (Σ log p).
- τ sweep on a 0.50–0.99 grid (step 0.01) over the validation split:
  coverage(τ) = fraction of images not abstained;
  accuracy(τ) = exact-match rate among non-abstained.
- Selection rule: τ* = argmax (accuracy(τ) + coverage(τ) − 1); ties break to
  the lower τ (higher coverage). The rule is recorded in the output; the
  curve itself is always reported, per the plan.
- Test is evaluated exactly once with τ*.

### D5: Metrics definitions
- Character accuracy: per plate, 1 − edit_distance(pred, gt)/max(len),
  averaged over the split (abstained plates count as 0).
- Exact match: normalized string equality.
- Plate P/R/F1 (edit-distance based): TP = read with edit distance ≤ 1;
  FP = read with edit distance > 1; FN = abstained or no read. This makes
  abstention trade recall for precision, which is what the policy analysis
  needs.
- Normalization: trim whitespace, uppercase Latin letters, CJK unchanged.
- Reported per split × protocol × mode, plus per size bucket.

### D6: Raw-result caching and runtime
First OCR pass writes per-crop raw results (text + char confs + timing) to
`results/<run_id>/raw_*.json`; all metric/τ work reads the cache, so sweeps
and re-runs are free. Expected first-run cost on CPU: roughly 0.5–2 s per
crop (≈ 2.4k crops per split per protocol × mode) — the ablation is run on
val for mode selection, then the selected mode on test, keeping total first
runtime to a few hours.

### D7: Outputs
```
baseline/ocr/results/<run_id>/
├── raw_<protocol>_<split>_<mode>.json     # per-crop text + char confs
├── metrics_<protocol>_<split>_<mode>.json # D5 metrics + D4 policy fields
├── accuracy_coverage_val.json             # sweep curve + chosen tau
└── accuracy_coverage_val.png
```
Every JSON carries engine name, weight sha256s, protocol, mode, score
function, τ, split, timestamp (traceability requirement).

### D8: CRNN readiness
- `charset.py`: derives the character set from `ocr_train.csv` plate texts
  (province CJK + A-Z + 0-9), writes `charset.json` (tracked); asserts no
  character outside the expected categories.
- `export_train.py`: validates `ocr_train.csv` (crop files exist, texts use
  charset characters only) and writes a training manifest (crop path + text).

## Module layout

```
baseline/
├── requirements-ocr.txt        # easyocr + matplotlib pins (torch shared)
├── ocr/
│   ├── __init__.py
│   ├── run.py                  # CLI entry
│   ├── engine.py               # EasyOCR wrapper
│   ├── ocr_only.py             # Path A
│   ├── end_to_end.py           # Path B (uses baseline/detection)
│   ├── confidence.py           # score fns, tau sweep, selection
│   ├── metrics.py              # char acc, exact match, edit P/R/F1
│   ├── charset.py              # CRNN prep
│   ├── export_train.py         # CRNN prep
│   ├── charset.json            # tracked, generated once
│   ├── results/                # gitignored
│   └── README.md
```

## Risks / Trade-offs

- [EasyOCR 1.7.2 may conflict with newest numpy/torch] → pin a known-good
  combination during implementation; a one-crop smoke test is a task before
  any full run. If no combination works, stop and surface the engine choice
  to the user (would change the plan).
- [Slow CPU inference] → raw-result caching (D6); val-only ablation; test
  run exactly once; runtime documented in README.
- [EasyOCR auto-download bypassing verification] → D2 pre-populates
  `model_directory`; the reader never reaches the network.
- [End-to-end inherits detector misses] → expected; recorded as abstains and
  quantified as the Δ vs. OCR-only (E2b).
- [Canny mode likely degrades OCR] → that is the ablation's purpose;
  reported, not avoided.

## Migration Plan

New code only. Rollback = delete `baseline/ocr/`, `baseline/requirements-ocr.txt`,
the recognizer entries in `models.json`, and the related `.gitignore` lines.

## Open Questions

- Exact EasyOCR weight file names/URLs and the exact easyocr/torch pins:
  recorded at implementation time during the first verified download.
- The τ selection rule (D4) is a reasonable default; if the val curve has a
  clear knee, the rule can be adjusted without changing specs or tasks.
