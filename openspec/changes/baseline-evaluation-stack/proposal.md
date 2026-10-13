# Proposal

## Why

The project plan (docs/CV_Plan.pdf) and the course guidelines require a
consolidated evaluation for the baseline models: one comparison table/figure,
per-stage latency and peak memory, and at least 3 failure cases with likely
causes. Today the stage metrics exist separately (`baseline/detection/`,
`baseline/ocr/`, two REPORT.md files and gitignored caches), but nothing ties
them together, and three graded requirements (timing/memory reporting, the
single comparison artifact, failure analysis) are uncovered. Plan rows E2b
(delta vs. OCR-only), E3 (per-step ablation delta) and E5 (OCR per tilt bin)
are also not produced as artifacts.

## What Changes

- New `baseline/evaluation/` package: a consolidation + probe layer over the
  existing stage code. No refactoring of `baseline/detection/` or
  `baseline/ocr/`; models stay fixed to the baseline detector (YOLO11s) and
  the EasyOCR adapter.
- One entry point (`python -m baseline.evaluation.run`) that:
  - reads the existing frozen results (detection metrics/predictions for
    val/test; OCR validation caches for all three preprocessing modes and
    both protocols, the validation selection, and the frozen test reports),
    failing fast if any are missing — it never re-runs metrics inference
  - computes the consolidated comparison table (E1 detection; E2 OCR-only;
    E2b end-to-end with delta vs. OCR-only; E3 ablation with per-step delta;
    E4 per size bucket; E5 per tilt bin; E6 accuracy-coverage at the
    selected tau) for val and test
  - extends OCR metrics with a per-tilt-bin breakdown (tilt_magnitude
    carried through OCR samples, reusing the existing by_size_bucket pattern)
  - dumps at least 3 detection failure cases (from predictions vs. ground
    truth) and at least 3 recognition failure cases (from validation caches,
    per the existing val-for-diagnostics convention), each auto-tagged with a
    likely cause
  - runs a sampled timing probe — a fixed, seeded sample of N images
    (default 100) — measuring per-stage latency (median/p95 ms) and peak
    memory for detection, each OCR preprocessing mode, and preprocessing
    itself; the probe never touches frozen test metrics and is safely
    re-runnable
- Outputs: gitignored `baseline/evaluation/results/<run>/` JSON artifacts
  (table, failures, timing), a tracked `baseline/evaluation/REPORT.md` with
  the one comparison table and interpretation, and one compact committed
  figure under `baseline/evaluation/figures/` for the two-page summary.
- Result inputs are plain paths (`--detection-results`, `--ocr-results`), so
  a future fine-tuned model's results directory plugs in without harness
  changes. The full-split re-run with per-image timing is out of scope and is
  recorded as a follow-up TODO/issue.

## Capabilities

### New Capabilities

- `baseline-evaluation`: consolidated baseline evaluation — a read-only
  consolidation of the frozen detection and recognition results into one
  comparison table with ablation/sensitivity breakdowns, failure-case dumps
  with causes, a seeded sampled timing probe, and reproducible report
  artifacts.

### Modified Capabilities

(None — `openspec/specs/` is empty; the `baseline-detection` and
`baseline-recognition` changes are in flight and their specs are not yet
archived, so no existing spec requirements change.)

## Impact

- New code: `baseline/evaluation/` only, plus small additive extensions in
  `baseline/ocr/io.py` (tilt field on samples) and `baseline/ocr/metrics.py`
  (tilt-bin breakdown).
- `.gitignore`: `baseline/evaluation/results/` ignored; `figures/` tracked.
- Read-only use of `data/dataset/` and of existing `baseline/*/results/`
  caches. No data pipeline changes, no model changes, no dependency
  additions (matplotlib is already pinned for the baseline venv).
- The timing probe needs the shared baseline venv and verified weights to
  run; all other outputs are derivable offline from existing caches.
- Downstream: the two-page technical summary and demo consume the
  consolidated table, figure, and failure cases; the future fine-tuning
  changes consume the path-based input convention.
