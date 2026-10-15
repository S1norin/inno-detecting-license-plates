# Design

## Context

`main` already contains frozen, report-backed stage results for both baselines:

- `baseline/detection/results/<run>/predictions_{val,test}.json` and
  `metrics_{val,test}.json` — YOLO11s inference records (boxes, scores) and
  mAP@0.5 / P / R / F1 overall, per size bucket, per tilt bin. Provenance
  carries dataset fingerprints, inference settings, and weight digests.
- `baseline/ocr/results/{ocr_only,end_to_end}/` — raw OCR caches for all
  three preprocessing modes on val, the validation selection (mode + tau per
  protocol), and frozen one-shot test reports with metrics and per-bucket
  breakdowns. Raw records carry `sample_id`, ground truth, raw read,
  per-character probabilities, status, and `size_bucket`.

Constraints: the OCR test split is frozen (test-once discipline — a test
re-run is forbidden); full CPU inference of EasyOCR over 700+800 images is
hours of wall time; the grading requirements that are uncovered are
per-stage latency + peak memory, one comparison table/figure, and failure
cases with causes; plan rows E2b (delta vs. OCR-only), E3 (per-step delta),
and E5 (OCR per tilt bin) are not produced as artifacts. OCR raw records do
not carry tilt; `metadata.csv` does (join key: `sample_id` -> parent_id +
scale).

## Goals / Non-Goals

**Goals:**

- One command that turns the existing frozen results into the complete
  evaluation artifact set (table, failures, timing, report, figure).
- Deterministic, byte-identical re-runs for everything except the timing
  probe and timestamps/run ids.
- A cheap, safe timing probe that is the only new inference.
- Additive, non-breaking extensions to `baseline/ocr/` for tilt bins.
- A path-based input convention so a future model's results directory plugs
  in without harness changes.

**Non-Goals:**

- No model registry, no `--model` flag, no engine abstraction (explicitly
  deferred; option A).
- No refactoring of `baseline/detection/` or `baseline/ocr/` CLIs or their
  existing outputs.
- No full-split re-run with per-image timing (recorded as a follow-up
  TODO/issue).
- No changes to `data/dataset/`, splits, thresholds, or preprocessing.

## Decisions

### D1: Consolidation layer, not re-orchestration

`baseline/evaluation/` reads the existing result JSONs and computes
everything derivable from them; the only new inference is the timing probe.

- Why: the metrics are frozen (test-once), re-derivation risks diverging
  from the reported numbers, and a full re-run is hours on this CPU.
- Alternative considered: the harness re-runs detection + OCR end to end
  with caching. Rejected: collides with the test freeze, and "resume" would
  collapse to the same read-only behavior when all caches exist.

### D2: Input contract is path + provenance validation

Inputs are directories/paths: `--detection-results <dir>` (containing
`predictions_{val,test}.json` + `metrics_{val,test}.json` for one run) and
`--ocr-results <dir>` (containing both protocol folders with val caches,
selection, and frozen test reports). Before use, the harness checks split,
scale, dataset fingerprint (`input_sha256` / `dataset_digests`), and model
identity against `data/dataset/`; any mismatch fails the run naming the
field.

- Why: zero new interchange schema; the stage schemas are stable and already
  provenance-rich.
- Alternative considered: a new unified intermediate schema written by the
  stages. Rejected: duplicates what provenance already carries and forces
  stage changes.

### D3: Tilt bins are joined from metadata at consolidation time

The consolidated table gets `by_tilt_bin` for OCR by joining raw records to
`metadata.csv` on `sample_id` (parent_id + scale -> `tilt_magnitude`) and
binning with the same `tilt_bin()` helper the detection metrics use.
Additively, `baseline/ocr/io.py` `load_samples` gains `tilt_magnitude` on
each sample and `baseline/ocr/metrics.py` gains a `by_tilt_bin` breakdown
mirroring `by_size_bucket`, so future caches/metrics carry it natively.
Existing caches are never regenerated.

- Why: E5 requires OCR per tilt bin; the data is already in `metadata.csv`;
  touching `io.py`/`metrics.py` additively keeps stage code self-consistent
  without breaking the 48 passing tests (pure additions).
- Alternative considered: keep all tilt logic inside the harness. Rejected:
  would leave stage metrics permanently missing E5 for any new run.

### D4: Timing probe = seeded sample, sequential stages, process-level memory

`timing.py`: `random.Random(seed).sample` over the original-scale image list
(default N=100, seed 42) selects the probe set once and records the sample
ids in `timing.json`. Stages run sequentially with the frozen inference
settings (detection: imgsz 640, conf 0.001, iou 0.7, 1 thread, seed 42; OCR:
existing EasyOCR adapter settings): (1) detection inference, (2) OCR
recognition per mode on GT crops (ocr-only geometry), (3) preprocessing per
mode. A 5-image warmup is excluded from timing and reported separately.
Latency: per-image wall clock, median + p95 per stage. Peak memory: process
max RSS (`resource.getrusage`) recorded after each stage, reported as
cumulative stage-end values.

- Why: N=100 gives tight enough latency estimates for "approximate
  inference time" in minutes; max RSS is the honest, cheap memory signal on
  a single-process CPU box.
- Alternatives considered: per-image `tracemalloc` (slow, noisy on
  torch/OpenCV allocations — rejected); a fresh subprocess per stage for
  isolated memory (heavier, and the report only needs approximate values —
  deferred to the full-run follow-up).

### D5: Failure selection is deterministic ranking over existing data

Detection: per image, best predicted box by IoU against the single GT box
(threshold 0.5, matching the metrics definition). Failures ranked by (is a
miss, best IoU ascending, score ascending); top 5 written. Auto-tags:
`miss`, `low_iou`, `multiple_detections`, plus `low_confidence` when best
score < 0.25. Recognition: from the val raw caches (both protocols, selected
mode), failures = accepted reads with edit distance > 1, plus abstained and
unread samples; ranked by (accepted-wrong first, then distance descending,
then coverage of status types); top 5 per protocol. Auto-tags: `no_detection`,
`unread`, `abstained`, `substitution`, `missing_chars`, `extra_chars` (from
the edit alignment). Test samples never appear in the failure dump. The
prose "likely cause" explanations are written in the report from these tags
at authoring time; the JSON carries tags, not prose.

- Why: deterministic (byte-identical re-runs), minimum 3 guaranteed, and the
  tags make the report's cause analysis auditable.

### D6: Report and figure are fully generated

`table.json` is the single source of truth. `REPORT.md` and the figure are
rendered from it by the harness: the markdown table, the interpretation
paragraphs (assembled from the numbers, e.g. delta magnitudes, worst bucket),
and one 2x2 figure (detection mAP50/F1 by bucket; OCR char-accuracy and exact
match by bucket per protocol; accuracy-coverage curves at selected tau;
per-tilt-bin comparison). `REPORT.md` goes to `baseline/evaluation/REPORT.md`
(tracked, overwritten on each run); the figure to
`baseline/evaluation/figures/baseline_comparison.png` (tracked, 160 dpi).
Raw JSONs go to the gitignored `--out` directory.

- Why: byte-identical reproducibility of the tracked artifacts; no
  drift between table and report.
- Trade-off: the interpretation is template-driven, not free-form prose.
  Accepted: the two-page summary is where the full interpretation lives.

### D7: Harness is stdlib-only offline; probe imports are guarded

Table, failures, and rendering use only the standard library (json, csv,
math, random, pathlib). `torch`/`ultralytics`/`easyocr` are imported inside
the probe code path only, so an offline consolidation run works without the
baseline venv; `--no-probe` makes the run fully offline.

## Risks / Trade-offs

- [Probe peak memory is cumulative process max RSS, not per-stage isolated]
  -> Methodology recorded in `timing.json` and reported honestly; isolated
  per-stage memory is part of the full-run follow-up.
- [Existing OCR caches lack tilt] -> joined from `metadata.csv` at
  consolidation time; the join key is validated to be unique, so a bad join
  fails the run instead of mis-binning.
- [EasyOCR probe is slow on CPU] -> N=100 keeps it to minutes; `--no-probe`
  exists for offline use; probe never blocks the table.
- [Detection failure cases need `predictions_*.json`, not just metrics]
  -> The spec names predictions as a required input; a run given only
  metrics files fails fast with the missing file named.
- [Generated REPORT.md limits narrative] -> accepted (D6); the two-page
  summary carries the full interpretation.
- [Tilt helper import from `baseline/detection/metrics.py` into
  `baseline/ocr/`] -> one small cross-import between sibling baseline
  packages; if it proves awkward, the 3-line bin function is duplicated
  instead (flagged in tasks).

## Migration Plan

Pure addition. No migration. Rollback = delete `baseline/evaluation/` and
revert the two additive `baseline/ocr/` edits; no existing artifact is
modified by the change itself (the tracked report/figure are new files).

## Open Questions

- Final figure layout (2x2 vs. a single combined panel) — cosmetic, decided
  during implementation.
- Probe warmup length (5) — a detail, not a spec commitment.
