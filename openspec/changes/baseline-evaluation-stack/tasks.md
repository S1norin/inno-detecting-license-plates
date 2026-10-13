# Tasks

## 1. Package skeleton and input contract

- [x] 1.1 Create the `baseline/evaluation/` package (`__init__.py`, `run.py` CLI stub with `--dataset`, `--detection-results`, `--ocr-results`, `--out`, `--no-probe`) and verify `uv run ruff check baseline/evaluation/` passes
- [x] 1.2 Implement input loading and provenance validation for the detection results directory (`predictions_{val,test}.json` + `metrics_{val,test}.json`, one run) and the OCR results directory (both protocol folders: val caches for all three modes, selection, frozen test reports), checking split, scale, dataset fingerprints, and model identity against `data/dataset/`, and verify synthetic-fixture tests reject each mismatched field by name and accept valid fixtures
- [x] 1.3 Write the synthetic fixture generator (small fake detection/OCR result trees under `baseline/evaluation/tests/fixtures/` with hand-computable numbers) and verify the fixtures load through the validator from a test

## 2. Consolidated comparison table

- [x] 2.1 Implement table construction for E1 (detection overall, per size bucket, per tilt bin), E2/E2b (OCR-only and end-to-end overall + buckets + tilt, with delta vs. OCR-only), E3 (ablation per mode on val with per-step delta vs. none), and E6 (accuracy/coverage at the selected tau), and verify fixture tests assert every row against hand-computed values
- [x] 2.2 Add tilt bins for OCR: extend `baseline/ocr/io.py` `load_samples` with `tilt_magnitude` (joined from `metadata.csv`, uniqueness-checked) and `baseline/ocr/metrics.py` with a `by_tilt_bin` breakdown mirroring `by_size_bucket` (reusing the detection `tilt_bin()` helper, or duplicating the 3-line bin function if the cross-import is awkward), and verify the existing baseline test suite still passes and a fixture test checks tilt binning against hand computation
- [x] 2.3 Write `table.json` with input digests, run id, and timestamp, and verify two runs against the same fixtures produce byte-identical table contents apart from timestamp and run id
- [x] 2.4 Make the run fail fast with the missing input named when any required file is absent, and verify a fixture test covers each missing-file case

## 3. Failure case dumps

- [x] 3.1 Implement detection failure selection from predictions vs. ground truth (best box by IoU at 0.5; ranked by miss, then IoU ascending, then score ascending; top 5; tags `miss`, `low_iou`, `multiple_detections`, `low_confidence`) and verify a fixture containing a miss, a low-IoU box, and a multi-box image produces the expected ordering and tags
- [x] 3.2 Implement recognition failure selection from the validation raw caches for both protocols (accepted-wrong by edit distance > 1, abstained, unread; top 5 per protocol; tags `no_detection`, `unread`, `abstained`, `substitution`, `missing_chars`, `extra_chars`) and verify fixture tests check tags, ordering, and that no test sample appears
- [x] 3.3 Write `failures.json` with at least 3 cases per stage including identifiers, boxes/text, and tags, and verify a run against the fixtures writes the file with the minimum counts

## 4. Sampled timing probe

- [x] 4.1 Implement the probe sample selection (`random.Random(seed).sample` over the original-scale image list, default N=100, seed 42, sample ids recorded) and verify a test shows the same seed and size select the identical image set and a different seed does not
- [x] 4.2 Implement stage timing (detection inference; OCR recognition per preprocessing mode on GT crops; preprocessing per mode) with the frozen inference settings, a 5-image warmup excluded and reported, per-stage median/p95 wall-clock latency and cumulative process max RSS, with torch/ultralytics/easyocr imported only inside the probe code path, and verify `--no-probe` completes fully offline without the baseline venv and a small real probe (N=10) writes a self-describing `timing.json` (per-stage stats, sample size, seed, settings)
- [x] 4.3 Verify probe safety: after a probe run, all frozen input files have unchanged digests (asserted in a test or documented check)

## 5. Report, figure, and end-to-end CLI

- [x] 5.1 Render `REPORT.md` from `table.json` (markdown tables plus interpretation paragraphs assembled from the numbers) and verify a fixture run produces a deterministic report whose tables match the hand-computed values
- [x] 5.2 Render the single committed 2x2 figure (detection mAP50/F1 by bucket; OCR char-accuracy and exact match by bucket per protocol; accuracy-coverage curves at selected tau; per-tilt-bin comparison) as a 160 dpi PNG and verify the file is produced and regenerated identically from the same table
- [x] 5.3 Wire the full `run` CLI (including `--probe-n`, `--probe-seed`, defaults for report and figure paths under `baseline/evaluation/`) and verify an end-to-end run against the real frozen results with `--no-probe` produces the complete artifact set and the table matches the numbers in the committed detection and OCR REPORT.md files
- [x] 5.4 Add `.gitignore` for `baseline/evaluation/results/` and verify after a run that `git status` shows only the new tracked report and figure

## 6. Real run, documentation, and follow-up

- [x] 6.1 Run the full stack on the real frozen results including the probe (N=100) and commit `baseline/evaluation/REPORT.md` and the figure, and verify the committed table matches both stage reports and the probe latencies are plausible for the CPU box
- [x] 6.2 Write `baseline/evaluation/README.md` with exact run instructions, output layout, probe methodology (sample, warmup, cumulative RSS), and limitations, and verify every documented command runs as written
- [x] 6.3 Record the deferred full-split re-run with per-image timing as a follow-up (README TODO plus a GitHub issue) and verify the issue exists and the README links it
- [x] 6.4 Run the full test suite and linter and verify `python -m unittest discover -s baseline -v` passes and `uv run ruff check .` is clean
