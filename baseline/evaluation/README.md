# Baseline evaluation stack

Consolidates the frozen baseline stage results (YOLO detection + EasyOCR) into
one comparable report: one metrics table, failure cases, and a sampled timing
probe. It reads existing result JSONs and never re-trains or re-selects
anything; the only new inference is the optional timing probe.

Scope is the baselines only (detector + EasyOCR are hardcoded). Extending it
to a fine-tuned model means pointing `--detection-results` / `--ocr-results`
at that model's results directory; the provenance checks below reject anything
that does not match the dataset.

## Inputs

- `--dataset` — dataset directory with `metadata.csv` and `ocr_{val,test}.csv`
- `--detection-results` — directory with `predictions_{val,test}.json` and
  `metrics_{val,test}.json` (from `baseline/detection/run.py`)
- `--ocr-results` — directory with `ocr_only/` and `end_to_end/` protocol
  folders, each holding `raw_{protocol}_val_{mode}.json` for all three modes,
  `selection_{protocol}_val.json`, and `report_{protocol}_test.json`
  (from `baseline/ocr/run.py` + `baseline/ocr/evaluate.py`; the stage outputs
  `accuracy_coverage_val.json` and `metrics_{protocol}_test_{mode}.json` are
  copied under these canonical names — the stage files themselves are never
  modified)

Every input is validated against the dataset (splits, scales, sample counts,
dataset digests, model identity, selection-to-cache binding) before anything
is computed.

## Usage

The OCR stage writes its selection and test report under stage-specific
names (`accuracy_coverage_val.json`, `metrics_<protocol>_test_<mode>.json`).
Assemble a canonical input directory by copying (never modifying) the stage
files:

```bash
mkdir -p baseline/evaluation/results/inputs/ocr
for protocol in ocr_only end_to_end; do
  mkdir -p baseline/evaluation/results/inputs/ocr/$protocol
  cp baseline/ocr/results/$protocol/raw_${protocol}_val_*.json \
     baseline/evaluation/results/inputs/ocr/$protocol/
  cp baseline/ocr/results/$protocol/accuracy_coverage_val.json \
     baseline/evaluation/results/inputs/ocr/$protocol/selection_${protocol}_val.json
  mode=$(.venv-baseline/bin/python -c "import json;\
    print(json.load(open('baseline/ocr/results/$protocol/accuracy_coverage_val.json'))['protocols']['$protocol']['mode'])")
  cp baseline/ocr/results/$protocol/metrics_${protocol}_test_${mode}.json \
     baseline/evaluation/results/inputs/ocr/$protocol/report_${protocol}_test.json
done
```

Offline consolidation only (stdlib, no model loading):

```bash
python -m baseline.evaluation.run \
  --dataset data/dataset \
  --detection-results baseline/detection/results/baseline \
  --ocr-results baseline/evaluation/results/inputs/ocr \
  --out baseline/evaluation/results/<run> \
  --no-probe
```

With the sampled timing probe (needs the baseline venv, detector weights, and
offline EasyOCR weights; defaults to `baseline/weights/license-plate-finetune-v1s.pt`):

```bash
.venv-baseline/bin/python -m baseline.evaluation.run \
  --dataset data/dataset \
  --detection-results baseline/detection/results/baseline \
  --ocr-results baseline/evaluation/results/inputs/ocr \
  --out baseline/evaluation/results/<run>
```

The probe samples `--probe-n` (default 100) original-scale images with
`--probe-seed` (default 42), warms up 5 images per stage, and reports
median/p95/mean plus peak RSS per stage. It is read-only: the probe never
writes into the dataset or results directories, so the frozen inputs keep
their digests.

## Outputs

Per run folder (`baseline/evaluation/results/<run>/`, gitignored):

- `table.json` — all metrics (E1 detection, E2 OCR-only, E2b end-to-end +
  delta, E3 ablation at fixed tau, E6 threshold) with input provenance and a
  deterministic run id
- `failures.json` — ranked failure cases with cause tags
- `timing.json` — probe results (absent with `--no-probe`)
- `REPORT.md` — the full human-readable report
- `figures/baseline_comparison.png` — 2x2 summary figure (160 dpi)

The tracked `REPORT.md` and `figures/baseline_comparison.png` at this package
level are copies of the latest accepted run's artifacts.

## Conventions

- E3 ablation is reported at the baseline default tau = 0.50 for every mode;
  per-mode selected taus are only used for the E2/E2b protocol rows.
- Recognition tilt bins are reported on validation only: the frozen test
  reports predate per-sample tilt records, and re-running test is not allowed.
- Failure cases are ranked deterministically (detection: misses first, then
  best IoU ascending, then score; recognition: edit distance descending, then
  sample id). Recognition cases come from validation so the test split stays
  untouched.
- Full-split per-image timing (all 1500 val+test images) is intentionally not
  done here; it is tracked as a follow-up issue:
  [Full-split timing with per-image breakdowns](https://github.com/S1norin/inno-detecting-license-plates/issues/14).

## Tests

`python -m unittest discover -s baseline` — the evaluation tests build a
synthetic result tree in a temporary directory and assert hand-computed
metrics, so they run offline in the project venv.
