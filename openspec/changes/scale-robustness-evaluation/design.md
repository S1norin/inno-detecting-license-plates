# Design: Scale-robustness evaluation

## Context

See proposal.md for motivation. Verified constraints that shape the approach:

- `baseline/detection/evaluate.py` hard-filters metadata to `scale == 1.0`
  and requires the split list to cover exactly those rows; its default data
  config is the list-form `dataset_original.yaml`.
- `data/dataset/dataset.yaml` is directory-form (`images/{split}`) and
  covers all 15,000 variants; every variant has its own metadata row, label
  file, `size_bucket`, and `tilt_magnitude`.
- `baseline/ocr` already supports `--scale {0.5,0.75,1.0}` end to end:
  `load_samples` filters `ocr_{split}.csv` by scale, the test-inference guard
  requires the test run to match a validation selection made for the same
  scale, and `end_to_end` joins detector predictions by image path.
- Scaled crops and images exist for all splits (800 test variants per scale).
- The `baseline-detection` spec pins the primary evaluation to original
  scale; scaled evaluation is an additive requirement.

## Goals / Non-Goals

**Goals:**
- Reproducible scaled detection runs for both detectors with the same
  matching, metrics, breakdowns, and provenance as the primary run.
- CRNN recognition numbers on scaled crops and scaled end-to-end images,
  selected on val and reported on test per scale.
- One report section and one figure that quantify scale degradation.

**Non-Goals:**
- No dataset, split, seed, threshold, or scale-factor changes.
- No training or fine-tuning on scaled variants.
- No ultralytics `--validate` cross-check for scaled runs.
- No timing/latency probe for scaled runs (latency is scale-invariant at a
  fixed 640px input; not needed for the report).

## Decisions

- **D1: Additive `--scales` option in the detection evaluator.**
  `--scales 0.75,0.5` (comma-separated subset of the dataset scales,
  default `1.0`). For the default, the code path is untouched. For a
  non-default subset, the data config must be directory-form
  (`dataset.yaml`); the loader enumerates the split directory (sorted)
  instead of reading a list file, filters metadata rows by
  `scale in scales`, and applies the existing per-variant validation
  (label/metadata agreement, bucket, tilt) unchanged. Unknown scales are
  refused before inference. The evaluated scale subset is recorded in the
  run provenance. Alternative considered: a separate script for scaled
  runs — rejected because it would duplicate the matching/metrics code.

- **D2: `--validate` is refused for scaled runs.** The ultralytics
  cross-check needs a scale-filtered list-form YAML; generating one is
  out of scope. The harness metrics use the same code path as the primary
  run, so they remain authoritative. A clear error message explains this.

- **D3: OCR runs need no code changes.** Each (protocol, scale, detector)
  combination gets its own output directory under
  `models/crnn/results/scaled/` with the standard flow: validation for all
  three preprocessing modes, selection on val, then one-shot test with the
  selected mode/tau. Per-scale selection is expected and honest; the
  selected mode/tau per scale is recorded in the report.

- **D4: End-to-end on scaled images uses each detector's own scaled
  predictions** (top-1 box per image, as in the original-scale protocol),
  so detector quality and recognizer quality are combined per scale for
  both the frozen and fine-tuned detectors.

- **D5: Report as a section of `docs/md/final_results.md`** plus one
  figure `docs/figures/scale_robustness.png` (detection mAP@0.5/F1 by
  scale per detector; CRNN character accuracy by scale per protocol).
  The dataset note "scaled variants were not evaluated" is replaced with a
  pointer to the new section.

- **D6: Identical deterministic settings to the primary runs** — IoU 0.5,
  confidence floor 0.001, NMS 0.7, 640px input, CPU single thread, seed 42,
  canonical CRNN checkpoint `92d1fbea...`, fine-tuned `best.pt`
  `0a4f080f...`, frozen `license-plate-finetune-v1s.pt` `95e50c25...`.

## Risks / Trade-offs

- [Scaled small plates degrade sharply] → Expected; it is the measurement
  target. Breakdowns per size bucket make the degradation explicit.
- [No ultralytics cross-check for scaled runs] → Mitigated by reusing the
  exact harness matching code validated for the primary run.
- [Per-scale OCR selection may pick different modes/taus] → Document the
  selected policy per scale; selection stays val-only, test is one-shot.
- [Directory listing order must be stable] → Loader sorts directory
  entries; fingerprints (config, metadata, image/label hashes) are
  recorded as in the primary run.

## Migration Plan

Additive only. Rollback: delete the new gitignored result directories and
revert the `evaluate.py`/test changes; no data artifacts are affected.
