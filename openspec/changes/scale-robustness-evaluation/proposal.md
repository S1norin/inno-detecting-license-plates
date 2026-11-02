# Proposal: Scale-robustness evaluation on scaled dataset variants

## Why

All final experiments so far used original-scale (1.0) images and crops only.
The dataset contains 7,500 scaled variants per split set (scales 0.75 and 0.5,
15,000 variants total), and the final report currently states that scale
robustness was not evaluated. The course report needs numbers showing how
detection and recognition performance degrade as image scale decreases, so
the scaled variants must be evaluated with the same frozen and fine-tuned
systems that produced the final results.

## What Changes

- Add an additive `--scales` option to `baseline/detection/evaluate.py`
  (default `1.0`, primary behavior unchanged) so the detection harness can
  evaluate explicit scale subsets (0.75, 0.5) of the dataset variants using
  the directory-form `data/dataset/dataset.yaml`.
- Run detection evaluations on the scaled test variants for both the frozen
  YOLO11s and the fine-tuned YOLO11s (4 runs: 2 detectors x 2 scales).
- Run CRNN recognition evaluations on scaled ground-truth crops
  (`ocr_only`, scales 0.75 and 0.5) and end-to-end on scaled test images
  using each detector's own scaled predictions (`end_to_end`, 2 detectors x
  2 scales), using the existing `--scale` support in `baseline/ocr` with no
  code changes.
- Add a "Scale robustness" section and one comparison figure to
  `docs/md/final_results.md`, replacing the "not evaluated here" note.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `baseline-detection`: new requirement for scaled-variant evaluation — the
  detection baseline gains an explicit `--scales` option for evaluating the
  0.75/0.5 dataset variants, while the primary evaluation remains
  original-scale only.

## Impact

- Code: `baseline/detection/evaluate.py` (additive CLI option and metadata
  filter), new/updated tests in `baseline/tests/`.
- Data: new gitignored result caches under `baseline/detection/results/` and
  `models/crnn/results/scaled/`; no dataset, split, seed, or threshold
  changes.
- Docs: `docs/md/final_results.md` gains a scale-robustness section and
  figure; the dataset note is updated accordingly.
- No spec changes to `baseline-recognition` or `crnn-recognition`: the OCR
  harness already supports `--scale` and no requirement changes.
