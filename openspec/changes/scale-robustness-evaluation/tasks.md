# Tasks: Scale-robustness evaluation

## 1. Detection harness: additive `--scales` option

- [x] 1.1 Add `--scales` (comma-separated subset of `0.5,0.75,1.0`, default `1.0`) to `baseline/detection/evaluate.py`; for a non-default subset require the directory-form `data/dataset/dataset.yaml`, enumerate the split directory sorted instead of reading a list file, filter metadata rows by `scale in scales`, keep all existing per-variant validation, record the scale subset in the run provenance, and refuse unknown scales before inference
- [x] 1.2 Refuse `--validate` for non-default scale subsets with a clear error (harness metrics remain authoritative)
- [x] 1.3 Add tests: scaled split loads the right variant rows with per-variant bucket/tilt; default run is byte-identical to before (original-scale only); unknown scale refused; `--validate` refused for scaled runs
- [x] 1.4 Verify `uv run ruff check .` passes and the full `baseline` test suite passes

## 2. Scaled detection runs (test split, both detectors)

- [x] 2.1 Frozen detector on scaled test variants: `--split test --scales 0.75,0.5 --run-id frozen-test-s075-s050` (single run covering both scales, per-scale metrics in the output)
- [x] 2.2 Fine-tuned detector on scaled test variants: `--split test --scales 0.75,0.5 --run-id finetuned-test-s075-s050 --model baseline/weights/yolo11_finetune/best.pt`
- [x] 2.3 Verify both runs: per-scale overall + size-bucket + tilt-bin metrics present, provenance records the scale subset, weight digests match `95e50c25...` / `0a4f080f...`

## 3. Scaled CRNN recognition runs (canonical checkpoint)

- [x] 3.1 `ocr_only` scale 0.75: validation for `none`/`gray_contrast`/`canny`, selection on val, one-shot test → `models/crnn/results/scaled/ocr_only_s075/`
- [x] 3.2 `ocr_only` scale 0.5: same flow → `models/crnn/results/scaled/ocr_only_s050/`
- [x] 3.3 `end_to_end` scale 0.75 with the fine-tuned detector's scaled test predictions → `models/crnn/results/scaled/end_to_end_finetuned_s075/` (val + selection + test)
- [x] 3.4 `end_to_end` scale 0.5 with the fine-tuned detector's scaled test predictions → `models/crnn/results/scaled/end_to_end_finetuned_s050/`
- [x] 3.5 `end_to_end` scale 0.75 with the frozen detector's scaled test predictions → `models/crnn/results/scaled/end_to_end_frozen_s075/`
- [x] 3.6 `end_to_end` scale 0.5 with the frozen detector's scaled test predictions → `models/crnn/results/scaled/end_to_end_frozen_s050/`
- [x] 3.7 Verify all six runs: test metrics use the val-selected mode/tau, engine is the canonical CRNN checkpoint `92d1fbea...`, sample counts are 700 val / 800 test per scale

## 4. Report section and figure

- [x] 4.1 Add `docs/figures/scale_robustness.png`: detection mAP@0.5 and F1 by scale (1.0/0.75/0.5) per detector, and CRNN test character accuracy by scale per protocol, with the 1.0 values from the existing final results
- [x] 4.2 Add a "Scale robustness" section to `docs/md/final_results.md`: per-scale detection table (both detectors), per-scale CRNN table (ocr_only and end_to_end, both detectors), selected mode/tau per scale, and a short degradation analysis; replace the "scaled variants were not evaluated" note with a pointer to the section
- [x] 4.3 Verify every number in the new section and figure against the run outputs

## 5. Integration checks

- [x] 5.1 `uv run ruff check .` passes
- [x] 5.2 Full test suites pass: `baseline` (110+ tests), `models/crnn/tests`, `models/yolo11_finetune/tests`
- [x] 5.3 `openspec validate scale-robustness-evaluation` passes
