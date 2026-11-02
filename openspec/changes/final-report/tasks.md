# Tasks

## 1. Weights and manifest

- [x] 1.1 Add the `yolo11s-license-plate-finetuned` entry to `baseline/common/models.json` (URL `https://huggingface.co/Sinorin/inno-detecting-license-plates-weights/resolve/main/detection/yolo11_finetune/best.pt`, filename `best.pt`, cache subdir `yolo11_finetune`, SHA-256 `0a4f080fe3749b8be145101468a0dd98455cf50117d2acd37575f33e0d261055`) and verify `python -m baseline.common.weights verify yolo11s-license-plate-finetuned` resolves the entry and `uv run ruff check .` stays clean
- [x] 1.2 Fetch all four assets through the digest-verifying downloader (`weights fetch yolo11s-license-plate-finetuned`, `weights fetch crnn-full-training`, plus the two `training_record.json` files into the matching cache subdirs) and verify `weights verify` passes for both weight files and each fetched file's SHA-256 matches the digest recorded in its training record
- [x] 1.3 Run the existing baseline common tests (`.venv-baseline/bin/python -m unittest discover -s baseline/tests -v`) and verify they all pass with the new manifest entry present

## 2. Fine-tuned detector evaluation

- [x] 2.1 Evaluate the fine-tuned checkpoint on validation with `baseline/detection/evaluate.py --model` and `--validate`, output under `baseline/detection/results/finetuned-val/`, and verify the metrics JSON exists and the independent Ultralytics cross-check agrees within 0.01
- [x] 2.2 Run the one-shot test evaluation with identical settings, output under `baseline/detection/results/finetuned-test/`, and verify the metrics JSON exists with 800 image records
- [x] 2.3 Build the frozen-vs-fine-tuned comparison JSONs (`models/yolo11_finetune/results/finetune/comparison_{val,test}.json`) and verify every scope (overall, 3 size buckets, 3 tilt bins x 4 metrics x 2 splits) matches the two metrics JSONs

## 3. CRNN evaluations with the canonical checkpoint

- [x] 3.1 Run `baseline/ocr/run.py --engine crnn --protocol ocr_only` for all three modes on validation and verify each raw cache passes `validate_raw` with 700 distinct samples
- [x] 3.2 Run the end-to-end protocol on frozen YOLO validation predictions for all three modes and verify each raw cache passes `validate_raw` with 700 samples
- [x] 3.3 Run the end-to-end protocol on fine-tuned YOLO validation predictions for all three modes and verify each raw cache passes `validate_raw` with 700 samples
- [x] 3.4 Run validation selection for all three protocol groups (ocr-only, end-to-end frozen, end-to-end fine-tuned) and verify each selection JSON records mode/tau chosen from validation only
- [x] 3.5 Run the one-shot test inference for each protocol under its selected policy (3 runs, 800 samples each) and verify each test metrics JSON exists and the test-once reservation binds the policy to its cache

## 4. Timing probes

- [x] 4.1 Add `--probe-engine {easyocr,crnn}` (default `easyocr`) and `--crnn-checkpoint` to `baseline/evaluation/run.py` and wire the engine choice through `run_probe` in `timing.py`, and verify a new unit test covers the CRNN path with a synthetic digest-verified checkpoint while all existing evaluation tests stay green
- [x] 4.2 Run the probe with the fine-tuned detector weights and the CRNN engine (n=100, seed 42) and verify `timing.json` reports per-stage median/p95 latency and peak RSS for detection, each CRNN mode, and preprocessing
- [x] 4.3 Run the probe with the frozen detector weights and the CRNN engine and verify `timing.json` exists for the second configuration and both files record inference settings, sample size, seed, and machine

## 5. Failure cases and final-results document

- [x] 5.1 Extract failure cases per design D6 (at least 3 per stage: detection misses plus one multi-detection case per detector, CRNN recognition failures including one large-plate and one 10-19-degree tilt case) from the regenerated JSON caches and verify the selected sample IDs are recorded and reproducible from the caches alone
- [x] 5.2 Generate the comparison figures under `docs/figures/` (detection comparison, recognition comparison across the five configurations, preprocessing ablation, size/tilt breakdown, failure-case montage) and verify each figure exists and its values match the source metrics JSONs
- [x] 5.3 Write `docs/md/final_results.md` with dataset/split summary, detection comparison, recognition comparison, ablation, machine-labelled runtime table (training from the fetched training records, inference from the probes and the committed EasyOCR probe), failure analysis, and a reproducibility section, and verify every reported number matches its source JSON or committed report
- [x] 5.4 Update `baseline/ocr/README.md` (and `baseline/detection/README.md` where it documents weight acquisition) so the fine-tuned detector weights are documented as fetchable via the manifest, and verify the documented fetch/verify commands run as written

## 6. Integration checks

- [x] 6.1 Run `uv run ruff check .` and verify it passes for the whole repository
- [x] 6.2 Run the full test suite (`.venv-baseline/bin/python -m unittest discover -s baseline -v`, `models/crnn/tests`, `models/yolo11_finetune/tests`) and verify all tests pass
- [x] 6.3 Run `openspec validate final-report` and verify the change validates

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Write the two-page technical summary from `docs/md/final_results.md` (separate deliverable, out of scope for this change).
