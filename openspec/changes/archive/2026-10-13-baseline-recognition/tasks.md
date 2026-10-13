# Tasks

## 1. Recognizer environment and weights

- [x] 1.1 Create the `baseline/ocr/` package skeleton (`__init__.py`, `engine.py`, `ocr_only.py`, `end_to_end.py`, `confidence.py`, `metrics.py`, `charset.py`, `export_train.py`) and verify `uv run ruff check baseline/` passes. Verified: `uv run ruff check .` clean, re-checked on macOS arm64 (2026-10-12) after the #8/#9 rebase.
- [x] 1.2 Write `baseline/requirements-ocr.txt` with exact pins (easyocr, matplotlib; torch shared with the detection stage) and verify every pin imports from `.venv-baseline`. Verified by the author's machine: `baseline.ocr.setup` rejects conflicting pins, first install plus no-op re-run, all pins import.
- [x] 1.3 Add the EasyOCR detection and recognition weight entries (URLs + sha256) to `baseline/common/models.json` and verify `weights.py` fetches and verifies them into `baseline/weights/easyocr/`. Verified: archive + extracted SHA-256 pinning unit-tested (extraction, duplicate members, fail-closed corruption); real fetch verified by the author's machine.
- [x] 1.4 Implement `engine.py` (EasyOCR wrapper returning text + per-character confidences, reader built offline from the verified `model_directory`) and verify a one-crop smoke test on a `crops_gt` image returns a string and per-character confidences with no network access. Verified: offline Reader (`download_enabled=False`) unit-tested; one-crop smoke test verified by the author's machine; adapter limitations documented in REPORT.md.

## 2. Two recognition protocols

- [x] 2.1 Implement `ocr_only.py` (Path A: `ocr_<split>.csv` → GT crops → preprocess → OCR) and verify a run over the validation split writes raw results for every listed crop. Verified: all six validation caches have 700 distinct records; dataset-join and path-escape checks unit-tested.
- [x] 2.2 Implement `end_to_end.py` (Path B: full image → baseline detector → highest-score box → clamped crop → OCR; no box recorded as unread without failing) and verify a run over the validation split processes every image and records no-box images as unread. Verified: 17 `no_detection` test samples stay in the denominator; clamping and prediction-coverage checks unit-tested.
- [x] 2.3 Implement `run.py` CLI (`--protocol`, `--split`, `--mode`) and verify a full `ocr_only` val run with mode none writes `raw_ocr_only_val_none.json` with per-crop text and confidences. Verified: resumable atomic caches unit-tested (resume, changed-input rejection); full runs verified by the author's machine.

## 3. Confidence policy and metrics

- [x] 3.1 Implement `confidence.py` (mean/product/sumlog score functions, 0.50-0.99 τ sweep, selection rule accuracy+coverage−1) and verify a val run produces the accuracy-coverage curve data, a plot, and a selected τ recorded in the output. Verified: both protocols select `none` / mean / τ 0.50 on validation; sweep and log-domain handling unit-tested.
- [x] 3.2 Implement `metrics.py` (character accuracy, exact match, edit-distance-based plate P/R/F1, per size bucket) and verify correctness on a hand-computed synthetic set of predictions. Verified: hand-computed metric tests pass; REPORT tables re-computed independently.
- [x] 3.3 Wire the preprocessing ablation (none / gray_contrast / canny via `data/preprocessing.py`) and verify all three modes produce distinct raw results and per-mode metrics on val. Verified: six distinct validation caches with per-mode metrics in REPORT.md.
- [x] 3.4 Run the full evaluation: val for mode selection, then test exactly once with the selected mode for both protocols and the val-selected τ, and verify every `metrics_*.json` contains the traceability fields (engine, weight digests, protocol, mode, score function, τ, split, timestamp). Verified: one frozen-policy test run per protocol with test-inference reservations; traceability and binding checks unit-tested.

## 4. CRNN readiness and documentation

- [x] 4.1 Implement `charset.py` and generate `charset.json` and verify the set equals the distinct characters of `ocr_train.csv` plate texts and contains nothing outside CJK-province/Latin/digits. Verified: committed charset has 58 characters plus CTC blank; training-only construction unit-tested.
- [x] 4.2 Implement `export_train.py` and verify the exported training manifest lists every training crop, all crop files exist, and all texts use only charset characters. Verified by the author's machine: all 10,500 training crops validated.
- [x] 4.3 Write `baseline/ocr/README.md` (run instructions for both protocols, ablation, confidence policy, expected runtimes, outputs) and verify each documented command runs as written. Verified by the author's machine end-to-end; independent commands re-checked on macOS arm64 (2026-10-12).

## Workflow follow-up

- Requires the `baseline-detection` change to be implemented first (shared env module + detector for Path B).
- Open a PR per CONTRIBUTING.md stating how the change was verified (val/test metrics per protocol, τ selection on val, test evaluated once, ruff clean).
- Archive the change after the PR is merged.
