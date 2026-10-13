# Tasks

## 1. Shared environment and weights module

- [x] 1.1 Create the `baseline/common/` package (`__init__.py`, `env.py`, `weights.py`, `models.json`) and verify `uv run ruff check baseline/` passes
- [x] 1.2 Implement `env.py`: create `.venv-baseline` if missing, compare installed versions against a pinned requirements file, install only missing/mismatched pins via uv, and verify a first run installs everything while a second run is a no-op
- [x] 1.3 Implement `weights.py`: download a manifest entry to `baseline/weights/`, verify sha256, use cache on match, and verify a mismatched file fails with expected-vs-actual digests and is not usable
- [x] 1.4 Write `baseline/requirements-detection.txt` with exact pins (torch, ultralytics, opencv-python-headless, numpy, newest mutually compatible) and verify all pins import from `.venv-baseline`
- [x] 1.5 Record the YOLOv11s plate detector from `huggingface.co/morsetechlab/yolov11-license-plate-detection` in `models.json` (exact file URL + sha256) and verify `weights.py` fetches and verifies it
- [x] 1.6 Add `.gitignore` entries for `.venv-baseline/`, `baseline/weights/`, `baseline/detection/results/` and verify `git status` shows no generated artifacts

Verified on macOS arm64 (2026-10-12): first run installed all 40 pins, second run
was a no-op, all direct pins import, and `weights fetch` + `weights verify`
succeed against the manifest sha256.

## 2. Detection baseline evaluation

_Deferred to the follow-up detection PR; this PR ships section 1 only._

- [ ] 2.1 Implement `baseline/detection/evaluate.py` (`--split {val,test}`): load the hash-verified weights, run CPU inference over all split images, and verify `predictions_<split>.json` contains one record per split image with boxes and scores
- [ ] 2.2 Implement `baseline/detection/metrics.py`: single-class IoU matching at 0.5, overall mAP@0.5 / precision / recall / F1@0.5, per size bucket and per tilt bin (0-9, 10-19, 20+) joined from `metadata.csv`, and verify overall mAP@0.5 matches ultralytics' built-in `val()` on the same split within 0.01
- [ ] 2.3 Run evaluation on val and test, writing `metrics_<split>.json` with model id, weight sha256, dataset config, split, and timestamp, and verify each file contains all traceability fields
- [ ] 2.4 Re-run the evaluation and verify the metrics are identical to the previous run

## 3. Fine-tuning readiness and documentation

_Deferred to the follow-up detection PR; this PR ships section 1 only._

- [ ] 3.1 Add `baseline/detection/train_config.yaml` (cached weight path, `dataset_original.yaml`, default hyperparameters) and verify it parses and every referenced path exists
- [ ] 3.2 Write `baseline/detection/README.md` with exact run instructions (env setup, evaluation, outputs) and verify each documented command runs as written from a clean checkout

## Workflow follow-up

- Open a PR per CONTRIBUTING.md stating how the change was verified (val/test metrics produced, determinism re-run, ruff clean).
- Archive the change after the PR is merged.
