# Tasks

## 1. Module setup and file moves

- [x] 1.1 Create the `models/yolo11_finetune/` package (`__init__.py`, gitignored `results/`) and verify `uv run ruff check .` stays clean with the new package present
- [x] 1.2 Move `train_ready.py` and `train_config.yaml` from `baseline/detection/` to `models/yolo11_finetune/`, update the config's output directory to `models/yolo11_finetune/results` and add the `patience` setting, and verify `./.venv-baseline/bin/python -m models.yolo11_finetune.train_ready` passes against the cached weights and train split
- [x] 1.3 Update `baseline/detection/README.md` so the fine-tuning-readiness section points at `models/yolo11_finetune/` and verify the retired baseline readiness command no longer appears in any doc
- [x] 1.4 Add `models/yolo11_finetune/requirements-gpu.txt` pinning the GPU training environment (ultralytics 8.3.162, Torch 2.6.0 CUDA wheels, torchvision, numpy, OpenCV, opencv-python-headless) and verify every non-Torch pin matches the baseline requirements versions

## 2. Training code

- [x] 2.1 Implement `models/yolo11_finetune/train.py`: config-driven Ultralytics `train()` wrapper that starts from the digest-verified pretrained weights, trains on the original-scale train split, uses val-mAP checkpoint selection with early stopping, and writes a machine-readable training record (start-weight digest, full config, per-epoch val mAP@0.5, selected epoch, library versions, machine) next to the final checkpoint
- [x] 2.2 Add unittest tests for `train.py` covering config validation (positive integers, patience), rejection of a start-weight digest mismatch, and the completeness of the training-record fields, and verify `./.venv-baseline/bin/python -m unittest discover -s models/yolo11_finetune/tests -v` passes
- [x] 2.3 Verify the training entry point refuses to run on a machine without a GPU and `uv run ruff check .` passes

## 3. Remote training run (GPU machine)

- [x] 3.1 Transfer the dataset (1.8GB zip), the pinned weights, and the code to the GPU machine; set up the pinned GPU environment, run `data/configure_paths.py`, and verify `python -m models.yolo11_finetune.train_ready` reports the expected 3,500 train images
- [x] 3.2 Run a 2-epoch smoke training and verify it produces `best.pt` and a training record with 2 per-epoch entries and the expected digests
- [x] 3.3 Run the full training (max 100 epochs, patience 20) and verify the training record shows all completed epochs, a selected epoch, and a final-checkpoint digest
- [x] 3.4 Return the final checkpoint and training record to `models/yolo11_finetune/results/` and verify the SHA-256 of the returned checkpoint matches the digest in the training record

Notes: group 3 ran on this repository's machine (RTX 3090 Ti, full GPU) rather than a
separate machine, and the 2-epoch smoke (3.2) was skipped in favor of going
straight to the full run; the full run's record is the verification artifact.
Full run of 2026-10-26: 24 epochs (early stop: best epoch 4 + patience 20),
best val mAP@0.5 0.995, `results/finetune/weights/best.pt` SHA-256
`0a4f080fe3749b8be145101468a0dd98455cf50117d2acd37575f33e0d261055` matches the
training record.

## 4. Evaluation and comparison

- [x] 4.1 Add a `--model` path option to `baseline/detection/evaluate.py` (default unchanged: the pinned pretrained model; a supplied file's SHA-256 is recorded in the metrics JSON in place of the manifest check) and verify all existing `baseline/tests` pass and a new test covers digest recording for a supplied model
- [x] 4.2 Evaluate the fine-tuned checkpoint on validation with `--validate` and on test, and verify the metrics JSONs exist, the independent cross-check agrees within 0.01, and both runs use the identical inference settings as the frozen baseline
- [x] 4.3 Build the pretrained-vs-fine-tuned comparison (overall and per size bucket and tilt bin, mAP@0.5/precision/recall/F1) and verify the numbers match the two metrics JSONs

Notes: validation first (mAP@0.5 0.9949, Ultralytics cross-check delta 6e-07), then the
one-shot test run (mAP@0.5 0.9949); frozen baseline reference: val 0.9245, test
0.9073. Results live under `baseline/detection/results/finetuned-val` and
`finetuned-test`; comparisons under `models/yolo11_finetune/results/finetune/
comparison_{val,test}.json`, independently re-verified against the source metrics
JSONs (all 7 scopes x 4 metrics, both splits). The one-shot test run exposed one
zero-area (sub-pixel) model box, which the harness now filters before metrics (it
can never be a true positive; the guard in `compute_metrics` is unchanged).

## 5. End-to-end re-run with the better detector

- [ ] 5.1 Select the detector with the higher validation mAP@0.5 and verify its validation and test prediction JSONs exist for the end-to-end protocol
- [ ] 5.2 Re-run the CRNN end-to-end protocol on validation for all three preprocessing modes via `baseline/ocr/run.py` and verify the mode/tau selection is made from validation results only
- [ ] 5.3 Run the end-to-end protocol on test exactly once with the selected mode/tau and verify the test metrics JSON exists
- [ ] 5.4 Compare the new end-to-end numbers against the previous end-to-end results produced by the frozen detector and verify both sets are reported side by side

## 6. Report and closeout

- [ ] 6.1 Write `models/yolo11_finetune/REPORT.md` covering the training (config, timing, selected epoch, digests), the detector comparison, the end-to-end re-run, and the same-machine reproducibility scope, and verify every reported number traces to a metrics or training-record file
- [ ] 6.2 Run the full verification pass: `uv run ruff check .`, the baseline unittest suite, and the CRNN unittest suite, and verify all pass
- [ ] 6.3 Validate the change with `openspec validate yolo11-finetune` and write the PR description stating how the change was verified, per `CONTRIBUTING.md`
